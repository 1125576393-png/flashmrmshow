#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations
import gc
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import argparse
import tracemalloc
import sys
import pickle
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class to centralize all parameters"""
    # Data file paths
    DEMO_DATA_PATH: str = '375pesticides_inchikey.csv'
    PESUDO_TQDB_PATH: str = 'Pesudo-TQDB'  # Folder path
    INTF_TQDB_PATH: str = 'INTF_TQDB_EXPOS'  # Folder path, default to EXPOS data
    OUTPUT_PATH: str = 'optimization_results.csv'
    
    # Processing parameters
    CHUNK_SIZE: int = 100000
    MAX_COMPOUNDS: int = 375  # Process all compounds, None means process all
    MZ_TOLERANCE: float = 0.7
    RT_TOLERANCE: float = 2.0  # 2 minutes tolerance (RT converted to minutes)
    MSMS_TOLERANCE: float = 0.7
    PRECURSOR_MZ_MIN_DIFF: float = 14.0126
    ION_PAIR_MIN_DIFF: float = 2.0
    MAX_IONS_PER_CE: int = 10
    RT_OFFSET: float = 0.0  # Do not use RT offset
    
    # Batch processing parameters
    BATCH_SIZE: int = 50  # Number of compounds processed per batch
    SAVE_INTERVAL: int = 100  # Save intermediate results after processing this many compounds
    
    # Scoring parameters
    SENSITIVITY_WEIGHT: float = 0.5
    SPECIFICITY_WEIGHT: float = 0.5
    TOP_COMBINATIONS: int = 10  # Number of top combinations to return (applies to both EXPER and EXPOS methods)

    #specificity calculation mode
    SPECIFICITY_CALC_MODE: str = "Standard mode"   # "Standard mode" or "Stabilized mode"
    SPECIFICITY_EPS: float = 1.2e-4          # epsilon used in stabilized mode (you can tune)
    
    # QQQ conversion parameters
    CE_SLOPE: float = 0.5788
    CE_INTERCEPT: float = 9.4452
    
    # Interference calculation method selection
    USE_EXPOS_METHOD: bool = True  # True uses EXPOS method, False uses EXPER method
    
    # Input mode selection
    SINGLE_COMPOUND_MODE: bool = False  # True for single compound input mode
    TARGET_INCHIKEY: str = ""
    
    # In-memory custom database (for uploaded files)
    CUSTOM_DB_DF: Optional[pd.DataFrame] = None  # DataFrame for in-memory custom database  # Target InChIKey for single compound mode

    # In-memory demo/batch InChIKey list (from uploaded file)
    CUSTOM_DEMO_DF: Optional[pd.DataFrame] = None

class DataLoader:
    """Data loader with optimized memory usage"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_demo_data(self) -> pd.DataFrame:
        """Load demo data(file or in-memory uploaded DF)"""

        # NEW: if uploaded batch file has been parsed into a DataFrame, use it directly
        if self.config.CUSTOM_DEMO_DF is not None:
            logger.info("Using in-memory uploaded demo data (CUSTOM_DEMO_DF).")
            df = self.config.CUSTOM_DEMO_DF.copy()

            # Safety check: must contain InChIKey column
            if "InChIKey" not in df.columns:
                raise ValueError('Uploaded data must contain an "InChIKey" column.')
            return df
        
        logger.info("Reading demo_data.csv...")
        try:
            df = pd.read_csv(self.config.DEMO_DATA_PATH, low_memory=False, encoding='ISO-8859-1')
            logger.info(f"demo_data.csv contains {len(df)} rows of data")
            return df
        except Exception as e:
            logger.error(f"Failed to read demo_data.csv: {e}")
            raise
    
    def load_large_csv(self, folder_path: str, desc: str) -> pd.DataFrame:
        """Load large files from folder (all CSV files) and merge them"""
        logger.info(f"Reading {desc} from folder: {folder_path}...")
        
        # Check if it's a file or folder
        if os.path.isfile(folder_path):
            # If it's a file, use original method
            chunks = []
            try:
                for chunk in tqdm(
                    pd.read_csv(folder_path, chunksize=self.config.CHUNK_SIZE, encoding='utf-8'), 
                    desc=f"Reading {desc}"
                ):
                    chunks.append(chunk)
                    
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"{desc} contains {len(df)} rows of data")
                
                del chunks
                gc.collect()
                
                return df
            except Exception as e:
                logger.error(f"Failed to read {desc}: {e}")
                raise
        else:
            # If it's a folder, read all CSV files
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            # Get all CSV files in the folder
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            csv_files.sort()  # Sort for consistent order
            
            if not csv_files:
                raise ValueError(f"No CSV files found in folder: {folder_path}")
            
            logger.info(f"Found {len(csv_files)} CSV files in {folder_path}")
            
            all_chunks = []
            
            try:
                for csv_file in tqdm(csv_files, desc=f"Reading {desc} files"):
                    file_path = os.path.join(folder_path, csv_file)
                    for chunk in pd.read_csv(file_path, chunksize=self.config.CHUNK_SIZE, encoding='utf-8'):
                        all_chunks.append(chunk)
                        
                df = pd.concat(all_chunks, ignore_index=True)
                logger.info(f"{desc} contains {len(df)} rows of data (from {len(csv_files)} files)")
                
                # Clean up memory
                del all_chunks
                gc.collect()
                
                return df
            except Exception as e:
                logger.error(f"Failed to read {desc} from folder {folder_path}: {e}")
                raise


class LazyFileLoader:
    """Lazy file loader that queries data on-demand without loading everything into memory"""
    
    def __init__(self, config: Config):
        self.config = config
        self.index_cache_dir = '.index_cache'
        os.makedirs(self.index_cache_dir, exist_ok=True)
        self.file_indexes = {}  # Cache file indexes
    
    def _get_index_path(self, source_path: str) -> str:
        """Get index file path"""
        path_hash = hashlib.md5(source_path.encode()).hexdigest()[:16]
        index_name = f"index_{os.path.basename(source_path)}_{path_hash}.pkl"
        return os.path.join(self.index_cache_dir, index_name)
    
    def _build_file_index(self, folder_path: str, desc: str) -> Dict:
        """Build index: InChIKey -> list of file_paths"""
        index_path = self._get_index_path(folder_path)
        
        # Check if index exists
        if os.path.exists(index_path):
            logger.info(f"Loading existing index for {desc}...")
            try:
                with open(index_path, 'rb') as f:
                    index = pickle.load(f)
                # Check if index format is correct (should be dict with list values)
                if isinstance(index, dict) and len(index) > 0:
                    sample_key = list(index.keys())[0]
                    # New format: list of file paths (strings)
                    # Old format: list of tuples (file_path, row_offset)
                    if isinstance(index[sample_key], list) and len(index[sample_key]) > 0:
                        if isinstance(index[sample_key][0], str):
                            # New format - OK
                            logger.info(f"Index loaded successfully ({len(index)} unique InChIKeys)")
                            return index
                        else:
                            # Old format - need to rebuild
                            logger.info("Detected old index format, rebuilding with new format...")
                            os.remove(index_path)
                    else:
                        # Invalid format - rebuild
                        logger.info("Invalid index format, rebuilding...")
                        os.remove(index_path)
            except Exception as e:
                logger.warning(f"Error loading index: {e}, rebuilding...")
                if os.path.exists(index_path):
                    os.remove(index_path)
        
        logger.info(f"Building index for {desc} (this may take a while, but only needs to be done once)...")
        index = {}
        
        csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
        logger.info(f"Indexing {len(csv_files)} files...")
        
        for csv_file in tqdm(csv_files, desc=f"Indexing {desc}"):
            file_path = os.path.join(folder_path, csv_file)
            
            # Read file in chunks and index InChIKeys
            try:
                chunk_count = 0
                for chunk in pd.read_csv(file_path, chunksize=50000, encoding='utf-8', low_memory=False):
                    chunk_count += 1
                    if 'InChIKey' in chunk.columns:
                        # Get unique InChIKeys in this chunk
                        # Convert to string and normalize
                        chunk['InChIKey'] = chunk['InChIKey'].astype(str).str.strip()
                        # Also create cleaned version (without whitespace)
                        chunk['InChIKey_clean'] = chunk['InChIKey'].str.replace(r'\s+', '', regex=True)
                        
                        # Index both original and cleaned versions
                        for inchikey in chunk['InChIKey'].dropna().unique():
                            inchikey = str(inchikey).strip()
                            if inchikey and inchikey != 'nan' and inchikey.lower() != 'none':
                                # Index original format
                                if inchikey not in index:
                                    index[inchikey] = []
                                if file_path not in index[inchikey]:
                                    index[inchikey].append(file_path)
                                
                                # Also index cleaned version (without whitespace)
                                inchikey_clean = inchikey.replace(' ', '').replace('\t', '').replace('\n', '')
                                if inchikey_clean != inchikey and inchikey_clean:
                                    if inchikey_clean not in index:
                                        index[inchikey_clean] = []
                                    if file_path not in index[inchikey_clean]:
                                        index[inchikey_clean].append(file_path)
                        
                        # Also index cleaned versions directly
                        for inchikey_clean in chunk['InChIKey_clean'].dropna().unique():
                            inchikey_clean = str(inchikey_clean).strip()
                            if inchikey_clean and inchikey_clean != 'nan' and inchikey_clean.lower() != 'none':
                                if inchikey_clean not in index:
                                    index[inchikey_clean] = []
                                if file_path not in index[inchikey_clean]:
                                    index[inchikey_clean].append(file_path)
                    # Log progress for large files
                    if chunk_count % 100 == 0:
                        logger.debug(f"  Indexed {chunk_count} chunks from {csv_file}")
            except Exception as e:
                logger.warning(f"Error indexing {csv_file}: {e}")
                continue
        
        # Save index
        with open(index_path, 'wb') as f:
            pickle.dump(index, f)
        logger.info(f"Index saved to {index_path} (contains {len(index)} unique InChIKeys)")
        
        return index
    
    def query_by_inchikey(self, folder_path: str, inchikey: str, desc: str = "") -> pd.DataFrame:
        """Query data by InChIKey - only loads relevant files"""
        # Normalize InChIKey
        inchikey_original = str(inchikey).strip()
        inchikey = inchikey_original
        
        # Build or load index
        if folder_path not in self.file_indexes:
            self.file_indexes[folder_path] = self._build_file_index(folder_path, desc)
        
        index = self.file_indexes[folder_path]
        
        # Try exact match first
        if inchikey not in index:
            # Try case-insensitive search
            inchikey_lower = inchikey.lower()
            found = False
            matching_key = None
            similar_keys = []
            
            for key in index.keys():
                key_str = str(key).strip()
                if key_str.lower() == inchikey_lower:
                    matching_key = key_str
                    found = True
                    break
                # Also collect similar keys for debugging
                if inchikey_lower in key_str.lower() or key_str.lower() in inchikey_lower:
                    similar_keys.append(key_str)
            
            if found:
                inchikey = matching_key
                logger.info(f"Found InChIKey with case-insensitive match: {inchikey}")
            else:
                # Log similar keys for debugging
                if similar_keys:
                    logger.warning(f"InChIKey '{inchikey_original}' not found in index, but found {len(similar_keys)} similar keys (first 5): {similar_keys[:5]}")
                else:
                    logger.warning(f"InChIKey '{inchikey_original}' not found in index (index contains {len(index)} keys)")
                # Don't do full file search to avoid memory issues
                return pd.DataFrame()
        
        # Load only relevant files
        all_data = []
        file_paths = index[inchikey]
        logger.info(f"Found InChIKey '{inchikey}' in {len(file_paths)} file(s)")
        
        for file_path in file_paths:
            try:
                # Read file in chunks and filter
                chunk_count = 0
                for chunk in pd.read_csv(file_path, chunksize=50000, encoding='utf-8', low_memory=False):
                    chunk_count += 1
                    if 'InChIKey' in chunk.columns:
                        # Normalize InChIKey column for comparison
                        chunk['InChIKey'] = chunk['InChIKey'].astype(str).str.strip()
                        # Also create cleaned version for matching
                        chunk['InChIKey_clean'] = chunk['InChIKey'].str.replace(r'\s+', '', regex=True)
                        inchikey_clean = inchikey.replace(' ', '').replace('\t', '').replace('\n', '')
                        
                        # Try multiple matching strategies
                        mask = (
                            (chunk['InChIKey'] == inchikey) |
                            (chunk['InChIKey_clean'] == inchikey_clean) |
                            (chunk['InChIKey'].str.lower() == inchikey.lower())
                        )
                        filtered = chunk[mask]
                        if len(filtered) > 0:
                            all_data.append(filtered)
                            logger.debug(f"Found {len(filtered)} rows in chunk {chunk_count} of {os.path.basename(file_path)}")
            except Exception as e:
                logger.warning(f"Error reading {file_path} for InChIKey {inchikey}: {e}")
                # Fallback: try reading entire file
                try:
                    logger.info(f"Trying to read entire file {os.path.basename(file_path)}...")
                    chunk = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                    if 'InChIKey' in chunk.columns:
                        chunk['InChIKey'] = chunk['InChIKey'].astype(str).str.strip()
                        filtered = chunk[chunk['InChIKey'] == inchikey]
                        if len(filtered) > 0:
                            all_data.append(filtered)
                            logger.info(f"Found {len(filtered)} rows in {os.path.basename(file_path)}")
                except Exception as e2:
                    logger.warning(f"Fallback read also failed for {file_path}: {e2}")
                    continue
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully found {len(result)} total rows for InChIKey {inchikey}")
            return result
        else:
            logger.warning(f"No data found for InChIKey {inchikey} in indexed files")
            return pd.DataFrame()
    
    def query_interference_by_range(self, folder_or_file_path: str, precursormz: float, rt: float,
                                   mz_tolerance: float, rt_tolerance: float, 
                                   use_avg_mz: bool = False, desc: str = "") -> pd.DataFrame:
        """Query interference data by m/z and RT range - loads files in chunks
        Supports both folder path (multiple CSV files) and single file path
        """
        # Check if it's a file or folder
        if os.path.isfile(folder_or_file_path):
            # Single file mode
            csv_files = [os.path.basename(folder_or_file_path)]
            folder_path = os.path.dirname(folder_or_file_path)
        elif os.path.isdir(folder_or_file_path):
            # Folder mode
            folder_path = folder_or_file_path
            csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
        else:
            logger.error(f"Path does not exist: {folder_or_file_path}")
            return pd.DataFrame()
        
        if not csv_files:
            logger.warning(f"No CSV files found in {folder_or_file_path}")
            return pd.DataFrame()
        
        all_data = []
        
        # Process files in smaller batches to control memory
        batch_size = 375  # Process files in batches
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i:i+batch_size]
            batch_data = []
            
            for csv_file in batch_files:
                if os.path.isfile(folder_or_file_path):
                    # Single file mode - use the original path
                    file_path = folder_or_file_path
                else:
                    # Folder mode - join folder and file
                    file_path = os.path.join(folder_path, csv_file)
                
                try:
                    # Read file in chunks and filter
                    for chunk in pd.read_csv(file_path, chunksize=50000, encoding='utf-8', low_memory=False):
                        if use_avg_mz:
                            if 'Average Mz' in chunk.columns and 'Average Rt(min)' in chunk.columns:
                                mask = (
                                    (abs(chunk['Average Mz'] - precursormz) <= mz_tolerance) &
                                    (abs(chunk['Average Rt(min)'] - rt) <= rt_tolerance)
                                )
                        else:
                            if 'PrecursorMZ' in chunk.columns and 'RT' in chunk.columns:
                                mask = (
                                    (abs(chunk['PrecursorMZ'] - precursormz) <= mz_tolerance) &
                                    (abs(chunk['RT'] - rt) <= rt_tolerance)
                                )
                            else:
                                continue
                        
                        filtered = chunk[mask]
                        if len(filtered) > 0:
                            batch_data.append(filtered)
                except Exception as e:
                    logger.warning(f"Error reading {csv_file}: {e}")
                    continue
                
                # If single file mode, break after first file
                if os.path.isfile(folder_or_file_path):
                    break
            
            if batch_data:
                all_data.append(pd.concat(batch_data, ignore_index=True))
            
            # Clean up after each batch
            del batch_data
            gc.collect()
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            return result
        return pd.DataFrame()


class InterferenceCalculatorEXPER:
    """Interference calculator for EXPER method (experimentally acquired data)"""
    
    def __init__(self, config: Config):
        self.config = config
        self._msms_cache = {}  # Cache for parsed MS/MS spectra
    
    def extract_intensity_from_msms_cached(self, msms_spectrum: str, target_ion: float) -> float:
        """Extract intensity for a specific ion from MS/MS spectrum (with caching)"""
        if pd.isna(msms_spectrum) or msms_spectrum == '':
            return 0.0
        
        # Use cache to avoid repeated parsing
        cache_key = f"{msms_spectrum}_{target_ion}"
        if cache_key in self._msms_cache:
            return self._msms_cache[cache_key]
        
        try:
            peaks = msms_spectrum.split()
            total_intensity = 0.0
            
            for peak in peaks:
                if ':' in peak:
                    parts = peak.split(':', 1)
                    if len(parts) == 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            
                            if abs(mz - target_ion) <= self.config.MSMS_TOLERANCE:
                                total_intensity += intensity
                        except (ValueError, IndexError):
                            continue
            
            # Cache result
            self._msms_cache[cache_key] = total_intensity
            return total_intensity
            
        except Exception:
            self._msms_cache[cache_key] = 0.0
            return 0.0

class InterferenceCalculatorEXPOS:
    """Interference calculator for EXPOS method"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def process_combination(self, index, row, different_inchikey_rows_low, different_inchikey_rows_medium, 
                           different_inchikey_rows_high, coverage_low, coverage_medium, coverage_high, coverage_all):
        """Interference calculation for EXPOS method"""
        quan_ion = row['MSMS1']
        quan_ion_intensity = row['intensity1']
        quan_ion_nce = row['NCE1']
        quan_ion_ce = row['CE1']

        qual_ion = row['MSMS2']
        qual_ion_intensity = row['intensity2']
        qual_ion_nce = row['NCE2']
        qual_ion_ce = row['CE2']
        
        # Select coverage based on NCE
        if quan_ion_nce <= 60.0:    
            coverage1 = coverage_low
        elif 60.0 < quan_ion_nce <= 120.0:
            coverage1 = coverage_medium
        elif quan_ion_nce > 120.0:
            coverage1 = coverage_high
        else:
            coverage1 = 0
            
        if qual_ion_nce <= 60.0:    
            coverage2 = coverage_low
        elif 60.0 < qual_ion_nce <= 120.0:
            coverage2 = coverage_medium
        elif qual_ion_nce > 120.0:
            coverage2 = coverage_high
        else:
            coverage2 = 0
        
        coverage = coverage_all

        # Process data for different CE ranges
        result_rows1 = self.process_ce_range(different_inchikey_rows_low, different_inchikey_rows_medium, 
                                           different_inchikey_rows_high, quan_ion, quan_ion_nce)
        result_rows2 = self.process_ce_range(different_inchikey_rows_low, different_inchikey_rows_medium, 
                                           different_inchikey_rows_high, qual_ion, qual_ion_nce)

        common_inchikeys = set(result_rows1["InChIKey"]).union(set(result_rows2["InChIKey"]))
        hit_num = len(common_inchikeys)
        hit_rate = 0
        if coverage != 0:
            hit_rate = len(common_inchikeys)/coverage

        return hit_num, hit_rate

    def process_single_ion(self, row, different_inchikey_rows_low, different_inchikey_rows_medium, 
                           different_inchikey_rows_high, coverage_low, coverage_medium, coverage_high, coverage_all):
        """Interference calculation for single ion in EXPOS method"""
        ion = row['MSMS']
        ion_nce = row['NCE']
        
        # Select coverage based on NCE
        if ion_nce <= 60.0:    
            coverage = coverage_low
        elif 60.0 < ion_nce <= 120.0:
            coverage = coverage_medium
        elif ion_nce > 120.0:
            coverage = coverage_high
        else:
            coverage = 0
        
        # Process data for the CE range
        result_rows = self.process_ce_range(different_inchikey_rows_low, different_inchikey_rows_medium, 
                                           different_inchikey_rows_high, ion, ion_nce)
        
        hit_num = len(result_rows["InChIKey"].unique()) if len(result_rows) > 0 else 0
        hit_rate = 0
        if coverage != 0:
            hit_rate = hit_num / coverage
        
        return hit_num, hit_rate
    
    def process_ce_range(self, different_inchikey_rows_low, different_inchikey_rows_medium, 
                        different_inchikey_rows_high, ion, nce):
        """CE range processing for EXPOS method"""
        if nce <= 60.0:    
            return different_inchikey_rows_low[abs(ion - different_inchikey_rows_low['MSMS']) <= 1]
        elif 60.0 < nce <= 120.0:
            return different_inchikey_rows_medium[abs(ion - different_inchikey_rows_medium['MSMS']) <= 1]
        elif nce > 120.0:
            return different_inchikey_rows_high[abs(ion - different_inchikey_rows_high['MSMS']) <= 1]
        else:
            return pd.DataFrame()

class IonPairOptimizerEXPER:
    """Ion pair optimizer for EXPER method (experimentally acquired data)"""
    
    def __init__(self, config: Config, interference_calc: InterferenceCalculatorEXPER):
        self.config = config
        self.interference_calc = interference_calc
    
    def filter_and_rank_ions(self, working_group: pd.DataFrame) -> pd.DataFrame:
        """Filter and rank ions"""
        # Group by CE
        ce_groups = {
            'low': working_group[working_group['CE'] <= 20.0],
            'medium': working_group[
                (working_group['CE'] > 20.0) & 
                (working_group['CE'] <= 40.0)
            ],
            'high': working_group[working_group['CE'] > 40.0]
        }
        
        # Determine which name column to use
        name_col = 'Name_x' if 'Name_x' in working_group.columns else 'Name'
        
        filtered_ions = []
        
        for ce_level, group in ce_groups.items():
            if len(group) > 0:
                # Sort by intensity
                group_sorted = group.sort_values('intensity', ascending=False)
                # Deduplicate
                group_dedup = group_sorted.drop_duplicates([name_col, 'MSMS'], keep='first')
                # Take top N
                group_filtered = group_dedup.head(self.config.MAX_IONS_PER_CE)
                filtered_ions.append(group_filtered)
        
        return pd.concat(filtered_ions, ignore_index=True) if filtered_ions else pd.DataFrame()
    
    def generate_ion_pairs(self, ions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate ion pair combinations"""
        if len(ions_df) < 2:
            return pd.DataFrame()
        
        combinations_list = list(combinations(ions_df.iterrows(), 2))
        candidate_data = []
        
        for (index1, row1), (index2, row2) in combinations_list:
            if (row1['MSMS'] != row2['MSMS'] and 
                abs(row1['MSMS'] - row2['MSMS']) >= self.config.ION_PAIR_MIN_DIFF):
                candidate_data.append([
                    row1['MSMS'], row1['intensity'], row1['CE'],
                    row2['MSMS'], row2['intensity'], row2['CE']
                ])
        
        if not candidate_data:
            return pd.DataFrame()
        
        candidate_df = pd.DataFrame(candidate_data, columns=[
            'MSMS1', 'intensity1', 'CE1', 'MSMS2', 'intensity2', 'CE2'
        ])
        
        return candidate_df
    
    def calculate_scores(self, 
                        candidate_df: pd.DataFrame, 
                        interference_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate scores"""
        # Calculate interference levels
        target_ions_1 = candidate_df['MSMS1'].values
        target_ions_2 = candidate_df['MSMS2'].values
        
        interference_levels_1 = np.zeros(len(candidate_df))
        interference_levels_2 = np.zeros(len(candidate_df))
        
        for index, row in candidate_df.iterrows():
            ce1 = row['CE1']
            ce2 = row['CE2']
            
            # Select interference data based on CE
            if ce1 <= 20.0:
                intf_data_1 = interference_data['low']
            elif ce1 <= 40.0:
                intf_data_1 = interference_data['medium']
            else:
                intf_data_1 = interference_data['high']
            
            if ce2 <= 20.0:
                intf_data_2 = interference_data['low']
            elif ce2 <= 40.0:
                intf_data_2 = interference_data['medium']
            else:
                intf_data_2 = interference_data['high']
            
            # Calculate interference levels
            interference_levels_1[index] = sum(
                self.interference_calc.extract_intensity_from_msms_cached(
                    intf_row['MS/MS spectrum'], row['MSMS1']
                ) for _, intf_row in intf_data_1.iterrows()
            )
            
            interference_levels_2[index] = sum(
                self.interference_calc.extract_intensity_from_msms_cached(
                    intf_row['MS/MS spectrum'], row['MSMS2']
                ) for _, intf_row in intf_data_2.iterrows()
            )
        
        candidate_df['interference_level1'] = interference_levels_1
        candidate_df['interference_level2'] = interference_levels_2
        candidate_df['intensity_sum'] = candidate_df['intensity1'] + candidate_df['intensity2']
        candidate_df['interference_level_sum'] = interference_levels_1 + interference_levels_2
        
        # Calculate scores
        max_intensity = candidate_df['intensity_sum'].max()
        max_interference = candidate_df['interference_level_sum'].max()

        mode = getattr(self.config, "SPECIFICITY_CALC_MODE", "Standard mode")
        eps = float(getattr(self.config, "SPECIFICITY_EPS", 1.2e-4))
        
        if max_intensity > 0 and max_interference > 0:
            # Calculate scoring metrics
            candidate_df['sensitivity_score'] = candidate_df['intensity_sum'] / max_intensity
            candidate_df['intensity_score'] = candidate_df['intensity_sum'] / max_intensity
            
            # Specificity: mode switch
            if mode == "Standard mode":
                # ✅ Keep original formula
                candidate_df['specificity_score'] = -(1 + candidate_df['interference_level_sum']) / (1 + max_interference)
                candidate_df['interference_score'] = -(1 + candidate_df['interference_level_sum']) / (1 + max_interference)

            elif mode == "Stabilized mode":
                # ✅ Stabilized mode
                # Example stabilized denominator:
                denom = (1 + max_interference + eps)
                candidate_df['specificity_score'] = -(1 + candidate_df['interference_level_sum'] + eps) / denom
                candidate_df['interference_score'] = -(1 + candidate_df['interference_level_sum'] + eps) / denom

            else:
                # Fallback to Standard if unknown mode
                candidate_df['specificity_score'] = -(1 + candidate_df['interference_level_sum']) / (1 + max_interference)
                candidate_df['interference_score'] = -(1 + candidate_df['interference_level_sum']) / (1 + max_interference)
            
            # Combined score
            candidate_df['score'] = (
                candidate_df['sensitivity_score'] * self.config.SENSITIVITY_WEIGHT +
                candidate_df['specificity_score'] * self.config.SPECIFICITY_WEIGHT
            )
        else:
            candidate_df['score'] = candidate_df['intensity_sum']
            candidate_df['sensitivity_score'] = candidate_df['intensity_sum']
            candidate_df['specificity_score'] = -candidate_df['interference_level_sum']
            candidate_df['intensity_score'] = candidate_df['intensity_sum']
            candidate_df['interference_score'] = -candidate_df['interference_level_sum']
        
        return candidate_df
    
    def select_best_pairs(self, candidate_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Select best ion pairs"""
        # Deduplication
        candidate_df['MSMS_combined'] = candidate_df.apply(
            lambda row: tuple(sorted([row['MSMS1'], row['MSMS2']])), axis=1
        )
        candidate_df = candidate_df.loc[candidate_df.groupby('MSMS_combined')['score'].idxmax()]
        candidate_df = candidate_df.drop(columns=['MSMS_combined'])
        
        # Get best combination
        max_row = candidate_df.loc[candidate_df["score"].idxmax()]
        
        # Get top N best combinations (CE conversion will be done in MRMOptimizer)
        top_n = min(self.config.TOP_COMBINATIONS, len(candidate_df))
        top_rows = candidate_df.nlargest(top_n, 'score').copy()
        top_rows = top_rows.reset_index(drop=True)
        
        return max_row, top_rows

class IonPairOptimizerEXPOS:
    """Ion pair optimizer for EXPOS method"""
    
    def __init__(self, config: Config, interference_calc: InterferenceCalculatorEXPOS):
        self.config = config
        self.interference_calc = interference_calc
    
    def filter_and_rank_ions(self, working_group_inchikey: pd.DataFrame) -> pd.DataFrame:
        """Filter and rank ions for EXPOS method"""
        # Determine which name column to use
        name_col = 'Name_x' if 'Name_x' in working_group_inchikey.columns else 'Name'
        
        # Split working_group_inchikey into three sub-tables by NCE
        working_group_inchikey_low = working_group_inchikey[working_group_inchikey['NCE'] <= 60.0]
        working_group_inchikey_low = working_group_inchikey_low.sort_values('intensity', ascending=False)
        working_group_inchikey_low = working_group_inchikey_low.drop_duplicates([name_col, 'MSMS'], keep='first')
        working_group_inchikey_low = working_group_inchikey_low.head(10)
        
        working_group_inchikey_medium = working_group_inchikey[
            (working_group_inchikey['NCE'] > 60.0) & (working_group_inchikey['NCE'] <= 120.0)
        ]
        working_group_inchikey_medium = working_group_inchikey_medium.sort_values('intensity', ascending=False)
        working_group_inchikey_medium = working_group_inchikey_medium.drop_duplicates([name_col, 'MSMS'], keep='first')
        working_group_inchikey_medium = working_group_inchikey_medium.head(10)
        
        working_group_inchikey_high = working_group_inchikey[working_group_inchikey['NCE'] > 120.0]
        working_group_inchikey_high = working_group_inchikey_high.sort_values('intensity', ascending=False)
        working_group_inchikey_high = working_group_inchikey_high.drop_duplicates([name_col, 'MSMS'], keep='first')
        working_group_inchikey_high = working_group_inchikey_high.head(10)
        
        working_group = pd.concat([working_group_inchikey_low, working_group_inchikey_medium, working_group_inchikey_high], ignore_index=True)
        
        return working_group
    
    def generate_ion_pairs(self, working_group: pd.DataFrame) -> pd.DataFrame:
        """Generate ion pair combinations for EXPOS method"""
        if len(working_group) < 1:
            return pd.DataFrame()
        
        # Sort by intensity
        working_group_sorted = working_group.sort_values('intensity', ascending=False)
        
        # Deduplicate MSMS with tolerance 0.001 da (keep the one with highest intensity)
        msms_tolerance = 0.001
        unique_ions = []
        used_msms = []
        
        for index, row in working_group_sorted.iterrows():
            msms = row['MSMS']
            # Check if this MSMS is too close to any already selected MSMS
            is_too_close = False
            for used_msms_val in used_msms:
                if abs(msms - used_msms_val) < msms_tolerance:
                    is_too_close = True
                    break
            
            if not is_too_close:
                unique_ions.append(row.to_dict())
                used_msms.append(msms)
        
        if len(unique_ions) < 2:
            return pd.DataFrame()
        
        # Generate ion pair combinations from unique ions
        unique_ions_df = pd.DataFrame(unique_ions).reset_index(drop=True)
        combinations_list = list(combinations(unique_ions_df.iterrows(), 2))
        
        candidate_columns = ['MSMS1', 'intensity1', 'NCE1', 'CE1', 'MSMS2', 'intensity2', 'NCE2', 'CE2']
        candidate_data = []
        
        for (index1, row1), (index2, row2) in combinations_list:
            if row1['MSMS'] != row2['MSMS'] and abs(row1['MSMS'] - row2['MSMS']) >= 2.0:
                candidate_data.append([
                    row1['MSMS'], row1['intensity'], row1['NCE'], row1['CE'],
                    row2['MSMS'], row2['intensity'], row2['NCE'], row2['CE']
                ])
        
        candidate_df = pd.DataFrame(candidate_data, columns=candidate_columns)
        return candidate_df
    
    def calculate_scores(self, candidate_df: pd.DataFrame, different_inchikey_rows_low, 
                        different_inchikey_rows_medium, different_inchikey_rows_high,
                        coverage_low, coverage_medium, coverage_high, coverage_all) -> pd.DataFrame:
        """Calculate scores for EXPOS method (ion pair mode)"""
        # Calculate interference for each ion pair
        hit_nums = []
        hit_rates = []
        
        for index, row in candidate_df.iterrows():
            hit_num, hit_rate = self.interference_calc.process_combination(
                index, row, different_inchikey_rows_low, different_inchikey_rows_medium, 
                different_inchikey_rows_high, coverage_low, coverage_medium, coverage_high, coverage_all
            )
            hit_nums.append(hit_num)
            hit_rates.append(hit_rate)
        
        candidate_df['hit_num'] = hit_nums
        candidate_df['hit_rate'] = hit_rates
        
        # Calculate intensity sum (two channels combined)
        candidate_df['intensity_sum'] = candidate_df['intensity1'] + candidate_df['intensity2']
        
        # Calculate Sensitivity Score and Specificity Score
        max_intensity_sum = candidate_df['intensity_sum'].max()
        max_hit_num = candidate_df['hit_num'].max()

         # Sensitivity Score = 当前intensity_sum / 所有组合中最大的intensity_sum
        if max_intensity_sum > 0:
            candidate_df['sensitivity_score'] = candidate_df['intensity_sum'] / max_intensity_sum
        else:
            candidate_df['sensitivity_score'] = 0

        #cal mode
        mode = getattr(self.config, "SPECIFICITY_CALC_MODE", "Standard mode")
        eps = float(getattr(self.config, "SPECIFICITY_EPS", 1.2e-6))

        # Specificity score (mode switch)
        if max_hit_num > 0:
            if mode == "Standard mode":
            #Keep original formula
                candidate_df['specificity_score'] = 1 - candidate_df['hit_num'] / max_hit_num

            elif mode == "Stabilized mode":
                #Stabilized mode
                denom = (max_hit_num + eps)

                #Replace this line with your stabilized specificity formula
                candidate_df['specificity_score'] = 1 - ( candidate_df['hit_num'] + eps ) / denom

            else:
                # Fallback to Standard
                candidate_df['specificity_score'] = 1 - candidate_df['hit_num'] / max_hit_num
        else:
            candidate_df['specificity_score'] = 1

        # Score = weighted combination of sensitivity_score and specificity_score (unchanged)
        candidate_df['score'] = (
            candidate_df['sensitivity_score'] * self.config.SENSITIVITY_WEIGHT +
            candidate_df['specificity_score'] * self.config.SPECIFICITY_WEIGHT
        )

        return candidate_df
    
    def select_best_pairs(self, candidate_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Select best ion pairs for EXPOS method"""
        # Deduplication by MSMS pair
        candidate_df['MSMS_combined'] = candidate_df.apply(
            lambda row: tuple(sorted([row['MSMS1'], row['MSMS2']])), axis=1
        )
        candidate_df = candidate_df.drop_duplicates(subset='MSMS_combined')
        candidate_df = candidate_df.drop(columns=['MSMS_combined'])
        
        # Get best combination
        max_row = candidate_df.loc[candidate_df["score"].idxmax()]
        top_n = min(self.config.TOP_COMBINATIONS, len(candidate_df))
        top_rows = candidate_df.nlargest(top_n, 'score')
        top_rows = top_rows.reset_index(drop=True)
        
        return max_row, top_rows

class MemoryMonitor:
    """Memory usage monitor"""
    
    def __init__(self):
        self.max_memory_mb = 0
        self.memory_snapshots = []
        tracemalloc.start()
    
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        current, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)  # Convert to MB
    
    def snapshot(self, label: str = ""):
        """Take a memory snapshot"""
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)
        
        if peak_mb > self.max_memory_mb:
            self.max_memory_mb = peak_mb
        
        self.memory_snapshots.append({
            'label': label,
            'current_mb': current_mb,
            'peak_mb': peak_mb
        })
        
        return current_mb, peak_mb
    
    def log_snapshot(self, label: str = ""):
        """Take snapshot and log it"""
        current_mb, peak_mb = self.snapshot(label)
        logger.info(f"[内存监控] {label}: 当前={current_mb:.2f} MB, 峰值={peak_mb:.2f} MB")
        return current_mb, peak_mb
    
    def get_summary(self) -> Dict:
        """Get memory usage summary"""
        return {
            'max_memory_mb': self.max_memory_mb,
            'max_memory_gb': self.max_memory_mb / 1024,
            'snapshots': self.memory_snapshots
        }


class MRMOptimizer:
    """Main optimizer class"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # NEW: 加一个映射
        # If frontend passes TOP_PRODUCT_IONS, map it to existing TOP_COMBINATIONS
        if hasattr(self.config, "TOP_PRODUCT_IONS") and getattr(self.config, "TOP_PRODUCT_IONS") is not None:
            self.config.TOP_COMBINATIONS = int(getattr(self.config, "TOP_PRODUCT_IONS"))
            
        self.data_loader = DataLoader(self.config)
        
        # Initialize lazy file loader for memory-efficient queries
        self.lazy_loader = LazyFileLoader(self.config)
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.log_snapshot("初始化完成")
        
        # Initialize different components based on method selection
        if self.config.USE_EXPOS_METHOD:
            self.interference_calc = InterferenceCalculatorEXPOS(self.config)
            self.ion_optimizer = IonPairOptimizerEXPOS(self.config, self.interference_calc)
        else:
            self.interference_calc = InterferenceCalculatorEXPER(self.config)
            self.ion_optimizer = IonPairOptimizerEXPER(self.config, self.interference_calc)
        
        # Data storage - use lazy loading mode
        self.demo_df = None
        self.pesudo_df = None  # Will not be loaded in lazy mode
        self.intf_df = None    # Will not be loaded in lazy mode
        self.matched_df = None
        self.unique_inchikeys = None  # Cache unique InChIKeys from demo_data
    
    def load_all_data(self):
        """Load all data - using lazy loading to reduce memory"""
        logger.info("使用内存优化模式：按需加载数据，不一次性加载所有文件")
        
        # In single compound mode, we don't need demo_data
        if not self.config.SINGLE_COMPOUND_MODE:
            self.demo_df = self.data_loader.load_demo_data()
            self.unique_inchikeys = self.demo_df['InChIKey'].unique().tolist()
            self.memory_monitor.log_snapshot("加载 demo_data 完成")
            logger.info(f"找到 {len(self.unique_inchikeys)} 个唯一的 InChIKey")
        else:
            # Still load demo_data but we won't use it for matching
            self.demo_df = None
            self.unique_inchikeys = None
        
        # Don't load large files - will query on-demand
        self.pesudo_df = None
        self.intf_df = None
        self.matched_df = None
        
        logger.info("大文件将按需查询，内存占用将保持在较低水平")
        self.memory_monitor.log_snapshot("数据加载准备完成")
    
    def check_inchikey_exists(self, target_inchikey: str) -> bool:
        """Check if a specific InChIKey exists"""
        # Check in demo_data first (if available)
        if self.demo_df is not None and target_inchikey in self.demo_df['InChIKey'].values:
            return True
        
        # Query on-demand from Pesudo-TQDB
        test_df = self.lazy_loader.query_by_inchikey(
            self.config.PESUDO_TQDB_PATH, target_inchikey, "Pesudo-TQDB"
        )
        return len(test_df) > 0
    
    def _save_intermediate_results(self, results: List[Dict], processed_count: int):
        """Save intermediate results"""
        if results:
            method_suffix = "expos" if self.config.USE_EXPOS_METHOD else "exper"
            intermediate_path = f"MRM_optimization_intermediate_{method_suffix}_{processed_count}.csv"
            result_df = pd.DataFrame(results)
            result_df.to_csv(intermediate_path, index=False, encoding='utf-8')
            logger.info(f"Intermediate results saved to {intermediate_path}")
    
    def process_compound_expos(self, inchikey: str) -> Optional[Dict]:
        """Process a single compound using EXPOS method"""
        logger.info(f"Processing InChIKey: {inchikey}")
        
        # Query data on-demand using lazy loader
        working_group_inchikey = self.lazy_loader.query_by_inchikey(
            self.config.PESUDO_TQDB_PATH, inchikey, "Pesudo-TQDB"
        )
        
        if len(working_group_inchikey) == 0:
            logger.warning(f"  No data found for InChIKey, skipping")
            return None
        
        # Keep only [M+H]+ type data
        working_group_inchikey = working_group_inchikey[working_group_inchikey['Precursor_type'] == '[M+H]+']
        
        if len(working_group_inchikey) == 0:
            logger.warning(f"  No [M+H]+ type data found, skipping")
            return None
        
        # Get basic information
        first_row = working_group_inchikey.iloc[0]
        precursormz = first_row['PrecursorMZ']
        rt = first_row['RT']  # Use RT
        ion_mode = first_row['Ion_mode']
        # In single compound mode, use 'Name' from pesudo_df if 'Name_x' doesn't exist
        if 'Name_x' in first_row and pd.notna(first_row['Name_x']):
            chemical = first_row['Name_x']
        elif 'Name' in first_row:
            chemical = first_row['Name']
        else:
            chemical = inchikey  # Fallback to InChIKey if no name available
        
        logger.info(f"  Compound: {chemical}")
        logger.info(f"  Precursor m/z: {precursormz}")
        logger.info(f"  RT: {rt}")
        
        # Filter fragment ions
        working_group_inchikey = working_group_inchikey[
            abs(working_group_inchikey['MSMS'] - precursormz) > self.config.PRECURSOR_MZ_MIN_DIFF
        ]
        
        if len(working_group_inchikey) < 2:
            logger.warning(f"  Insufficient available ions, skipping")
            return None
        
        # Filter and rank ions using EXPOS optimizer
        working_group = self.ion_optimizer.filter_and_rank_ions(working_group_inchikey)
        
        if len(working_group) < 1:
            logger.warning(f"  Insufficient ions after filtering, skipping")
            return None
        
        # Generate ion pairs using EXPOS optimizer
        candidate_df = self.ion_optimizer.generate_ion_pairs(working_group)
        
        if len(candidate_df) < 1:
            logger.warning(f"  No valid ion pair combinations found")
            return {
                'chemical': chemical,
                'Precursor_mz': precursormz, 
                'InChIKey': inchikey, 
                'RT': rt,
                'coverage_low': 0,
                'coverage_medium': 0,
                'coverage_high': 0,
                'coverage_all': 0,
                'best_combinations': "no combination",
                'max_score': 0
            }
        
        # Prepare interference data - query on-demand
        different_inchikey_rows = self.lazy_loader.query_interference_by_range(
            self.config.INTF_TQDB_PATH, precursormz, rt, 0.7, self.config.RT_TOLERANCE,
            use_avg_mz=False, desc="Interference Database"
        )
        
        # Filter by ion_mode
        if len(different_inchikey_rows) > 0 and 'Ion_mode' in different_inchikey_rows.columns:
            different_inchikey_rows = different_inchikey_rows[
                different_inchikey_rows['Ion_mode'] == ion_mode
            ]
        
        different_inchikey_rows_low = different_inchikey_rows[different_inchikey_rows['NCE'] <= 60.0]
        different_inchikey_rows_medium = different_inchikey_rows[
            (different_inchikey_rows['NCE'] <= 120.0) & (different_inchikey_rows['NCE'] > 60.0)
        ]
        different_inchikey_rows_high = different_inchikey_rows[different_inchikey_rows['NCE'] > 120.0]
        
        coverage_low = len(different_inchikey_rows_low["InChIKey"].unique())
        coverage_medium = len(different_inchikey_rows_medium["InChIKey"].unique())
        coverage_high = len(different_inchikey_rows_high["InChIKey"].unique())
        coverage_all = len(different_inchikey_rows["InChIKey"].unique())
        
        logger.info(f"  Interference coverage - Low NCE: {coverage_low}, Medium NCE: {coverage_medium}, High NCE: {coverage_high}, Total: {coverage_all}")
        
        # Calculate scores using EXPOS optimizer
        candidate_df = self.ion_optimizer.calculate_scores(
            candidate_df, different_inchikey_rows_low, different_inchikey_rows_medium, 
            different_inchikey_rows_high, coverage_low, coverage_medium, coverage_high, coverage_all
        )
        
        # Select best ion pairs using EXPOS optimizer
        max_row, top_rows = self.ion_optimizer.select_best_pairs(candidate_df)
        
        # Calculate QQQ collision energy
        if pd.notna(max_row['CE1']) and pd.notna(max_row['CE2']):
            CE1 = self.config.CE_SLOPE * float(max_row['CE1']) + self.config.CE_INTERCEPT
            CE2 = self.config.CE_SLOPE * float(max_row['CE2']) + self.config.CE_INTERCEPT
        else:
            CE1 = 0
            CE2 = 0
        
        # Add QQQ collision energy to top_rows
        top_rows['CE_QQQ1'] = self.config.CE_SLOPE * top_rows['CE1'] + self.config.CE_INTERCEPT
        top_rows['CE_QQQ2'] = self.config.CE_SLOPE * top_rows['CE2'] + self.config.CE_INTERCEPT
        
        logger.info(f"  Best ion pair: {max_row['MSMS1']:.1f} (CE: {CE1:.1f}) / {max_row['MSMS2']:.1f} (CE: {CE2:.1f})")
        logger.info(f"  Max score: {max_row['score']:.4f}")
        logger.info(f"  Sensitivity score: {max_row['sensitivity_score']:.4f}")
        logger.info(f"  Specificity score: {max_row['specificity_score']:.4f}")
        logger.info(f"  Intensity sum: {max_row['intensity_sum']:.4f}")
        
        return {
            'chemical': chemical,
            'Precursor_mz': precursormz,
            'InChIKey': inchikey,
            'RT': rt,
            'coverage_all': coverage_all,
            'coverage_low': coverage_low,
            'coverage_medium': coverage_medium,
            'coverage_high': coverage_high,
            'MSMS1': max_row['MSMS1'],
            'MSMS2': max_row['MSMS2'],
            'CE_QQQ1': CE1,
            'CE_QQQ2': CE2,
            'best_combinations': top_rows.to_dict('records'),
            'max_score': max_row['score'],
            'max_sensitivity_score': max_row['sensitivity_score'],
            'max_specificity_score': max_row['specificity_score'],
            'max_intensity_sum': max_row['intensity_sum'],
        }
    
    def process_compound_exper(self, inchikey: str) -> Optional[Dict]:
        """Process a single compound using EXPER method (experimentally acquired data)"""
        logger.info(f"Processing InChIKey: {inchikey}")
        
        # Query data on-demand using lazy loader
        working_group = self.lazy_loader.query_by_inchikey(
            self.config.PESUDO_TQDB_PATH, inchikey, "Pesudo-TQDB"
        )
        
        if len(working_group) == 0:
            logger.warning(f"  No data found for InChIKey, skipping")
            return None
        
        # Keep only [M+H]+ type data
        working_group = working_group[working_group['Precursor_type'] == '[M+H]+']
        
        if len(working_group) == 0:
            logger.warning(f"  No [M+H]+ type data found, skipping")
            return None
        
        # Get basic information
        first_row = working_group.iloc[0]
        precursormz = first_row['PrecursorMZ']
        rt = first_row['RT'] + self.config.RT_OFFSET
        # In single compound mode, use 'Name' from pesudo_df if 'Name_x' doesn't exist
        if 'Name_x' in first_row and pd.notna(first_row['Name_x']):
            chemical = first_row['Name_x']
        elif 'Name' in first_row:
            chemical = first_row['Name']
        else:
            chemical = inchikey  # Fallback to InChIKey if no name available
        
        logger.info(f"  Compound: {chemical}")
        logger.info(f"  Precursor m/z: {precursormz}")
        logger.info(f"  RT: {rt}")
        
        # Filter fragment ions
        working_group = working_group[
            abs(working_group['MSMS'] - precursormz) > self.config.PRECURSOR_MZ_MIN_DIFF
        ]
        
        if len(working_group) < 2:
            logger.warning(f"  Insufficient available ions, skipping")
            return None
        
        # Filter and rank ions
        filtered_ions = self.ion_optimizer.filter_and_rank_ions(working_group)
        
        if len(filtered_ions) < 2:
            logger.warning(f"  Insufficient ions after filtering, skipping")
            return None
        
        # Generate ion pairs
        candidate_df = self.ion_optimizer.generate_ion_pairs(filtered_ions)
        
        if len(candidate_df) == 0:
            logger.warning(f"  No valid ion pair combinations found")
            return None
        
        logger.info(f"  Generated {len(candidate_df)} candidate ion pairs")
        
        # Prepare interference data
        interference_data = self.prepare_interference_data_exper(precursormz, rt)
        
        # Calculate coverage
        coverage = {
            'low': len(interference_data['low']['Alignment ID'].unique()) if len(interference_data['low']) > 0 else 0,
            'medium': len(interference_data['medium']['Alignment ID'].unique()) if len(interference_data['medium']) > 0 else 0,
            'high': len(interference_data['high']['Alignment ID'].unique()) if len(interference_data['high']) > 0 else 0,
            'all': len(interference_data['low']['Alignment ID'].unique()) + 
                   len(interference_data['medium']['Alignment ID'].unique()) + 
                   len(interference_data['high']['Alignment ID'].unique())
        }
        
        logger.info(f"  Interference coverage - Low CE: {coverage['low']}, Medium CE: {coverage['medium']}, High CE: {coverage['high']}, Total: {coverage['all']}")
        
        # Calculate scores
        candidate_df = self.ion_optimizer.calculate_scores(candidate_df, interference_data)
        
        # Select best ion pairs
        max_row, top_rows = self.ion_optimizer.select_best_pairs(candidate_df)
        
        # Calculate QQQ collision energy
        CE1 = self.config.CE_SLOPE * float(max_row['CE1']) + self.config.CE_INTERCEPT
        CE2 = self.config.CE_SLOPE * float(max_row['CE2']) + self.config.CE_INTERCEPT
        
        # Add QQQ collision energy to top_rows
        top_rows['CE_QQQ1'] = self.config.CE_SLOPE * top_rows['CE1'] + self.config.CE_INTERCEPT
        top_rows['CE_QQQ2'] = self.config.CE_SLOPE * top_rows['CE2'] + self.config.CE_INTERCEPT
        
        logger.info(f"  Best ion pair: {max_row['MSMS1']:.1f} (CE: {CE1:.1f}) / {max_row['MSMS2']:.1f} (CE: {CE2:.1f})")
        logger.info(f"  Max score: {max_row['score']:.4f}")
        
        return {
            'chemical': chemical,
            'Precursor_mz': precursormz,
            'InChIKey': inchikey,
            'RT': rt,
            'coverage_all': coverage['all'],
            'coverage_low': coverage['low'],
            'coverage_medium': coverage['medium'],
            'coverage_high': coverage['high'],
            'MSMS1': max_row['MSMS1'],
            'MSMS2': max_row['MSMS2'],
            'CE_QQQ1': CE1,
            'CE_QQQ2': CE2,
            'best_combinations': top_rows.to_dict('records'),
            'max_score': max_row['score'],
            'max_sensitivity_score': max_row['sensitivity_score'],
            'max_specificity_score': max_row['specificity_score'],
        }
    
    def prepare_interference_data_exper(self, precursormz: float, rt: float) -> Dict[str, pd.DataFrame]:
        """Prepare interference data (EXPER method) - query on-demand
        Supports both file-based and in-memory DataFrame
        """
        # Check if using in-memory custom database
        if self.config.CUSTOM_DB_DF is not None:
            # Use in-memory DataFrame directly
            df = self.config.CUSTOM_DB_DF
            # Filter by m/z and RT
            if 'Average Mz' in df.columns and 'Average Rt(min)' in df.columns:
                mask = (
                    (abs(df['Average Mz'] - precursormz) <= self.config.MZ_TOLERANCE) &
                    (abs(df['Average Rt(min)'] - rt) <= self.config.RT_TOLERANCE)
                )
                rt_filtered_rows = df[mask].copy()
            else:
                logger.warning("Custom database DataFrame missing required columns (Average Mz, Average Rt(min))")
                rt_filtered_rows = pd.DataFrame()
        else:
            # Query interference data on-demand from file/folder
            rt_filtered_rows = self.lazy_loader.query_interference_by_range(
                self.config.INTF_TQDB_PATH, precursormz, rt,
                self.config.MZ_TOLERANCE, self.config.RT_TOLERANCE,
                use_avg_mz=True, desc="Interference Database"
            )
        
        # Group by CE
        if len(rt_filtered_rows) > 0 and 'CE' in rt_filtered_rows.columns:
            interference_data = {
                'low': rt_filtered_rows[rt_filtered_rows['CE'] <= 20.0].copy(),
                'medium': rt_filtered_rows[
                    (rt_filtered_rows['CE'] > 20.0) & 
                    (rt_filtered_rows['CE'] <= 40.0)
                ].copy(),
                'high': rt_filtered_rows[rt_filtered_rows['CE'] > 40.0].copy()
            }
        else:
            interference_data = {
                'low': pd.DataFrame(),
                'medium': pd.DataFrame(),
                'high': pd.DataFrame()
            }
        
        return interference_data
    
    def run_optimization(self):
        """Run optimization"""
        method_name = "EXPOS" if self.config.USE_EXPOS_METHOD else "EXPER"
        logger.info(f"Starting MRM transition optimization calculation (using {method_name} method)...")
        
        # Load data
        self.load_all_data()
        
        # Initialize results table
        results = []
        
        # Handle single compound mode
        if self.config.SINGLE_COMPOUND_MODE:
            target_inchikey = self.config.TARGET_INCHIKEY.strip()
            if not target_inchikey:
                logger.error("Single compound mode enabled but no target InChIKey provided")
                return
            
            logger.info(f"Single compound mode: searching for InChIKey: {target_inchikey}")
            
            # Check if InChIKey exists
            if not self.check_inchikey_exists(target_inchikey):
                logger.warning(f"InChIKey '{target_inchikey}' not found in the database")
                # Create a not found result
                not_found_result = {
                    'chemical': 'not found',
                    'Precursor_mz': 0,
                    'InChIKey': target_inchikey,
                    'RT': 0,
                    'coverage_all': 0,
                    'coverage_low': 0,
                    'coverage_medium': 0,
                    'coverage_high': 0,
                    'MSMS1': 0,
                    'MSMS2': 0,
                    'CE_QQQ1': 0,
                    'CE_QQQ2': 0,
                    'best_combinations': "not found",
                    'max_score': 0,
                    'max_sensitivity_score': 0,
                    'max_specificity_score': 0,
                }
                results.append(not_found_result)
                
                # Save not found result
                result_df = pd.DataFrame(results)
                result_df.to_csv(self.config.OUTPUT_PATH, index=False, encoding='utf-8')
                logger.info(f"Result saved to {self.config.OUTPUT_PATH}")
                return
            
            compounds_to_process = [target_inchikey]
            logger.info(f"Found InChIKey '{target_inchikey}', starting processing...")
        else:
            # Get unique InChIKeys to process from demo_data
            if self.unique_inchikeys is None:
                raise ValueError("unique_inchikeys not initialized. Make sure demo_data is loaded.")
            unique_inchikeys_0 = self.unique_inchikeys
            logger.info(f"Need to process {len(unique_inchikeys_0)} unique InChIKeys")
            
            # Process compounds
            compounds_to_process = unique_inchikeys_0[:self.config.MAX_COMPOUNDS] if self.config.MAX_COMPOUNDS else unique_inchikeys_0
            logger.info(f"Starting processing for {len(compounds_to_process)} compounds")
        
        processed_count = 0
        error_count = 0
        start_time = pd.Timestamp.now()
        
        for i, inchikey in enumerate(tqdm(compounds_to_process, desc='Processing compounds')):
            try:
                # Select processing function based on method
                if self.config.USE_EXPOS_METHOD:
                    result = self.process_compound_expos(inchikey)
                else:
                    result = self.process_compound_exper(inchikey)
                
                if result:
                    results.append(result)
                    processed_count += 1
                else:
                    error_count += 1
                
                # Monitor memory after each compound
                if (i + 1) % 5 == 0:  # Check memory every 5 compounds
                    self.memory_monitor.log_snapshot(f"处理 {i + 1} 个化合物后")
                
                # Periodic saving of intermediate results and progress display
                if (i + 1) % self.config.SAVE_INTERVAL == 0:
                    elapsed_time = pd.Timestamp.now() - start_time
                    avg_time_per_compound = elapsed_time.total_seconds() / (i + 1)
                    remaining_compounds = len(compounds_to_process) - (i + 1)
                    estimated_remaining_time = remaining_compounds * avg_time_per_compound
                    
                    logger.info(f"Progress report:")
                    logger.info(f"  Processed: {i + 1}/{len(compounds_to_process)} compounds")
                    logger.info(f"  Successful: {processed_count}")
                    logger.info(f"  Failed/skipped: {error_count}")
                    logger.info(f"  Average time per compound: {avg_time_per_compound:.2f} seconds")
                    logger.info(f"  Estimated remaining time: {estimated_remaining_time/3600:.2f} hours")
                    logger.info(f"Saving intermediate results...")
                    self._save_intermediate_results(results, i + 1)
                    self.memory_monitor.log_snapshot("保存中间结果后")
                
                # Periodic memory cleanup
                if (i + 1) % self.config.BATCH_SIZE == 0:
                    gc.collect()
                    self.memory_monitor.log_snapshot("内存清理后")
                    
            except Exception as e:
                logger.error(f"Error processing compound {inchikey}: {e}")
                error_count += 1
                continue
        
        # Final statistics
        total_time = pd.Timestamp.now() - start_time
        logger.info(f"\nProcessing completed!")
        logger.info(f"Total time: {total_time.total_seconds()/3600:.2f} hours")
        logger.info(f"Total compounds: {len(compounds_to_process)}")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Failed/skipped: {error_count}")
        logger.info(f"Success rate: {processed_count/len(compounds_to_process)*100:.1f}%")
        
        # Final memory snapshot
        self.memory_monitor.log_snapshot("处理完成")
        
        # Save results
        if results:
            result_df = pd.DataFrame(results)
            result_df.to_csv(self.config.OUTPUT_PATH, index=False, encoding='utf-8')
            logger.info(f"Final results saved to {self.config.OUTPUT_PATH}")
            self.memory_monitor.log_snapshot("保存最终结果后")
            
            # Display results summary
            logger.info("\nResults summary:")
            summary_columns = ['chemical', 'MSMS1', 'MSMS2', 'CE_QQQ1', 'CE_QQQ2', 'max_score', 'max_sensitivity_score', 'max_specificity_score']
            available_columns = [col for col in summary_columns if col in result_df.columns]
            print(result_df[available_columns].head(10))
        else:
            logger.warning("No compounds processed successfully")
        
        # Display memory usage summary
        summary = self.memory_monitor.get_summary()
        logger.info("\n" + "="*60)
        logger.info("内存使用情况总结:")
        logger.info("="*60)
        logger.info(f"最大内存占用: {summary['max_memory_mb']:.2f} MB ({summary['max_memory_gb']:.3f} GB)")
        logger.info(f"是否超过2GB限制: {'是' if summary['max_memory_gb'] > 2.0 else '否'}")
        logger.info("\n关键节点内存使用:")
        for snapshot in summary['snapshots']:
            logger.info(f"  {snapshot['label']}: 峰值={snapshot['peak_mb']:.2f} MB")
        logger.info("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MRM Transition Optimization Tool')
    parser.add_argument('--intf-db', choices=['expos', 'exper', 'custom', 'exposome_explorer', 'experimental', 'nist', 'qe'], default='expos',
                       help='Select interference database: expos (or exposome_explorer/nist), exper (or experimental/qe), or custom (default: expos)')
    parser.add_argument('--custom-db-path', type=str, default='',
                       help='Custom database file or folder path (required when --intf-db is custom). If file, must be EXPER format CSV. If folder, specify format with --custom-db-method')
    parser.add_argument('--custom-db-method', choices=['expos', 'exper'], default='exper',
                       help='Database format for custom database folder: expos or exper (default: exper, only used when --custom-db-path is a folder)')
    parser.add_argument('--max-compounds', type=int, default=375,
                       help='Maximum number of compounds to process (default: 375)')
    parser.add_argument('--output', type=str, default='optimization_results.csv',
                       help='Output filename (default: optimization_results.csv)')
    parser.add_argument('--single-compound', action='store_true',
                       help='Enable single compound input mode')
    parser.add_argument('--inchikey', type=str, default='',
                       help='Target InChIKey for single compound mode')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = Config()
        # Support both old and new parameter names for backward compatibility
        is_expos = args.intf_db in ['expos', 'exposome_explorer', 'nist']
        is_custom = args.intf_db == 'custom'
        
        config.MAX_COMPOUNDS = args.max_compounds
        config.OUTPUT_PATH = args.output
        config.SINGLE_COMPOUND_MODE = args.single_compound
        config.TARGET_INCHIKEY = args.inchikey
        
        # Set INTF_TQDB_PATH and USE_EXPOS_METHOD based on selection
        if is_custom:
            # Custom database mode - supports both single file and folder
            if not args.custom_db_path:
                logger.error("--custom-db-path is required when --intf-db is custom")
                return
            if not os.path.exists(args.custom_db_path):
                logger.error(f"Custom database path not found: {args.custom_db_path}")
                return
            
            config.INTF_TQDB_PATH = args.custom_db_path
            
            # Custom mode: if it's a file, use EXPER method; if it's a folder, use specified method
            if os.path.isfile(args.custom_db_path):
                # Single file mode - always use EXPER method (EXPER format)
                config.USE_EXPOS_METHOD = False
                logger.info(f"Using custom interference database file: {config.INTF_TQDB_PATH}")
                logger.info(f"Custom database format: EXPER (single file mode)")
            elif os.path.isdir(args.custom_db_path):
                # Folder mode - use specified method
                config.USE_EXPOS_METHOD = (args.custom_db_method == 'expos')
                csv_files = [f for f in os.listdir(args.custom_db_path) if f.endswith('.csv')]
                if not csv_files:
                    logger.warning(f"No CSV files found in custom database path: {args.custom_db_path}")
                logger.info(f"Using custom interference database folder: {config.INTF_TQDB_PATH}")
                logger.info(f"Custom database format: {args.custom_db_method.upper()}")
            else:
                logger.error(f"Custom database path is neither a file nor a directory: {args.custom_db_path}")
                return
        elif is_expos:
            config.INTF_TQDB_PATH = 'INTF_TQDB_EXPOS'
            config.USE_EXPOS_METHOD = True
            logger.info(f"Using interference database: {config.INTF_TQDB_PATH}")
        else:
            config.INTF_TQDB_PATH = 'INTF_TQDB_EXPER'
            config.USE_EXPOS_METHOD = False
            logger.info(f"Using interference database: {config.INTF_TQDB_PATH}")
        
        logger.info(f"Using method: {'EXPOS' if config.USE_EXPOS_METHOD else 'EXPER'}")
        
        if config.SINGLE_COMPOUND_MODE:
            if not config.TARGET_INCHIKEY:
                logger.error("Single compound mode requires --inchikey parameter")
                return
            logger.info(f"Single compound mode: Target InChIKey = {config.TARGET_INCHIKEY}")
        else:
            logger.info(f"Batch mode: Processing up to {config.MAX_COMPOUNDS} compounds")
        
        # Run optimization
        optimizer = MRMOptimizer(config)
        optimizer.run_optimization()
        
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        raise

if __name__ == "__main__":
    main()



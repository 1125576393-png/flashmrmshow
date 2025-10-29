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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class to centralize all parameters"""
    # Data file paths
    DEMO_DATA_PATH: str = 'demo_data.csv'
    PESUDO_TQDB_PATH: str = 'Pesudo-TQDB.csv'
    INTF_TQDB_PATH: str = 'INTF-TQDB(from NIST).csv'  # Default to NIST data
    OUTPUT_PATH: str = 'optimization_results.csv'
    
    # Processing parameters
    CHUNK_SIZE: int = 100000
    MAX_COMPOUNDS: int = 5  # Process all compounds, None means process all
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
    SENSITIVITY_WEIGHT: float = 0.8
    SPECIFICITY_WEIGHT: float = 0.2
    
    # QQQ conversion parameters
    CE_SLOPE: float = 0.5788
    CE_INTERCEPT: float = 9.4452
    
    # Interference calculation method selection
    USE_NIST_METHOD: bool = True  # True uses NIST method, False uses QE method
    
    # Input mode selection
    SINGLE_COMPOUND_MODE: bool = False  # True for single compound input mode
    TARGET_INCHIKEY: str = ""  # Target InChIKey for single compound mode

class DataLoader:
    """Data loader with optimized memory usage"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_demo_data(self) -> pd.DataFrame:
        """Load demo data"""
        logger.info("Reading demo_data.csv...")
        try:
            df = pd.read_csv(self.config.DEMO_DATA_PATH, low_memory=False, encoding='ISO-8859-1')
            logger.info(f"demo_data.csv contains {len(df)} rows of data")
            return df
        except Exception as e:
            logger.error(f"Failed to read demo_data.csv: {e}")
            raise
    
    def load_large_csv(self, file_path: str, desc: str) -> pd.DataFrame:
        """Load large files in chunks"""
        # Check if it's a Pesudo-TQDB file
        if "Pesudo-TQDB" in file_path:
            # Try to load split files
            base_path = os.path.splitext(file_path)[0]  # Get base path without extension
            part_files = [
                f"{base_path}_split_part1.csv",
                f"{base_path}_split_part2.csv",
                f"{base_path}_split_part3.csv"
            ]
            
            # Check if all part files exist
            if all(os.path.exists(f) for f in part_files):
                logger.info(f"Found split files for {desc}, loading them instead")
                chunks = []  # List to store all data chunks
                
                # Process each part file
                for part_file in part_files:
                    logger.info(f"Reading {part_file}...")
                    try:
                        part_chunks = []  # List to store chunks for current file
                        # Read current file in chunks
                        for chunk in tqdm(
                            pd.read_csv(part_file, chunksize=self.config.CHUNK_SIZE, encoding='utf-8'), 
                            desc=f"Reading {os.path.basename(part_file)}"
                        ):
                            part_chunks.append(chunk)
                        
                        # Combine all chunks for current file
                        part_df = pd.concat(part_chunks, ignore_index=True)
                        chunks.append(part_df)  # Add to main chunks list
                        logger.info(f"Loaded {len(part_df)} rows from {os.path.basename(part_file)}")
                        
                        # Clean up memory
                        del part_chunks
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"Failed to read {part_file}: {e}")
                        raise
                
                # Combine data from all part files
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Combined {desc} contains {len(df)} rows of data")
                
                # Clean up memory
                del chunks
                gc.collect()
                
                return df
        
        # If no split files or not Pesudo-TQDB, load normally
        logger.info(f"Reading {desc}...")
        chunks = []  # List to store data chunks
        
        try:
            # Read file in chunks
            for chunk in tqdm(
                pd.read_csv(file_path, chunksize=self.config.CHUNK_SIZE, encoding='utf-8'), 
                desc=f"Reading {desc}"
            ):
                chunks.append(chunk)
                
            # Combine all chunks
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"{desc} contains {len(df)} rows of data")
            
            # Clean up memory
            del chunks
            gc.collect()
            
            return df
        except Exception as e:
            logger.error(f"Failed to read {desc}: {e}")
            raise


class InterferenceCalculatorQE:
    """Interference calculator for QE method"""
    
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

class InterferenceCalculatorNIST:
    """Interference calculator for NIST method"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def process_combination(self, index, row, different_inchikey_rows_low, different_inchikey_rows_medium, 
                           different_inchikey_rows_high, coverage_low, coverage_medium, coverage_high, coverage_all):
        """Interference calculation for NIST method"""
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

        common_inchikeys = set(result_rows1["InChIKey"]).intersection(set(result_rows2["InChIKey"]))
        hit_num = len(common_inchikeys)
        hit_rate = 0
        if coverage != 0:
            hit_rate = len(common_inchikeys)/coverage

        return hit_num, hit_rate

    def process_ce_range(self, different_inchikey_rows_low, different_inchikey_rows_medium, 
                        different_inchikey_rows_high, ion, nce):
        """CE range processing for NIST method"""
        if nce <= 60.0:    
            return different_inchikey_rows_low[abs(ion - different_inchikey_rows_low['MSMS']) <= 1]
        elif 60.0 < nce <= 120.0:
            return different_inchikey_rows_medium[abs(ion - different_inchikey_rows_medium['MSMS']) <= 1]
        elif nce > 120.0:
            return different_inchikey_rows_high[abs(ion - different_inchikey_rows_high['MSMS']) <= 1]
        else:
            return pd.DataFrame()

class IonPairOptimizerQE:
    """Ion pair optimizer for QE method"""
    
    def __init__(self, config: Config, interference_calc: InterferenceCalculatorQE):
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
        
        filtered_ions = []
        
        for ce_level, group in ce_groups.items():
            if len(group) > 0:
                # Sort by intensity
                group_sorted = group.sort_values('intensity', ascending=False)
                # Deduplicate
                group_dedup = group_sorted.drop_duplicates(['Name_x', 'MSMS'], keep='first')
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
        
        if max_intensity > 0 and max_interference > 0:
            # Calculate scoring metrics
            candidate_df['sensitivity_score'] = candidate_df['intensity_sum'] / max_intensity
            candidate_df['specificity_score'] = -(1 + candidate_df['interference_level_sum']) / (1 + max_interference)
            candidate_df['intensity_score'] = candidate_df['intensity_sum'] / max_intensity
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
        
        # Get top 5 best combinations (CE conversion will be done in MRMOptimizer)
        max5_rows = candidate_df.nlargest(5, 'score').copy()
        max5_rows = max5_rows.reset_index(drop=True)
        
        return max_row, max5_rows

class IonPairOptimizerNIST:
    """Ion pair optimizer for NIST method"""
    
    def __init__(self, config: Config, interference_calc: InterferenceCalculatorNIST):
        self.config = config
        self.interference_calc = interference_calc
    
    def filter_and_rank_ions(self, working_group_inchikey: pd.DataFrame) -> pd.DataFrame:
        """Filter and rank ions for NIST method"""
        # Split working_group_inchikey into three sub-tables by NCE
        working_group_inchikey_low = working_group_inchikey[working_group_inchikey['NCE'] <= 60.0]
        working_group_inchikey_low = working_group_inchikey_low.sort_values('intensity', ascending=False)
        working_group_inchikey_low = working_group_inchikey_low.drop_duplicates(['Name_x', 'MSMS'], keep='first')
        working_group_inchikey_low = working_group_inchikey_low.head(10)
        
        working_group_inchikey_medium = working_group_inchikey[
            (working_group_inchikey['NCE'] > 60.0) & (working_group_inchikey['NCE'] <= 120.0)
        ]
        working_group_inchikey_medium = working_group_inchikey_medium.sort_values('intensity', ascending=False)
        working_group_inchikey_medium = working_group_inchikey_medium.drop_duplicates(['Name_x', 'MSMS'], keep='first')
        working_group_inchikey_medium = working_group_inchikey_medium.head(10)
        
        working_group_inchikey_high = working_group_inchikey[working_group_inchikey['NCE'] > 120.0]
        working_group_inchikey_high = working_group_inchikey_high.sort_values('intensity', ascending=False)
        working_group_inchikey_high = working_group_inchikey_high.drop_duplicates(['Name_x', 'MSMS'], keep='first')
        working_group_inchikey_high = working_group_inchikey_high.head(10)
        
        working_group = pd.concat([working_group_inchikey_low, working_group_inchikey_medium, working_group_inchikey_high], ignore_index=True)
        
        return working_group
    
    def generate_ion_pairs(self, working_group: pd.DataFrame) -> pd.DataFrame:
        """Generate ion pair combinations for NIST method"""
        if len(working_group) < 2:
            return pd.DataFrame()
        
        # Generate ion pair combinations
        combinations_list = list(combinations(working_group.iterrows(), 2))
        
        candidate_columns = ['MSMS1','intensity1', 'NCE1', 'CE1', 'MSMS2','intensity2', 'NCE2', 'CE2']
        candidate_data = []
        
        for (index1, row1), (index2, row2) in combinations_list:
            if row1['MSMS'] != row2['MSMS'] and abs(row1['MSMS']-row2['MSMS']) >= 2.0:
                candidate_data.append([
                    row1['MSMS'], row1['intensity'], row1['NCE'], row1['CE'], 
                    row2['MSMS'], row2['intensity'], row2['NCE'], row2['CE']
                ])
        
        candidate_df = pd.DataFrame(candidate_data, columns=candidate_columns)
        return candidate_df
    
    def calculate_scores(self, candidate_df: pd.DataFrame, different_inchikey_rows_low, 
                        different_inchikey_rows_medium, different_inchikey_rows_high,
                        coverage_low, coverage_medium, coverage_high, coverage_all) -> pd.DataFrame:
        """Calculate scores for NIST method"""
        # Calculate interference
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
        
        # Calculate scores
        if coverage_all != 0:
            score = (candidate_df['intensity1'] + candidate_df['intensity2']) * 0.8 - 2 * candidate_df['hit_rate'] * 0.2
        else:
            score = (candidate_df['intensity1'] + candidate_df['intensity2']) * 0.5
        
        # Calculate sensitivity and specificity scores
        max_intensity = candidate_df['intensity1'] + candidate_df['intensity2']
        max_intensity_sum = max_intensity.max()
        max_hit_rate = candidate_df['hit_rate'].max()
        
        if max_intensity_sum > 0:
            candidate_df['sensitivity_score'] = max_intensity / max_intensity_sum
        else:
            candidate_df['sensitivity_score'] = 0
        
        if max_hit_rate > 0:
            candidate_df['specificity_score'] = -(1 + candidate_df['hit_rate']) / (1 + max_hit_rate)
        else:
            candidate_df['specificity_score'] = 0
        
        candidate_df['score'] = score
        
        return candidate_df
    
    def select_best_pairs(self, candidate_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Select best ion pairs for NIST method"""
        # Deduplication
        candidate_df['MSMS_combined'] = candidate_df.apply(
            lambda row: tuple(sorted([row['MSMS1'], row['MSMS2']])), axis=1
        )
        candidate_df = candidate_df.drop_duplicates(subset='MSMS_combined')
        candidate_df = candidate_df.drop(columns=['MSMS_combined'])
        
        # Get best combination
        max_row = candidate_df.loc[candidate_df["score"].idxmax()]
        max5_rows = candidate_df.nlargest(5, 'score')
        max5_rows = max5_rows.reset_index(drop=True)
        
        return max_row, max5_rows

class MRMOptimizer:
    """Main optimizer class"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        
        # Initialize different components based on method selection
        if self.config.USE_NIST_METHOD:
            self.interference_calc = InterferenceCalculatorNIST(self.config)
            self.ion_optimizer = IonPairOptimizerNIST(self.config, self.interference_calc)
        else:
            self.interference_calc = InterferenceCalculatorQE(self.config)
            self.ion_optimizer = IonPairOptimizerQE(self.config, self.interference_calc)
        
        # Data storage
        self.demo_df = None
        self.pesudo_df = None
        self.intf_df = None
        self.matched_df = None
    
    def load_all_data(self):
        """Load all data"""
        self.demo_df = self.data_loader.load_demo_data()
        self.pesudo_df = self.data_loader.load_large_csv(
            self.config.PESUDO_TQDB_PATH, "Pesudo-TQDB"
        )
        self.intf_df = self.data_loader.load_large_csv(
            self.config.INTF_TQDB_PATH, "Interference Database"
        )
        
        # Match data
        logger.info("Matching InChIKeys...")
        self.matched_df = pd.merge(self.demo_df, self.pesudo_df, on='InChIKey', how='inner')
        logger.info(f"Matched {len(self.matched_df)} rows of data")
        
        if len(self.matched_df) == 0:
            raise ValueError("No matching InChIKeys found, please check data")
    
    def check_inchikey_exists(self, target_inchikey: str) -> bool:
        """Check if a specific InChIKey exists in the matched data"""
        return target_inchikey in self.matched_df['InChIKey'].values
    
    def _save_intermediate_results(self, results: List[Dict], processed_count: int):
        """Save intermediate results"""
        if results:
            method_suffix = "nist" if self.config.USE_NIST_METHOD else "qe"
            intermediate_path = f"MRM_optimization_intermediate_{method_suffix}_{processed_count}.csv"
            result_df = pd.DataFrame(results)
            result_df.to_csv(intermediate_path, index=False, encoding='utf-8')
            logger.info(f"Intermediate results saved to {intermediate_path}")
    
    def process_compound_nist(self, inchikey: str) -> Optional[Dict]:
        """Process a single compound using NIST method"""
        logger.info(f"Processing InChIKey: {inchikey}")
        
        # Get all data for current InChIKey
        working_group_inchikey = self.matched_df[self.matched_df['InChIKey'] == inchikey].copy()
        
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
        chemical = first_row['Name_x']
        
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
        
        # Filter and rank ions using NIST optimizer
        working_group = self.ion_optimizer.filter_and_rank_ions(working_group_inchikey)
        
        if len(working_group) < 2:
            logger.warning(f"  Insufficient ions after filtering, skipping")
            return None
        
        # Generate ion pairs using NIST optimizer
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
                'best5_combinations': "no combination",
                'max_score': 0
            }
        
        # Prepare interference data
        precursor_covered_rows = self.intf_df[abs(self.intf_df['PrecursorMZ'] - precursormz) <= 0.7]
        different_inchikey_rows = precursor_covered_rows[precursor_covered_rows['Ion_mode'] == ion_mode]
        different_inchikey_rows = different_inchikey_rows[abs(rt - different_inchikey_rows['RT']) <= self.config.RT_TOLERANCE]
        
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
        
        # Calculate scores using NIST optimizer
        candidate_df = self.ion_optimizer.calculate_scores(
            candidate_df, different_inchikey_rows_low, different_inchikey_rows_medium, 
            different_inchikey_rows_high, coverage_low, coverage_medium, coverage_high, coverage_all
        )
        
        # Select best pairs using NIST optimizer
        max_row, max5_rows = self.ion_optimizer.select_best_pairs(candidate_df)
        
        # Calculate QQQ collision energy
        if max_row['CE1'] and max_row['CE2']:
            CE1 = self.config.CE_SLOPE * float(max_row['CE1']) + self.config.CE_INTERCEPT
            CE2 = self.config.CE_SLOPE * float(max_row['CE2']) + self.config.CE_INTERCEPT
        else:
            CE1 = 0
            CE2 = 0
        
        # Add QQQ collision energy to max5_rows
        max5_rows['CE_QQQ1'] = self.config.CE_SLOPE * max5_rows['CE1'] + self.config.CE_INTERCEPT
        max5_rows['CE_QQQ2'] = self.config.CE_SLOPE * max5_rows['CE2'] + self.config.CE_INTERCEPT
        
        logger.info(f"  Best ion pair: {max_row['MSMS1']:.1f} (CE: {CE1:.1f}) / {max_row['MSMS2']:.1f} (CE: {CE2:.1f})")
        logger.info(f"  Max score: {max_row['score']:.4f}")
        logger.info(f"  Sensitivity score: {max_row['sensitivity_score']:.4f}")
        logger.info(f"  Specificity score: {max_row['specificity_score']:.4f}")
        
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
            'best5_combinations': max5_rows.to_dict('records'),
            'max_score': max_row['score'],
            'max_sensitivity_score': max_row['sensitivity_score'],
            'max_specificity_score': max_row['specificity_score'],
        }
    
    def process_compound_qe(self, inchikey: str) -> Optional[Dict]:
        """Process a single compound using QE method"""
        logger.info(f"Processing InChIKey: {inchikey}")
        
        # Get all data for current InChIKey
        working_group = self.matched_df[self.matched_df['InChIKey'] == inchikey].copy()
        
        # Keep only [M+H]+ type data
        working_group = working_group[working_group['Precursor_type'] == '[M+H]+']
        
        if len(working_group) == 0:
            logger.warning(f"  No [M+H]+ type data found, skipping")
            return None
        
        # Get basic information
        first_row = working_group.iloc[0]
        precursormz = first_row['PrecursorMZ']
        rt = first_row['RT'] + self.config.RT_OFFSET
        chemical = first_row['Name_x']
        
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
        interference_data = self.prepare_interference_data_qe(precursormz, rt)
        
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
        max_row, max5_rows = self.ion_optimizer.select_best_pairs(candidate_df)
        
        # Calculate QQQ collision energy
        CE1 = self.config.CE_SLOPE * float(max_row['CE1']) + self.config.CE_INTERCEPT
        CE2 = self.config.CE_SLOPE * float(max_row['CE2']) + self.config.CE_INTERCEPT
        
        # Add QQQ collision energy to max5_rows
        max5_rows['CE_QQQ1'] = self.config.CE_SLOPE * max5_rows['CE1'] + self.config.CE_INTERCEPT
        max5_rows['CE_QQQ2'] = self.config.CE_SLOPE * max5_rows['CE2'] + self.config.CE_INTERCEPT
        
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
            'best5_combinations': max5_rows.to_dict('records'),
            'max_score': max_row['score'],
            'max_sensitivity_score': max_row['sensitivity_score'],
            'max_specificity_score': max_row['specificity_score'],
        }
    
    def prepare_interference_data_qe(self, precursormz: float, rt: float) -> Dict[str, pd.DataFrame]:
        """Prepare interference data (QE method)"""
        # Filter data within precursor m/z and RT range
        precursor_covered_rows = self.intf_df[
            abs(self.intf_df['Average Mz'] - precursormz) <= self.config.MZ_TOLERANCE
        ]
        rt_filtered_rows = precursor_covered_rows[
            abs(rt - precursor_covered_rows['Average Rt(min)']) <= self.config.RT_TOLERANCE
        ]
        
        # Group by CE
        interference_data = {
            'low': rt_filtered_rows[rt_filtered_rows['CE'] <= 20.0],
            'medium': rt_filtered_rows[
                (rt_filtered_rows['CE'] > 20.0) & 
                (rt_filtered_rows['CE'] <= 40.0)
            ],
            'high': rt_filtered_rows[rt_filtered_rows['CE'] > 40.0]
        }
        
        return interference_data
    
    def run_optimization(self):
        """Run optimization"""
        method_name = "NIST" if self.config.USE_NIST_METHOD else "QE"
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
                    'best5_combinations': "not found",
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
            # Get unique InChIKeys to process
            unique_inchikeys_0 = self.matched_df['InChIKey'].unique()
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
                if self.config.USE_NIST_METHOD:
                    result = self.process_compound_nist(inchikey)
                else:
                    result = self.process_compound_qe(inchikey)
                
                if result:
                    results.append(result)
                    processed_count += 1
                else:
                    error_count += 1
                
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
                
                # Periodic memory cleanup
                if (i + 1) % self.config.BATCH_SIZE == 0:
                    gc.collect()
                    
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
        
        # Save results
        if results:
            result_df = pd.DataFrame(results)
            result_df.to_csv(self.config.OUTPUT_PATH, index=False, encoding='utf-8')
            logger.info(f"Final results saved to {self.config.OUTPUT_PATH}")
            
            # Display results summary
            logger.info("\nResults summary:")
            summary_columns = ['chemical', 'MSMS1', 'MSMS2', 'CE_QQQ1', 'CE_QQQ2', 'max_score', 'max_sensitivity_score', 'max_specificity_score']
            available_columns = [col for col in summary_columns if col in result_df.columns]
            print(result_df[available_columns].head(10))
        else:
            logger.warning("No compounds processed successfully")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MRM Transition Optimization Tool')
    parser.add_argument('--intf-db', choices=['nist', 'qe'], default='nist',
                       help='Select interference database: nist or qe (default: nist)')
    parser.add_argument('--max-compounds', type=int, default=5,
                       help='Maximum number of compounds to process (default: 5)')
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
        config.USE_NIST_METHOD = (args.intf_db == 'nist')
        config.MAX_COMPOUNDS = args.max_compounds
        config.OUTPUT_PATH = args.output
        config.SINGLE_COMPOUND_MODE = args.single_compound
        config.TARGET_INCHIKEY = args.inchikey
        
        # Set INTF_TQDB_PATH based on selection
        if args.intf_db == 'nist':
            config.INTF_TQDB_PATH = 'INTF-TQDB(from NIST).csv'
        else:
            config.INTF_TQDB_PATH = 'INTF-TQDB(from QE).csv'
        
        logger.info(f"Using interference database: {config.INTF_TQDB_PATH}")
        logger.info(f"Using method: {'NIST' if config.USE_NIST_METHOD else 'QE'}")
        
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
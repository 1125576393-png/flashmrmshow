import streamlit as st
import pandas as pd
import time
from FlashMRM import Config, MRMOptimizer
import os

# Note: st_yled is a custom module, using standard streamlit components instead
# If you have st_yled module, you can uncomment the following lines:
# import st_yled
# st_yled.init()

st.set_page_config(
    page_title="FlashMRM",
    page_icon="786a50646609813e89cc2017082525a3.png",
    layout="wide"
)

# 自定义CSS样式（放大字体）
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: Arial, sans-serif !important;
        font-size: 18px !important;
    }
    .main-header {
        font-family: Arial, sans-serif !important;
        font-size: 28px !important;
        font-weight: bold;
        margin-bottom: 30px;
        color: #1f77b4;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .section-header {
        font-family: Arial, sans-serif !important;
        font-size: 22px !important;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .input-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    .input-label, label, .stTextInput label, .stNumberInput label {
        font-family: Arial, sans-serif !important;
        width: 150px;
        font-weight: bold;
        font-size: 64px !important;
    }
    .result-container {
        margin-top: 30px;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
    }
    .calculate-button {
        margin-top: 40px;
    }
    .param-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 30px;
    }
    .upload-status {
        padding: 8px;
        border-radius: 4px;
        margin-top: 10px;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .calculate-container {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-top: 40px;
    }
    .progress-container {
        flex-grow: 1;
    }

    /* 新增：为主要区块添加额外间距 */
    .stRadio {
        margin-bottom: 40px !important;  /* 输入模式选择区块 */
    }
    /* 为文件上传区域添加底部间距 */
    .uploadedFile {
        margin-bottom: 40px !important;
    }
    /* 新增：表格与按钮字体放大 */
    .stDataFrame th, .stDataFrame td {
        font-family: Arial, sans-serif !important;
        font-size: 16px !important;
    }
    .stButton>button {
        font-family: Arial, sans-serif !important;
        font-size: 18px !important;
        height: 45px !important;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "Single mode"
if 'inchikey_value' not in st.session_state:
    st.session_state.inchikey_value = "Input InChIKey"
if 'batch_file' not in st.session_state:
    st.session_state.batch_file = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'upload_status' not in st.session_state:
    st.session_state.upload_status = None
if 'calculation_in_progress' not in st.session_state:
    st.session_state.calculation_in_progress = False
if 'calculation_complete' not in st.session_state:
    st.session_state.calculation_complete = False
if 'progress_value' not in st.session_state:
    st.session_state.progress_value = 0
if 'show_help' not in st.session_state:
    st.session_state.show_help = False
if 'result_df' not in st.session_state:
    st.session_state.result_df = pd.DataFrame()
if 'top_combinations' not in st.session_state:
    st.session_state.top_combinations = 10

def process_uploaded_data():
    """处理上传的数据"""
    try:
        if st.session_state.input_mode == "Single mode":
            inchikey = st.session_state.inchikey_value.strip()
            if not inchikey:
                st.session_state.upload_status = ("error", "Please enter a valid InChIKey！")
                return False
            if inchikey.count('-') != 2:
                st.session_state.upload_status = ("error", "InChIKey format is invalid! Standard format example: KXRPCFINVWWFHQ-UHFFFAOYSA-N")
                return False
            st.session_state.uploaded_data = {
                "type": "single_inchikey",
                "data": inchikey,
                "timestamp": time.time()
            }
            st.session_state.upload_status = ("success", f"Successfully uploaded InChIKey: {inchikey}")
            return True
            
        else: 
            batch_file = st.session_state.batch_file
            if batch_file is None:
                st.session_state.upload_status = ("error", "Please upload the file!")
                return False
            try:
                if batch_file.name.endswith('.csv'):
                    df = pd.read_csv(batch_file)
                    if "InChIKey" not in df.columns:
                        st.session_state.upload_status = ("error", "The CSV file must contain an \"InChIKey\" column!")
                        return False
                elif batch_file.name.endswith('.txt'):
                    content = batch_file.getvalue().decode('utf-8')
                    inchikeys = [line.strip() for line in content.split('\n') if line.strip()]
                    df = pd.DataFrame({"InChIKey": inchikeys})
                else:
                    st.session_state.upload_status = ("error", "Unsupported file format! Only CSV and TXT formats are supported")
                    return False
            except Exception as e:
                st.session_state.upload_status = ("error", f"File parsing failed: {str(e)}")
                return False
                
            valid_inchikeys = [ik for ik in df["InChIKey"].dropna().unique() if ik.count('-') == 2]
            if len(valid_inchikeys) == 0:
                st.session_state.upload_status = ("error", "No valid InChIKey found in the file！")
                return False
            
            st.session_state.uploaded_data = {
                "type": "batch_file",
                "data": pd.DataFrame({"InChIKey": valid_inchikeys}),
                "filename": batch_file.name,
                "timestamp": time.time(),
                "record_count": len(valid_inchikeys),
                "original_count": len(df)
            }
            st.session_state.upload_status = (
                "success", 
                f"File successfully uploaded: {batch_file.name}, containing {len(df)} original records and {len(valid_inchikeys)} valid InChIKeys."
            )
            return True
            
    except Exception as e:
        st.session_state.upload_status = ("error", f"Upload processing failed: {str(e)}")
        return False

def run_flashmrm_calculation():
    try:
        st.session_state.calculation_in_progress = True
        st.session_state.calculation_complete = False
        st.session_state.progress_value = 0
        st.session_state.result_df = pd.DataFrame()
        
        config = Config()
        config.MZ_TOLERANCE = st.session_state.get("mz_tolerance", 0.7)
        config.RT_TOLERANCE = st.session_state.get("rt_tolerance", 2.0)
        config.RT_OFFSET = st.session_state.get("rt_offset", 0.0)
        config.SPECIFICITY_WEIGHT = st.session_state.get("specificity_weight", 0.2)
        config.TOP_COMBINATIONS = st.session_state.get("top_combinations", 10)
        config.OUTPUT_PATH = "flashmrm_output.csv"
        
        # 设置干扰数据库
        intf_data_selection = st.session_state.get("intf_data", "EXPOS")
        if intf_data_selection == "EXPOS":
            config.INTF_TQDB_PATH = 'INTF_TQDB_EXPOS'
            config.USE_EXPOS_METHOD = True
        else:
            config.INTF_TQDB_PATH = 'INTF_TQDB_EXPER'
            config.USE_EXPOS_METHOD = False
        
        # 2. 获取目标InChIKey列表
        uploaded_data = st.session_state.uploaded_data
        if uploaded_data["type"] == "single_inchikey":
            target_inchikeys = [uploaded_data["data"]]
            config.SINGLE_COMPOUND_MODE = True
            config.TARGET_INCHIKEY = target_inchikeys[0]
        else:
            target_inchikeys = uploaded_data["data"]["InChIKey"].tolist()
            config.SINGLE_COMPOUND_MODE = False
            config.MAX_COMPOUNDS = len(target_inchikeys)  # 按有效数量设置最大处理数
        
        # 3. 加载基础数据
        try:
            optimizer = MRMOptimizer(config)
            optimizer.load_all_data()  # 加载demo、Pesudo-TQDB和INTF-TQDB数据
        except ValueError as e:
            if "No matching InChIKeys found" in str(e):
                results = []
                for inchikey in target_inchikeys:
                    results.append({
                        'chemical': 'not found',
                        'Precursor_mz': 0.0,
                        'InChIKey': inchikey,
                        'RT': 0.0,
                        'coverage_all': 0,
                        'coverage_low': 0,
                        'coverage_medium': 0,
                        'coverage_high': 0,
                        'MSMS1': 0.0,
                        'MSMS2': 0.0,
                        'CE_QQQ1': 0.0,
                        'CE_QQQ2': 0.0,
                        'best_combinations': "no matching data in database",
                        'max_score': 0.0,
                        'max_sensitivity_score': 0.0,
                        'max_specificity_score': 0.0,
                    })
                st.session_state.result_df = pd.DataFrame(results)
                st.session_state.progress_value = 100
                st.session_state.upload_status = ("error", "No matches found for all InChIKey entries in the database. Please verify your data.")
                st.session_state.calculation_in_progress = False
                st.session_state.calculation_complete = True
                return
            else:
                raise  
        
        # 4. 遍历计算所有目标InChIKey
        results = []
        total_compounds = len(target_inchikeys)
        process_func = optimizer.process_compound_expos if config.USE_EXPOS_METHOD else optimizer.process_compound_exper
        
        for idx, inchikey in enumerate(target_inchikeys):
            try:
                # 检查当前InChIKey是否存在于匹配数据中
                if not optimizer.check_inchikey_exists(inchikey):
                    results.append({
                        'chemical': 'not found',
                        'Precursor_mz': 0.0,
                        'InChIKey': inchikey,
                        'RT': 0.0,
                        'coverage_all': 0,
                        'coverage_low': 0,
                        'coverage_medium': 0,
                        'coverage_high': 0,
                        'MSMS1': 0.0,
                        'MSMS2': 0.0,
                        'CE_QQQ1': 0.0,
                        'CE_QQQ2': 0.0,
                        'best_combinations': "inchikey not found",
                        'max_score': 0.0,
                        'max_sensitivity_score': 0.0,
                        'max_specificity_score': 0.0,
                    })
                    st.session_state.progress_value = int((idx + 1) / total_compounds * 100)
                    time.sleep(0.1)
                    continue
                
                compound_result = process_func(inchikey)
                if compound_result:
                    results.append(compound_result)
                else:
                    results.append({
                        'chemical': 'calculation failed',
                        'Precursor_mz': 0.0,
                        'InChIKey': inchikey,
                        'RT': 0.0,
                        'coverage_all': 0,
                        'coverage_low': 0,
                        'coverage_medium': 0,
                        'coverage_high': 0,
                        'MSMS1': 0.0,
                        'MSMS2': 0.0,
                        'CE_QQQ1': 0.0,
                        'CE_QQQ2': 0.0,
                        'best_combinations': "processing failed",
                        'max_score': 0.0,
                        'max_sensitivity_score': 0.0,
                        'max_specificity_score': 0.0,
                    })
            
            except Exception as e:
                results.append({
                    'chemical': 'error',
                    'Precursor_mz': 0.0,
                    'InChIKey': inchikey,
                    'RT': 0.0,
                    'coverage_all': 0,
                    'coverage_low': 0,
                    'coverage_medium': 0,
                    'coverage_high': 0,
                    'MSMS1': 0.0,
                    'MSMS2': 0.0,
                    'CE_QQQ1': 0.0,
                    'CE_QQQ2': 0.0,
                    'best_combinations': f"error: {str(e)[:50]}...", 
                    'max_score': 0.0,
                    'max_sensitivity_score': 0.0,
                    'max_specificity_score': 0.0,
                })
            
            # 更新进度条
            st.session_state.progress_value = int((idx + 1) / total_compounds * 100)
            time.sleep(0.1)  # 避免前端进度条卡顿
        
        # 5. 整理最终结果
        st.session_state.result_df = pd.DataFrame(results) if results else pd.DataFrame()
        st.session_state.progress_value = 100
        st.session_state.calculation_complete = True
        st.session_state.calculation_in_progress = False
        st.session_state.upload_status = ("success", f"Calculation complete! A total of {total_compounds} compounds have been processed.")
    
    except Exception as e:
        # 全局异常处理
        st.session_state.calculation_in_progress = False
        st.session_state.calculation_complete = True
        error_msg = f"Calculation Overview Error: {str(e)}"
        st.session_state.upload_status = ("error", error_msg)
        
        # 生成兜底结果（确保前端有数据显示）
        fallback_results = []
        target_inchikeys = []
        if st.session_state.uploaded_data:
            if st.session_state.uploaded_data["type"] == "Single mode":
                target_inchikeys = [st.session_state.uploaded_data["data"]]
            else:
                target_inchikeys = st.session_state.uploaded_data["data"]["InChIKey"].tolist()
        
        for inchikey in target_inchikeys[:1]:  # 仅显示第一个化合物的错误兜底
            fallback_results.append({
                'chemical': 'global error',
                'Precursor_mz': 0.0,
                'InChIKey': inchikey,
                'RT': 0.0,
                'coverage_all': 0,
                'coverage_low': 0,
                'coverage_medium': 0,
                'coverage_high': 0,
                'MSMS1': 0.0,
                'MSMS2': 0.0,
                'CE_QQQ1': 0.0,
                'CE_QQQ2': 0.0,
                'best_combinations': error_msg[:50] + "...",
                'max_score': 0.0,
                'max_sensitivity_score': 0.0,
                'max_specificity_score': 0.0,
            })
        st.session_state.result_df = pd.DataFrame(fallback_results)

# 主标题和Help按钮
col_title, col_help = st.columns([3, 1])
with col_title:
   # st.image("786a50646609813e89cc2017082525a3.png", width=200)  # Image file not found, commented out
   st.title("FlashMRM")
with col_help:
   with col_help:
    if st.button("Help", width='stretch', key="help_btn"):  
        st.session_state.show_help = not st.session_state.get('show_help', False)

# 显示帮助信息
if st.session_state.get('show_help', False):
    st.info("""
    **Instruction for Use**
1. **Select Input mode**  
   - *Single mode*: Enter a standard InChIKey (e.g., KXRPCFINVWWFHQ-UHFFFAOYSA-N).  
   - *Batch mode*: Upload a CSV (containing column InChIKey) or a TXT file (one InChIKey per line).  
2. Click **Upload** to validate and upload the data.  
3. **Parameter setting (optional)**  
   - *m/z tolerance*: mass-to-charge ratio tolerance (default 0.7)  
   - *RT tolerance*: retention time tolerance in minutes (default 2.0)  
   - *RT offset*: retention time offset in minutes (default 0.0)  
   - *Specificity weight*: (0–1), default 0.2  
   - *Select INTF data*: choose interference database  
     - **EXPOS** = EXPOS-format DB  
     - **EXPER** = EXPER-format DB  
     - **Upload custom** = upload a CSV interference data file to be used instead of built-in databases
   - *Top combinations*: number of top combinations to return (default 10)  
4. Click **Calculate** to start; a progress bar will show completion status.  
5. When finished, view the results table and download a CSV.

---

📂 **Demo Data Example**  
You can download the demo dataset used for testing here:  
👉 [Download demo_data.csv](https://github.com/1125576393-png/flashmrmshow/blob/main/demo_data.csv)
""")

    st.caption("**Preview (first 2 rows):**")

    demo_sources = ["demo_data.csv"]
    demo_df = None
    error_msgs = []

    for src in demo_sources:
        try:
            if src.startswith("http"):
                demo_df = pd.read_csv(src)
            else:
                if os.path.exists(src):
                    demo_df = pd.read_csv(src)
            if demo_df is not None:
                break
        except Exception as e:
            error_msgs.append(f"{src} -> {e}")

    if demo_df is not None:
        st.dataframe(
            demo_df.head(2),
            width="stretch",
            hide_index=True
        )
    else:
        st.warning("Failed to load demo_data.csv for preview. Please check the link or place the file locally.")
        if error_msgs:
            with st.expander("Debug details"):
                st.write("\n".join(error_msgs))

# 输入模式选择
with st.expander("Select Input mode"):
    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.markdown(
            """
            <div style="display:flex; height:100%; align-items:center; justify-content:flex-end; padding-right:8px;">
            """,
            unsafe_allow_html=True
        )
        selected_mode = st.radio(
            "Select Input mode:",
            ["Single mode", "Batch mode"],
            index=0 if st.session_state.get("input_mode", "Single mode") == "Single mode" else 1,
            key="mode_selector",
            label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; gap:0.35rem; width:100%;">
            """,
            unsafe_allow_html=True
        )
        if selected_mode == "Single mode":
            inchikey_input = st.text_input(
                "Single mode:",
                key="inchikey_input_active",
                value=st.session_state.get("inchikey_value", ""),
                placeholder="Input InChIKey",
                label_visibility="collapsed",
            )
            if inchikey_input:
                st.session_state.inchikey_value = inchikey_input
            st.file_uploader(
                "Batch mode:",
                type=['txt', 'csv'],
                label_visibility="collapsed",
                key="batch_input_disabled",
                disabled=True,
                help="Disable batch uploads in single-file mode"
            )
        else:
            st.text_input(
                "Single mode:",
                value="",
                placeholder="Disable individual input in batch mode",
                label_visibility="collapsed",
                key="inchikey_input_disabled",
                disabled=True
            )
            batch_input = st.file_uploader(
                "Batch mode:",
                type=['txt', 'csv'],
                help='Drag and drop files here. Supported formats: CSV (including an "InChIKey" column) and TXT (one InChIKey per line).',
                label_visibility="collapsed",
                key="batch_input_active"
            )
            if batch_input is not None:
                st.session_state.batch_file = batch_input
        st.markdown("</div>", unsafe_allow_html=True)  
    # 更新输入模式
    if selected_mode != st.session_state.input_mode:
        st.session_state.input_mode = selected_mode
        st.session_state.uploaded_data = None  
        st.session_state.upload_status = None
        st.rerun()
    # 上传按钮
    col_u1, col_u2, col_u3 = st.columns([1, 1, 1])
    with col_u2:
        upload_clicked = st.button(
            "Upload",
            key="upload_button",
            disabled=st.session_state.calculation_in_progress
        )
    if upload_clicked:
        process_uploaded_data()

# 显示上传状态
if st.session_state.upload_status:
    status_type, message = st.session_state.upload_status
    st.markdown(f'<div class="upload-status {status_type}">{message}</div>', unsafe_allow_html=True)

# 显示已上传的数据信息（展开面板）
if st.session_state.uploaded_data:
    with st.expander("Data information has been uploaded", expanded=False):
        ud = st.session_state.uploaded_data
        st.write(f"Data type: {'Single InChIKey' if ud['type'] == 'single_inchikey' else 'Batch file'}")
        st.write(f"Upload time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ud['timestamp']))}")
        
        if ud["type"] == "single_inchikey":
            st.write(f"InChIKey: {ud['data']}")
        else:
            st.write(f"filename: {ud['filename']}")
            st.write(f"Number of original records: {ud.get('original_count', 0)}")
            st.write(f"Valid InChIKey count: {ud['record_count']}")
            st.write("Valid InChIKey Preview:")
            st.dataframe(ud['data'].head(10), width="content")  # 非必要宽度，用默认content
            if len(ud['data']) > 10:
                st.write(f"... totle{len(ud['data'])}valid records")

with st.container():
    col0, col00 = st.columns([1, 1])
    with col0:
        st.write("")
    with col00:
        st.write("")


# 参数设置部分
if "specificity_weight" not in st.session_state and "sensitivity_weight" not in st.session_state:
    st.session_state.specificity_weight = 0.2
    st.session_state.sensitivity_weight = 0.8  # 1 - 0.2
def sync_weights(changed: str):
    if changed == "specificity":
        st.session_state.sensitivity_weight = 1 - st.session_state.specificity_weight
    elif changed == "sensitivity":
        st.session_state.specificity_weight = 1 - st.session_state.sensitivity_weight

with st.expander("Parameter Setting"):  
    with st.container():
        # 第一行参数：数据库选择 + M/z容差
        col1, col2 = st.columns([2, 2])
        with col1:
            se_bg = "#D9E4FF"
            intf_data = st.selectbox(
                "Select INTF data:",
                ["EXPOS", "EXPER"],
                index=0,
                key="intf_data",
                help="EXPOS: Using EXPOS Format Interference Database；EXPER: Using EXPER format to interference with the database",
            )
        with col2:
            mz_tolerance = st.number_input(
                "*m/z* tolerance:",
                min_value=0.0,
                max_value=10.0,
                value=0.7,
                step=0.1,
                help="Mass-to-charge ratio matching tolerance, default 0.7",
                key="mz_tolerance",
            )

    # 第二行参数：RT容差 + RT偏移
        col4, col5 = st.columns([1, 1])
        with col4:
            rt_tolerance = st.number_input(
                "RT tolerance:",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Retention time matching tolerance, default 2.0 minutes",
                key="rt_tolerance",
            )
        with col5:
            rt_offset = st.number_input(
                "RT offset:",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                help="Retention time offset, default 0.0 minutes",
                key="rt_offset",
            )

    # 第三行参数：特异性权重 + 灵敏度权重
        col6, col7 = st.columns([1, 1])
        with col6:
            specificity_weight = st.number_input(
                "Specificity weight:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.specificity_weight,
                step=0.05,
                help="Specificity weight (0–1), default 0.2",
                key="specificity_weight",
                on_change=sync_weights,
                args=("specificity",),
            )
        with col7:
            st.number_input(
                "Sensitivity weight:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.sensitivity_weight,
                step=0.05,
                key="sensitivity_weight",
                help="Automatically calculated as 1 - Specificity weight",
                on_change=sync_weights,
                args=("sensitivity",),
            )

    # 第四行参数：Top combinations
        col8 = st.columns([1])[0]
        with col8:
            top_combinations = st.number_input(
                "Top combinations:",
                min_value=1,
                max_value=50,
                value=st.session_state.get("top_combinations", 10),
                step=1,
                help="Number of top combinations to return (default 10)",
                key="top_combinations",
            )

# 计算区域：按钮 + 进度条
st.markdown('<div class="section-header">Calculate</div>', unsafe_allow_html=True)
col_calc, col_prog = st.columns([1, 3])
with col_calc:
    calculate_clicked = st.button(
        "Calculate", 
        width='stretch',  
        type="primary", 
        key="calculate_main",
        disabled=st.session_state.calculation_in_progress or st.session_state.uploaded_data is None
    )
with col_prog:
    # 实时更新的进度条
    progress_bar = st.progress(st.session_state.progress_value, text=f"Processing progress: {st.session_state.progress_value}%")

# 若进度值变化，更新进度条文本
if st.session_state.progress_value != progress_bar.value:
    progress_bar.progress(st.session_state.progress_value, text=f"Processing progress: {st.session_state.progress_value}%")

# 运行计算逻辑
if calculate_clicked:
    if st.session_state.uploaded_data is None:
        st.error("Please first use the 'Upload' button to upload and verify the data！")
    else:
        run_flashmrm_calculation()

# 显示计算结果
if st.session_state.calculation_complete:
    st.markdown('<div class="section-header">Calculation results</div>', unsafe_allow_html=True)
    result_df = st.session_state.result_df
    if not result_df.empty:
        CALCULATION_COLUMNS = [
            'chemical','Precursor_mz','InChIKey','RT',
            'MSMS1','MSMS2','CE_QQQ1','CE_QQQ2',
            'max_score',
        ]
        for col in CALCULATION_COLUMNS:
            if col not in result_df.columns:
                result_df[col] = None

         # ⭐（显示用）格式化副本 —— 保留 1 位小数
        display_df = result_df[CALCULATION_COLUMNS].copy()
        EXCLUDE_FORMAT = ['max_score']

        for col in CALCULATION_COLUMNS:
            if col not in EXCLUDE_FORMAT:
                if hasattr(display_df[col], "dtype") and display_df[col].dtype.kind in "fc":
                    display_df[col] = display_df[col].round(1)

        st.dataframe(display_df, width="content")

        csv_data = result_df[CALCULATION_COLUMNS].to_csv(index=False, encoding='utf-8').encode('utf-8')
        st.download_button(
            label="📥 Download results CSV",
            data=csv_data,
            file_name=f"FlashMRM_results_{time.strftime('%Y%m%d%H%M%S')}.csv",
            mime="text/csv",
            width='stretch',
            key="download_result"
        )

    st.markdown('<div class="section-header">Best 5 ion-pair combinations</div>', unsafe_allow_html=True)

    IONPAIR_COLUMNS = [
        'MSMS1','MSMS2',
        'sensitivity_score','specificity_score',
        'CE_QQQ1','CE_QQQ2'
    ]

    def _normalize_top5_rows(raw_list):
        import pandas as pd
        if not isinstance(raw_list, (list, tuple)):
            df = pd.DataFrame(columns=IONPAIR_COLUMNS)
        else:
            df = pd.DataFrame(raw_list)

        for c in IONPAIR_COLUMNS:
            if c not in df.columns:
                df[c] = 0

        df = df[IONPAIR_COLUMNS].head(5)

        if len(df) < 5:
            import pandas as pd
            n_missing = 5 - len(df)
            zero_row = {c: 0 for c in IONPAIR_COLUMNS}
            df = pd.concat([df, pd.DataFrame([zero_row]*n_missing)], ignore_index=True)
        return df

    result_df = st.session_state.result_df.copy()
    result_df['_display_key'] = result_df.apply(
        lambda r: f"{str(r.get('chemical',''))} | {str(r.get('InChIKey',''))}", axis=1
    )
    selected_key = st.selectbox(
        "",
        options=result_df['_display_key'].tolist(),
        index=0
    )
    sel_row = result_df[result_df['_display_key'] == selected_key].iloc[0]
    top5_df = _normalize_top5_rows(sel_row.get('best_combinations'))

    # ⭐ 展示用：格式化副本（仅显示 1 位小数）
    display_top5_df = top5_df.copy()
    EXCLUDE_TOP5 = ['sensitivity_score','specificity_score']

    for col in IONPAIR_COLUMNS:
        if col not in EXCLUDE_TOP5:
            if hasattr(display_top5_df[col], "dtype") and display_top5_df[col].dtype.kind in "fc":
                display_top5_df[col] = display_top5_df[col].round(1)
    display_top5_df = display_top5_df.rename(columns={
        "MSMS1": "MRM1",
        "MSMS2": "MRM2"
    })
    st.dataframe(display_top5_df, width="stretch", hide_index=True)

    st.download_button(
        label="📥 Download Top-5 ion pairs (CSV)",
        data=top5_df.to_csv(index=False).encode('utf-8'),
        file_name="best5_ion_pairs.csv",
        mime="text/csv",
        key="download_best5"
    )
        
    success_conditions = (
        result_df['chemical'].notna() & 
        ~result_df['chemical'].isin(['not found', 'calculation failed', 'error', 'global error'])
    )
    success_count = success_conditions.sum()  # 用sum()统计True的数量，避免len()的歧义
        
    st.success(f"Calculation complete ✅ | Successfully processed: {success_count}| Overall processing: {len(result_df)}")













    


import streamlit as st
import pandas as pd
import time
import random
from FlashMRM import Config, MRMOptimizer
import os

# 页面配置
st.set_page_config(
    page_title="FlashMRM",
    page_icon="🧪",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1f77b4;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .input-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .input-label {
        width: 150px;
        font-weight: bold;
    }
    .result-container {
        margin-top: 20px;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
    }
    .calculate-button {
        margin-top: 20px;
    }
    .param-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .upload-status {
        padding: 8px;
        border-radius: 4px;
        margin-top: 5px;
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
        margin-top: 20px;
    }
    .progress-container {
        flex-grow: 1;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "Input InChIKey"
if 'inchikey_value' not in st.session_state:
    st.session_state.inchikey_value = "KXRPCFINVWWFHQ-UHFFFAOYSA-N"
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
    
def process_uploaded_data():
    """处理上传的数据"""
    try:
        if st.session_state.input_mode == "Input InChIKey":
            # 处理单个InChIKey
            inchikey = st.session_state.inchikey_value.strip()
            if not inchikey:
                st.session_state.upload_status = ("error", "请输入有效的InChIKey！")
                return False
            
            # 这里可以添加InChIKey格式验证
            st.session_state.uploaded_data = {
                "type": "single_inchikey",
                "data": inchikey,
                "timestamp": time.time()
            }
            st.session_state.upload_status = ("success", f"成功上传InChIKey: {inchikey}")
            return True
            
        else:  # Batch mode
            # 处理批量文件
            batch_file = st.session_state.batch_file
            if batch_file is None:
                st.session_state.upload_status = ("error", "请上传文件！")
                return False
            
            # 根据文件类型处理
            if batch_file.name.endswith('.csv'):
                df = pd.read_csv(batch_file)
            elif batch_file.name.endswith('.txt'):
                # 假设txt文件每行一个InChIKey
                content = batch_file.getvalue().decode('utf-8')
                inchikeys = [line.strip() for line in content.split('\n') if line.strip()]
                df = pd.DataFrame({"InChIKey": inchikeys})
            else:
                st.session_state.upload_status = ("error", "不支持的文件格式！")
                return False
            
            st.session_state.uploaded_data = {
                "type": "batch_file",
                "data": df,
                "filename": batch_file.name,
                "timestamp": time.time(),
                "record_count": len(df)
            }
            st.session_state.upload_status = ("success", f"成功上传文件: {batch_file.name}，包含 {len(df)} 条记录")
            return True
            
    except Exception as e:
        st.session_state.upload_status = ("error", f"上传处理失败: {str(e)}")
        return False


def run_flashmrm_calculation():
    """运行 FlashMRM.py 的真实后端计算"""
    try:
        st.session_state.calculation_in_progress = True
        st.session_state.calculation_complete = False
        st.session_state.progress_value = 0

        config = Config()
        config.MZ_TOLERANCE = st.session_state.get("mz_tolerance", 0.7)
        config.RT_TOLERANCE = st.session_state.get("rt_tolerance", 2.0)
        config.RT_OFFSET = st.session_state.get("rt_offset", 0.0)
        config.SPECIFICITY_WEIGHT = st.session_state.get("specificity_weight", 0.2)
        config.MAX_COMPOUNDS = 5  # 可按需调整
        config.OUTPUT_PATH = "flashmrm_output.csv"

        # 选择干扰数据库
        intf_data_selection = st.session_state.get("intf_data", "Default")
        if intf_data_selection == "Default":
            config.INTF_TQDB_PATH = 'INTF-TQDB(from NIST).csv'
            config.USE_NIST_METHOD = True
        else:
            config.INTF_TQDB_PATH = 'INTF-TQDB(from QE).csv'
            config.USE_NIST_METHOD = False

        # 输入模式判断
        if st.session_state.input_mode == "Input InChIKey":
            config.SINGLE_COMPOUND_MODE = True
            config.TARGET_INCHIKEY = st.session_state.inchikey_value.strip()
        else:
            config.SINGLE_COMPOUND_MODE = False

        # 初始化优化器
        optimizer = MRMOptimizer(config)

        # 加载数据
        optimizer.load_all_data()

        # === 防止空数据死循环 ===
        if getattr(optimizer, "matched_df", None) is None or optimizer.matched_df.empty:
            st.session_state.progress_value = 100
            st.session_state.calculation_complete = True
            st.session_state.calculation_in_progress = False

            st.session_state.result_df = pd.DataFrame([{
                'chemical': 'not found',
                'Precursor_mz': 0,
                'InChIKey': config.TARGET_INCHIKEY if config.SINGLE_COMPOUND_MODE else "N/A",
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
            }])

            st.session_state.upload_status = ("error", "❌ 未在数据库中找到匹配数据。")
            return

        # === 单个化合物模式 ===
        if config.SINGLE_COMPOUND_MODE:
            inchikey = config.TARGET_INCHIKEY
            if not optimizer.check_inchikey_exists(inchikey):
                st.session_state.progress_value = 100
                st.session_state.calculation_complete = True
                st.session_state.calculation_in_progress = False

                not_found_result = {
                    'chemical': 'not found',
                    'Precursor_mz': 0,
                    'InChIKey': inchikey,
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
                st.session_state.result_df = pd.DataFrame([not_found_result])
                st.session_state.upload_status = ("error", f"未找到化合物：{inchikey}")
                return

            # 处理并输出结果
            result = optimizer.process_compound_nist(inchikey)
            st.session_state.progress_value = 100
            time.sleep(0.5)

            if result:
                st.session_state.result_df = pd.DataFrame([result])
                st.session_state.upload_status = ("success", f"✅ 成功处理 {inchikey}")
            else:
                failed_result = {
                    'chemical': 'processing failed',
                    'Precursor_mz': 0,
                    'InChIKey': inchikey,
                    'RT': 0,
                    'coverage_all': 0,
                    'coverage_low': 0,
                    'coverage_medium': 0,
                    'coverage_high': 0,
                    'MSMS1': 0,
                    'MSMS2': 0,
                    'CE_QQQ1': 0,
                    'CE_QQQ2': 0,
                    'best5_combinations': "processing failed",
                    'max_score': 0,
                    'max_sensitivity_score': 0,
                    'max_specificity_score': 0,
                }
                st.session_state.result_df = pd.DataFrame([failed_result])
                st.session_state.upload_status = ("error", f"❌ {inchikey} 处理失败。")

        # === 批量模式 ===
        else:
            inchikeys = optimizer.matched_df["InChIKey"].unique()
            total = len(inchikeys)
            if total == 0:
                st.session_state.result_df = pd.DataFrame()
                st.session_state.progress_value = 100
                st.session_state.calculation_complete = True
                st.session_state.calculation_in_progress = False
                st.session_state.upload_status = ("error", "未找到任何匹配化合物。")
                return

            results = []
            for i, inchikey in enumerate(inchikeys[:config.MAX_COMPOUNDS]):
                result = optimizer.process_compound_nist(inchikey)
                if result:
                    results.append(result)
                progress = int((i + 1) / total * 100)
                st.session_state.progress_value = progress
                time.sleep(0.1)

            st.session_state.result_df = pd.DataFrame(results) if results else pd.DataFrame()
            st.session_state.upload_status = ("success", f"✅ 批量处理完成，共 {len(results)} 条结果。")

        # === 最终状态更新 ===
        st.session_state.progress_value = 100
        st.session_state.calculation_complete = True
        st.session_state.calculation_in_progress = False

    except Exception as e:
        st.session_state.calculation_in_progress = False
        st.session_state.calculation_complete = False
        st.session_state.upload_status = ("error", f"运行错误: {e}")

# 主标题和Help按钮
col_title, col_help = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">FlashMRM</div>', unsafe_allow_html=True)
with col_help:
    if st.button("Help", use_container_width=True):
        st.session_state.show_help = not st.session_state.get('show_help', False)

# 显示帮助信息
if st.session_state.get('show_help', False):
    st.info("""
    **使用说明:**
    - 选择输入模式: 单个InChIKey或批量上传
    - 在输入模式部分输入数据
    - 点击Upload按钮上传数据到后台
    - 设置参数: M/z容差、RT容差等
    - 点击Calculate开始计算
    - 查看结果并下载
    """)

# 输入模式选择
st.markdown('<div class="section-header">输入模式</div>', unsafe_allow_html=True)

# 使用自定义布局实现单选按钮在左侧，输入框在右侧
col_a, col_b = st.columns([1, 2])

with col_a:
    # 单选按钮（垂直排列）
    selected_mode = st.radio(
        "选择输入模式:",
        ["Input InChIKey", "Batch mode"],
        index=0 if st.session_state.input_mode == "Input InChIKey" else 1,
        key="mode_selector",
        label_visibility="collapsed"
    )

with col_b:
    # 根据选择的模式显示相应的输入框
    if selected_mode == "Input InChIKey":
        inchikey_input = st.text_input(
            "Input InChIKey:",
            value=st.session_state.inchikey_value,
            placeholder="输入InChIKey...",
            label_visibility="collapsed",
            key="inchikey_input_active"
        )
        if inchikey_input:
            st.session_state.inchikey_value = inchikey_input
        
        # 禁用状态的Batch mode文件上传（占位）
        st.file_uploader(
            "Batch mode:",
            type=['txt', 'csv'],
            label_visibility="collapsed",
            key="batch_input_disabled",
            disabled=True
        )
    else:
        # 禁用状态的InChIKey输入框（占位）
        st.text_input(
            "Input InChIKey:",
            value="",
            placeholder="",
            label_visibility="collapsed",
            key="inchikey_input_disabled",
            disabled=True
        )
        
        batch_input = st.file_uploader(
            "Batch mode:",
            type=['txt', 'csv'],
            help="Drag and drop file here. Limit 200MB per file • TXT, CSV",
            label_visibility="collapsed",
            key="batch_input_active"
        )
        if batch_input is not None:
            st.session_state.batch_file = batch_input

# 更新session state
if selected_mode != st.session_state.input_mode:
    st.session_state.input_mode = selected_mode
    st.rerun()

# 参数设置部分
st.markdown('<div class="section-header">参数设置</div>', unsafe_allow_html=True)

# 创建参数设置容器
with st.container():
    # 第一行参数
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # 选择INTF数据
        intf_data = st.selectbox(
            "Select INTF data:",
            ["Default", "QE"],
            index=0,
            key="intf_data"
        )

    with col2:
        # 空的列用于布局对齐
        st.write("")  # 占位符
        
    with col3:
        # Upload按钮
        upload_clicked = st.button(
            "Upload", 
            use_container_width=True,
            key="upload_button"
        )

with st.container():
    # 第二行参数
    col4, col5 = st.columns([1,1])
    
    with col4:
        # M/z tolerance
        mz_tolerance = st.number_input(
            "M/z tolerance:",
            min_value=0.0,
            max_value=10.0,
            value=0.7,
            step=0.1,
            help="M/z容差设置"
        )
    
    with col5:
        # RT offset
        rt_offset = st.number_input(
            "RT offset:",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="RT偏移量"
        )
    

    
    # 第三行参数
    col6, col7 = st.columns([1, 1])

    with col6:
        # RT tolerance
        rt_tolerance = st.number_input(
            "RT tolerance:",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="RT容差设置"
        )
        
    with col7:
        # Specificity weight
        specificity_weight = st.number_input(
            "Specificity weight:",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="特异性权重"
        )

# 处理Upload按钮点击
if upload_clicked:
    process_uploaded_data()

# 显示上传状态
if st.session_state.upload_status:
    status_type, message = st.session_state.upload_status
    if status_type == "success":
        st.markdown(f'<div class="upload-status success">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="upload-status error">{message}</div>', unsafe_allow_html=True)

# 显示已上传的数据信息（用于调试）
if st.session_state.uploaded_data:
    with st.expander("已上传数据信息"):
        st.write("数据类型:", st.session_state.uploaded_data["type"])
        st.write("上传时间:", time.strftime('%Y-%m-%d %H:%M:%S', 
                                         time.localtime(st.session_state.uploaded_data["timestamp"])))
        
        if st.session_state.uploaded_data["type"] == "single_inchikey":
            st.write("InChIKey:", st.session_state.uploaded_data["data"])
        else:
            st.write("文件名:", st.session_state.uploaded_data["filename"])
            st.write("记录数:", st.session_state.uploaded_data["record_count"])
            st.write("数据预览:")
            st.dataframe(st.session_state.uploaded_data["data"].head())

# Calculate按钮和进度条在同一排
st.markdown('<div class="section-header">计算</div>', unsafe_allow_html=True)

col_calc, col_prog = st.columns([1, 3])

with col_calc:
    calculate_clicked = st.button(
        "Calculate", 
        use_container_width=True, 
        type="primary", 
        key="calculate_main",
        disabled=st.session_state.calculation_in_progress
    )

with col_prog:
    # 始终显示进度条
    progress_bar = st.progress(st.session_state.progress_value)
        
# 如果点击了Calculate按钮
if calculate_clicked:
    if st.session_state.uploaded_data is None:
        st.error("请先使用 Upload 按钮上传数据！")
    else:
        # 直接调用，不要用多线程
        run_flashmrm_calculation()

# 如果计算完成，显示结果
if st.session_state.get("calculation_complete", False):
    st.markdown('<div class="section-header">计算结果</div>', unsafe_allow_html=True)

    if "result_df" in st.session_state:
        df = st.session_state.result_df.copy()
        st.dataframe(df, use_container_width=True)

        # 创建下载文件
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载结果 CSV",
            data=csv,
            file_name="FlashMRM_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.success("✅ 计算完成，结果已生成。")
    else:
        st.warning("⚠️ 未生成任何有效结果，请检查输入或参数。")

    # 防止页面不刷新（强制 rerun 一次）
    st.button("🔁 重新开始", on_click=lambda: st.session_state.update({
        "uploaded_data": None,
        "upload_status": None,
        "calculation_complete": False,
        "calculation_in_progress": False,
        "progress_value": 0
    }))


# 页脚信息
st.sidebar.markdown("---")
st.sidebar.markdown("**FlashMRM** - 质谱数据分析工具")


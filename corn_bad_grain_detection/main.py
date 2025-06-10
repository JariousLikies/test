import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import os
from config.logging_config import setup_logging
from models.model_loader import load_model
from models.model_tester import test_model_inference
from utils.image_processor import process_image

# 配置日志
logger = setup_logging()

# 设置页面配置
st.set_page_config(
    page_title="玉米坏粒识别平台",
    page_icon="🌽",
    layout="wide"
)

# 应用样式
def apply_custom_styles():
    """应用自定义CSS样式"""
    st.markdown("""
    <style>
        /* 整体页面样式 */
        .main-header {
            color: #2c3e50;
            font-family: 'Segoe UI', sans-serif;
        }
        
        /* 侧边栏样式 */
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* 卡片样式 */
        .card {
            background-color: white;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            transition: transform 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
        }
        
        /* 按钮样式 */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton>button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(76, 175, 80, 0.25);
        }
        
        /* 统计卡片样式 */
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .stat-card {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 1rem;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.75rem;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #7f8c8d;
        }
        
        /* 深色模式样式 */
        .dark-mode {
            background-color: #1e1e1e;
            color: white;
        }
        
        .dark-mode .card {
            background-color: #2d2d2d;
            color: white;
        }
        
        .dark-mode .stat-card {
            background-color: #3d3d3d;
            color: white;
        }
        
        /* 图像容器样式 */
        .image-container {
            border-radius: 0.75rem;
            overflow: hidden;
            margin-bottom: 1rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }
        
        /* 分隔线样式 */
        .divider {
            border-top: 1px solid #e0e0e0;
            margin: 1.5rem 0;
        }
        
        /* 下载按钮样式 */
        .download-btn {
            text-align: center;
            margin-top: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

# 应用自定义样式
apply_custom_styles()

# 标题和介绍
st.markdown("<h1 class='main-header'>🌽 玉米坏粒识别平台</h1>", unsafe_allow_html=True)
st.markdown("本平台基于深度学习技术，能够自动识别玉米中的坏粒，帮助您快速评估玉米质量。")

# 主题选择器
with st.sidebar:
    st.header("界面设置")
    theme = st.selectbox(
        "选择主题",
        ["亮色模式", "深色模式"],
        index=0
    )
    
    # 应用主题
    if theme == "深色模式":
        st.markdown("<body class='dark-mode'>", unsafe_allow_html=True)
    
    # 侧边栏 - 模型设置
    st.header("模型设置")

    # 默认模型路径
    DEFAULT_MODEL_PATH = 'model/best.pt'

    # 检查默认模型是否存在
    default_model_exists = os.path.exists(DEFAULT_MODEL_PATH)

    if default_model_exists:
        st.info(f"检测到默认模型: {DEFAULT_MODEL_PATH}")
    else:
        st.warning(f"未找到默认模型: {DEFAULT_MODEL_PATH}")

    # 模型选择方式
    model_choice = st.radio(
        "选择模型来源",
        ["默认模型", "上传自定义模型"]
    )

    # 根据选择设置模型文件和类型
    model_file = None
    model_type = None

    if model_choice == "默认模型" and default_model_exists:
        try:
            model_file = open(DEFAULT_MODEL_PATH, 'rb')
            file_ext = os.path.splitext(DEFAULT_MODEL_PATH)[1].lower()
            if file_ext == '.onnx':
                model_type = "ONNX"
            else:
                model_type = "PyTorch"
            st.success("已选择默认模型")
        except Exception as e:
            st.error(f"无法加载默认模型: {e}")
            model_file = None
    elif model_choice == "上传自定义模型":
        # 上传模型权重文件
        model_file = st.file_uploader("上传模型文件", type=["pt", "pth", "onnx"])

        if model_file:
            file_ext = os.path.splitext(model_file.name)[1].lower()
            if file_ext == '.onnx':
                default_model_type = "ONNX"
            else:
                default_model_type = "PyTorch"

            model_type = st.selectbox(
                "模型类型",
                ["PyTorch", "TorchScript", "ONNX"],
                index=["PyTorch", "TorchScript", "ONNX"].index(default_model_type)
            )
            st.success(f"已上传模型: {model_file.name}")
        else:
            st.info("请上传模型文件")
    else:
        st.info("请选择模型来源")

    # 只有在选择了模型后才显示其他设置
    if model_file and model_type:
        confidence_threshold = st.slider(
            "置信度阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )

        # 高级设置
        with st.expander("高级设置"):
            draw_bbox = st.checkbox("显示边界框", value=True)
            draw_label = st.checkbox("显示标签", value=True)
            draw_confidence = st.checkbox("显示置信度", value=True)
            line_thickness = st.slider("边界框线条粗细", min_value=1, max_value=10, value=2)
            detection_color = st.color_picker("坏粒标记颜色", "#FF0000")

    st.header("关于")
    st.info("""
    本平台使用深度学习模型识别玉米坏粒，支持多种格式的图像输入。
    上传图像后，系统将自动检测并标记出坏粒区域。
    """)

# 主界面 - 优化布局结构
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("图像输入")
    
    # 创建卡片式布局
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # 上传图像文件
    uploaded_file = st.file_uploader(
        "选择一张图片",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )
    
    # 或者从摄像头捕获
    use_camera = st.checkbox("使用摄像头拍摄")
    if use_camera:
        uploaded_file = st.camera_input("拍摄玉米照片", label_visibility="collapsed")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # 创建卡片式布局
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("原始图像")
        
        # 显示原始图像
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # 如果图像是RGBA格式，转换为RGB
        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.subheader("分析结果")
    
    # 结果区域占位符
    result_placeholder = st.empty()
    
    # 只有在选择了模型和上传了图像后才显示分析按钮
    if uploaded_file is not None and model_file and model_type:
        # 加载模型
        if 'model' not in st.session_state:
            with st.spinner("加载模型中..."):
                st.session_state.model = load_model(model_file, model_type)
                model = st.session_state.model
                
                # 模型测试推理（仅在上传模型后执行）
                if model and test_model_inference(model, model_type):
                    st.success("模型测试推理成功，准备就绪！")
                elif model:
                    st.warning("模型测试推理返回意外结果，但继续运行。")
        else:
            model = st.session_state.model

        if st.button("开始分析", type="primary"):
            if model is None:
                st.error("模型加载失败，请检查模型文件。")
            else:
                with st.spinner("正在分析图像..."):
                    start_time = time.time()
                    result_img, bad_count = process_image(
                        img_array, model, model_type, confidence_threshold, detection_color,
                        draw_bbox, draw_label, draw_confidence, line_thickness
                    )
                    end_time = time.time()

                    # 显示处理时间
                    processing_time = end_time - start_time
                    
                    # 在结果占位符中显示内容
                    with result_placeholder.container():
                        # 创建卡片式布局
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        
                        # 显示结果图像
                        st.subheader("分析结果")
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        st.image(
                            cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                            use_column_width=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # 显示统计信息
                        st.subheader("统计信息")
                        st.markdown("""
                        <div class="stats-container">
                            <div class="stat-card">
                                <div class="stat-value">%d</div>
                                <div class="stat-label">坏粒数量</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">%.2fs</div>
                                <div class="stat-label">处理时间</div>
                            </div>
                        </div>
                        """ % (bad_count, processing_time), unsafe_allow_html=True)
                        
                        # 下载结果
                        st.markdown("""
                        <style>
                        .download-btn {
                            text-align: center;
                            margin-top: 1.5rem;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('<div class="download-btn">', unsafe_allow_html=True)
                        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                            result_pil.save(tmp.name)
                            st.download_button(
                                label="下载分析结果",
                                data=open(tmp.name, 'rb').read(),
                                file_name="corn_analysis_result.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)

# 底部信息
st.markdown("""
<div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
    <p>© 2025 玉米坏粒识别平台 | 检测结果仅供参考，实际应用请结合专业质检流程</p>
</div>
""", unsafe_allow_html=True)

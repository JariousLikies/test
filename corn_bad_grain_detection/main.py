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
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用样式 - 优化布局和图像显示
def apply_custom_styles():
    st.markdown("""
    <style>
        /* 侧边栏样式 */
        .css-1lsmgbg {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
            padding: 2rem 1.5rem;
        }
        
        .css-10oheav {
            color: #1e293b;
            font-weight: 600;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }
        
        /* 主内容区 */
        .main-content {
            padding: 2rem;
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
        
        /* 图像容器 - 优化布局 */
        .image-container {
            border-radius: 0.75rem;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #f8fafc;
            min-height: 200px;
        }
        
        /* 统计卡片 */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .stat-card {
            background-color: #f8fafc;
            border-radius: 0.5rem;
            padding: 1rem;
            text-align: center;
            transition: all 0.2s ease;
        }
        
        .stat-card:hover {
            background-color: #edf2f7;
        }
        
        .stat-value {
            font-size: 1.75rem;
            font-weight: 600;
            color: #2d3748;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #718096;
        }
        
        /* 按钮样式 */
        .stButton>button {
            background-color: #48bb78;
            color: white;
            border-radius: 0.5rem;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            background-color: #38a169;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(72, 187, 120, 0.25);
        }
        
        /* 深色模式适配 */
        .dark-mode {
            background-color: #1a202c;
            color: #e2e8f0;
        }
        
        .dark-mode .card {
            background-color: #2d3748;
            color: #e2e8f0;
        }
        
        .dark-mode .stat-card {
            background-color: #4a5568;
            color: #e2e8f0;
        }
        
        .dark-mode .image-container {
            background-color: #2d3748;
        }
        
        /* 下载按钮 */
        .download-btn {
            text-align: center;
            margin-top: 1.5rem;
        }
        
        /* 图片对齐辅助类 */
        .img-wrapper {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
    </style>
    """, unsafe_allow_html=True)

# 应用自定义样式
apply_custom_styles()

# 标题和介绍
st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>🌽 玉米坏粒识别平台</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #718096; margin-bottom: 2rem;'>基于深度学习技术的玉米质量智能评估系统</p>", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("### 🌐 系统设置")
    
    # 主题切换
    theme = st.radio(
        "选择主题",
        ["亮色模式", "深色模式"],
        index=0,
        horizontal=True
    )
    
    # 应用主题
    if theme == "深色模式":
        st.markdown("<body class='dark-mode'>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🧠 模型设置")
    
    # 模型选择逻辑保持不变
    DEFAULT_MODEL_PATH = 'model/best.pt'
    default_model_exists = os.path.exists(DEFAULT_MODEL_PATH)
    
    if default_model_exists:
        st.info(f"✅ 默认模型可用: {DEFAULT_MODEL_PATH}")
    else:
        st.warning(f"⚠️ 未找到默认模型: {DEFAULT_MODEL_PATH}")
    
    model_choice = st.radio(
        "模型来源",
        ["默认模型", "上传自定义模型"]
    )
    
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
            st.success("✅ 已加载默认模型")
        except Exception as e:
            st.error(f"❌ 加载失败: {e}")
            model_file = None
    elif model_choice == "上传自定义模型":
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
            st.success(f"✅ 已上传模型: {model_file.name}")
        else:
            st.info("请上传模型文件")
    else:
        st.info("请选择模型来源")
    
    # 只有在选择了模型后才显示其他设置
    if model_file and model_type:
        st.markdown("---")
        st.markdown("### 🎯 检测参数")
        
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
    
    st.markdown("---")
    st.markdown("### ℹ️ 关于系统")
    st.info("""
    本平台使用深度学习技术识别玉米坏粒，支持：
    - 多格式图像上传
    - 自定义模型加载
    - 实时摄像头拍摄
    """)

# 主内容区 - 优化图像布局
with st.container():
    # 输入区域和结果区域分栏展示
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📥 图像输入")
        
        # 上传图像文件
        uploaded_file = st.file_uploader(
            "选择图片",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed"
        )
        
        # 或者从摄像头捕获
        use_camera = st.checkbox("使用摄像头拍摄")
        if use_camera:
            uploaded_file = st.camera_input("拍摄玉米照片", label_visibility="collapsed")
        
        # 显示原始图像
        if uploaded_file is not None:
            st.markdown("#### 原始图像")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # 处理图像格式
            if img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            else:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(image, use_column_width=True, output_format="PNG")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 分析结果")
        
        # 结果区域占位符
        result_placeholder = st.empty()
        
        # 分析按钮 - 只有在上传图像且选择模型后显示
        if uploaded_file is not None and model_file and model_type:
            # 加载模型（使用会话状态缓存）
            if 'model' not in st.session_state:
                with st.spinner("加载模型中..."):
                    st.session_state.model = load_model(model_file, model_type)
                    model = st.session_state.model
                    
                    # 模型测试推理
                    if model and test_model_inference(model, model_type):
                        st.success("模型加载成功，准备就绪！")
                    elif model:
                        st.warning("模型测试推理返回意外结果，但继续运行。")
            else:
                model = st.session_state.model
            
            # 只有在未分析时显示分析按钮
            if 'analysis_done' not in st.session_state or not st.session_state.analysis_done:
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
                            
                            # 计算处理时间
                            processing_time = end_time - start_time
                            
                            # 更新会话状态，表示已完成分析
                            st.session_state.analysis_done = True
                            
                            # 显示结果
                            with result_placeholder.container():
                                st.markdown("<div class='card'>", unsafe_allow_html=True)
                                
                                st.subheader("分析结果")
                                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                                
                                # 将OpenCV格式的结果图像转换为PIL格式并显示
                                result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                                st.image(result_pil, use_column_width=True, output_format="PNG")
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # 显示统计信息
                                st.subheader("检测统计")
                                st.markdown("""
                                <div class="stats-grid">
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
                                st.markdown('<div class="download-btn">', unsafe_allow_html=True)
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                                    result_pil.save(tmp.name)
                                    st.download_button(
                                        label="下载分析结果",
                                        data=open(tmp.name, 'rb').read(),
                                        file_name=f"corn_analysis_{time.strftime('%Y%m%d_%H%M%S')}.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
            else:
                # 如果已经分析过，直接显示结果
                result_placeholder.markdown("分析已完成，结果如下：")

# 底部信息
st.markdown("""
<div style="text-align: center; color: #718096; margin-top: 2rem; padding: 1rem; border-top: 1px solid #e2e8f0;">
    <p>© 2025 玉米坏粒识别平台 | 检测结果仅供参考，实际应用请结合专业质检流程</p>
</div>
""", unsafe_allow_html=True)

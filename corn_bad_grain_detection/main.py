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

# 优化后的应用样式
def apply_custom_styles():
    st.markdown("""
    <style>
        /* 整体布局 */
        .main {max-width: 1200px; margin: 0 auto;}
        .container {display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;}
        
        /* 卡片样式升级 */
        .card {
            background-color: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
            transition: transform 0.2s ease;
        }
        .card:hover {transform: translateY(-5px); box-shadow: 0 8px 18px rgba(0,0,0,0.12);}
        
        /* 图像容器优化 */
        .image-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        
        /* 结果统计卡片 */
        .stats-card {
            display: flex;
            justify-content: space-around;
            gap: 1.5rem;
            margin-top: 2rem;
        }
        .stat-item {
            text-align: center;
            padding: 1.2rem;
            background-color: #f8f9fa;
            border-radius: 8px;
            flex: 1;
        }
        .stat-value {
            font-size: 2.2rem;
            font-weight: 600;
            color: #2c3e50;
        }
        
        /* 按钮样式增强 */
        .stButton>button {
            background-color: #2ecc71;
            padding: 0.8rem 2rem;
            font-size: 1.1rem;
            border-radius: 20px;
            box-shadow: 0 2px 4px rgba(46, 204, 113, 0.2);
        }
        .stButton>button:hover {
            background-color: #27ae60;
            box-shadow: 0 4px 8px rgba(46, 204, 113, 0.3);
            transform: translateY(-1px);
        }
        
        /* 深色模式适配 */
        .dark-mode .card {background-color: #2d2d2d;}
        .dark-mode .image-container {box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
        .dark-mode .stats-card .stat-item {background-color: #3d3d3d; color: white;}
    </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# 标题区域
st.markdown(
    "<div style='text-align: center; margin-bottom: 2rem;'>"
    "<h1 class='main-header'>🌽 玉米坏粒识别平台</h1>"
    "<p>基于深度学习的玉米质量智能评估系统</p>"
    "</div>", 
    unsafe_allow_html=True
)

# 侧边栏保留原有功能，优化排版
with st.sidebar:
    st.header("系统设置", divider='rainbow')
    
    # 主题切换
    theme = st.radio("选择主题", ["亮色模式", "深色模式"], horizontal=True)
    if theme == "深色模式":
        st.markdown("<body class='dark-mode'>", unsafe_allow_html=True)
    
    # 模型设置
    st.header("模型管理", divider='gray')
    # ...（原有模型选择逻辑保持不变）
    
    # 关于信息
    st.header("关于", divider='gray')
    st.info("""
    本平台支持：<br>
    ✅ 多格式图像上传<br>
    ✅ 自定义模型加载<br>
    ✅ 实时摄像头拍摄<br>
    """, unsafe_allow_html=True)

# 主内容区域采用容器布局
with st.container():
    col1, col2 = st.columns([1, 1], gap='large')
    
    with col1:
        st.subheader("图像输入", divider='blue')
        # 上传组件
        uploaded_file = st.file_uploader(
            "选择图片", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="支持JPG/PNG格式，或点击下方摄像头拍摄"
        )
        # 摄像头选项
        if st.checkbox("使用摄像头拍摄", key="camera_check"):
            uploaded_file = st.camera_input("拍摄玉米照片", key="camera_input")
        
        # 原始图像展示
        if uploaded_file:
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>原始图像</h3>", unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(image, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("分析结果", divider='green')
        result_placeholder = st.empty()
        
        # 模型执行逻辑
        if uploaded_file and model_file and model_type:
            if st.button("开始分析", key="analyze_btn", help="点击进行坏粒检测"):
                # 模型加载与推理
                if not model: model = load_model(model_file, model_type)
                if model and test_model_inference(model, model_type):
                    with st.spinner("正在进行坏粒检测..."):
                        start_time = time.time()
                        result_img, bad_count = process_image(
                            np.array(Image.open(uploaded_file)), 
                            model, model_type, 
                            confidence_threshold, detection_color,
                            draw_bbox, draw_label, draw_confidence, line_thickness
                        )
                        processing_time = time.time() - start_time
                        
                        # 结果展示
                        result_placeholder.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("<h3 style='text-align: center;'>检测结果</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                        st.image(result_img, channels="BGR", use_column_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # 统计信息
                        st.markdown("""
                        <div class="stats-card">
                            <div class="stat-item">
                                <div class="stat-value">%d</div>
                                <div style="color: #7f8c8d;">坏粒数量</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">%.2f秒</div>
                                <div style="color: #7f8c8d;">处理耗时</div>
                            </div>
                        </div>
                        """ % (bad_count, processing_time), unsafe_allow_html=True)
                        
                        # 下载按钮
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
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            result_pil.save(tmp.name)
                            st.download_button(
                                "下载标注图像",
                                data=open(tmp.name, 'rb').read(),
                                file_name=f"corn_analysis_{time.strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                        result_placeholder.markdown("</div>", unsafe_allow_html=True)

# 底部提示
st.markdown(
    "<div style='text-align: center; margin: 2rem 0; color: #7f8c8d;'>"
    "提示：检测结果仅供参考，实际应用请结合专业质检流程"
    "</div>", 
    unsafe_allow_html=True
)

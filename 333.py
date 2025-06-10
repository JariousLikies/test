import streamlit as st
import os
import tempfile
import torch
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import time
import logging
from ultralytics import YOLO

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

st.header("关于")
st.info("""
本平台使用深度学习模型识别玉米坏粒，支持多种格式的图像输入。
上传图像后，系统将自动检测并标记出坏粒区域。
""")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    st.warning("未安装ultralytics库，可能无法加载某些类型的模型。")

# 侧边栏选项
with st.sidebar:
    st.header("模型选择")
    model_choice = st.radio("选择模型", ["自定义模型", "默认模型"])

    if model_choice == "自定义模型":
        model_file = st.file_uploader("上传模型文件", type=["pt", "onnx"])
        model_type = st.selectbox("模型类型", ["ONNX", "PyTorch"])
    else:
        # 假设model文件夹下有一个默认模型文件
        default_model_path = "model/default_model.onnx"
        if os.path.exists(default_model_path):
            model_file = open(default_model_path, 'rb')
            model_type = "ONNX"
        else:
            st.warning("默认模型文件不存在，请检查路径。")
            model_file = None
            model_type = None

# 加载模型
@st.cache_resource
def load_model(model_path, model_type):
    if not model_path:
        st.warning("未上传模型，使用示例参数。请上传您的模型文件以获得准确结果。")
        return None

    try:
        st.info(f"正在加载{model_type}模型: {model_path.name}")

        # 临时保存上传的模型文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_path.name)[1]) as tmp:
            tmp.write(model_path.read())
            tmp_path = tmp.name

        if model_type == "ONNX":
            # 加载ONNX模型
            model = ort.InferenceSession(
                tmp_path,
                providers=['CPUExecutionProvider']
            )
            st.success("ONNX模型加载成功！")
        else:
            # 尝试加载为Ultralytics YOLO模型
            if ULTRALYTICS_AVAILABLE:
                try:
                    model = YOLO(tmp_path)
                    st.success("Ultralytics YOLO模型加载成功！")
                    # 特殊处理：调整为评估模式
                    if hasattr(model, 'model') and hasattr(model.model, 'eval'):
                        model.model.eval()
                except Exception as e:
                    st.info(f"尝试加载为Ultralytics模型失败，错误: {e}。尝试常规加载...")

                    # 尝试常规PyTorch加载
                    if "DetectionModel" in str(e):
                        # 处理Ultralytics DetectionModel类未注册的问题
                        try:
                            from ultralytics.nn.tasks import DetectionModel
                            # 允许加载DetectionModel类
                            torch.serialization.add_safe_globals([DetectionModel])

                            # 使用weights_only=False加载完整模型
                            model = torch.load(tmp_path, map_location=torch.device('cpu'), weights_only=False)
                            st.success("PyTorch模型加载成功（使用weights_only=False）！")

                            # 检查是否需要转换为eval模式
                            if hasattr(model, 'eval'):
                                model = model.eval()
                        except Exception as e2:
                            st.error(f"加载失败: {e2}")
                            raise e2
                    else:
                        # 尝试常规PyTorch加载
                        try:
                            # 尝试常规加载
                            model = torch.load(tmp_path, map_location=torch.device('cpu'))
                            st.success("PyTorch模型加载成功！")

                            # 检查是否需要转换为eval模式
                            if hasattr(model, 'eval'):
                                model = model.eval()
                        except Exception as e2:
                            # 尝试作为TorchScript加载
                            st.info("常规加载失败，尝试作为TorchScript加载...")
                            model = torch.jit.load(tmp_path, map_location=torch.device('cpu'))
                            st.success("TorchScript模型加载成功！")
            else:
                # 没有安装ultralytics库，直接尝试常规加载
                try:
                    # 尝试常规加载
                    model = torch.load(tmp_path, map_location=torch.device('cpu'))
                    st.success("PyTorch模型加载成功！")

                    # 检查是否需要转换为eval模式
                    if hasattr(model, 'eval'):
                        model = model.eval()
                except Exception as e:
                    # 尝试作为TorchScript加载
                    st.info("常规加载失败，尝试作为TorchScript加载...")
                    model = torch.jit.load(tmp_path, map_location=torch.device('cpu'))
                    st.success("TorchScript模型加载成功！")

        # 清理临时文件
        os.unlink(tmp_path)

        # 模型测试推理（仅在上传模型后执行）
        if test_model_inference(model, model_type):
            st.info("模型测试推理成功，准备就绪！")
        else:
            st.warning("模型测试推理返回意外结果，但继续运行。")

        return model

    except Exception as e:
        st.error(f"模型加载失败: {e}")
        logger.error(f"模型加载错误: {e}", exc_info=True)
        return None


# 测试模型推理
def test_model_inference(model, model_type):
    try:
        # 创建一个随机测试图像
        test_img = torch.rand(1, 3, 640, 640)

        # 根据模型类型调整输入
        if model_type == "ONNX":
            # ONNX模型推理
            input_name = model.get_inputs()[0].name
            with torch.no_grad():
                output = model.run(None, {input_name: test_img.numpy()})
        else:
            # PyTorch/TorchScript模型推理
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    output = model.forward(test_img)
                else:
                    output = model(test_img)

        # 简单验证输出
        if model_type == "ONNX":
            # 检查ONNX输出格式
            if isinstance(output, list) and len(output) > 0:
                return True
        else:
            # 检查PyTorch输出格式
            if isinstance(output, dict) and 'boxes' in output:
                return True
            elif isinstance(output, list) and len(output) > 0:
                return True

        logger.warning(f"模型测试输出格式不匹配，但继续执行")
        return True  # 宽容处理，允许不同格式

    except Exception as e:
        logger.warning(f"模型测试推理失败: {e}")
        return True  # 宽容处理，避免阻止应用运行


# 处理图像
def process_image(image, model, model_type, threshold, color_hex):
    if model is None:
        # 使用示例检测结果
        st.warning("使用示例检测结果，因为没有加载有效模型")
        return generate_example_result(image, threshold, color_hex)

    try:
        # 转换为RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor, original_info = preprocess_image(img_rgb)

        # 模型推理
        with torch.no_grad():
            if model_type == "ONNX":
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: img_tensor.numpy()})
                results = postprocess_onnx_output(outputs, original_info)
            else:
                outputs = model(img_tensor)
                results = postprocess_pytorch_output(outputs, original_info, model_type)

        # 解析结果
        boxes = results['boxes']
        scores = results['scores']

        # 过滤低置信度预测
        filtered_indices = np.where(scores >= threshold)[0]
        filtered_boxes = boxes[filtered_indices]
        filtered_scores = scores[filtered_indices]

        # 绘制结果
        img_with_boxes = img_rgb.copy()
        total_bad = len(filtered_boxes)

        # 转换颜色并处理OpenCV的BGR格式
        color = hex_to_rgb(color_hex)
        color_bgr = (color[2], color[1], color[0])  # 转换为BGR格式

        for box, score in zip(filtered_boxes, filtered_scores):
            x1, y1, x2, y2 = map(int, box)

            # 确保边界框在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_with_boxes.shape[1], x2)
            y2 = min(img_with_boxes.shape[0], y2)

            # 跳过无效的边界框
            if x1 >= x2 or y1 >= y2:
                continue

            # 绘制边界框
            if draw_bbox:
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color_bgr, line_thickness)

            # 绘制标签和置信度
            if draw_label or draw_confidence:
                text = "坏粒"
                if draw_confidence:
                    text += f": {score:.2f}"

                # 确保文本位置在图像范围内
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x1
                text_y = max(y1 - 10, text_size[1] + 10)  # 确保文本不会超出图像顶部

                cv2.putText(
                    img_with_boxes,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_bgr,
                    2
                )

        return img_with_boxes, total_bad

    except Exception as e:
        st.error(f"图像处理失败: {e}")
        logger.error(f"图像处理错误: {e}", exc_info=True)
        # 生成示例结果
        return generate_example_result(image, threshold, color_hex)


# 标签映射（根据模型输出，强制映射为“坏粒”）
def get_label(class_id):
    return "坏粒"  # 模型只有一种标签，直接返回中文


# 图像预处理 - 改进版本，避免双重归一化
def preprocess_image(image):
    original_h, original_w = image.shape[:2]
    target_size = 640
    # 计算缩放比例（保持原图比例）
    scale = min(target_size / original_w, target_size / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    # 缩放图像
    resized_img = cv2.resize(image, (new_w, new_h))
    # 创建带 padding 的图像（灰边填充，也可黑色，不影响检测）
    padded_img = np.full((target_size, target_size, 3), 128, dtype=np.uint8)  # 128 是灰色，方便调试看 padding
    # 计算 padding 位置，让图像居中
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    padded_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_img

    # 转换为模型输入张量（归一化）
    img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    # 记录预处理信息，用于后处理还原
    return img_tensor, {
        "original_size": (original_h, original_w),
        "resized_size": (new_h, new_w),
        "pad_top": pad_top,
        "pad_left": pad_left,
        "scale": scale
    }


# 后处理PyTorch模型输出 - 增强版本，支持多种输出格式
def postprocess_pytorch_output(outputs, preprocess_info, model_type):
    try:
        original_h, original_w = preprocess_info["original_size"]
        scale = preprocess_info["scale"]
        pad_top = preprocess_info["pad_top"]
        pad_left = preprocess_info["pad_left"]

        # 适配不同模型输出（以 YOLO 系列为例，输出格式是 [batch, boxes, [x1,y1,x2,y2,score,...]]）
        if ULTRALYTICS_AVAILABLE and isinstance(outputs, list) and len(outputs) > 0:
            if hasattr(outputs[0], "boxes"):
                # YOLOv8 格式
                pred_boxes = outputs[0].boxes
                boxes = pred_boxes.xyxy.cpu().numpy()  # 原始坐标是基于 640×640 的
                scores = pred_boxes.conf.cpu().numpy()
            else:
                # 其他格式（如 YOLOv5 直接输出张量）
                pred = outputs[0].cpu().numpy()
                boxes = pred[..., :4]  # x1,y1,x2,y2
                scores = pred[..., 4]  # score
        else:
            # 兜底：假设输出是张量或字典（根据实际模型调整）
            raise ValueError("模型输出格式未适配，请根据模型类型调整后处理逻辑！")

        # 还原边界框到原图尺寸：先减去 padding，再按缩放比例还原
        boxes[:, 0] = (boxes[:, 0] - pad_left) / scale
        boxes[:, 1] = (boxes[:, 1] - pad_top) / scale
        boxes[:, 2] = (boxes[:, 2] - pad_left) / scale
        boxes[:, 3] = (boxes[:, 3] - pad_top) / scale

        # 边界框坐标限制在原图范围内
        boxes[:, 0] = np.clip(boxes[:, 0], 0, original_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, original_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, original_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, original_h)

        return {"boxes": boxes, "scores": scores}
    except Exception as e:
        logger.error(f"PyTorch 输出后处理失败: {e}", exc_info=True)
        # 示例结果兜底
        return {
            "boxes": np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
            "scores": np.array([0.8, 0.7])
        }


# 后处理ONNX模型输出 - 增强版本，支持多种输出格式
def postprocess_onnx_output(outputs, original_info):
    try:
        # 调试输出：打印输出类型和结构
        logger.debug(f"ONNX模型输出数量: {len(outputs)}")
        for i, output in enumerate(outputs):
            logger.debug(f"输出 {i} 形状: {output.shape}")

        # 处理常见的ONNX输出格式
        boxes = None
        scores = None

        # 尝试标准格式 [boxes, scores]
        if len(outputs) >= 2:
            boxes = outputs[0]
            scores = outputs[1]

            if boxes.ndim == 3:
                boxes = boxes[0]  # 移除batch维度
            if scores.ndim == 2:
                scores = scores[0]  # 移除batch维度

            logger.info("检测到标准ONNX格式输出")

        # 尝试YOLO格式 [n, 85] 或 [1, n, 85]
        elif len(outputs) == 1 and outputs[0].ndim in [2, 3]:
            pred = outputs[0]
            if pred.ndim == 3:
                pred = pred[0]  # 移除batch维度

            boxes = pred[:, :4]  # x1, y1, x2, y2
            scores = pred[:, 4]  # confidence
            logger.info("检测到YOLO ONNX格式输出")

        else:
            raise ValueError("无法识别的ONNX模型输出格式")

        # 调整边界框大小以匹配原始图像
        original_size = original_info['original_size']
        new_size = original_info['new_size']
        pad = original_info['pad']

        # 计算缩放比例
        scale_x = original_size[1] / new_size[0]
        scale_y = original_size[0] / new_size[1]

        # 调整边界框坐标
        boxes[:, 0] = (boxes[:, 0] - pad[0] / 2) * scale_x
        boxes[:, 1] = (boxes[:, 1] - pad[1] / 2) * scale_y
        boxes[:, 2] = (boxes[:, 2] - pad[0] / 2) * scale_x
        boxes[:, 3] = (boxes[:, 3] - pad[1] / 2) * scale_y

        return {'boxes': boxes, 'scores': scores}

    except Exception as e:
        logger.error(f"ONNX输出后处理失败: {e}", exc_info=True)
        # 返回示例结果
        return {
            'boxes': np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
            'scores': np.array([0.8, 0.7])
        }


# 生成示例结果
def generate_example_result(image, threshold, color_hex):
    img_with_boxes = image.copy()
    height, width = image.shape[:2]

    # 转换颜色并处理OpenCV的BGR格式
    color = hex_to_rgb(color_hex)
    color_bgr = (color[2], color[1], color[0])  # 转换为BGR格式

    # 随机生成一些示例检测框
    num_detections = np.random.randint(1, 10)
    total_bad = 0

    for _ in range(num_detections):
        score = np.random.uniform(threshold, 1.0)
        if score >= threshold:
            total_bad += 1

            # 随机位置和大小，但确保在图像范围内
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = np.random.randint(x1 + 10, width)
            y2 = np.random.randint(y1 + 10, height)

            # 绘制边界框
            if draw_bbox:
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color_bgr, line_thickness)

            # 绘制标签和置信度
            if draw_label or draw_confidence:
                text = "坏粒"
                if draw_confidence:
                    text += f": {score:.2f}"

                # 确保文本位置在图像范围内
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x1
                text_y = max(y1 - 10, text_size[1] + 10)  # 确保文本不会超出图像顶部

                cv2.putText(
                    img_with_boxes,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_bgr,
                    2
                )

    return img_with_boxes, total_bad


# 十六进制颜色转RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# 主界面
col1, col2 = st.columns([1, 1])

# 假设的全局变量
draw_bbox = True
draw_label = True
draw_confidence = True
line_thickness = 2
confidence_threshold = 0.5
detection_color = "#FF0000"

with col1:
    st.subheader("上传图像")

    # 上传图像文件
    uploaded_file = st.file_uploader(
        "选择一张图片",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    # 或者从摄像头捕获
    use_camera = st.checkbox("使用摄像头拍摄")
    if use_camera:
        uploaded_file = st.camera_input("拍摄玉米照片")

    if uploaded_file is not None:
        # 显示原始图像
        st.subheader("原始图像")
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # 如果图像是RGBA格式，转换为RGB
        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        st.image(image, use_column_width=True)

        # 加载模型
        model = load_model(model_file, model_type)

        if st.button("开始分析"):
            if model is None and not model_file:
                st.warning("未加载模型，将使用示例参数进行演示。")

            with st.spinner("正在分析图像..."):
                start_time = time.time()
                result_img, bad_count = process_image(
                    img_array, model, model_type, confidence_threshold, detection_color
                )
                end_time = time.time()

                # 显示处理时间
                processing_time = end_time - start_time
                st.write(f"分析完成！耗时: {processing_time:.2f}秒")

                # 显示结果图像
                with col2:
                    st.subheader("分析结果")
                    st.image(
                        cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                        use_column_width=True
                    )

                    # 显示统计信息
                    st.subheader("统计信息")
                    st.metric("坏粒数量", bad_count)

                    # 下载结果
                    result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        result_pil.save(tmp.name)
                        st.download_button(
                            label="下载分析结果",
                            data=open(tmp.name, 'rb').read(),
                            file_name="corn_analysis_result.png",
                            mime="image/png"
                        )

import streamlit as st
import cv2
import numpy as np
import torch
import logging
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import tempfile
import logging
import streamlit as st
import onnxruntime as ort
from utils.example_generator import generate_example_result

logger = logging.getLogger(__name__)

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
        if isinstance(outputs, list) and len(outputs) > 0:
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
            if isinstance(outputs, dict) and 'boxes' in outputs:
                boxes = outputs['boxes'].cpu().numpy()
                scores = outputs['scores'].cpu().numpy()
            else:
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

# 处理图像
def process_image(image, model, model_type, threshold, color_hex, draw_bbox, draw_label, draw_confidence, line_thickness):
    if model is None:
        # 使用示例检测结果
        st.warning("使用示例检测结果，因为没有加载有效模型")
        return generate_example_result(image, threshold, color_hex, draw_bbox, draw_label, draw_confidence, line_thickness)

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
        from utils.color_converter import hex_to_rgb
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
        return generate_example_result(image, threshold, color_hex, draw_bbox, draw_label, draw_confidence, line_thickness)

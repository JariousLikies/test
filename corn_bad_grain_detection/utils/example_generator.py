import cv2
import numpy as np
from utils.color_converter import hex_to_rgb
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import tempfile
import logging
import streamlit as st
import onnxruntime as ort
def generate_example_result(image, threshold, color_hex, draw_bbox, draw_label, draw_confidence, line_thickness):
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
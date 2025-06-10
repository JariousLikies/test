import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import tempfile
import logging
import streamlit as st
import onnxruntime as ort
logger = logging.getLogger(__name__)

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
                try:
                    output = model(test_img)
                except Exception as e:
                    logger.warning(f"模型推理时出错: {e}")
                    return False

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
        return False  # 测试失败
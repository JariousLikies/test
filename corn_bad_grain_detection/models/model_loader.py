import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import tempfile
import logging
import streamlit as st
import onnxruntime as ort

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

def load_model(model_path, model_type):
    if not model_path:
        st.error("没有提供模型文件，请先选择模型。")
        return None

    try:
        st.info(f"正在加载{model_type}模型...")

        # 临时保存上传的模型文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_path.name)[1]) as tmp:
            if hasattr(model_path, 'read'):
                model_content = model_path.read()
                tmp.write(model_content)
                model_path.seek(0)  # 重置文件指针，以便后续使用
            else:
                with open(model_path.name, 'rb') as f:
                    model_content = f.read()
                    tmp.write(model_content)
            tmp_path = tmp.name

        st.info(f"模型文件临时保存路径: {tmp_path}")

        if model_type == "ONNX":
            try:
                model = ort.InferenceSession(
                    tmp_path,
                    providers=['CPUExecutionProvider']
                )
                st.success("ONNX模型加载成功！")
            except Exception as e:
                st.error(f"ONNX模型加载失败: {e}")
                raise
        else:
            try:
                if ULTRALYTICS_AVAILABLE:
                    try:
                        model = YOLO(tmp_path)
                        st.success("Ultralytics YOLO模型加载成功！")
                        if hasattr(model, 'model') and hasattr(model.model, 'eval'):
                            model.model.eval()
                    except Exception as e:
                        st.info(f"尝试加载为Ultralytics模型失败，错误: {e}。尝试常规加载...")
                        model = torch.load(tmp_path, map_location=torch.device('cpu'))
                        st.success("PyTorch模型加载成功！")
                        if hasattr(model, 'eval'):
                            model = model.eval()
                else:
                    model = torch.load(tmp_path, map_location=torch.device('cpu'))
                    st.success("PyTorch模型加载成功！")
                    if hasattr(model, 'eval'):
                        model = model.eval()
            except Exception as e:
                st.info("常规加载失败，尝试作为TorchScript加载...")
                try:
                    model = torch.jit.load(tmp_path, map_location=torch.device('cpu'))
                    st.success("TorchScript模型加载成功！")
                except Exception as e2:
                    st.error(f"TorchScript模型加载失败: {e2}")
                    raise

        # 清理临时文件
        os.unlink(tmp_path)

        return model

    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None
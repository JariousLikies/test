import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import tempfile
import logging
import streamlit as st
import onnxruntime as ort
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import tempfile
import logging
import streamlit as st
import onnxruntime as ort
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
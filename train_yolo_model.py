import torch
import os
from yolov5 import train

def train_yolo_model(data_yaml, epochs=100, batch_size=16):
    # Train the YOLOv5 model
    os.system(f"python train.py --img 640 --batch {batch_size} --epochs {epochs} --data {data_yaml} --weights yolov5s.pt")


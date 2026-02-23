import warnings
warnings.filterwarnings('ignore')
import os
import tempfile

# Workaround for "FileNotFoundError: [Errno 2] No usable temporary directory found"
# This must be done before importing ultralytics/torch
try:
    tempfile.gettempdir()
except FileNotFoundError:
    os.environ['TMPDIR'] = '/tmp'

from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize model with the RGBT mid-fusion configuration
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion.yaml')
    
    # Start training
    model.train(
        data='/home/radar/Desktop/data.yaml', 
        epochs=100, 
        batch=16, 
        imgsz=640, 
        device='cuda'  # Optimized for jetson
    )
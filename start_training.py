import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Mac workaround for temporary directories
try:
    tempfile.gettempdir()
except FileNotFoundError:
    os.environ['TMPDIR'] = '/tmp'

from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize standard YOLO model (handles both RGB and Grayscale images as 3-channel)
    model = YOLO('yolo11n.pt')
    
    # Start training
    model.train(
        data='data.yaml', 
        epochs=100, 
        batch=16, 
        imgsz=640, 
        device='mps'
    )
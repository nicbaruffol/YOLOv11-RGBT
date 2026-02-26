import torch
from ultralytics import YOLO
from torch2trt import torch2trt

def convert_model(pt_path, out_path, channels):
    print(f"Loading {pt_path}...")
    # Load the Ultralytics model
    yolo = YOLO(pt_path)
    
    # Extract the raw PyTorch neural network, set to eval mode, and move to Jetson GPU
    model = yolo.model.eval().cuda()
    
    # Create a dummy input tensor (Batch Size 1, Channels, Height, Width)
    # Ensure this matches your actual camera/video resolution!
    dummy_input = torch.randn(1, channels, 640, 640).cuda()

    print("Converting to TensorRT (FP16)... This will take a few minutes.")
    # Run the torch2trt compiler
    model_trt = torch2trt(model, [dummy_input], fp16_mode=True)
    
    # Save the optimized model
    torch.save(model_trt.state_dict(), out_path)
    print(f"Saved optimized model to {out_path}\n")

# Convert the 3-channel RGB and IR models
convert_model("runs/Anti-UAV/yolo11n-RGB-Only2/weights/best.pt", "rgb_trt.pth", channels=3)
convert_model("runs/Anti-UAV/yolo11n-IR-Only/weights/best.pt", "ir_trt.pth", channels=3)

# Convert the 6-channel RGBT model
convert_model("runs/Anti-UAV/yolo11n-RGBRGB/weights/best.pt", "rgbt_trt.pth", channels=6)  
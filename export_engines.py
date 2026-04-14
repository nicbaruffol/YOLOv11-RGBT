import torch
from ultralytics import YOLO

# 1. We create a global hook to intercept torch.zeros
original_zeros = torch.zeros
TARGET_CHANNELS = 3

def patched_zeros(*args, **kwargs):
    global TARGET_CHANNELS
    try:
        # If the exporter tries to build the dummy image like: torch.zeros(1, 1, 640, 640)
        if len(args) == 4 and args[2] == 640 and args[3] == 640:
            new_args = list(args)
            new_args[1] = TARGET_CHANNELS  # Force it to 3 or 6 channels
            return original_zeros(*new_args, **kwargs)
            
        # If the exporter tries to build it with a tuple: torch.zeros((1, 1, 640, 640))
        if len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 4:
            shape = list(args[0])
            if shape[2] == 640 and shape[3] == 640:
                shape[1] = TARGET_CHANNELS  # Force it to 3 or 6 channels
                return original_zeros(tuple(shape), **kwargs)
    except Exception:
        pass
        
    # Let all other normal torch.zeros calls pass through untouched
    return original_zeros(*args, **kwargs)

# Apply the patch to PyTorch globally
torch.zeros = patched_zeros


def export_model(weights_path, channels):
    global TARGET_CHANNELS
    TARGET_CHANNELS = channels  # Set the target before exporting
    
    print(f"\n--- Loading {weights_path} ---")
    model = YOLO(weights_path)
    
    print(f"Exporting with {channels} channels... This will take 10-15 minutes.")
    # Because we use the native export here, it WILL embed the metadata 
    # required to fix the NoneType Gradio error!
    model.export(
        format="engine", 
        half=True, 
        workspace=4, 
        imgsz=640
    )
    print(f"--- Finished exporting {weights_path} ---")

if __name__ == "__main__":
    # Export RGB Model (Force 3 channels)       
    export_model("YOLOv11-RGBT/runs/Anti-UAV/yolo11n-RGB-Only3/weights/best.pt", 3)
    
    # Export Thermal Model (Force 3 channe ls)
    export_model("YOLOv11-RGBT/runs/Anti-UAV/yolo11n-IR-Only2/weights/best.pt", 3)
    
    # Export Combined RGBT Model (Force 6 channels)
    export_model("YOLOv11-RGBT/runs/Anti-UAV/yolo11n-RGBRGB2/weights/best.pt", 6)
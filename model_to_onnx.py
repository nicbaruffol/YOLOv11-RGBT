import torch
import torch.nn as nn
from ultralytics import YOLO

# 1. Define a Wrapper Class
class RGBTWrapper(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model
        
        # Force the detection head into export mode
        for m in self.model.modules():
            if type(m).__name__ == "Detect":
                m.export = True
                m.format = 'onnx'

    def forward(self, rgb, ir):
        # Concatenate the two 3-channel inputs into one 6-channel tensor
        # This gives the model the single tensor it expects to slice apart
        x = torch.cat([rgb, ir], dim=1) 
        
        return self.model(x)

# 2. Load the model
model = YOLO("runs/Anti-UAV/yolo11n-RGBRGB/weights/best.pt")
model.model.eval()

# 3. Wrap it
wrapped_model = RGBTWrapper(model).to("cuda")
wrapped_model.eval()

# 4. Prepare dummy inputs
imgsz = 640
dummy_rgb = torch.randn(1, 3, imgsz, imgsz, device="cuda")
dummy_ir = torch.randn(1, 3, imgsz, imgsz, device="cuda")

# 5. Export to ONNX
onnx_file = "yolo11-rgbt.onnx"
torch.onnx.export(
    wrapped_model, 
    (dummy_rgb, dummy_ir),  # Now perfectly matches the wrapper's forward(self, rgb, ir)
    onnx_file,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input_rgb', 'input_ir'],
    output_names=['output0'],
    dynamic_axes={
        'input_rgb': {0: 'batch'},
        'input_ir': {0: 'batch'},
        'output0': {0: 'batch'}
    }
)

print(f"Successfully exported to {onnx_file}")
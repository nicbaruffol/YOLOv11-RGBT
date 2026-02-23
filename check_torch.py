import torch
from ultralytics import checks

# 1. Check base PyTorch CUDA support
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")

# 2. Run YOLO's comprehensive environment check
print("\n--- Running Ultralytics Checks ---")
checks()
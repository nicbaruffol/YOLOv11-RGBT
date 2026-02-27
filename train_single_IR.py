from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load the STANDARD pre-trained weights (NOT your 6-channel weights)
    model = YOLO('YOLOv11-RGBT/runs/Anti-UAV/yolo11n-IR-Only/weights/best.pt') # This is the standard YOLOv11n model, not the gray one. We will specify channels in training.') 

    # 2. Train the model (Example: RGB)
    model.train(
        data="/cluster/home/nbaruffol/YOLOv11-RGBT/data_ir.yaml", # Change to data_ir.yaml for Thermal
        epochs=100,
        batch=16,          
        imgsz=640,
        device='cuda',
        workers=4,
        fraction=0.5,      # Keeping your speed trick!
        project='runs/Anti-UAV',
        name='yolo11n-IR-Only2', # Change to 'yolo11n-IR-Only' when running thermal
        patience=20,
    )
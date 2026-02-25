from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load the STANDARD pre-trained weights (NOT your 6-channel weights)
    model = YOLO('/cluster/home/nbaruffol/YOLOv11-RGBT/ultralytics/cfg/models/11/yolo11.yaml') # This is the standard YOLOv11n model, not the gray one. We will specify channels in training.') 

    # 2. Train the model (Example: RGB)
    model.train(
        data="/cluster/home/nbaruffol/YOLOv11-RGBT/data_ir.yaml", # Change to data_ir.yaml for Thermal
        epochs=100,
        batch=16,          
        imgsz=640,
        device='cuda',
        workers=4,
        fraction=0.1,      # Keeping your speed trick!
        patience=10,       # Stop early if it stops improving
        project='runs/Anti-UAV',
        name='yolo11n-IR-Only', # Change to 'yolo11n-IR-Only' when running thermal
    )
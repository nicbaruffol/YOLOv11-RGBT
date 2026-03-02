import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/Anti-UAV/yolo11n-RGBRGB23/weights/last.pt')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
    # model.load(r'yolo11n-RGBRGB6C-midfussion.pt') # loading pretrain weights 网盘下载
    model.train(data='data_rgb_ir.yaml',
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=0,
                workers=4,
                device='cuda',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                fraction=0.5,
                use_simotm="RGBRGB6C",
                channels=6,  #
                project='runs/Anti-UAV',
                name='yolo11n-RGBRGB2',  # name of the training run
                patience=20,  # early stopping patience
                # val=True,
                )
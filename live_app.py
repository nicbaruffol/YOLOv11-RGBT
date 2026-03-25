import gradio as gr
import cv2
import tempfile
import numpy as np
import threading
import time
from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor

# Import PySpin for the Point Grey / FLIR industrial camera
try:
    import PySpin
except ImportError:
    print("⚠️ PySpin not found! Spinnaker SDK cameras will not work.")

# --- METADATA BYPASS PATCH ---
original_setup_model = BasePredictor.setup_model

def patched_setup_model(self, model, verbose=False):
    original_setup_model(self, model, verbose)
    try:
        for k, v in self.model.bindings.items():
            shape = v.shape if hasattr(v, 'shape') else v.get('shape')
            if shape and len(shape) == 4:
                self.channels = shape[1] 
                break
    except Exception:
        weights = str(getattr(self.model, 'weights', ''))
        self.channels = 6 if 'RGBRGB' in weights else 3

BasePredictor.setup_model = patched_setup_model
# -----------------------------

print("Loading TensorRT engines...")
model_rgb = YOLO("runs/Anti-UAV/yolo11n-RGB-Only3/weights/best.engine", task="detect")
model_ir = YOLO("runs/Anti-UAV/yolo11n-IR-Only2/weights/best.engine", task="detect")
model_rgbt = YOLO("runs/Anti-UAV/yolo11n-RGBRGB2/weights/best.engine", task="detect")
print("Engines loaded successfully!")

# --- THREADED V4L2 CAMERA CLASS (For the IR Boson) ---
class CameraStream:
    def __init__(self, src=0):
        try:
            src = int(src)
        except ValueError:
            pass 

        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            self.grabbed, self.frame, self.stopped = False, None, True
            return

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        if not self.stopped:
            threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if hasattr(self, 'stream') and self.stream.isOpened():
            self.stream.release()

# --- THREADED SPINNAKER CAMERA CLASS (For Point Grey RGB) ---
class SpinnakerStream:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        
        if self.cam_list.GetSize() == 0:
            print("⚠️ No Spinnaker cameras found!")
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            self.grabbed, self.frame, self.stopped = False, None, True
            return

        # Grab the first available Spinnaker camera
        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()
        
        # 1. HARDWARE LEVEL FIX: Force the camera to output BGR8 natively
        try:
            # If the camera node is writable, switch it from Raw/Bayer to BGR8
            if self.cam.PixelFormat.GetAccessMode() == PySpin.RW:
                self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
                print("✅ Spinnaker hardware successfully set to BGR8 mode.")
            else:
                print("⚠️ Spinnaker PixelFormat is Read-Only. Falling back to software conversion.")
        except Exception as e:
            print(f"⚠️ Could not set hardware BGR8 mode: {e}")

        # 2. SOFTWARE FALLBACK: Setup Image Processor (if it exists in your SDK version)
        self.processor = None
        if hasattr(PySpin, 'ImageProcessor'):
            self.processor = PySpin.ImageProcessor()
            try:
                self.processor.SetColorProcessing(PySpin.HQ_LINEAR)
            except AttributeError:
                pass
        
        # Set acquisition mode to continuous
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.BeginAcquisition()

        self.grabbed = True
        self.frame = None
        self.stopped = False

    def start(self):
        if not self.stopped:
            threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                # Grab image from camera buffer (1000ms timeout)
                image_result = self.cam.GetNextImage(1000)
                
                if not image_result.IsIncomplete():
                    # Extract the raw NumPy array from the buffer
                    raw_array = image_result.GetNDArray()
                    
                    # Check if the hardware gave us a 3-channel color image natively
                    if len(raw_array.shape) == 3 and raw_array.shape[2] == 3:
                        self.frame = raw_array
                        
                    # If the hardware gave us a 2D Grayscale/Bayer image, we must convert it
                    else:
                        # Attempt SDK 4.x Processor Conversion
                        if self.processor is not None:
                            converted = self.processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                            self.frame = converted.GetNDArray()
                            
                        # Attempt SDK 2.x Legacy Conversion
                        elif hasattr(image_result, 'Convert'):
                            converted = image_result.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
                            self.frame = converted.GetNDArray()
                            
                        # ULTIMATE BULLETPROOF FALLBACK: OpenCV Conversion
                        # This guarantees the app will not crash so YOLO can run!
                        else:
                            self.frame = cv2.cvtColor(raw_array, cv2.COLOR_GRAY2BGR)
                            
                    self.grabbed = True
                
                # We MUST release the image to free the buffer
                image_result.Release()
                
            except PySpin.SpinnakerException as e:
                print(f"Spinnaker Stream Error: {e}")
                self.grabbed = False

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        try:
            if hasattr(self, 'cam') and self.cam.IsInitialized():
                self.cam.EndAcquisition()
                self.cam.DeInit()
                del self.cam
            if hasattr(self, 'cam_list'):
                self.cam_list.Clear()
            if hasattr(self, 'system'):
                self.system.ReleaseInstance()
        except Exception as e:
            print(f"Error shutting down Spinnaker: {e}")

# --- LIVE PROCESSING FUNCTION ---
def live_stream(mode, rgb_cam_id, ir_cam_id):
    cams = []
    rgb_stream = None
    ir_stream = None
    
    # Initialize RGB Camera (Spinnaker or Standard USB)
    if mode in ["RGB Only", "Combined RGB-T"]:
        if str(rgb_cam_id).strip().lower() == "spinnaker":
            rgb_stream = SpinnakerStream().start()
        else:
            rgb_stream = CameraStream(src=rgb_cam_id).start()
        cams.append(rgb_stream)
        
    # Initialize IR Camera (Boson via V4L2)
    if mode in ["Thermal Only", "Combined RGB-T"]:
        ir_stream = CameraStream(src=ir_cam_id).start()
        cams.append(ir_stream)

    time.sleep(1.0) # Hardware warmup

    try:
        while True:
            final_frame = None
            
            if mode == "RGB Only" and rgb_stream and not rgb_stream.stopped:
                rgb_frame = rgb_stream.read()
                if rgb_frame is None: continue
                
                results = model_rgb(rgb_frame, verbose=False)
                final_frame = results[0].plot()

            elif mode == "Thermal Only" and ir_stream and not ir_stream.stopped:
                ir_frame = ir_stream.read()
                if ir_frame is None: continue
                
                results = model_ir(ir_frame, verbose=False)
                final_frame = results[0].plot()

            elif mode == "Combined RGB-T" and rgb_stream and ir_stream:
                rgb_frame = rgb_stream.read()
                ir_frame = ir_stream.read()
                
                if rgb_frame is None or ir_frame is None: continue
                
                # Resize IR to match Point Grey resolution
                if rgb_frame.shape != ir_frame.shape:
                    ir_frame = cv2.resize(ir_frame, (rgb_frame.shape[1], rgb_frame.shape[0]))
                
                combined_input = np.concatenate((rgb_frame, ir_frame), axis=-1)
                results = model_rgbt(combined_input, verbose=False)
                final_frame = results[0].plot(img=rgb_frame.copy())

            if final_frame is not None:
                ui_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                yield ui_frame
            else:
                yield np.zeros((640, 640, 3), dtype=np.uint8)

    finally:
        for cam in cams:
            if cam: cam.stop()

def update_inputs(mode):
    if mode == "RGB Only":
        return gr.update(visible=True), gr.update(visible=False)
    elif mode == "Thermal Only":
        return gr.update(visible=False), gr.update(visible=True)
    else:  
        return gr.update(visible=True), gr.update(visible=True)

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🚁 LIVE Multi-Modal Anti-UAV Detection")
    
    with gr.Row():
        with gr.Column(scale=1):
            mode_selector = gr.Radio(
                choices=["RGB Only", "Thermal Only", "Combined RGB-T"],
                value="Combined RGB-T",
                label="Select Detection Mode"
            )
            
            # The word 'spinnaker' triggers the PySpin class!
            rgb_input = gr.Textbox(label="RGB Camera ID", value="spinnaker", visible=True)
            ir_input = gr.Textbox(label="Thermal Camera ID (e.g., 0)", value="0", visible=True)
            
            start_btn = gr.Button("▶️ Start Live Stream", variant="primary")
            stop_btn = gr.Button("⏹️ Stop Stream", variant="stop")
            
        with gr.Column(scale=2):
            live_feed = gr.Image(label="Live Inference Feed", streaming=True)

    mode_selector.change(fn=update_inputs, inputs=mode_selector, outputs=[rgb_input, ir_input])
    
    stream_event = start_btn.click(
        fn=live_stream, 
        inputs=[mode_selector, rgb_input, ir_input], 
        outputs=[live_feed]
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[stream_event])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
import gradio as gr
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor

# --- METADATA BYPASS PATCH ---
# This intercepts the YOLO initialization and forces it to use the correct 
# channel counts directly from the TensorRT hardware bindings, ignoring buggy metadata.
original_setup_model = BasePredictor.setup_model

def patched_setup_model(self, model, verbose=False):
    # Run the standard initialization
    original_setup_model(self, model, verbose)
    
    # Force the Predictor to use the actual engine shapes
    try:
        for k, v in self.model.bindings.items():
            shape = v.shape if hasattr(v, 'shape') else v.get('shape')
            # Look for the 4-dimensional input binding [Batch, Channels, Height, Width]
            if shape and len(shape) == 4:
                self.channels = shape[1]  # Automatically sets to 3 or 6!
                break
    except Exception:
        # Bulletproof fallback based on the file path just in case
        weights = str(getattr(self.model, 'weights', ''))
        self.channels = 6 if 'RGBRGB' in weights else 3

BasePredictor.setup_model = patched_setup_model
# -----------------------------

# Load the optimized TensorRT engines
print("Loading TensorRT engines... This may take a moment.")
model_rgb = YOLO("runs/Anti-UAV/yolo11n-RGB-Only2/weights/best.engine", task="detect")
model_ir = YOLO("runs/Anti-UAV/yolo11n-IR-Only/weights/best.engine", task="detect")
model_rgbt = YOLO("runs/Anti-UAV/yolo11n-RGBRGB/weights/best.engine", task="detect")
print("Engines loaded successfully!")

def process_video(mode, rgb_path, ir_path):
    if mode in ["RGB Only", "Combined RGB-T"] and not rgb_path:
        raise gr.Error("Please upload an RGB video.")
    if mode in ["Thermal Only", "Combined RGB-T"] and not ir_path:
        raise gr.Error("Please upload a Thermal (IR) video.")

    caps = []
    if mode == "RGB Only":
        caps.append(cv2.VideoCapture(rgb_path))
    elif mode == "Thermal Only":
        caps.append(cv2.VideoCapture(ir_path))
    else:  # Combined
        caps.append(cv2.VideoCapture(rgb_path))
        caps.append(cv2.VideoCapture(ir_path))

    cap_ref = caps[0]
    width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_ref.get(cv2.CAP_PROP_FPS)) or 30

    output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    final_frame = None
    ui_frame = None

    while True:
        frames = []
        ret_flags = []
        
        for cap in caps:
            ret, frame = cap.read()
            ret_flags.append(ret)
            frames.append(frame)
            
        if not all(ret_flags):
            break

        # --- INFERENCE ---
        if mode == "RGB Only":
            results = model_rgb(frames[0], verbose=False)
            final_frame = results[0].plot() 
            
        elif mode == "Thermal Only":
            results = model_ir(frames[0], verbose=False)
            final_frame = results[0].plot()
            
        elif mode == "Combined RGB-T":
            if frames[0].shape != frames[1].shape:
                frames[1] = cv2.resize(frames[1], (frames[0].shape[1], frames[0].shape[0]))
            
            # Combine into 6 channels [H, W, 6]
            combined_input = np.concatenate((frames[0], frames[1]), axis=-1)
            
            results = model_rgbt(combined_input, verbose=False)
            final_frame = results[0].plot(img=frames[0].copy())

        out.write(final_frame)

        # Convert for Gradio UI
        ui_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        yield ui_frame, None

    for cap in caps:
        cap.release()
    out.release()

    if ui_frame is not None:
        yield ui_frame, output_video_path
    else:
        yield None, None

def update_inputs(mode):
    if mode == "RGB Only":
        return gr.update(visible=True), gr.update(visible=False)
    elif mode == "Thermal Only":
        return gr.update(visible=False), gr.update(visible=True)
    else:  
        return gr.update(visible=True), gr.update(visible=True)

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🚁 Multi-Modal Anti-UAV Detection Dashboard")
    
    with gr.Row():
        with gr.Column(scale=1):
            mode_selector = gr.Radio(
                choices=["RGB Only", "Thermal Only", "Combined RGB-T"],
                value="RGB Only",
                label="Select Detection Mode"
            )
            
            rgb_input = gr.Video(label="RGB Video", visible=True)
            ir_input = gr.Video(label="Thermal (IR) Video", visible=False)
            
            process_btn = gr.Button("Detect Drones", variant="primary")
            
        with gr.Column(scale=2):
            live_feed = gr.Image(label="Live Inference Feed")
            final_output = gr.Video(label="Final Downloadable Video")

    mode_selector.change(fn=update_inputs, inputs=mode_selector, outputs=[rgb_input, ir_input])
    process_btn.click(fn=process_video, inputs=[mode_selector, rgb_input, ir_input], outputs=[live_feed, final_output])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
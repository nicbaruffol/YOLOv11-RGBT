import gradio as gr
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO  # NEW: Import YOLO

model_rgb = YOLO("runs/train/rgb_training/weights/best.pt")
model_ir = YOLO("runs/train/ir_training/weights/best.pt")

# If you trained the unified dual-stream RGBT model, load it here
model_rgbt = YOLO("runs/train/rgbt_training/weights/best.pt")

def process_video(mode, rgb_path, ir_path):
    # Error checking to ensure the user uploaded the required videos
    if mode in ["RGB Only", "Combined RGB-T"] and not rgb_path:
        return None
    if mode in ["Thermal Only", "Combined RGB-T"] and not ir_path:
        return None

    caps = []
    if mode == "RGB Only":
        caps.append(cv2.VideoCapture(rgb_path))
    elif mode == "Thermal Only":
        caps.append(cv2.VideoCapture(ir_path))
    else:  # Combined
        caps.append(cv2.VideoCapture(rgb_path))
        caps.append(cv2.VideoCapture(ir_path))

    # Extract properties from the first valid video stream
    cap_ref = caps[0]
    width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_ref.get(cv2.CAP_PROP_FPS))

    # If combined, the output video will be twice as wide (side-by-side)
    out_width = width * 2 if mode == "Combined RGB-T" else width
    
    output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, height))

    while True:
        frames = []
        ret_flags = []
        
        # Read a frame from all active video captures
        for cap in caps:
            ret, frame = cap.read()
            ret_flags.append(ret)
            frames.append(frame)
            
        # If any video stream ends, stop processing
        if not all(ret_flags):
            break

        # ---------------------------------------------------------
        # --- INFERENCE PLACEMENT ---
        # If RGB Only: results = rgb_model(frames[0])
        # If IR Only: results = ir_model(frames[0])
        # If Combined: results = rgbt_model(frames[0], frames[1])
        # ---------------------------------------------------------

        # Mock drawing bounding boxes on the active frames
        for frame in frames:
            cv2.rectangle(frame, (200, 200), (400, 400), (0, 255, 0), 3)
            cv2.putText(frame, 'UAV - 0.95', (200, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if mode == "RGB Only":
            # Run inference on the visible frame
            # verbose=False stops it from printing to the console every single frame
            results = model_rgb(frames[0], verbose=False)
            
            # .plot() automatically draws the bounding boxes and confidence scores
            frames[0] = results[0].plot() 
            
        elif mode == "Thermal Only":
            # Run inference on the infrared frame
            results = model_ir(frames[0], verbose=False)
            frames[0] = results[0].plot()
            
        elif mode == "Combined RGB-T":
            # If you are running two separate models side-by-side:
            res_rgb = model_rgb(frames[0], verbose=False)
            res_ir = model_ir(frames[1], verbose=False)
            
            frames[0] = res_rgb[0].plot()
            frames[1] = res_ir[0].plot()

        out.write(final_frame)

    for cap in caps:
        cap.release()
    out.release()

    return output_video_path

# --- UI Visibility Logic ---
def update_inputs(mode):
    """Dynamically shows/hides video input fields based on the selected mode."""
    if mode == "RGB Only":
        return gr.update(visible=True), gr.update(visible=False)
    elif mode == "Thermal Only":
        return gr.update(visible=False), gr.update(visible=True)
    else:  # Combined RGB-T
        return gr.update(visible=True), gr.update(visible=True)


# --- Build the UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🚁 Multi-Modal Anti-UAV Detection Dashboard")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Mode Selection
            mode_selector = gr.Radio(
                choices=["RGB Only", "Thermal Only", "Combined RGB-T"],
                value="RGB Only",
                label="Select Detection Mode"
            )
            
            # Input Videos
            rgb_input = gr.Video(label="RGB Video", visible=True)
            ir_input = gr.Video(label="Thermal (IR) Video", visible=False)
            
            process_btn = gr.Button("Detect Drones", variant="primary")
            
        with gr.Column(scale=2):
            # Output Video
            output_video = gr.Video(label="Detection Output")

    # Link the radio buttons to the UI update function
    mode_selector.change(
        fn=update_inputs, 
        inputs=mode_selector, 
        outputs=[rgb_input, ir_input]
    )

    # Link the run button to the processing function
    process_btn.click(
        fn=process_video, 
        inputs=[mode_selector, rgb_input, ir_input], 
        outputs=output_video
    )

if __name__ == "__main__":
    app.launch()
import gradio as gr
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# Load all three models at the top of your script
model_rgbt = YOLO("runs/Anti-UAV/yolo11n-RGBRGB6C-midfusion/weights/best.pt")
model_rgb = YOLO("runs/train/rgb_training/weights/best.pt") # Update with your real path
model_ir = YOLO("runs/train/ir_training/weights/best.pt")   # Update with your real path

# --- UPDATED: Added conf_thresh to the function signature ---
def process_video(mode, rgb_path, ir_path, conf_thresh):
    if mode in ["RGB Only", "Combined RGB-T"] and not rgb_path:
        return None
    if mode in ["Thermal Only", "Combined RGB-T"] and not ir_path:
        return None

    caps = []
    if mode == "RGB Only":
        caps.append(cv2.VideoCapture(rgb_path))
    elif mode == "Thermal Only":
        caps.append(cv2.VideoCapture(ir_path))
    else:  
        caps.append(cv2.VideoCapture(rgb_path))
        caps.append(cv2.VideoCapture(ir_path))

    cap_ref = caps[0]
    width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_ref.get(cv2.CAP_PROP_FPS))

    out_width = width * 2 if mode == "Combined RGB-T" else width
    output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_width, height))

    while True:
        frames = []
        ret_flags = []
        
        for cap in caps:
            ret, frame = cap.read()
            ret_flags.append(ret)
            frames.append(frame)
            
        if not all(ret_flags):
            break

        # ---------------------------------------------------------
        # --- MODE 1: COMBINED RGB-T (6-Channel Fusion) ---
        # ---------------------------------------------------------
        if mode == "Combined RGB-T":
            rgb_frame = frames[0]
            ir_frame = frames[1]

            if rgb_frame.shape[:2] != ir_frame.shape[:2]:
                ir_frame = cv2.resize(ir_frame, (rgb_frame.shape[1], rgb_frame.shape[0]))

            six_channel_frame = np.concatenate((rgb_frame, ir_frame), axis=-1)

            # --- UPDATED: Pass conf_thresh into the model ---
            results = model_rgbt(six_channel_frame, conf=conf_thresh, verbose=False)

            rgb_plotted = rgb_frame.copy()
            ir_plotted = ir_frame.copy()
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = f'UAV {conf:.2f}'
                
                cv2.rectangle(rgb_plotted, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb_plotted, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                cv2.rectangle(ir_plotted, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(ir_plotted, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            final_frame = np.hstack((rgb_plotted, ir_plotted))
            
        # ---------------------------------------------------------
        # --- MODE 2: RGB ONLY ---
        # ---------------------------------------------------------
        elif mode == "RGB Only":
            # --- UPDATED: Pass conf_thresh into the model ---
            results = model_rgb(frames[0], conf=conf_thresh, verbose=False)
            final_frame = results[0].plot()
            
        # ---------------------------------------------------------
        # --- MODE 3: THERMAL ONLY ---
        # ---------------------------------------------------------
        elif mode == "Thermal Only":
            # --- UPDATED: Pass conf_thresh into the model ---
            results = model_ir(frames[0], conf=conf_thresh, verbose=False)
            final_frame = results[0].plot()

        out.write(final_frame)

    for cap in caps:
        cap.release()
    out.release()

    return output_video_path

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
            
            # --- NEW: Confidence Slider ---
            conf_slider = gr.Slider(
                minimum=0.01, 
                maximum=1.0, 
                value=0.25, 
                step=0.01, 
                label="Confidence Threshold",
                info="Higher values = stricter detections"
            )
            
            process_btn = gr.Button("Detect Drones", variant="primary")
            
        with gr.Column(scale=2):
            output_video = gr.Video(label="Detection Output")

    mode_selector.change(
        fn=update_inputs, 
        inputs=mode_selector, 
        outputs=[rgb_input, ir_input]
    )

    # --- UPDATED: Added conf_slider to the inputs array ---
    process_btn.click(
        fn=process_video, 
        inputs=[mode_selector, rgb_input, ir_input, conf_slider], 
        outputs=output_video
    )

if __name__ == "__main__":
    app.launch()
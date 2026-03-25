import csv
import gradio as gr
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Wrap your existing TensorRT models for SAHI (We'll do RGB as an example)
sahi_model_rgb = AutoDetectionModel.from_pretrained(
    model_type='yolov8', 
    model_path="runs/Anti-UAV/yolo11n-RGB-Only2/weights/best.engine",
    confidence_threshold=0.25,
    device="cuda:0"
)

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
model_rgb = YOLO("runs/Anti-UAV/yolo11n-RGB-Only3/weights/best.engine", task="detect")
model_ir = YOLO("runs/Anti-UAV/yolo11n-IR-Only2/weights/best.engine", task="detect")
model_rgbt = YOLO("runs/Anti-UAV/yolo11n-RGBRGB2/weights/best.engine", task="detect")
print("Engines loaded successfully!")

CSV_HEADER = ["frame", "class", "confidence", "x1", "y1", "x2", "y2", "width_px", "height_px", "area_px2"]

def write_detections(csv_writer, frame_idx, results, model):
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return
    names = model.names
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        w = x2 - x1
        h = y2 - y1
        csv_writer.writerow([
            frame_idx,
            names[cls],
            f"{conf:.4f}",
            f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
            f"{w:.1f}", f"{h:.1f}",
            f"{w * h:.1f}",
        ])

def process_video(mode, rgb_path, ir_path):
    if mode in ["RGB Only", "Combined RGB-T"] and not rgb_path:
        raise gr.Error("Please upload an RGB video.")
    if mode in ["Thermal Only", "Combined RGB-T"] and not ir_path:
        raise gr.Error("Please upload a Thermal (IR) video.")

    output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    csv_path = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    final_frame = None
    ui_frame = None
    ui_frame_count = 0

    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADER)

<<<<<<< HEAD
        # --- SINGLE MODE PROCESSING ---
        if mode == "RGB Only" or mode == "Thermal Only":
            video_path = rgb_path if mode == "RGB Only" else ir_path
            cap = cv2.VideoCapture(video_path)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
=======
        # --- INFERENCE ---
        if mode == "RGB Only":
            if use_sahi:
                # 🚀 LONG RANGE MODE: Slice the frame into 640x640 chunks
                result = get_sliced_prediction(
                    frames[0],
                    sahi_model_rgb,
                    slice_height=640,
                    slice_width=640,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2
                )
                # Convert SAHI result back to a plottable image array
                final_frame = result.image.copy()
                for object_prediction in result.object_prediction_list:
                    # Draw bounding boxes manually or use SAHI's export
                    bbox = object_prediction.bbox.to_xyxy()
                    cv2.rectangle(final_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            else:
                # 🏃 FAST MODE: Standard YOLO resize
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
>>>>>>> cdefc6bc3c2d4a54e1a2dd372e0e0a91db5b8959

            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            model = model_rgb if mode == "RGB Only" else model_ir
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=0.15, verbose=False)
                write_detections(csv_writer, frame_idx, results, model)
                frame_idx += 1
                final_frame = results[0].plot()
                out.write(final_frame)

                ui_frame_count += 1
                if ui_frame_count % 5 == 0:  # Throttle UI updates
                    ui_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    yield ui_frame, None, None

            cap.release()
            out.release()

        # --- COMBINED MODE PROCESSING (TIME-SYNCED) ---
        elif mode == "Combined RGB-T":
            cap_rgb = cv2.VideoCapture(rgb_path)
            cap_ir = cv2.VideoCapture(ir_path)

            width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_rgb = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0
            fps_ir = cap_ir.get(cv2.CAP_PROP_FPS) or 9.0

            out = cv2.VideoWriter(output_video_path, fourcc, fps_rgb, (width, height))

            # Read the first IR frame to start
            ret_ir, current_ir_frame = cap_ir.read()
            ir_frames_read = 1 if ret_ir else 0
            rgb_frames_read = 0

            while True:
                ret_rgb, frame_rgb = cap_rgb.read()
                if not ret_rgb:
                    break  # RGB video ended

                rgb_frames_read += 1
                current_time_sec = rgb_frames_read / fps_rgb

                # Fast-forward the IR video until its timestamp catches up to the RGB timestamp
                while (ir_frames_read / fps_ir) < current_time_sec:
                    ret_next_ir, next_ir_frame = cap_ir.read()
                    if not ret_next_ir:
                        break  # IR video ended
                    current_ir_frame = next_ir_frame
                    ir_frames_read += 1

                if current_ir_frame is None:
                    break

                # Ensure spatial dimensions match
                if frame_rgb.shape != current_ir_frame.shape:
                    current_ir_frame_resized = cv2.resize(current_ir_frame, (frame_rgb.shape[1], frame_rgb.shape[0]))
                else:
                    current_ir_frame_resized = current_ir_frame

                # Combine into 6 channels [H, W, 6]
                combined_input = np.concatenate((frame_rgb, current_ir_frame_resized), axis=-1)

                # Inference
                results = model_rgbt(combined_input, conf=0.15, verbose=False)
                write_detections(csv_writer, rgb_frames_read - 1, results, model_rgbt)
                final_frame = results[0].plot(img=frame_rgb.copy())

                out.write(final_frame)

                ui_frame_count += 1
                if ui_frame_count % 5 == 0:  # Throttle UI updates
                    ui_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    yield ui_frame, None, None

            cap_rgb.release()
            cap_ir.release()
            out.release()

    # Final yield to provide the downloadable files
    if ui_frame is not None:
        yield ui_frame, output_video_path, csv_path
    else:
        yield None, None, None

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
            
<<<<<<< HEAD
            rgb_input = gr.File(label="Upload RGB Video", file_types=["video"], visible=True)
            ir_input = gr.File(label="Upload Thermal (IR) Video", file_types=["video"], visible=False)
=======
            # Add a toggle for long-range detection
            sahi_toggle = gr.Checkbox(
                label="Enable High-Res Tiling (For distant drones)", 
                value=False,
                info="Slower, but prevents small objects from being downscaled."
            )

            rgb_input = gr.Video(label="RGB Video", visible=True)
            ir_input = gr.Video(label="Thermal (IR) Video", visible=False)
>>>>>>> cdefc6bc3c2d4a54e1a2dd372e0e0a91db5b8959
            
            process_btn = gr.Button("Detect Drones", variant="primary")
            
        with gr.Column(scale=2):
            live_feed = gr.Image(label="Live Inference Feed")
            final_output = gr.Video(label="Final Downloadable Video")
            csv_output = gr.File(label="Download Detections CSV")

    mode_selector.change(fn=update_inputs, inputs=mode_selector, outputs=[rgb_input, ir_input])
<<<<<<< HEAD
    process_btn.click(fn=process_video, inputs=[mode_selector, rgb_input, ir_input], outputs=[live_feed, final_output, csv_output])

=======
    process_btn.click(
            fn=process_video, 
            inputs=[mode_selector, rgb_input, ir_input, sahi_toggle], 
            outputs=[live_feed, final_output]
        )
    
>>>>>>> cdefc6bc3c2d4a54e1a2dd372e0e0a91db5b8959
if __name__ == "__main__":
    app.queue() # <-- ADD THIS LINE
    app.launch(server_name="0.0.0.0", server_port=7860)
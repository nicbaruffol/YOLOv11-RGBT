import cv2
import json
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = Path("Anti-UAV-RGBT")
OUTPUT_DIR = Path("Anti-UAV-YOLO")
SPLITS = ["train", "val", "test"]
MODALITIES = ["visible", "infrared"]
CLASS_ID = 0  # 0 represents 'UAV'

def convert_to_yolo_bbox(bbox, img_width, img_height):
    """Converts top-left (x, y, w, h) to normalized YOLO center (x, y, w, h)."""
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / img_width
    y_center = (y + h / 2.0) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [x_center, y_center, w_norm, h_norm]

def process_video(video_path, json_path, images_out_dir, labels_out_dir, prefix):
    # Ensure output directories exist
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load JSON annotations
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    # Standard Anti-UAV JSONs use 'exist' flags and 'gt_rect' for bounding boxes
    exist_flags = annotations.get('exist', [])
    bboxes = annotations.get('gt_rect', [])
    
    cap = cv2.VideoCapture(str(video_path))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_name = f"{prefix}_{frame_idx:06d}"
        img_out_path = images_out_dir / f"{frame_name}.jpg"
        txt_out_path = labels_out_dir / f"{frame_name}.txt"
        
        # Save the extracted frame
        cv2.imwrite(str(img_out_path), frame)
        
        # Determine if the UAV exists in this frame
        if frame_idx < len(exist_flags) and exist_flags[frame_idx] == 1:
            bbox = bboxes[frame_idx]
            yolo_bbox = convert_to_yolo_bbox(bbox, frame_width, frame_height)
            
            # Write the YOLO label file
            with open(txt_out_path, 'w') as f:
                f.write(f"{CLASS_ID} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
        else:
            # Save an empty text file for background frames (no target)
            open(txt_out_path, 'w').close()
            
        frame_idx += 1
        
    cap.release()
    print(f"Processed {frame_idx} frames from {video_path.name}")

# --- Main Execution Loop ---
for split in SPLITS:
    split_dir = SOURCE_DIR / split
    if not split_dir.exists():
        continue
        
    for seq_dir in split_dir.iterdir():
        if not seq_dir.is_dir():
            continue
            
        for modality in MODALITIES:
            vid_file = seq_dir / f"{modality}.mp4"
            json_file = seq_dir / f"{modality}.json"
            
            if vid_file.exists() and json_file.exists():
                # Route images and labels to their respective YOLO folders
                images_dst = OUTPUT_DIR / "images" / split / modality
                labels_dst = OUTPUT_DIR / "labels" / split / modality
                prefix = f"{seq_dir.name}_{modality}"
                
                process_video(vid_file, json_file, images_dst, labels_dst, prefix)

print("Dataset conversion to YOLO format is complete!")
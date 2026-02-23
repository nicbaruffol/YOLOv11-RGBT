import cv2
import glob
import os
from tqdm import tqdm

# Define path to your IR images
# Update this path to match your data.yaml 'train_ir' and 'val_ir' paths
ir_paths = [
    "/home/radar/Desktop/dataset/infrared/test",
    "/home/radar/Desktop/dataset/infrared/train",
    "/home/radar/Desktop/dataset/infrared/val"
]


def convert_to_grayscale(folder_path):
    print(f"Processing {folder_path}...")
    # Find all images
    images = glob.glob(os.path.join(folder_path, "*.*"))
    
    for img_path in tqdm(images):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Check if already 1-channel
        if len(img.shape) == 2:
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Overwrite the file
        cv2.imwrite(img_path, gray)

if __name__ == "__main__":
    for path in ir_paths:
        if os.path.exists(path):
            convert_to_grayscale(path)
        else:
            print(f"Path not found: {path}")

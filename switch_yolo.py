import shutil
from pathlib import Path

# --- Configuration ---
OLD_DIR = Path("Anti-UAV-YOLO") # The folder where your files currently are
NEW_DIR = Path("dataset")       # The new root folder you want

SPLITS = ["train", "val", "test"]
MODALITIES = ["visible", "infrared"]

moved_count = 0

for split in SPLITS:
    for modality in MODALITIES:
        # Define the old paths based on your first script
        old_images_dir = OLD_DIR / "images" / split / modality
        old_labels_dir = OLD_DIR / "labels" / split / modality
        
        # Define the new combined path: dataset/modality/split
        new_combined_dir = NEW_DIR / modality / split
        
        # Skip if there are no images for this split/modality combo
        if not old_images_dir.exists():
            continue
            
        # Create the new target directory structure
        new_combined_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate through all .jpg files in the old images folder
        for img_path in old_images_dir.glob("*.jpg"):
            # Find the corresponding .txt file in the old labels folder
            txt_path = old_labels_dir / f"{img_path.stem}.txt"
            
            # Define where they should go
            new_img_path = new_combined_dir / img_path.name
            new_txt_path = new_combined_dir / txt_path.name
            
            # Move the image
            shutil.move(str(img_path), str(new_img_path))
            
            # Move the label (if it exists)
            if txt_path.exists():
                shutil.move(str(txt_path), str(new_txt_path))
            
            moved_count += 1

print(f"Success! Reorganized {moved_count} image/label pairs into the new structure.")
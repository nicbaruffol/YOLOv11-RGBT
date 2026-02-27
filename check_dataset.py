import cv2
import os
from pathlib import Path

def check_rgbt_dataset(rgb_dir, ir_dir):
    rgb_path = Path(rgb_dir)
    ir_path = Path(ir_dir)

    if not rgb_path.exists() or not ir_path.exists():
        print("Error: One or both directories do not exist. Please check your paths.")
        return

    mismatched_pairs = []
    missing_ir_files = []

    # Get all image files from the RGB directory
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    rgb_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith(valid_extensions)]

    print(f"Scanning {len(rgb_files)} image pairs...")

    for filename in rgb_files:
        rgb_file_path = rgb_path / filename
        
        # --- THE FIX: Translate the filename from 'visible' to 'infrared' ---
        ir_filename = filename.replace('visible', 'infrared')
        ir_file_path = ir_path / ir_filename

        # Check if the matching IR file exists
        if not ir_file_path.exists():
            missing_ir_files.append((filename, ir_filename))
            continue

        # Load images using IMREAD_UNCHANGED to capture exact bit-depth (e.g., 16-bit thermal)
        rgb_img = cv2.imread(str(rgb_file_path), cv2.IMREAD_UNCHANGED)
        ir_img = cv2.imread(str(ir_file_path), cv2.IMREAD_UNCHANGED)

        if rgb_img is None or ir_img is None:
            print(f"Warning: Could not read {filename} or {ir_filename}. Corrupt file?")
            continue

        # Check dimensions (height, width)
        rgb_shape = rgb_img.shape[:2]
        ir_shape = ir_img.shape[:2]
        
        # Check data type (e.g., uint8 vs uint16)
        rgb_dtype = rgb_img.dtype
        ir_dtype = ir_img.dtype

        if rgb_shape != ir_shape or rgb_dtype != ir_dtype:
            mismatched_pairs.append({
                'rgb_name': filename,
                'ir_name': ir_filename,
                'rgb_shape': rgb_shape,
                'ir_shape': ir_shape,
                'rgb_dtype': rgb_dtype,
                'ir_dtype': ir_dtype
            })

    # --- Print Diagnostic Report ---
    print("\n" + "="*30)
    print("      DIAGNOSTIC REPORT")
    print("="*30)
    
    if missing_ir_files:
        print(f"⚠️ Found {len(missing_ir_files)} RGB images with no matching IR file.")
        print(f"   Example missing match: Looked for '{missing_ir_files[0][1]}' (based on '{missing_ir_files[0][0]}')")
    
    if not mismatched_pairs and not missing_ir_files:
        print("✅ SUCCESS: All paired images perfectly match in resolution and data type!")
    elif not mismatched_pairs:
        print("✅ The paired images that WERE found perfectly match in resolution and data type!")
    else:
        print(f"❌ Found {len(mismatched_pairs)} mismatched pairs out of {len(rgb_files)}.")
        print("-" * 30)
        
        # Print up to the first 10 mismatched files to avoid flooding the console
        for pair in mismatched_pairs[:10]:
            print(f"File: {pair['rgb_name']}")
            if pair['rgb_shape'] != pair['ir_shape']:
                print(f"  > Size Mismatch: RGB {pair['rgb_shape']} vs IR {pair['ir_shape']}")
            if pair['rgb_dtype'] != pair['ir_dtype']:
                print(f"  > Type Mismatch: RGB {pair['rgb_dtype']} vs IR {pair['ir_dtype']}")
        
        if len(mismatched_pairs) > 10:
            print(f"\n... and {len(mismatched_pairs) - 10} more.")

if __name__ == "__main__":
    # REPLACE THESE with the actual paths to your training images
    RGB_FOLDER = "/cluster/scratch/nbaruffol/dataset/visible/test"
    IR_FOLDER  = "/cluster/scratch/nbaruffol/dataset/infrared/test"
    
    check_rgbt_dataset(RGB_FOLDER, IR_FOLDER)
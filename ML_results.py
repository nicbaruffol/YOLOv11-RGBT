import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_yolo_progress(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_path}")
        return

    # 1. FIX: Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Use index for epochs if 'epoch' column doesn't exist
    if 'epoch' not in df.columns:
        df['epoch'] = df.index + 1

    # Create a figure with 3 columns (Box Loss, Class/DFL Loss, Metrics)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Box Loss (Regression) ---
    # Looks for 'train/box_loss' and 'val/box_loss'
    if 'train/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2)
    if 'val/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2)
    
    ax1.set_title('Box Loss (Localization)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)
    if ax1.get_legend_handles_labels()[0]: # Only show legend if data exists
        ax1.legend()

    # --- Plot 2: Class or DFL Loss ---
    # Looks for 'train/cls_loss' or 'train/dfl_loss'
    if 'train/cls_loss' in df.columns:
        ax2.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', linewidth=2)
    if 'val/cls_loss' in df.columns:
        ax2.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', linewidth=2)
    
    # Fallback/Addition for DFL loss if it exists
    if 'train/dfl_loss' in df.columns:
         ax2.plot(df['epoch'], df['train/dfl_loss'], label='Train DFL Loss', linestyle=':', alpha=0.7)

    ax2.set_title('Class/DFL Loss')
    ax2.set_xlabel('Epoch')
    ax2.grid(True, linestyle='--', alpha=0.5)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend()

    # --- Plot 3: Metrics (mAP) ---
    # YOLO usually tracks metrics/mAP50(B) and metrics/mAP50-95(B)
    map50_col = [c for c in df.columns if 'mAP50' in c and '95' not in c]
    map5095_col = [c for c in df.columns if 'mAP50-95' in c]

    if map50_col:
        ax3.plot(df['epoch'], df[map50_col[0]], label='mAP@50', linewidth=2, color='green')
    if map5095_col:
        ax3.plot(df['epoch'], df[map5095_col[0]], label='mAP@50-95', linewidth=2, color='orange')

    ax3.set_title('Performance Metrics (mAP)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.grid(True, linestyle='--', alpha=0.5)
    if ax3.get_legend_handles_labels()[0]:
        ax3.legend()

    plt.tight_layout()
    
    # Save automatically so you don't have to rely on X11 forwarding
    output_filename = csv_path.replace('.csv', '_plot.png')
    plt.savefig(output_filename, dpi=150)
    print(f"Plot saved to: {output_filename}")
    
    # Only try to show if a display is available (prevents errors on headless servers)
    try:
        if sys.platform != 'linux' or 'DISPLAY' in sys.modules:
             plt.show()
    except:
        pass

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
    plot_yolo_progress(file_path)
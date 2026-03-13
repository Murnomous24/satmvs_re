import argparse
import os
import sys
import numpy as np
import cv2
import re
from tqdm import tqdm

# Add workspace root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---- Import or Define Loading Functions ----

try:
    from dataset.data_io import load_pfm, load_rpc_as_array
except ImportError:
    # Standalone fallback for load_pfm
    def load_pfm(file_name):
        file = open(file_name, 'rb')
        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().decode('utf-8').rstrip())
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data

    # Standalone fallback for load_rpc_as_array
    # Parses height range from .rpc file: h_min = height_off - height_scale, h_max = height_off + height_scale
    def load_rpc_as_array(file_name):
        with open(file_name, 'r') as f:
            lines = f.read().splitlines()
        data = np.array([line.split(' ')[1] for line in lines], dtype=np.float64)
        h_min = data[4] - data[9]
        h_max = data[4] + data[9]
        return data, h_min, h_max

# Try importing GDAL, but don't fail if missing
try:
    from osgeo import gdal
except ImportError:
    gdal = None
    print("Warning: GDAL not found, TIFF support will rely on OpenCV.")

def load_depth(path):
    """
    Load depth map from path. 
    Supports .pfm, .tif, .tiff, .png, .npy
    Returns: numpy array of shape (H, W) or None if failed.
    """
    if not os.path.exists(path):
        return None

    if path.endswith('.pfm'):
        return load_pfm(path)
    
    elif path.endswith(('.tif', '.tiff')):
        # Try GDAL first for TIF as it handles multi-band/bitdepth robustly
        if gdal is not None:
            ds = gdal.Open(path)
            if ds is not None:
                data = ds.ReadAsArray()
                # Handle shapes like (1, H, W) or (H, W, 1) to return (H, W)
                if data.ndim == 3:
                    if data.shape[0] == 1:
                        data = data[0, :, :]
                    elif data.shape[2] == 1: 
                        data = data[:, :, 0]
                return data
        
        # Fallback to OpenCV
        # cv2.imread usually loads as uint8 or uint16 if flag is passed, but float tiff might be tricky
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return img
        
    elif path.endswith('.npy'):
        return np.load(path)
        
    elif path.endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return img
        
    return None


def map_error_piecewise(diff):
    """
    Piecewise map absolute depth error (meter) into [0, 255].
    Intervals are aligned with metrics thresholds: 1.0m, 2.5m, 7.5m.
    """
    mapped = np.zeros_like(diff, dtype=np.float32)

    # 0 ~ 1.0m -> 0 ~ 96 (Blue -> Cyan/Green)
    m1 = diff < 1.0
    mapped[m1] = (diff[m1] / 1.0) * 96.0

    # 1.0 ~ 2.5m -> 96 ~ 160 (Green/Cyan -> Yellow)
    m2 = (diff >= 1.0) & (diff < 2.5)
    mapped[m2] = 96.0 + ((diff[m2] - 1.0) / 1.5) * (160.0 - 96.0)

    # 2.5 ~ 7.5m -> 160 ~ 224 (Yellow -> Red)
    m3 = (diff >= 2.5) & (diff < 7.5)
    mapped[m3] = 160.0 + ((diff[m3] - 2.5) / 5.0) * (224.0 - 160.0)

    # >= 7.5m -> 224 ~ 255 (Deep Red)
    m4 = diff >= 7.5
    mapped[m4] = np.clip(224.0 + ((diff[m4] - 7.5) / 2.5) * 31.0, 224.0, 255.0)

    return mapped.astype(np.uint8)


def colorize_error_map(diff, valid_mask):
    """
    Build piecewise colorized error map.
    Invalid pixels are painted black.
    """
    mapped = map_error_piecewise(diff)
    diff_color = cv2.applyColorMap(mapped, cv2.COLORMAP_JET)
    diff_color[~valid_mask] = 0
    return diff_color

def visualize_error(gt_dir, pred_dir, rpc_dir=None):
    """
    Visualize absolute depth error map.

    Args:
        gt_dir:   Ground Truth directory (source of file list).
        pred_dir: Prediction directory.
        rpc_dir:  Optional. Directory containing .rpc files with the same base
                  names as the GT files. When provided, the valid mask is
                  restricted to pixels whose GT depth lies within
                  [height_off - height_scale, height_off + height_scale] as
                  defined in the .rpc file — identical to the training mask
                  computed in dataset_rpc.py.
    """
    error_dir = os.path.join(pred_dir, "error")
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    print(f"GT Directory:            {gt_dir}")
    print(f"Pred Directory:          {pred_dir}")
    print(f"Output Error Directory:  {error_dir}")
    print("Color Mapping Scheme     : Piecewise mapping aligned with train metrics")
    print("                           [0, 1.0m)   -> Blue/Cyan        (high accuracy)")
    print("                           [1.0, 2.5m) -> Green/Yellow     (moderate error)")
    print("                           [2.5, 7.5m) -> Orange/Red       (large error)")
    print("                           [>= 7.5m]   -> Deep Red         (extreme error)")
    print("                           Invalid     -> Black")
    if rpc_dir is not None:
        print(f"RPC Directory:           {rpc_dir}  (height-range mask enabled)")
    else:
        print("RPC Directory:           not provided (height-range mask disabled)")

    # 1. Collect GT files
    gt_files = []
    # Extension filter
    valid_ext = ('.pfm', '.tif', '.tiff', '.png', '.npy')
    
    for root, dirs, files in os.walk(gt_dir):
        for file in files:
            if file.lower().endswith(valid_ext):
                gt_path = os.path.join(root, file)
                gt_files.append(gt_path)

    print(f"Found {len(gt_files)} ground truth files.")
    
    count = 0
    summary_valid_pixels = 0
    summary_bins = {
        "lt_1m": 0,
        "1m_2p5m": 0,
        "2p5m_7p5m": 0,
        "ge_7p5m": 0
    }
    # 2. Iterate and process
    for gt_path in tqdm(gt_files, desc="Processing"):
        # Get relative path to maintain structure
        rel_path = os.path.relpath(gt_path, gt_dir)
        base_name, ext = os.path.splitext(rel_path)
        
        # Find corresponding prediction
        # Candidate paths relative to pred_dir
        # Check original extension first, then likely substitutes like .pfm
        
        # Construct candidate absolute paths
        candidates = [
            os.path.join(pred_dir, rel_path),              # Exact match (e.g. both .tif)
            os.path.join(pred_dir, base_name + ".pfm"),    # Common for MVS output
            os.path.join(pred_dir, base_name + ".tif"),
            os.path.join(pred_dir, base_name + ".png"),
            os.path.join(pred_dir, base_name + ".npy")
        ]
        
        pred_path = None
        for cand in candidates:
            if os.path.exists(cand):
                pred_path = cand
                break
        
        if pred_path is None:
            # Prediction missing for this GT
            # print(f"Missing prediction for {rel_path}")
            continue
            
        # Load Data
        depth_gt = load_depth(gt_path)
        depth_est = load_depth(pred_path)
        
        if depth_gt is None or depth_est is None:
            continue
            
        # Check dimensions
        if depth_gt.shape != depth_est.shape:
            # print(f"Shape mismatch: {rel_path} GT{depth_gt.shape} vs Pred{depth_est.shape}")
            continue
            
        # Valid Mask Construction
        # Step 1 – basic validity (non-zero, finite)
        mask = (depth_gt > 0) & np.isfinite(depth_gt) & (depth_est > 0) & np.isfinite(depth_est)

        # Step 2 – height-range mask, identical to training mask in dataset_rpc.py:
        #   mask = (height_image >= height_min) & (height_image <= height_max)
        #   where height_min/max come from the .rpc file (height_off ± height_scale),
        #   NOT from dsm_info/ depth range files.
        if rpc_dir is not None:
            rpc_path = os.path.join(rpc_dir, base_name + ".rpc")
            if os.path.exists(rpc_path):
                try:
                    _, h_min, h_max = load_rpc_as_array(rpc_path)
                    mask = mask & (depth_gt >= h_min) & (depth_gt <= h_max)
                except Exception:
                    pass  # If RPC parsing fails, keep the basic mask
        
        # If no valid pixels, skip
        if not np.any(mask):
            continue
            
        # Calculate absolute depth error
        diff = np.abs(depth_est - depth_gt)

        # Update summary bins over valid pixels only.
        valid_diff = diff[mask]
        summary_valid_pixels += valid_diff.size
        summary_bins["lt_1m"] += int(np.sum(valid_diff < 1.0))
        summary_bins["1m_2p5m"] += int(np.sum((valid_diff >= 1.0) & (valid_diff < 2.5)))
        summary_bins["2p5m_7p5m"] += int(np.sum((valid_diff >= 2.5) & (valid_diff < 7.5)))
        summary_bins["ge_7p5m"] += int(np.sum(valid_diff >= 7.5))

        # Apply piecewise visualization and paint invalid pixels as black.
        diff_color = colorize_error_map(diff, mask)
        
        # Save output
        # Replicate subdirectory structure in error_dir
        # Always save as .png for visualization
        output_rel_path = base_name + ".png"
        output_path = os.path.join(error_dir, output_rel_path)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, diff_color)
        count += 1
        
    print(f"Processed {count} files. Results saved to {error_dir}")

    if summary_valid_pixels > 0:
        p_lt_1m = 100.0 * summary_bins["lt_1m"] / summary_valid_pixels
        p_1m_2p5m = 100.0 * summary_bins["1m_2p5m"] / summary_valid_pixels
        p_2p5m_7p5m = 100.0 * summary_bins["2p5m_7p5m"] / summary_valid_pixels
        p_ge_7p5m = 100.0 * summary_bins["ge_7p5m"] / summary_valid_pixels

        print("Error-bin summary on valid pixels:")
        print(f"  < 1.0m      : {summary_bins['lt_1m']} ({p_lt_1m:.2f}%)")
        print(f"  [1.0,2.5)m  : {summary_bins['1m_2p5m']} ({p_1m_2p5m:.2f}%)")
        print(f"  [2.5,7.5)m  : {summary_bins['2p5m_7p5m']} ({p_2p5m_7p5m:.2f}%)")
        print(f"  >= 7.5m     : {summary_bins['ge_7p5m']} ({p_ge_7p5m:.2f}%)")
    else:
        print("No valid pixels found across all processed pairs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Depth Error (Heatmap)")
    parser.add_argument("--gt_dir",  type=str, required=True,  help="Ground Truth root directory")
    parser.add_argument("--pred_dir", type=str, required=True,  help="Prediction root directory")
    parser.add_argument("--rpc_dir",  type=str, default=None,
                        help="Directory containing .rpc files (same base names as GT files). "
                             "Enables height-range mask identical to training (dataset_rpc.py).")

    args = parser.parse_args()

    visualize_error(args.gt_dir, args.pred_dir, args.rpc_dir)

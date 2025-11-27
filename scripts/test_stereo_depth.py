# live_realsense_flowformer_d435i_save.py

import pyrealsense2 as rs
import torch
import numpy as np
import cv2
import os
from types import SimpleNamespace
from Stereo_Depth.stereo_depth_cov import FlowFormerCovDepth

# -----------------------------
# 1. Model configuration
# -----------------------------
config = SimpleNamespace(
    weight="/work/models/MACVO_FrontendCov.pth",  # Replace with your actual path
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    enc_dtype="fp16",
    dec_dtype="fp16"
)

device = config.device
model = FlowFormerCovDepth(config)  # FlowFormerCovDepth does NOT have .parameters()

# -----------------------------
# 2. RealSense setup (D435i)
# -----------------------------
pipeline = rs.pipeline()
rs_config = rs.config()

width, height, fps = 640, 480, 30
rs_config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)  # left
rs_config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)  # right

pipeline.start(rs_config)

# -----------------------------
# 3. Camera calibration (D435i)
# -----------------------------
baseline = 0.087  # meters
fx = 615.0        # pixels

# -----------------------------
# 4. Output directory
# -----------------------------
output_dir = "/work/output"
os.makedirs(output_dir, exist_ok=True)
frame_count = 0

# -----------------------------
# 5. Live loop (saving frames)
# -----------------------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)
        if not left_frame or not right_frame:
            continue

        # Convert to numpy arrays
        imgL = np.asanyarray(left_frame.get_data())
        imgR = np.asanyarray(right_frame.get_data())

        # Convert grayscale IR to 3-channel RGB for the model
        imgL_rgb = np.stack([imgL]*3, axis=2)
        imgR_rgb = np.stack([imgR]*3, axis=2)

        # To torch tensor [B,C,H,W]
        imageL = torch.from_numpy(imgL_rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
        imageR = torch.from_numpy(imgR_rgb).permute(2,0,1).unsqueeze(0).float() / 255.0

        # Move tensors to device and FP16 if required
        imageL = imageL.to(device)
        imageR = imageR.to(device)
        if config.enc_dtype == "fp16":
            imageL = imageL.half()
            imageR = imageR.half()

        # -----------------------------
        # 6. Run FlowFormerCovDepth
        # -----------------------------
        with torch.no_grad():
            output = model.estimate(imageL, imageR, baseline=baseline, fx=fx)

        # -----------------------------
        # 7. Process output for saving
        # -----------------------------
        depth = output.depth[0,0].cpu().numpy()
        disp = output.disparity[0,0].cpu().numpy()

        # Normalize to 0-255 for saving
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply color map for depth
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Overlay depth on IR image
        overlay = cv2.addWeighted(cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR), 0.6, depth_color, 0.4, 0)

        # -----------------------------
        # 8. Save output files
        # -----------------------------
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:05d}_left.png"), imgL)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:05d}_depth.png"), depth_vis)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:05d}_disparity.png"), disp_vis)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:05d}_overlay.png"), overlay)

        frame_count += 1

        # Optional: stop after N frames
        # if frame_count >= 100:
        #     break

finally:
    pipeline.stop()
    print(f"Saved {frame_count} frames to {output_dir}")

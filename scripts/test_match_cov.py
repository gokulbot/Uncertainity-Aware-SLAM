# realsense_flowformer_matcher_capture.py
#
# 1. Captures a single stream (color) from Intel RealSense (e.g. D435i)
# 2. Runs FlowFormerCovMatcher between consecutive frames (t, t+1)
# 3. Saves inputs and outputs (flow, cov, overlays, flow arrows) to disk

import os
from datetime import datetime

import pyrealsense2 as rs
import torch
import numpy as np
import cv2
from types import SimpleNamespace

from Matching.matcher import FlowFormerCovMatcher  # adjust import if needed


def flow_to_color(flow):
    """
    Convert optical flow (HxWx2, float) to a BGR color image using HSV.
    """
    h, w = flow.shape[:2]
    fx = flow[..., 0]
    fy = flow[..., 1]

    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    # Normalize magnitude to [0, 1]
    mag_norm = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[..., 0] = ang / 2.0          # OpenCV HSV hue range [0,180]
    hsv[..., 1] = 1.0                # full saturation
    hsv[..., 2] = mag_norm           # value = normalized magnitude

    hsv_8u = (hsv * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)
    return bgr


def draw_flow_arrows(image, flow, step=16, scale=1.0):
    """
    Draw arrows representing optical flow on top of an image.

    image: HxWx3 uint8 BGR image (base frame)
    flow:  HxWx2 float (u,v) optical flow
    step:  sampling stride in pixels (bigger = fewer arrows)
    scale: scale factor for arrow length
    """
    h, w = image.shape[:2]
    vis = image.copy()

    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            end_x = int(x + fx * scale)
            end_y = int(y + fy * scale)

            # Only draw if the motion is non-trivial
            if abs(fx) < 0.2 and abs(fy) < 0.2:
                continue

            cv2.arrowedLine(
                vis,
                (x, y),
                (end_x, end_y),
                (0, 255, 0),  # green arrows
                1,
                tipLength=0.3,
            )

    return vis


def main():
    # -----------------------------
    # 1. Output folder
    # -----------------------------
    base_output_dir = "/work/realsense_matcher_cov"  # change if you like
    os.makedirs(base_output_dir, exist_ok=True)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Saving RealSense + FlowFormerMatcher outputs to: {output_dir}")

    # -----------------------------
    # 2. Matcher configuration
    # -----------------------------
    config = SimpleNamespace(
        weight="/work/models/MACVO_FrontendCov.pth",  # <-- update if needed
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        enc_dtype="fp16",
        dec_dtype="fp16",
    )

    device = config.device
    print(f"[INFO] Using device: {device}")

    # Validate config if your class supports it
    if hasattr(FlowFormerCovMatcher, "is_valid_config"):
        FlowFormerCovMatcher.is_valid_config(config)

    matcher = FlowFormerCovMatcher(config)

    num_params = sum(p.numel() for p in matcher.model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the model: {num_params}")

    # -----------------------------
    # 3. RealSense setup (single stream)
    # -----------------------------
    pipeline = rs.pipeline()
    rs_config = rs.config()

    width, height, fps = 640, 480, 30

    # ---- OPTION A: Color stream ----
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # ---- OPTION B: Single IR stream (uncomment if you prefer IR) ----
    # rs_config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)

    pipeline.start(rs_config)
    print("[INFO] RealSense pipeline started. Press Ctrl+C to stop.")

    # -----------------------------
    # 4. Matching between consecutive frames
    # -----------------------------
    frame_count = 0
    pair_count = 0
    max_pairs = None  # set e.g. 100 to stop after 100 pairs

    prev_frame_np = None  # HxWx3 for color, or HxW for grayscale

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # ---- For color ----
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame_np = np.asanyarray(color_frame.get_data())  # HxWx3, uint8 (BGR)

            # ---- If using IR instead, use this: ----
            # ir_frame = frames.get_infrared_frame(1)
            # if not ir_frame:
            #     continue
            # frame_np = np.asanyarray(ir_frame.get_data())  # HxW, uint8 (grayscale)
            # frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)  # convert to 3-channel for consistency

            # Save raw frame
            frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
            cv2.imwrite(frame_path, frame_np)
            frame_count += 1

            if prev_frame_np is None:
                prev_frame_np = frame_np
                continue

            # -----------------------------
            # 4.1 Prepare tensors for matcher
            # -----------------------------
            # Matcher expects [B, C, H, W], normalize to [0,1], RGB or BGR is fine if consistent

            # Convert from BGR (OpenCV) to RGB if your matcher expects RGB:
            prev_rgb = cv2.cvtColor(prev_frame_np, cv2.COLOR_BGR2RGB)
            curr_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

            img1 = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img2 = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            img1 = img1.to(device)
            img2 = img2.to(device)
            if config.enc_dtype == "fp16":
                img1 = img1.half()
                img2 = img2.half()

            # -----------------------------
            # 4.2 Run FlowFormerCovMatcher
            # -----------------------------
            with torch.no_grad():
                output = matcher.estimate(img1, img2)

            flow = output.flow[0].detach().cpu().numpy().transpose(1, 2, 0)  # HxWx2
            cov = output.cov[0].detach().cpu().numpy()  # CxHxW or HxW

            # -----------------------------
            # 4.3 Process outputs for saving
            # -----------------------------
            # Flow color visualization
            flow_color = flow_to_color(flow)  # HxWx3, BGR

            # Draw sparse arrows
            arrows_img = draw_flow_arrows(frame_np, flow, step=16, scale=1.0)

            # Handle covariance visualization
            if cov.ndim == 3:
                cov_first = cov[0]  # HxW
            else:
                cov_first = cov

            cov_vis = cv2.normalize(cov_first, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cov_color = cv2.applyColorMap(cov_vis, cv2.COLORMAP_PLASMA)

            # Overlay flow and cov on the current frame
            overlay_flow = cv2.addWeighted(frame_np, 0.6, flow_color, 0.4, 0)
            overlay_cov = cv2.addWeighted(frame_np, 0.6, cov_color, 0.4, 0)

            # -----------------------------
            # 4.4 Save outputs for this pair
            # -----------------------------
            flow_color_path = os.path.join(output_dir, f"pair_{pair_count:05d}_flowcolor.png")
            flow_arrows_path = os.path.join(output_dir, f"pair_{pair_count:05d}_flow_arrows.png")
            cov_vis_path = os.path.join(output_dir, f"pair_{pair_count:05d}_cov.png")
            overlay_flow_path = os.path.join(output_dir, f"pair_{pair_count:05d}_overlay_flow.png")
            overlay_cov_path = os.path.join(output_dir, f"pair_{pair_count:05d}_overlay_cov.png")

            cv2.imwrite(flow_color_path, flow_color)
            cv2.imwrite(flow_arrows_path, arrows_img)
            cv2.imwrite(cov_vis_path, cov_vis)
            cv2.imwrite(overlay_flow_path, overlay_flow)
            cv2.imwrite(overlay_cov_path, overlay_cov)

            pair_count += 1
            prev_frame_np = frame_np  # shift window

            # Optional stopping condition
            if max_pairs is not None and pair_count >= max_pairs:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")

    finally:
        pipeline.stop()
        print(f"[INFO] Saved {frame_count} frames and {pair_count} matcher pairs to {output_dir}")


if __name__ == "__main__":
    main()

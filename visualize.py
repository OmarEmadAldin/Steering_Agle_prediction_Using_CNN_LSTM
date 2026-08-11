"""
visualize.py
Draws a line (like a steering needle) on each frame showing the predicted
steering angle alongside the ground-truth angle, and saves the result as
an annotated MP4. Just run it — no command-line args needed.

Edit the CONFIG block below to point at your files.
"""

import os
import math
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from model import CNNLSTMSteering
from dataset import IMAGE_COL, ANGLE_COL

# ----------------------------- CONFIG ---------------------------------
MODEL_PATH = "steering_model_best.pt"
BACKBONE = "resnet18"
HIDDEN_SIZE = 128
SEQ_LEN = 5
IMG_SIZE = 224

CSV_PATH = "driving_log.csv"
IMG_DIR = "/home/omar/Steering_Agle_prediction/driving_dataset/val/images"
OUT_PATH = "annotated.mp4"
FPS = 20
SHOW_GROUND_TRUTH = True
# ------------------------------------------------------------------------

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def load_model(model_path, backbone, hidden_size, device):
    model = CNNLSTMSteering(backbone=backbone, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_frame(img_bgr, img_size=224):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(MEAN.tolist(), STD.tolist()),
    ])
    return transform(pil_img)


def draw_steering_line(frame_bgr, angle_rad, color=(0, 0, 255), label=None,
                        origin_ratio=0.9, length_ratio=0.35, angle_units="radians"):
    """
    Draws a line from the bottom-center of the image, rotated by `angle_rad`
    from vertical, representing the steering direction.
    Positive angle = turn right, negative = turn left (adjust sign if your
    dataset's convention is flipped).
    """
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    if angle_units == "degrees":
        angle_rad = math.radians(angle_rad)

    origin = (w // 2, int(h * origin_ratio))
    length = int(h * length_ratio)

    end_x = int(origin[0] + length * math.sin(angle_rad))
    end_y = int(origin[1] - length * math.cos(angle_rad))

    cv2.line(out, origin, (end_x, end_y), color, thickness=4, lineType=cv2.LINE_AA)
    cv2.circle(out, origin, 6, color, -1)

    if label:
        cv2.putText(out, label, (10, 30 if color == (0, 0, 255) else 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return out


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def render_video(model, csv_path, img_dir, out_path, seq_len, img_size,
                  device, fps, show_ground_truth):
    # Loop over whatever image files actually exist in img_dir, rather than
    # trusting every filename listed in the CSV (some may be missing/renamed).
    image_files = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTENSIONS)
    )
    if not image_files:
        print(f"No image files found in {img_dir}")
        return

    # Build a filename -> ground-truth lookup from the CSV, if available.
    gt_lookup = {}
    if show_ground_truth and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if IMAGE_COL in df.columns and ANGLE_COL in df.columns:
            for _, row in df.iterrows():
                gt_lookup[os.path.basename(str(row[IMAGE_COL]))] = float(row[ANGLE_COL])

    frames_buffer = []
    writer = None
    skipped = 0
    written = 0

    progress = tqdm(image_files, desc="Rendering", unit="frame")
    for fname in progress:
        img_path = os.path.join(img_dir, fname)
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise ValueError("cv2.imread returned None")

            tensor = preprocess_frame(img_bgr, img_size)
            frames_buffer.append(tensor)
            if len(frames_buffer) > seq_len:
                frames_buffer.pop(0)

            if len(frames_buffer) < seq_len:
                seq = torch.stack([frames_buffer[0]] * (seq_len - len(frames_buffer)) + frames_buffer, dim=0)
            else:
                seq = torch.stack(frames_buffer, dim=0)

            seq = seq.unsqueeze(0).to(device)  # [1, seq_len, C, H, W]
            with torch.no_grad():
                pred = model(seq).item()

            annotated = draw_steering_line(img_bgr, pred, color=(0, 0, 255), label=f"pred: {pred:.3f}")

            if fname in gt_lookup:
                gt = gt_lookup[fname]
                annotated = draw_steering_line(annotated, gt, color=(0, 255, 0), label=f"gt:   {gt:.3f}")

            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            writer.write(annotated)
            written += 1

        except Exception as e:
            skipped += 1
            progress.write(f"  [skip] {fname}: {e}")
            continue

        progress.set_postfix(written=written, skipped=skipped)

    if writer is not None:
        writer.release()
        print(f"Saved annotated video to {out_path} ({written} frames written, {skipped} skipped)")
    else:
        print("No frames were written — all images failed to load.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, BACKBONE, HIDDEN_SIZE, device)
    render_video(model, CSV_PATH, IMG_DIR, OUT_PATH, SEQ_LEN, IMG_SIZE,
                 device, FPS, SHOW_GROUND_TRUTH)

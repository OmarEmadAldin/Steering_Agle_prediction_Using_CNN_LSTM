"""
dataset.py
Builds sliding-window sequences of frames for CNN+LSTM steering prediction.

Expected CSV format (edit COLUMN NAMES below if yours differ):
    image_path, steering_angle
    center_0001.jpg, -0.023
    center_0002.jpg, -0.019
    ...
Images should be listed in temporal (recorded) order.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# ---- EDIT THESE if your CSV uses different column names ----
IMAGE_COL = "image_path"
ANGLE_COL = "steering_angle"
# --------------------------------------------------------------


class SteeringSequenceDataset(Dataset):
    """
    Returns (sequence_of_frames, angle_at_last_frame)
    sequence_of_frames: Tensor [seq_len, C, H, W]
    """

    def __init__(self, csv_path, img_dir, seq_len=5, img_size=224,
                 train=True, angle_scale=1.0):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.angle_scale = angle_scale  # multiply raw angle if it's normalized differently

        if IMAGE_COL not in self.df.columns or ANGLE_COL not in self.df.columns:
            raise ValueError(
                f"CSV must contain columns '{IMAGE_COL}' and '{ANGLE_COL}'. "
                f"Found: {list(self.df.columns)}. Edit IMAGE_COL/ANGLE_COL in dataset.py."
            )

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if train:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

    def __len__(self):
        return max(0, len(self.df) - self.seq_len + 1)

    def __getitem__(self, idx):
        rows = self.df.iloc[idx: idx + self.seq_len]
        frames = []
        for _, row in rows.iterrows():
            img_path = os.path.join(self.img_dir, str(row[IMAGE_COL]))
            img = Image.open(img_path).convert("RGB")
            frames.append(self.transform(img))
        seq = torch.stack(frames, dim=0)  # [seq_len, C, H, W]

        angle = float(rows.iloc[-1][ANGLE_COL]) * self.angle_scale
        return seq, torch.tensor(angle, dtype=torch.float32)


def make_train_val_split(csv_path, val_fraction=0.15):
    """
    Splits by TIME (not randomly!) — critical for driving data, since random
    shuffling of a temporal sequence leaks near-identical adjacent frames
    across train/val and gives a falsely good validation score.
    """
    df = pd.read_csv(csv_path)
    split_idx = int(len(df) * (1 - val_fraction))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df

"""
model.py
CNNEncoder (ResNet18/MobileNetV2, pretrained, frozen) + LSTM (from scratch)
+ regression head, for predicting a single steering angle from a sequence
of frames.

Reconstructed to match the architecture described in README.md:
    Frame sequence -> CNN backbone (frozen ImageNet weights)
                    -> per-frame feature vectors
                    -> LSTM (1 layer, trained from scratch)
                    -> FC head -> predicted steering angle (last frame)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """Wraps a pretrained CNN backbone and returns a flat feature vector per frame."""

    def __init__(self, backbone="resnet18", freeze_backbone=True, unfreeze_last_block=False):
        super().__init__()

        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_dim = net.fc.in_features
            # Drop the final classification layer; keep everything up to global pooling.
            self.features = nn.Sequential(*list(net.children())[:-1])
            last_block_name = "layer4"
        elif backbone == "mobilenet_v2":
            net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.feature_dim = net.last_channel
            self.features = nn.Sequential(
                net.features,
                nn.AdaptiveAvgPool2d(1),
            )
            last_block_name = None  # unfreeze handled differently below
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone_name = backbone

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

            if unfreeze_last_block:
                if backbone == "resnet18":
                    for name, module in net.named_children():
                        if name == last_block_name:
                            for p in module.parameters():
                                p.requires_grad = True
                elif backbone == "mobilenet_v2":
                    # Unfreeze the last few conv blocks of MobileNetV2's feature extractor.
                    for p in net.features[-3:].parameters():
                        p.requires_grad = True

    def forward(self, x):
        # x: [N, C, H, W] -> [N, feature_dim]
        feats = self.features(x)
        feats = torch.flatten(feats, 1)
        return feats


class CNNLSTMSteering(nn.Module):
    """
    Encodes each frame in a sequence with a frozen CNN, feeds the per-frame
    feature vectors through an LSTM, and regresses a single steering angle
    from the final hidden state (corresponding to the last frame).
    """

    def __init__(self, backbone="resnet18", hidden_size=128, lstm_layers=1,
                 freeze_backbone=True, unfreeze_last_block=False, dropout=0.2):
        super().__init__()

        self.encoder = CNNEncoder(
            backbone=backbone,
            freeze_backbone=freeze_backbone,
            unfreeze_last_block=unfreeze_last_block,
        )

        self.lstm = nn.LSTM(
            input_size=self.encoder.feature_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.encoder(x)                # [B*T, feature_dim]
        feats = feats.view(b, t, -1)            # [B, T, feature_dim]

        lstm_out, (h_n, c_n) = self.lstm(feats)  # h_n: [num_layers, B, hidden_size]
        last_hidden = h_n[-1]                    # [B, hidden_size], final layer's hidden state

        angle = self.head(last_hidden).squeeze(-1)  # [B]
        return angle

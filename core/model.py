"""
core/model.py

ExamSentinelNet: Multi-head forensics detection model.
FIX: stage1_channels read dynamically from backbone — no hardcoded 24.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import timm

from core.forensics_filters import ForensicsFeatureExtractor


class StochasticPurifier(nn.Module):
    """
    Randomly drops semantic (content-biased) channels during training,
    forcing the model to rely on forensics residuals instead.
    """
    def __init__(self, channels: int, drop_rate: float = 0.2):
        super().__init__()
        self.drop_rate = drop_rate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_weights = self.gate(x)
        if self.training:
            mask = (torch.rand_like(gate_weights) > self.drop_rate).float()
            gate_weights = gate_weights * mask
        return x * gate_weights.unsqueeze(-1).unsqueeze(-1)


class DetectionHead(nn.Module):
    """Binary classification head for one forgery type."""
    def __init__(self, in_features: int, name: str = "head"):
        super().__init__()
        self.name = name
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),          nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class RecaptureHead(nn.Module):
    """
    Recapture head with moiré feature fusion.
    Novel: backbone features + FFT frequency map combined.
    """
    def __init__(self, backbone_features: int, moire_flat: int):
        super().__init__()
        self.moire_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(moire_flat, 128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(backbone_features + 128, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, backbone_feat: torch.Tensor,
                moire_feat: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([backbone_feat, self.moire_encoder(moire_feat)], dim=1))


class ExamSentinelNet(nn.Module):
    """
    4-head forensics detector.
    Outputs: deepfake, recapture, splicing, forgery, combined logits.
    """
    def __init__(self, img_size: int = 256, pretrained: bool = True):
        super().__init__()
        self.img_size = img_size

        self.forensics = ForensicsFeatureExtractor(img_size=img_size)

        self.backbone = timm.create_model(
            'efficientnet_b3', pretrained=pretrained,
            features_only=True, out_indices=(1, 2, 3, 4)
        )

        # FIX: read channel counts dynamically — works across all timm versions
        backbone_channels = self.backbone.feature_info.channels()
        stage1_channels = backbone_channels[0]   # e.g. 24 or 32 depending on version
        final_channels  = backbone_channels[-1]  # e.g. 384 for B3

        # Project forensics 64ch → stage1_channels (exact match, no assumption)
        self.forensics_inject = nn.Sequential(
            nn.Conv2d(64, stage1_channels, kernel_size=1),
            nn.BatchNorm2d(stage1_channels),
            nn.ReLU(inplace=True)
        )

        self.purifier = StochasticPurifier(channels=final_channels, drop_rate=0.2)
        self.pool     = nn.AdaptiveAvgPool2d(1)

        self.head_deepfake = DetectionHead(final_channels, "deepfake")
        self.head_splicing = DetectionHead(final_channels, "splicing")
        self.head_forgery  = DetectionHead(final_channels, "forgery")

        # Moiré flat size depends on img_size: (img_size // 8)^2 * 2
        moire_flat = 2 * (img_size // 8) * (img_size // 8)
        self.head_recapture = RecaptureHead(final_channels, moire_flat)

        self.ensemble_weights = nn.Parameter(torch.ones(4) / 4.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        forensics_feat, moire_feat = self.forensics(x)

        backbone_features = self.backbone(x)

        # Inject forensics residuals at stage 1 (additive)
        stage1 = backbone_features[0]
        f_resized = F.interpolate(
            self.forensics_inject(forensics_feat),
            size=stage1.shape[2:], mode='bilinear', align_corners=False
        )
        backbone_features[0] = stage1 + f_resized

        feat        = backbone_features[-1]
        feat        = self.purifier(feat)
        feat_pooled = self.pool(feat).flatten(1)

        logit_deepfake  = self.head_deepfake(feat_pooled)
        logit_splicing  = self.head_splicing(feat_pooled)
        logit_forgery   = self.head_forgery(feat_pooled)
        logit_recapture = self.head_recapture(feat_pooled, moire_feat)

        w = F.softmax(self.ensemble_weights, dim=0)
        logit_combined = (
            w[0] * logit_deepfake + w[1] * logit_recapture +
            w[2] * logit_splicing + w[3] * logit_forgery
        )

        return {
            "deepfake":  logit_deepfake,
            "recapture": logit_recapture,
            "splicing":  logit_splicing,
            "forgery":   logit_forgery,
            "combined":  logit_combined
        }

    def get_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {k: torch.sigmoid(v) for k, v in self.forward(x).items()}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ExamSentinelLoss(nn.Module):
    """Multi-task BCE loss with per-head weighting."""
    def __init__(self, head_weights: Dict[str, float] = None):
        super().__init__()
        self.head_weights = head_weights or {
            "deepfake": 1.0, "recapture": 1.5, "splicing": 1.0, "forgery": 1.0
        }
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions: Dict[str, torch.Tensor],
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        head_map = {
            "deepfake": labels[:, 0:1], "recapture": labels[:, 1:2],
            "splicing": labels[:, 2:3], "forgery":   labels[:, 3:4]
        }
        losses = {}
        total  = torch.tensor(0.0, device=labels.device)
        for name, target in head_map.items():
            if name not in predictions:
                continue
            loss = self.bce(predictions[name], target)
            losses[f"loss_{name}"] = loss
            total = total + loss * self.head_weights.get(name, 1.0)

        any_manip = (labels.sum(dim=1, keepdim=True) > 0).float()
        comb_loss = self.bce(predictions["combined"], any_manip)
        losses["loss_combined"] = comb_loss
        losses["loss_total"] = total + comb_loss * 0.5
        return losses


if __name__ == "__main__":
    print("Testing ExamSentinelNet...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ExamSentinelNet(img_size=256, pretrained=False).to(device)
    print(f"Parameters: {model.count_parameters():,}")
    dummy = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(dummy)
    for k, v in out.items():
        print(f"  {k:12s}: {v.shape}")
    loss_fn = ExamSentinelLoss()
    labels  = torch.zeros(2, 4).to(device)
    labels[0, 0] = 1
    losses = loss_fn(out, labels)
    print(f"\nLoss total: {losses['loss_total'].item():.4f}")
    print("✓ Model OK")

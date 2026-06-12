"""
core/model.py  — Day 2

ExamSentinelNet: Multi-head forensics detection model.

Architecture:
  ┌─────────────────────────────────────┐
  │  Input frame (3, 256, 256)          │
  ├─────────────────────────────────────┤
  │  ForensicsFeatureExtractor          │
  │  (BayarConv + SRM → 64ch residuals) │
  ├─────────────────────────────────────┤
  │  EfficientNet-B3 backbone           │
  │  (fused with forensics features)    │
  ├─────────────────────────────────────┤
  │  Stochastic Purifier                │
  │  (drops content-biased features)    │
  ├─────────────────────────────────────┤
  │  4 Detection Heads:                 │
  │   ├ Deepfake Head                   │
  │   ├ Recapture Head (+ moiré feats)  │
  │   ├ Splicing Head                   │
  │   └ Forgery Head                    │
  └─────────────────────────────────────┘

RTX 3060 6GB optimised:
  - Mixed precision (AMP)
  - Gradient checkpointing on backbone
  - Batch size 16 @ 256×256 fits comfortably
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import timm

from core.forensics_filters import ForensicsFeatureExtractor


# ─── Stochastic Purifier ──────────────────────────────────────────────────────

class StochasticPurifier(nn.Module):
    """
    Drops content-biased (scene-level) features and retains manipulation traces.

    Intuition: A genuine frame and a deepfake frame of the same face differ
    only in subtle noise residuals. If a model learns 'this looks like a face',
    it'll miss the forgery. The purifier randomly zeroes high-activation
    semantic channels during training, forcing the network to rely on
    the forensics residuals (BayarConv/SRM outputs) instead.

    At inference, applies soft channel masking based on activation statistics.
    """

    def __init__(self, channels: int, drop_rate: float = 0.2):
        super().__init__()
        self.channels = channels
        self.drop_rate = drop_rate

        # Learnable gate: predicts which channels are 'semantic' vs 'forensic'
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Random channel dropout to prevent semantic shortcuts
            gate_weights = self.gate(x)  # (B, C)
            noise = torch.rand_like(gate_weights)
            mask = (noise > self.drop_rate).float()
            gate_weights = gate_weights * mask
            gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)
            return x * gate_weights
        else:
            # Soft masking at inference
            gate_weights = self.gate(x)
            gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)
            return x * gate_weights


# ─── Detection Heads ──────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """Generic binary classification head."""

    def __init__(self, in_features: int, name: str = "head"):
        super().__init__()
        self.name = name
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)  # (B, 1) — raw logit


class RecaptureHead(nn.Module):
    """
    Recapture head with moiré feature fusion.
    Uses both backbone features AND frequency-domain moiré features.
    This is the novel contribution — no proctoring system does this.
    """

    def __init__(self, backbone_features: int, moire_features: int):
        super().__init__()

        # Moiré feature encoder
        self.moire_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(moire_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Fusion + classification
        self.fusion = nn.Sequential(
            nn.Linear(backbone_features + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, backbone_feat: torch.Tensor,
                moire_feat: torch.Tensor) -> torch.Tensor:
        moire_encoded = self.moire_encoder(moire_feat)
        fused = torch.cat([backbone_feat, moire_encoded], dim=1)
        return self.fusion(fused)


# ─── Main Model ───────────────────────────────────────────────────────────────

class ExamSentinelNet(nn.Module):
    """
    ExamSentinel multi-head forensics detection model.

    Inputs:  x (B, 3, 256, 256)
    Outputs: dict of logits {
        'deepfake':  (B, 1),
        'recapture': (B, 1),
        'splicing':  (B, 1),
        'forgery':   (B, 1),
        'combined':  (B, 1)   ← weighted ensemble
    }
    """

    def __init__(self, img_size: int = 256, pretrained: bool = True):
        super().__init__()
        self.img_size = img_size

        # ── Forensics feature extractor (BayarConv + SRM + Moiré)
        self.forensics = ForensicsFeatureExtractor(img_size=img_size)

        # ── Backbone: EfficientNet-B3 (good accuracy/speed on RTX 3060)
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )

        # ── Forensics feature injection: fuse into backbone at early stage
        # forensics outputs 64ch @ 256x256 → downsample to match backbone stage 1
        self.forensics_inject = nn.Sequential(
            nn.Conv2d(64, 24, kernel_size=1),  # 24 = backbone stage 1 channels
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # ── Get backbone output channels (B3: 32, 48, 136, 384 for stages 1-4)
        backbone_channels = self.backbone.feature_info.channels()
        final_channels = backbone_channels[-1]  # 384 for B3

        # ── Stochastic Purifier
        self.purifier = StochasticPurifier(channels=final_channels, drop_rate=0.2)

        # ── Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ── Detection heads
        self.head_deepfake = DetectionHead(final_channels, "deepfake")
        self.head_splicing = DetectionHead(final_channels, "splicing")
        self.head_forgery = DetectionHead(final_channels, "forgery")

        # Recapture head also uses moiré features
        # Moiré: (B, 2, 32, 32) → 2*32*32 = 2048 features
        moire_flat = 2 * (img_size // 8) * (img_size // 8)
        self.head_recapture = RecaptureHead(final_channels, moire_flat)

        # ── Ensemble weight (learnable)
        self.ensemble_weights = nn.Parameter(
            torch.ones(4) / 4.0
        )

        # Enable gradient checkpointing for memory efficiency on RTX 3060
        self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Reduce VRAM usage — critical for RTX 3060 6GB."""
        try:
            self.backbone.set_grad_checkpointing(True)
        except AttributeError:
            pass  # Not all versions support this

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]

        # ── Extract forensics features
        forensics_feat, moire_feat = self.forensics(x)

        # ── Backbone forward with forensics injection
        # Resize forensics features to match backbone stage 1 spatial size
        backbone_features = self.backbone(x)  # list of feature maps

        # Inject forensics at stage 1 (additive fusion)
        stage1 = backbone_features[0]
        forensics_resized = F.interpolate(
            self.forensics_inject(forensics_feat),
            size=stage1.shape[2:],
            mode='bilinear', align_corners=False
        )
        backbone_features[0] = stage1 + forensics_resized

        # Use final backbone stage
        feat = backbone_features[-1]  # (B, 384, H', W')

        # ── Stochastic Purifier
        feat = self.purifier(feat)

        # ── Global pooling
        feat_pooled = self.pool(feat).flatten(1)  # (B, 384)

        # ── Detection heads
        logit_deepfake = self.head_deepfake(feat_pooled)
        logit_splicing = self.head_splicing(feat_pooled)
        logit_forgery = self.head_forgery(feat_pooled)
        logit_recapture = self.head_recapture(feat_pooled, moire_feat)

        # ── Weighted ensemble (softmax over learned weights)
        w = F.softmax(self.ensemble_weights, dim=0)
        logit_combined = (
            w[0] * logit_deepfake +
            w[1] * logit_recapture +
            w[2] * logit_splicing +
            w[3] * logit_forgery
        )

        return {
            "deepfake":  logit_deepfake,
            "recapture": logit_recapture,
            "splicing":  logit_splicing,
            "forgery":   logit_forgery,
            "combined":  logit_combined
        }

    def get_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience: returns sigmoid probabilities instead of logits."""
        logits = self.forward(x)
        return {k: torch.sigmoid(v) for k, v in logits.items()}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Loss function ────────────────────────────────────────────────────────────

class ExamSentinelLoss(nn.Module):
    """
    Multi-task loss with per-head weighting.

    Recapture head gets higher weight since it's the novel contribution
    and the hardest task (moiré patterns are subtle).
    """

    def __init__(self, head_weights: Dict[str, float] = None):
        super().__init__()
        self.head_weights = head_weights or {
            "deepfake": 1.0,
            "recapture": 1.5,   # Novel + harder task
            "splicing": 1.0,
            "forgery": 1.0
        }
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                labels: torch.Tensor  # (B, 4): [deepfake, recapture, splicing, forgery]
                ) -> Dict[str, torch.Tensor]:

        head_map = {
            "deepfake":  labels[:, 0:1],
            "recapture": labels[:, 1:2],
            "splicing":  labels[:, 2:3],
            "forgery":   labels[:, 3:4]
        }

        losses = {}
        total = torch.tensor(0.0, device=labels.device)

        for head_name, target in head_map.items():
            if head_name not in predictions:
                continue
            pred = predictions[head_name]
            loss = self.bce(pred, target)
            weighted = loss * self.head_weights.get(head_name, 1.0)
            losses[f"loss_{head_name}"] = loss
            total = total + weighted

        # Combined head loss (ensemble)
        any_manip = (labels.sum(dim=1, keepdim=True) > 0).float()
        combined_loss = self.bce(predictions["combined"], any_manip)
        losses["loss_combined"] = combined_loss
        total = total + combined_loss * 0.5

        losses["loss_total"] = total
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
        print(f"  {k:12s}: {v.shape}  (logit range: {v.min():.2f}..{v.max():.2f})")

    # Test loss
    loss_fn = ExamSentinelLoss()
    labels = torch.zeros(2, 4).to(device)
    labels[0, 0] = 1  # first sample is deepfake
    losses = loss_fn(out, labels)
    print(f"\nLoss breakdown:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    print("\n✓ Model OK")

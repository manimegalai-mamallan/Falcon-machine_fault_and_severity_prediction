"""
2D CNN cascaded classifiers for vibration fault diagnosis.

Two model classes:

  FaultClassifier     — predicts the fault class (4-way softmax over
                        Normal / Unbalance / Misalignment / Looseness).

  SeverityClassifier  — predicts the severity (3-way softmax over
                        Low / Medium / High). One instance is trained
                        PER fault type, so we end up with 3 severity
                        models (severity given Unbalance, severity given
                        Misalignment, severity given Looseness).

This cascaded design is much more data-efficient than a single multi-task
model when the dataset is small: each severity model sees ~9 files of
its own fault type rather than dividing ~9 files across 4 fault
buckets simultaneously.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


# ---------------------------------------------------------------------------
# Backbone — shared between fault and severity models
# ---------------------------------------------------------------------------
class _ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               padding=1, bias=False)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class _Backbone(nn.Module):
    """Spectrogram encoder + RPM fusion. Used by both classifiers."""

    def __init__(
        self,
        in_channels:    int = config.N_CHANNELS,
        base_channels:  int = config.BASE_CHANNELS,
        rpm_dim:        int = config.RPM_EMB_DIM,
        fusion_dim:     int = config.FUSION_DIM,
        dropout:        float = config.DROPOUT,
    ):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(_ResBlock2D(c, c), _ResBlock2D(c, c))
        self.stage2 = nn.Sequential(_ResBlock2D(c, 2*c, stride=2), _ResBlock2D(2*c, 2*c))
        self.stage3 = nn.Sequential(_ResBlock2D(2*c, 4*c, stride=2), _ResBlock2D(4*c, 4*c))
        self.final_bn = nn.BatchNorm2d(4 * c)
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 4 * c

        self.rpm_embed = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True),
            nn.Linear(16, rpm_dim), nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + rpm_dim, fusion_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.fusion_dim = fusion_dim

    def forward(self, x_spec: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        h = self.stem(x_spec)
        h = self.stage1(h); h = self.stage2(h); h = self.stage3(h)
        h = F.relu(self.final_bn(h), inplace=True)
        h = self.pool(h).flatten(1)
        r = self.rpm_embed(rpm)
        h = torch.cat([h, r], dim=-1)
        return self.fusion(h)


# ---------------------------------------------------------------------------
# Fault classifier — 4-way softmax
# ---------------------------------------------------------------------------
class FaultClassifier(nn.Module):
    def __init__(
        self,
        n_faults: int = len(config.FAULT_CLASSES),
        in_channels:   int = config.N_CHANNELS,
        base_channels: int = config.BASE_CHANNELS,
        rpm_dim:       int = config.RPM_EMB_DIM,
        fusion_dim:    int = config.FUSION_DIM,
        dropout:       float = config.DROPOUT,
    ):
        super().__init__()
        self.backbone = _Backbone(
            in_channels=in_channels, base_channels=base_channels,
            rpm_dim=rpm_dim, fusion_dim=fusion_dim, dropout=dropout,
        )
        self.head = nn.Linear(fusion_dim, n_faults)

    def forward(self, x_spec: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x_spec, rpm)
        return self.head(h)


# ---------------------------------------------------------------------------
# Severity classifier — 3-way softmax. One instance per non-Normal fault.
# ---------------------------------------------------------------------------
class SeverityClassifier(nn.Module):
    def __init__(
        self,
        n_severity: int = len(config.SEVERITY_LEVELS),
        in_channels:   int = config.N_CHANNELS,
        base_channels: int = config.BASE_CHANNELS,
        rpm_dim:       int = config.RPM_EMB_DIM,
        fusion_dim:    int = config.FUSION_DIM,
        dropout:       float = config.DROPOUT,
    ):
        super().__init__()
        self.backbone = _Backbone(
            in_channels=in_channels, base_channels=base_channels,
            rpm_dim=rpm_dim, fusion_dim=fusion_dim, dropout=dropout,
        )
        self.head = nn.Linear(fusion_dim, n_severity)

    def forward(self, x_spec: torch.Tensor, rpm: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x_spec, rpm)
        return self.head(h)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)
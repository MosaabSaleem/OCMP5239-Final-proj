"""
models.py
=========
Network definitions for the dark-knowledge ablation study.

Two networks are defined:

1. ResNet18Teacher  - a CIFAR-adapted ResNet-18 (3x3 stem, no max-pool) used as the
                      high-capacity teacher. This is the standard CIFAR ResNet-18 that
                      reaches ~95% test accuracy on CIFAR-10.

2. SmallStudentCNN  - a deliberately small convolutional student (~0.08M params) that
                      cannot, on its own, match the teacher. This is the network whose
                      behaviour we study under different (ablated) teacher signals.

The whole study keeps BOTH networks fixed across every experiment. The ONLY thing that
changes between conditions is how the teacher's logits are transformed before they are
handed to the distillation loss (see dark_knowledge_transforms.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Teacher: CIFAR-style ResNet-18
# ---------------------------------------------------------------------------
class _BasicBlock(nn.Module):
    """Standard ResNet basic block (two 3x3 convs + identity/projection shortcut)."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18Teacher(nn.Module):
    """CIFAR-adapted ResNet-18 teacher network."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        # CIFAR stem: 3x3 conv, stride 1, NO initial max-pool (images are only 32x32).
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * _BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)   # returns raw logits


# ---------------------------------------------------------------------------
# Student: small CNN
# ---------------------------------------------------------------------------
class SmallStudentCNN(nn.Module):
    """
    A compact 3-block CNN student (~0.08M parameters).

    It is intentionally under-capacity relative to the ResNet-18 teacher so that
    there is a meaningful teacher-student accuracy gap for distillation to close.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                  # 32 -> 16

            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                  # 16 -> 8

            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                  # 8 -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out).flatten(1)
        return self.fc(out)   # returns raw logits


def count_parameters(model):
    """Return the number of trainable parameters (millions)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == "__main__":
    t = ResNet18Teacher()
    s = SmallStudentCNN()
    x = torch.randn(2, 3, 32, 32)
    print(f"Teacher params: {count_parameters(t):.2f}M  | output {t(x).shape}")
    print(f"Student params: {count_parameters(s):.3f}M  | output {s(x).shape}")

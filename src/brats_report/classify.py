"""Tumour-type classification (glioma / meningioma / no tumour / pituitary).

This complements the segmentation+measurement side: segmentation says *where* and
*how big*, this says *what kind*. Reimplemented in PyTorch/torchvision (the original
graduation notebook used TensorFlow/EfficientNet, which has no wheels on this Python).

We use an ImageNet-pretrained backbone as a **fixed feature extractor** with a light
trainable head - accurate and fast enough to train on CPU. The same model runs on a
2-D MRI image, or on the key axial slice extracted from a 3-D volume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

TUMOR_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # ImageFolder order
TUMOR_NAMES = {"glioma": "Glioma", "meningioma": "Meningioma",
               "notumor": "No tumour", "pituitary": "Pituitary"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "mobilenet_v3_small": (models.mobilenet_v3_small,
                           models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, 576),
}


def build_backbone(arch: str = "resnet18", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return a frozen feature-extractor (global-pooled features) and its feature dim."""
    fn, weights, feat_dim = _BACKBONES[arch]
    net = fn(weights=weights if pretrained else None)
    if arch == "resnet18":
        net.fc = nn.Identity()
    else:  # mobilenet_v3_small
        net.classifier = nn.Identity()
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    return net, feat_dim


class ClassifierHead(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int = 4, hidden: int = 256, p: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(feat_dim), nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(p), nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class TumourClassifier(nn.Module):
    """Frozen backbone + trainable head, packaged for end-to-end inference."""

    def __init__(self, arch="resnet18", n_classes=4, pretrained=True):
        super().__init__()
        self.arch = arch
        self.backbone, feat_dim = build_backbone(arch, pretrained)
        self.head = ClassifierHead(feat_dim, n_classes)

    @torch.no_grad()
    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.backbone(x))


def prep_image(arr: np.ndarray, size: int = 224) -> torch.Tensor:
    """Any 2-D array (any range) or HxWx3 -> normalised (1,3,size,size) tensor."""
    from skimage.transform import resize
    a = arr.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    lo, hi = np.percentile(a, [1, 99]) if a.max() > a.min() else (0.0, 1.0)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    a = resize(a, (size, size), order=1, mode="constant", anti_aliasing=True)
    rgb = np.stack([a, a, a], axis=-1)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)[None].astype(np.float32))


def load_classifier(path: str | Path, device="cpu"):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = TumourClassifier(arch=ckpt.get("arch", "resnet18"),
                             n_classes=len(ckpt.get("classes", TUMOR_CLASSES)),
                             pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_type(model, image2d: np.ndarray, ckpt: dict | None = None,
                 device="cpu", size=224) -> dict:
    classes = (ckpt or {}).get("classes", TUMOR_CLASSES)
    x = prep_image(image2d, size).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    i = int(probs.argmax())
    return {"class": classes[i], "name": TUMOR_NAMES.get(classes[i], classes[i]),
            "confidence": round(float(probs[i]), 4),
            "probs": {c: round(float(p), 4) for c, p in zip(classes, probs, strict=True)}}

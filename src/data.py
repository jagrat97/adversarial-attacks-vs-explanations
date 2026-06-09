"""Model + data utilities.

Design choice: we run *all* attacks and explanations in [0, 1] pixel space.
The ImageNet normalization (mean/std) is folded into the model as its first
layer (`NormalizedModel`). This means:
  * adversarial budgets (epsilon) are expressed directly in pixel units (k/255),
  * we can clamp perturbed images to the valid [0, 1] range trivially,
  * Grad-CAM / Integrated Gradients receive the same [0, 1] inputs.
"""
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(HERE, "images")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NormalizedModel(nn.Module):
    """Wraps a backbone so it accepts images in [0, 1] and normalizes internally."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def load_model(device=None):
    """Return (model, categories). `model` takes [0,1] images and outputs logits."""
    device = device or get_device()
    weights = ResNet50_Weights.IMAGENET1K_V2
    backbone = resnet50(weights=weights)
    model = NormalizedModel(backbone).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    categories = weights.meta["categories"]  # 1000 human-readable class names
    return model, categories


# Preprocessing WITHOUT normalization -> tensor in [0, 1]
_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # -> [0,1], shape (3, 224, 224)
])


def load_image(path):
    """Load an image file to a (3, 224, 224) float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return _PREPROCESS(img)


def list_images():
    """Return [(name, path, true_label), ...] for the bundled sample images."""
    labels_path = os.path.join(IMAGES_DIR, "labels.json")
    labels = {}
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = json.load(f)
    items = []
    for fn in sorted(os.listdir(IMAGES_DIR)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(fn)[0]
            items.append((name, os.path.join(IMAGES_DIR, fn), labels.get(fn, name)))
    return items


def to_numpy_img(x):
    """(3, H, W) [0,1] tensor -> (H, W, 3) numpy for plotting."""
    return x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def predict(model, x):
    """x: (N,3,H,W) in [0,1]. Returns (probs (N,1000), top1_idx (N,))."""
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return probs, probs.argmax(dim=1)

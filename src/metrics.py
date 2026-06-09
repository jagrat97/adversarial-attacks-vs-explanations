"""Quantitative metrics.

Two families:
  1. How much an explanation *changes* between a clean and an adversarial image
     (cosine similarity, Spearman rank correlation, top-k region IoU).
  2. Whether an input is adversarial can be *detected* from the fragility of the
     model's behaviour under tiny random noise (prediction-instability feature),
     scored with ROC-AUC.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


# ----------------------------- explanation similarity ----------------------- #
def _flat(a):
    return np.asarray(a, dtype=np.float64).ravel()


def cosine_sim(a, b):
    a, b = _flat(a), _flat(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def spearman(a, b):
    """Rank correlation: are the *same pixels* salient, ignoring magnitude?"""
    a, b = _flat(a), _flat(b)
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12
    return float(np.dot(ra, rb) / denom)


def topk_iou(a, b, frac=0.1):
    """IoU of the top-`frac` most-salient pixels of each map."""
    a, b = _flat(a), _flat(b)
    ka = a >= np.quantile(a, 1 - frac)
    kb = b >= np.quantile(b, 1 - frac)
    inter = np.logical_and(ka, kb).sum()
    union = np.logical_or(ka, kb).sum() + 1e-12
    return float(inter / union)


def explanation_shift(clean_map, adv_map):
    """Bundle of similarity metrics between two saliency maps."""
    return {
        "cosine": cosine_sim(clean_map, adv_map),
        "spearman": spearman(clean_map, adv_map),
        "topk_iou": topk_iou(clean_map, adv_map),
    }


# ------------------------------- detection ---------------------------------- #
@torch.no_grad()
def prediction_instability(model, x, base_idx, sigma=0.05, n=16, seed=0):
    """Fraction of `n` Gaussian-noised copies of x whose top-1 != base_idx.

    Adversarial inputs sit near a decision boundary, so their prediction is
    fragile under tiny noise -> high instability. Clean inputs are stable.
    Forward-only (no gradients); cheap. x: (3,H,W). Returns float in [0,1].
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    batch = x.unsqueeze(0).repeat(n, 1, 1, 1).cpu()
    batch = (batch + sigma * torch.randn(batch.shape, generator=g)).clamp(0, 1)
    preds = model(batch.to(next(model.parameters()).device)).argmax(dim=1).cpu()
    return float((preds != int(base_idx)).float().mean())


def detector_auc(clean_scores, adv_scores):
    """ROC-AUC for separating clean (label 0) from adversarial (label 1)."""
    y = np.r_[np.zeros(len(clean_scores)), np.ones(len(adv_scores))]
    s = np.r_[np.asarray(clean_scores), np.asarray(adv_scores)]
    return float(roc_auc_score(y, s))

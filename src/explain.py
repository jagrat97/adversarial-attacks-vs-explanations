"""Post-hoc explanation methods: Grad-CAM (from scratch) and Integrated Gradients.

Grad-CAM            - Selvaraju et al. ICCV 2017. https://arxiv.org/abs/1610.02391
Integrated Gradients- Sundararajan et al. ICML 2017. https://arxiv.org/abs/1703.01365
                      (computed with the Captum library, the tool used in the
                      course's reference ViT-interpretability project).

Both return a per-sample saliency map normalized to [0, 1] and upsampled to the
input resolution, so clean vs. adversarial maps can be compared directly.
"""
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients


def _normalize_map(cam):
    """Per-sample min-max normalize a (N, H, W) map to [0, 1]."""
    n = cam.shape[0]
    flat = cam.view(n, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo + 1e-8)
    return flat.view_as(cam)


class GradCAM:
    """Grad-CAM on a chosen convolutional layer (default: ResNet layer4)."""

    def __init__(self, model, target_layer=None):
        self.model = model
        if target_layer is None:
            target_layer = model.backbone.layer4  # last conv stage of ResNet
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        """x: (N,3,H,W) in [0,1]. Returns (cam (N,H,W) in [0,1], class_idx (N,))."""
        # The model's params are frozen, so we route gradients through the input
        # to build the graph; the backward hook then captures the layer4 grads.
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(dim=1)
            class_idx = torch.as_tensor(class_idx, device=logits.device).view(-1)
            score = logits.gather(1, class_idx.view(-1, 1)).sum()
            score.backward()

        # Grad-CAM: channel weights = global-average-pooled gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))  # (N, h, w)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:],
                            mode="bilinear", align_corners=False).squeeze(1)
        return _normalize_map(cam).cpu(), class_idx.cpu()


def integrated_gradients(model, x, class_idx=None, n_steps=32, baseline=None):
    """Integrated Gradients heatmap via Captum.

    Returns (saliency (N,H,W) in [0,1], class_idx (N,)). The signed per-pixel,
    per-channel attributions are reduced to a magnitude heatmap (sum of |attr|
    over the colour channels), matching common IG visualizations.
    """
    if class_idx is None:
        with torch.no_grad():
            class_idx = model(x).argmax(dim=1)
    class_idx = torch.as_tensor(class_idx, device=x.device).view(-1)
    baseline = baseline if baseline is not None else torch.zeros_like(x)

    ig = IntegratedGradients(model)
    x_in = x.clone().detach().requires_grad_(True)
    attr = ig.attribute(x_in, baselines=baseline, target=class_idx,
                        n_steps=n_steps)
    sal = attr.abs().sum(dim=1)  # (N, H, W)
    return _normalize_map(sal).detach().cpu(), class_idx.cpu()

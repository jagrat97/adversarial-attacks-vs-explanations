"""White-box adversarial attacks, implemented from scratch.

Both attacks are *untargeted*: they push the model away from the label `y`
(by default the model's own clean top-1 prediction), maximizing the loss.
All tensors live in [0, 1] pixel space (see data.NormalizedModel), so the
perturbation budget `eps` is a fraction of the pixel range (e.g. 8/255).

References:
  FGSM  - Goodfellow, Shlens, Szegedy. "Explaining and Harnessing Adversarial
          Examples." ICLR 2015.  https://arxiv.org/abs/1412.6572
  PGD   - Madry et al. "Towards Deep Learning Models Resistant to Adversarial
          Attacks." ICLR 2018.   https://arxiv.org/abs/1706.06083
"""
import torch
import torch.nn.functional as F


def _labels(model, x, y):
    if y is not None:
        return y
    with torch.no_grad():
        return model(x).argmax(dim=1)


def fgsm(model, x, y=None, eps=8 / 255):
    """One-step Fast Gradient Sign Method. Returns adversarial images in [0,1]."""
    y = _labels(model, x, y)
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x + eps * grad.sign()
    return x_adv.clamp(0, 1).detach()


def pgd(model, x, y=None, eps=8 / 255, alpha=None, steps=10, random_start=True):
    """L-infinity Projected Gradient Descent (a.k.a. iterative FGSM).

    alpha defaults to eps/4 (a common, stable choice). The perturbation is
    projected back into the L-inf eps-ball and the image clamped to [0,1]
    after every step.
    """
    y = _labels(model, x, y)
    alpha = alpha if alpha is not None else max(eps / 4.0, 1 / 255)
    x_orig = x.clone().detach()
    x_adv = x_orig.clone()
    if random_start and eps > 0:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0, 1)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # project into L-inf eps-ball around the original, then to valid range
        x_adv = torch.min(torch.max(x_adv, x_orig - eps), x_orig + eps)
        x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()


ATTACKS = {"fgsm": fgsm, "pgd": pgd}


def linf(x_adv, x):
    """Realized L-infinity perturbation size (in pixel units)."""
    return (x_adv - x).abs().amax().item()

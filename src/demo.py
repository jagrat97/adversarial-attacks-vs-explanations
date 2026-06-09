"""Live demo: attack one image and watch its explanation + robustness collapse.

Used for the recorded walkthrough. Example:
    python src/demo.py                          # default: giant panda, PGD 4/255
    python src/demo.py --image images/zebra.jpg --attack pgd --eps 8/255
    python src/demo.py --image goldfish --attack fgsm

Prints a human-readable report and saves a clean/adversarial side-by-side with
Grad-CAM overlays to figures/demo.png.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attacks
import data
import explain
import metrics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_eps(s):
    if "/" in str(s):
        a, b = str(s).split("/")
        return float(a) / float(b)
    return float(s)


def resolve_image(arg):
    if os.path.exists(arg):
        return arg
    cand = os.path.join(HERE, "images", arg if arg.endswith((".jpg", ".png", ".jpeg")) else arg + ".jpg")
    if os.path.exists(cand):
        return cand
    raise SystemExit(f"Image not found: {arg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="giant_panda", help="path or bundled name (e.g. zebra)")
    ap.add_argument("--attack", default="pgd", choices=["pgd", "fgsm"])
    ap.add_argument("--eps", default="4/255", help="L-inf budget, e.g. 4/255 or 0.016")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "demo.png"))
    args = ap.parse_args()
    eps = parse_eps(args.eps)

    device = data.get_device()
    model, cats = data.load_model(device)
    path = resolve_image(args.image)
    x = data.load_image(path).unsqueeze(0).to(device)

    p, ci = data.predict(model, x)
    clean_pred, clean_conf = cats[ci.item()], p[0, ci].item()

    if args.attack == "fgsm":
        xa = attacks.fgsm(model, x, y=ci, eps=eps)
    else:
        xa = attacks.pgd(model, x, y=ci, eps=eps, steps=args.steps)
    pa, ai = data.predict(model, xa)
    adv_pred, adv_conf = cats[ai.item()], pa[0, ai].item()

    gc = explain.GradCAM(model)
    cam_c, _ = gc(x, class_idx=ci)
    cam_a, _ = gc(xa, class_idx=ci)        # same (original) class on both
    shift = metrics.explanation_shift(cam_c[0].numpy(), cam_a[0].numpy())

    inst_c = metrics.prediction_instability(model, x[0], ci.item(), sigma=args.sigma, n=32)
    inst_a = metrics.prediction_instability(model, xa[0], ai.item(), sigma=args.sigma, n=32)

    bar = "=" * 60
    print(f"\n{bar}\n  ADVERSARIAL ATTACKS vs. EXPLANATIONS  -  live demo\n{bar}")
    print(f"  image            : {os.path.basename(path)}")
    print(f"  attack           : {args.attack.upper()}  eps={args.eps}  "
          f"(L_inf={attacks.linf(xa, x):.4f})")
    print(f"  clean prediction : {clean_pred}  ({clean_conf:.3f})")
    print(f"  adv.  prediction : {adv_pred}  ({adv_conf:.3f})"
          f"   {'<<< FLIPPED' if ai.item() != ci.item() else '(unchanged)'}")
    print(f"  explanation shift: top-10% IoU={shift['topk_iou']:.2f}, "
          f"Spearman={shift['spearman']:.2f}  (1.0 = unchanged)")
    print(f"  instability      : clean={inst_c:.2f}  adv={inst_a:.2f}")
    verdict = "ADVERSARIAL" if inst_a > 0.5 else "clean"
    print(f"  detector verdict : input flagged as -> {verdict}  (threshold 0.5)\n{bar}\n")

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 4.4))
    for a, img, cam, ttl, col in (
        (ax[0], data.to_numpy_img(x[0]), cam_c[0], f"clean: {clean_pred} ({clean_conf:.2f})", "#2a3f8f"),
        (ax[1], data.to_numpy_img(xa[0]), cam_a[0], f"adversarial: {adv_pred} ({adv_conf:.2f})", "#c0392b")):
        a.imshow(img); a.imshow(cam, cmap="jet", alpha=0.45)
        a.set_title(ttl, color=col, fontsize=11); a.axis("off")
    fig.suptitle(f"Grad-CAM for '{clean_pred}'  -  top-10% IoU drops to {shift['topk_iou']:.2f}",
                 color="#2a3f8f")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"  figure saved -> {args.out}\n")


if __name__ == "__main__":
    main()

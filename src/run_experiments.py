"""Run the full MRP study and emit figures/ + results/results.json (REAL numbers).

Three experiments, mirroring the three research questions:
  A. Do attacks succeed?            attack-success vs. epsilon (FGSM vs PGD)
  B. Do explanations survive?       Grad-CAM / IG  clean-vs-adversarial shift
  C. Can fragility reveal attacks?  prediction-instability detector + ROC-AUC

Usage:  python src/run_experiments.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attacks
import data
import explain
import metrics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# ------------------------------- config ------------------------------------ #
EPS_SWEEP = [0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]
EPS_LABELS = ["0", "1/255", "2/255", "4/255", "8/255", "16/255"]
EPS_MAIN = 4 / 255          # imperceptible budget used for the detailed study
PGD_STEPS = 10
SIGMA = 0.05                # detector noise level (validated sweet spot)
N_NOISE = 32
SEED = 0

NAVY, RED, GREY = "#2a3f8f", "#c0392b", "#7f8c8d"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.titlesize": 11, "axes.spines.top": False,
                     "axes.spines.right": False})


def overlay(ax, img, cam=None, title="", color="black"):
    ax.imshow(img)
    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=0.45)
    ax.set_title(title, color=color, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    torch.manual_seed(SEED)
    device = data.get_device()
    print(f"device={device}")
    model, cats = data.load_model(device)
    items = data.list_images()
    names = [n for n, _, _ in items]
    trues = [t for _, _, t in items]
    X = torch.stack([data.load_image(p) for _, p, _ in items]).to(device)
    N = X.shape[0]
    print(f"{N} images loaded")

    clean_probs, CI = data.predict(model, X)
    clean_conf = clean_probs.gather(1, CI.view(-1, 1)).squeeze(1)
    clean_correct = sum(cats[CI[i]].lower() in trues[i].lower()
                        or trues[i].lower() in cats[CI[i]].lower() for i in range(N))
    print(f"clean top-1 matches the file label on {clean_correct}/{N} images")

    results = {
        "model": "torchvision resnet50 (IMAGENET1K_V2)", "device": str(device),
        "n_images": N, "names": names,
        "clean_accuracy_vs_filename": clean_correct / N,
        "config": {"eps_main": "4/255", "pgd_steps": PGD_STEPS,
                   "detector_sigma": SIGMA, "detector_n_noise": N_NOISE},
    }

    # ---------------- Experiment A: attack success vs epsilon --------------- #
    print("\n[A] epsilon sweep ...")
    A = {"epsilons": EPS_LABELS, "fgsm_success": [], "pgd_success": [],
         "fgsm_conf": [], "pgd_conf": []}
    for eps in EPS_SWEEP:
        for atk, key in ((attacks.fgsm, "fgsm"), (attacks.pgd, "pgd")):
            xa = atk(model, X, y=CI, eps=eps) if eps > 0 else X
            p, idx = data.predict(model, xa)
            success = float((idx != CI).float().mean())
            conf_true = float(p.gather(1, CI.view(-1, 1)).mean())  # conf in orig class
            A[f"{key}_success"].append(success)
            A[f"{key}_conf"].append(conf_true)
        print(f"  eps={eps:.4f}  FGSM={A['fgsm_success'][-1]:.2f}  "
              f"PGD={A['pgd_success'][-1]:.2f}")
    results["experiment_A_attack_success"] = A

    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = list(range(len(EPS_SWEEP)))
    ax.plot(xs, A["fgsm_success"], "o-", color=NAVY, label="FGSM (1 step)")
    ax.plot(xs, A["pgd_success"], "s-", color=RED, label=f"PGD ({PGD_STEPS} steps)")
    ax.set_xticks(xs); ax.set_xticklabels(EPS_LABELS)
    ax.set_xlabel(r"perturbation budget  $\epsilon$  (L$_\infty$, pixel units)")
    ax.set_ylabel("attack success rate")
    ax.set_ylim(-0.03, 1.03); ax.grid(alpha=.25); ax.legend()
    ax.set_title("Imperceptible perturbations flip the prediction", color=NAVY)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_attack_success.png"); plt.close(fig)

    # ---------------- Experiment B: explanations under attack --------------- #
    print(f"\n[B] explanations at eps={EPS_MAIN:.4f} (PGD) ...")
    Xadv = attacks.pgd(model, X, y=CI, eps=EPS_MAIN, steps=PGD_STEPS)
    adv_probs, AI = data.predict(model, Xadv)
    adv_conf = adv_probs.gather(1, AI.view(-1, 1)).squeeze(1)
    flipped = (AI != CI)
    print(f"  PGD flipped {int(flipped.sum())}/{N} predictions")

    gradcam = explain.GradCAM(model)
    cam_clean, _ = gradcam(X, class_idx=CI)          # explain ORIGINAL class
    cam_adv, _ = gradcam(Xadv, class_idx=CI)         # ... on the adversarial image
    cam_clean, cam_adv = cam_clean.numpy(), cam_adv.numpy()

    # Integrated Gradients, looped per-image (memory-friendly), original class
    ig_clean, ig_adv = [], []
    for i in range(N):
        c, _ = explain.integrated_gradients(model, X[i:i + 1], class_idx=CI[i:i + 1], n_steps=32)
        a, _ = explain.integrated_gradients(model, Xadv[i:i + 1], class_idx=CI[i:i + 1], n_steps=32)
        ig_clean.append(c[0].numpy()); ig_adv.append(a[0].numpy())

    def shift_stats(clean_maps, adv_maps):
        rows = [metrics.explanation_shift(clean_maps[i], adv_maps[i]) for i in range(N)]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        agg["per_image"] = rows
        return agg

    B = {"eps": "4/255", "pgd_flip_rate": float(flipped.float().mean()),
         "gradcam": shift_stats(cam_clean, cam_adv),
         "ig": shift_stats(ig_clean, ig_adv)}
    results["experiment_B_explanation_shift"] = B
    print(f"  Grad-CAM cosine={B['gradcam']['cosine']:.3f} "
          f"spearman={B['gradcam']['spearman']:.3f} IoU={B['gradcam']['topk_iou']:.3f}")
    print(f"  IG       cosine={B['ig']['cosine']:.3f} "
          f"spearman={B['ig']['spearman']:.3f} IoU={B['ig']['topk_iou']:.3f}")

    imgs_np = [data.to_numpy_img(X[i]) for i in range(N)]
    advs_np = [data.to_numpy_img(Xadv[i]) for i in range(N)]

    # --- Hero figure (panda triptych: clean | perturbation x N | adversarial) ---
    hi = names.index("giant_panda") if "giant_panda" in names else 0
    pert = (advs_np[hi] - imgs_np[hi]); amp = 10
    pert_vis = np.clip(0.5 + amp * pert, 0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
    overlay(ax[0], imgs_np[hi], None,
            f"prediction: {cats[CI[hi]]}\nconfidence {clean_conf[hi]:.2f}", NAVY)
    overlay(ax[1], pert_vis, None,
            f"adversarial perturbation (x{amp})\nL_inf = {attacks.linf(Xadv[hi:hi+1], X[hi:hi+1]):.4f}", GREY)
    overlay(ax[2], advs_np[hi], None,
            f"prediction: {cats[AI[hi]]}\nconfidence {adv_conf[hi]:.2f}", RED)
    fig.suptitle("Same picture to a human - different label to the model", y=1.02, color="black")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_hero.png", bbox_inches="tight"); plt.close(fig)

    # --- Explanations-break grid: pick 3 flipped images -------------------- #
    pick = [i for i in range(N) if flipped[i]][:3]
    while len(pick) < 3:
        pick.append([i for i in range(N) if i not in pick][0])
    fig, axes = plt.subplots(len(pick), 4, figsize=(11, 2.7 * len(pick)))
    col_titles = ["clean image", "Grad-CAM (clean)", "adversarial image", "Grad-CAM (adv.)"]
    for r, i in enumerate(pick):
        overlay(axes[r, 0], imgs_np[i], None, f"{cats[CI[i]]} ({clean_conf[i]:.2f})", NAVY)
        overlay(axes[r, 1], imgs_np[i], cam_clean[i], "")
        overlay(axes[r, 2], advs_np[i], None, f"{cats[AI[i]]} ({adv_conf[i]:.2f})", RED)
        overlay(axes[r, 3], advs_np[i], cam_adv[i],
                f"IoU vs clean = {B['gradcam']['per_image'][i]['topk_iou']:.2f}", GREY)
        if r == 0:
            for c in range(4):
                axes[r, c].set_title(col_titles[c] + ("" if c not in (0, 2) else ""),
                                     fontsize=9.5, color="black", pad=14)
    fig.suptitle("Explanations are fragile: an imperceptible attack relocates the saliency map",
                 y=1.01, color=NAVY)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_explanations_break.png", bbox_inches="tight"); plt.close(fig)

    # --- Explanation-shift bar chart (Grad-CAM vs IG) ---------------------- #
    fig, ax = plt.subplots(figsize=(7, 4.2))
    keys = ["cosine", "spearman", "topk_iou"]
    klabels = ["cosine sim.", "Spearman rank", "top-10% IoU"]
    gc_vals = [B["gradcam"][k] for k in keys]; ig_vals = [B["ig"][k] for k in keys]
    w = 0.35; xpos = np.arange(len(keys))
    ax.bar(xpos - w / 2, gc_vals, w, color=NAVY, label="Grad-CAM")
    ax.bar(xpos + w / 2, ig_vals, w, color="#e08e0b", label="Integrated Gradients")
    for j, v in enumerate(gc_vals): ax.text(xpos[j] - w / 2, v + .02, f"{v:.2f}", ha="center", fontsize=8)
    for j, v in enumerate(ig_vals): ax.text(xpos[j] + w / 2, v + .02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels(klabels)
    ax.set_ylabel("similarity  (clean vs. adversarial)\n1.0 = unchanged,  lower = more fragile")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.25); ax.legend()
    ax.set_title(f"How much explanations change under attack (eps={B['eps']})", color=NAVY)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_explanation_shift.png"); plt.close(fig)

    # ---------------- Experiment C: detection ------------------------------- #
    print(f"\n[C] detection (sigma={SIGMA}, n={N_NOISE}) ...")
    clean_inst = [metrics.prediction_instability(model, X[i], CI[i].item(),
                  sigma=SIGMA, n=N_NOISE, seed=SEED) for i in range(N)]
    adv_inst = [metrics.prediction_instability(model, Xadv[i], AI[i].item(),
                sigma=SIGMA, n=N_NOISE, seed=SEED) for i in range(N)]
    auc = metrics.detector_auc(clean_inst, adv_inst)
    C = {"sigma": SIGMA, "n_noise": N_NOISE, "auc": auc,
         "clean_inst": clean_inst, "adv_inst": adv_inst,
         "clean_inst_mean": float(np.mean(clean_inst)),
         "adv_inst_mean": float(np.mean(adv_inst))}
    results["experiment_C_detection"] = C
    print(f"  clean_inst={C['clean_inst_mean']:.2f}  adv_inst={C['adv_inst_mean']:.2f}  AUC={auc:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    rng = np.random.default_rng(SEED)
    ax[0].scatter(rng.normal(0, .04, N), clean_inst, color=NAVY, s=40, label="clean", zorder=3)
    ax[0].scatter(rng.normal(1, .04, N), adv_inst, color=RED, s=40, label="adversarial", zorder=3)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["clean", "adversarial (PGD)"])
    ax[0].set_ylabel("prediction instability under tiny noise")
    ax[0].set_ylim(-.05, 1.05); ax[0].grid(axis="y", alpha=.25)
    ax[0].set_title("Adversarial inputs are fragile under noise", color=NAVY)
    y = np.r_[np.zeros(N), np.ones(N)]; s = np.r_[clean_inst, adv_inst]
    fpr, tpr, _ = roc_curve(y, s)
    ax[1].plot(fpr, tpr, color=RED, lw=2, label=f"AUC = {auc:.3f}")
    ax[1].plot([0, 1], [0, 1], "--", color=GREY)
    ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate")
    ax[1].set_title("Detector ROC", color=NAVY); ax[1].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_detection.png"); plt.close(fig)

    # ---------------- per-example table ------------------------------------- #
    results["examples"] = [{
        "name": names[i], "true": trues[i],
        "clean_pred": cats[CI[i]], "clean_conf": round(float(clean_conf[i]), 3),
        "adv_pred": cats[AI[i]], "adv_conf": round(float(adv_conf[i]), 3),
        "linf": round(attacks.linf(Xadv[i:i + 1], X[i:i + 1]), 4),
        "flipped": bool(flipped[i]),
    } for i in range(N)]

    with open(f"{RES}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {RES}/results.json")
    print(f"Saved figures  -> {FIG}/ ({len(os.listdir(FIG))} files)")
    print("\n=== HEADLINE NUMBERS ===")
    print(f"PGD success @4/255 ........ {B['pgd_flip_rate']*100:.0f}%")
    print(f"Grad-CAM top-10% IoU ...... {B['gradcam']['topk_iou']:.2f} (1.0=unchanged)")
    print(f"IG top-10% IoU ............ {B['ig']['topk_iou']:.2f}")
    print(f"Detector ROC-AUC .......... {auc:.3f}")


if __name__ == "__main__":
    main()

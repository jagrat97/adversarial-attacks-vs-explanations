"""Generate MRP_presentation.pdf (a ~10-minute slide deck) from the REAL results.

All headline numbers are read from results/results.json, and the repo / YouTube
links from links.json, so the deck always matches the experiments. Rendered with
matplotlib (no LaTeX needed) into a 16:9 multi-page PDF.

    python src/make_slides.py
"""
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
W, H = 12.8, 7.2                      # 16:9 inches
NAVY, RED, INK, SUB, GOLD = "#22317a", "#c0392b", "#1a1a1a", "#5b6470", "#e08e0b"
plt.rcParams["font.family"] = "DejaVu Sans"

R = json.load(open(os.path.join(HERE, "results", "results.json")))
L = json.load(open(os.path.join(HERE, "links.json")))

# ----- pull real numbers -----
A = R["experiment_A_attack_success"]
B = R["experiment_B_explanation_shift"]
C = R["experiment_C_detection"]
PGD_MAX = int(round(A["pgd_success"][-1] * 100))
PGD_MAIN = int(round(B["pgd_flip_rate"] * 100))
GC_IOU, IG_IOU = B["gradcam"]["topk_iou"], B["ig"]["topk_iou"]
GC_SPE = B["gradcam"]["spearman"]
AUC = C["auc"]
CLEAN_INST, ADV_INST = C["clean_inst_mean"], C["adv_inst_mean"]
NIMG = R["n_images"]
# pick the panda example for the headline anecdote
PANDA = next((e for e in R["examples"] if e["name"] == "giant_panda"), R["examples"][0])


# --------------------------- slide primitives ------------------------------ #
def line(fig, x0, x1, y, color=NAVY, lw=2):
    fig.add_artist(Line2D([x0, x1], [y, y], color=color, lw=lw, transform=fig.transFigure))


def header(fig, title, kicker=None):
    if kicker:
        fig.text(0.06, 0.93, kicker.upper(), fontsize=12, color=RED, fontweight="bold", va="top")
    fig.text(0.06, 0.90, title, fontsize=24, color=NAVY, fontweight="bold", va="top")
    line(fig, 0.06, 0.94, 0.855, NAVY, 2.2)


def footer(fig, page):
    line(fig, 0.06, 0.94, 0.066, "#d3d7de", 1)
    fig.text(0.06, 0.035, "Adversarial Attacks vs. Explanations", fontsize=8, color=SUB)
    fig.text(0.5, 0.035, L["course"], fontsize=8, color=SUB, ha="center")
    fig.text(0.94, 0.035, str(page), fontsize=8, color=SUB, ha="right")


def bullets(fig, items, x=0.075, y0=0.80, size=17):
    """items: str or (str, level). level 0 = main bullet, 1 = sub-bullet."""
    y = y0
    for it in items:
        text, level = it if isinstance(it, tuple) else (it, 0)
        if text == "":
            y -= 0.028
            continue
        sz = size if level == 0 else size - 2.5
        marker, mcol = ("●", NAVY) if level == 0 else ("–", RED)
        indent = x + level * 0.045
        wrapped = textwrap.fill(text, int((92 - level * 16) * 17 / size))
        nlines = wrapped.count("\n") + 1
        fig.text(indent, y, marker, fontsize=sz - 4, color=mcol, va="top", fontweight="bold")
        fig.text(indent + 0.026, y, wrapped, fontsize=sz, color=INK, va="top", linespacing=1.4)
        y -= nlines * (sz * 1.72 / (72 * H)) + 0.020
    return y


def fitted_image(fig, img_path, top=0.815, target_w=0.88, max_h=0.60):
    img = plt.imread(img_path)
    ih, iw = img.shape[:2]
    ar = iw / ih
    w = target_w
    h = w * W / (H * ar)
    if h > max_h:
        h = max_h
        w = h * H * ar / W
    left = (1 - w) / 2
    ax = fig.add_axes([left, top - h, w, h])
    ax.axis("off")
    ax.imshow(img)


def takeaway(fig, text, cy=0.115, fontsize=13, max_chars=90):
    """Centered, auto-wrapped takeaway banner that grows to fit its text."""
    wrapped = textwrap.fill(text, max_chars)
    nlines = wrapped.count("\n") + 1
    line_h = fontsize * 1.5 / (72 * H)
    box_h = nlines * line_h + 0.030
    box = FancyBboxPatch((0.07, cy - box_h / 2), 0.86, box_h, boxstyle="round,pad=0.012",
                         linewidth=0, facecolor="#eef1fb", transform=fig.transFigure, zorder=0)
    fig.add_artist(box)
    fig.text(0.5, cy, wrapped, fontsize=fontsize, color=NAVY, ha="center",
             va="center", fontweight="bold", linespacing=1.45)


def new(pdf):
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white"); return fig


def close(pdf, fig, page):
    if page:
        footer(fig, page)
    pdf.savefig(fig); plt.close(fig)


# ------------------------------- build deck -------------------------------- #
def build():
    out = os.path.join(HERE, "MRP_presentation.pdf")
    with PdfPages(out) as pdf:
        page = 0

        # 1 — TITLE
        fig = new(pdf)
        line(fig, 0.08, 0.92, 0.80, NAVY, 3)
        fig.text(0.5, 0.70, "Do Explanations Survive Attacks?", fontsize=33,
                 color=NAVY, ha="center", fontweight="bold")
        fig.text(0.5, 0.615, "Adversarial Robustness of Saliency-Based Interpretability",
                 fontsize=18, color=INK, ha="center")
        line(fig, 0.30, 0.70, 0.57, RED, 2)
        fig.text(0.5, 0.50, "A Mini Research Problem bridging", fontsize=13, color=SUB, ha="center")
        fig.text(0.5, 0.455, "Lecture 07 (Robustness)   x   Lecture 09 (Interpretability)",
                 fontsize=14, color=NAVY, ha="center", fontweight="bold")
        fig.text(0.5, 0.30, L["author"], fontsize=17, color=INK, ha="center", fontweight="bold")
        fig.text(0.5, 0.255, L["course"], fontsize=12, color=SUB, ha="center")
        fig.text(0.5, 0.215, f"Instructor: {L['professor']}", fontsize=11, color=SUB, ha="center")
        fig.text(0.5, 0.13, f"model: {R['model']}    |    {NIMG} ImageNet images    |    code + demo linked on last slide",
                 fontsize=10, color=SUB, ha="center", style="italic")
        close(pdf, fig, 0); page += 1

        # 2 — TL;DR (required single slide)
        fig = new(pdf); page += 1
        header(fig, "TL;DR", "one slide")
        bullets(fig, [
            ("Motivation: deep classifiers are deployed with saliency-map explanations "
             "(Grad-CAM, Integrated Gradients) to justify decisions, e.g. in medicine and "
             "autonomous driving. But are those explanations trustworthy when an adversary "
             "is present?", 0),
            ("Main idea: take a pretrained ImageNet classifier; craft imperceptible "
             "adversarial perturbations (FGSM / PGD); then measure (a) does the prediction "
             "flip, (b) does the explanation survive, and (c) can the explanation's own "
             "fragility be used to DETECT the attack?", 0),
            (f"Results on {NIMG} images (ResNet-50):", 0),
            (f"PGD flips up to {PGD_MAX}% of predictions with an L-inf budget invisible to the eye.", 1),
            (f"Explanations are fragile: only ~{int(round(GC_IOU*100))}% of the top-salient region "
             f"survives the attack (Grad-CAM IoU {GC_IOU:.2f}, IG {IG_IOU:.2f}).", 1),
            (f"That fragility is a signal: a noise-instability detector separates clean vs. "
             f"adversarial inputs with ROC-AUC = {AUC:.2f}.", 1),
        ], y0=0.79, size=16.5)
        close(pdf, fig, page)

        # 3 — BACKGROUND / PROBLEM
        fig = new(pdf); page += 1
        header(fig, "Background: two ideas from the course", "problem setup")
        bullets(fig, [
            ("Adversarial example (Lecture 07 - Robustness): an input x' = x + delta with a tiny "
             "L-inf-bounded perturbation delta that a human cannot see, yet the model "
             "mis-classifies with high confidence.", 0),
            ("Saliency explanation (Lecture 09 - Interpretability): a heatmap over the input "
             "showing which pixels drive the prediction - Grad-CAM (gradients x activations of "
             "the last conv layer) and Integrated Gradients (path integral of gradients).", 0),
            ("Research question: explanations are sold as a transparency / debugging tool. If an "
             "imperceptible attack can silently relocate the explanation, the tool is unreliable "
             "exactly when we need it. Can we quantify this - and turn it into a defense?", 0),
        ], y0=0.79, size=17)
        close(pdf, fig, page)

        # 4 — LIT REVIEW 1
        fig = new(pdf); page += 1
        header(fig, "Related work (1/2): attacks & explanations", "literature review")
        bullets(fig, [
            ("Adversarial attacks", 0),
            ("Szegedy et al. 2014, 'Intriguing properties of neural networks' - discovered "
             "adversarial examples. arXiv:1312.6199", 1),
            ("Goodfellow et al. 2015, 'Explaining and Harnessing Adversarial Examples' - FGSM, "
             "the linear view. arXiv:1412.6572", 1),
            ("Madry et al. 2018, 'Towards Deep Learning Models Resistant to Adversarial Attacks' "
             "- PGD + adversarial training. arXiv:1706.06083", 1),
            ("Carlini & Wagner 2017, strong C&W attacks. arXiv:1608.04644", 1),
            ("Attribution / explanation methods", 0),
            ("Simonyan et al. 2014, gradient saliency maps. arXiv:1312.6034", 1),
            ("Selvaraju et al. 2017, 'Grad-CAM'. arXiv:1610.02391", 1),
            ("Sundararajan et al. 2017, 'Integrated Gradients' (Axiomatic Attribution). "
             "arXiv:1703.01365", 1),
        ], y0=0.80, size=14.5)
        close(pdf, fig, page)

        # 5 — LIT REVIEW 2
        fig = new(pdf); page += 1
        header(fig, "Related work (2/2): fragility & detection", "literature review")
        bullets(fig, [
            ("Fragility of explanations (this MRP builds on these)", 0),
            ("Ghorbani et al. 2019, 'Interpretation of Neural Networks is Fragile' - tiny "
             "perturbations change saliency while keeping the prediction. arXiv:1710.10547", 1),
            ("Kindermans et al. 2019, 'The (Un)reliability of saliency methods'. arXiv:1711.00867", 1),
            ("Dombrowski et al. 2019, 'Explanations can be manipulated and geometry is to blame'. "
             "arXiv:1906.07983", 1),
            ("Detecting adversarial examples", 0),
            ("Feinman et al. 2017, 'Detecting Adversarial Samples from Artifacts'. arXiv:1703.00410", 1),
            ("Xu et al. 2018, 'Feature Squeezing' - prediction disagreement under input "
             "transforms. arXiv:1704.01155", 1),
            ("Carlini & Wagner 2017, 'Adversarial Examples Are Not Easily Detected' - caveat: "
             "adaptive attacks bypass many detectors. arXiv:1705.07263", 1),
        ], y0=0.80, size=14.5)
        close(pdf, fig, page)

        # 6 — APPROACH
        fig = new(pdf); page += 1
        header(fig, "Approach & implementation", "method + links")
        bullets(fig, [
            (f"Model: torchvision ResNet-50 (ImageNet); all attacks/explanations run in [0,1] "
             f"pixel space with normalization folded into the model.", 0),
            ("Attacks (implemented from scratch, PyTorch): untargeted FGSM and L-inf PGD, "
             "pushing the model away from its own clean prediction.", 0),
            ("Explanations: Grad-CAM from scratch (forward/backward hooks on layer4) + Integrated "
             "Gradients via Captum (the library used by the course's reference ViT project).", 0),
            ("Explanation shift: cosine, Spearman rank, and top-10% region IoU between the clean "
             "and adversarial saliency maps for the same target class.", 0),
            ("Detector: 'prediction instability' = fraction of tiny Gaussian-noised copies whose "
             "label changes; scored by ROC-AUC. Forward-only, no training.", 0),
            ("Tooling: PyTorch, torchvision, Captum, scikit-learn, matplotlib. Run on Apple-Silicon "
             "MPS in ~40 s. Full code + figures in the repo (last slide).", 0),
        ], y0=0.80, size=15)
        close(pdf, fig, page)

        # 7 — RESULT A
        fig = new(pdf); page += 1
        header(fig, "Result A - attacks succeed with invisible noise", "experiment A")
        fitted_image(fig, os.path.join(FIG, "fig_attack_success.png"), top=0.80, max_h=0.60)
        takeaway(fig, f"PGD reaches {PGD_MAX}% success; FGSM is far weaker. The budget is "
                      f"imperceptible (L-inf <= a few /255).")
        close(pdf, fig, page)

        # 8 — RESULT B hero
        fig = new(pdf); page += 1
        header(fig, "Result B - same picture, different label", "experiment B")
        fitted_image(fig, os.path.join(FIG, "fig_hero.png"), top=0.815, max_h=0.52)
        takeaway(fig, f"'{PANDA['true']}' ({PANDA['clean_conf']:.2f}) -> "
                      f"'{PANDA['adv_pred']}' ({PANDA['adv_conf']:.2f}) at L-inf={PANDA['linf']:.3f}; "
                      f"the perturbation (centre, x10) is invisible at scale 1.")
        close(pdf, fig, page)

        # 9 — RESULT B grid
        fig = new(pdf); page += 1
        header(fig, "Result B - explanations relocate under attack", "experiment B")
        fitted_image(fig, os.path.join(FIG, "fig_explanations_break.png"), top=0.81, max_h=0.62)
        takeaway(fig, "Grad-CAM moves off the object after an imperceptible perturbation - "
                      "the 'explanation' no longer reflects the (now wrong) decision.")
        close(pdf, fig, page)

        # 10 — RESULT B quant
        fig = new(pdf); page += 1
        header(fig, "Result B - how fragile, quantitatively", "experiment B")
        fitted_image(fig, os.path.join(FIG, "fig_explanation_shift.png"), top=0.80, max_h=0.60)
        takeaway(fig, f"Only ~{int(round(GC_IOU*100))}% of the top-10% salient region survives "
                      f"(Grad-CAM IoU {GC_IOU:.2f}, IG {IG_IOU:.2f}). Both methods are fragile.")
        close(pdf, fig, page)

        # 11 — RESULT C
        fig = new(pdf); page += 1
        header(fig, "Result C - fragility becomes a detector", "experiment C")
        fitted_image(fig, os.path.join(FIG, "fig_detection.png"), top=0.80, max_h=0.58)
        takeaway(fig, f"Clean instability {CLEAN_INST:.2f} vs adversarial {ADV_INST:.2f} -> "
                      f"ROC-AUC = {AUC:.2f}. The very fragility that breaks explanations flags attacks.")
        close(pdf, fig, page)

        # 12 — DISCUSSION
        fig = new(pdf); page += 1
        header(fig, "Discussion & limitations", "honest caveats")
        bullets(fig, [
            ("Explanations alone are not a security guarantee: an imperceptible attack flips the "
             "label AND the saliency map, so a user reading the heatmap would be misled.", 0),
            ("But behaviour under noise is informative: adversarial inputs sit near a decision "
             "boundary and are unstable, which our simple detector exploits (AUC "
             f"{AUC:.2f}).", 0),
            (f"Limitations: small sample ({NIMG} images, single architecture); white-box, "
             "L-inf attacks only; the detector is non-adaptive.", 0),
            ("Carlini & Wagner (2017) caution: an adversary aware of the detector can craft "
             "noise-robust attacks - so this is a hurdle, not a solution.", 0),
        ], y0=0.79, size=16)
        close(pdf, fig, page)

        # 13 — CONCLUSION
        fig = new(pdf); page += 1
        header(fig, "Conclusion & future work", "wrap-up")
        bullets(fig, [
            ("Attacks succeed, explanations break, and the break is detectable - one pipeline "
             "links robustness (L07) and interpretability (L09).", 0),
            (f"Headline numbers: PGD up to {PGD_MAX}% success; explanation top-IoU ~{GC_IOU:.2f}; "
             f"detector AUC {AUC:.2f}.", 0),
            ("Future work:", 0),
            ("evaluate on a full validation set + more architectures (ViT vs CNN);", 1),
            ("adaptive attacks that preserve the explanation / evade the detector;", 1),
            ("adversarial training - do robust models also have robust explanations?", 1),
        ], y0=0.79, size=16.5)
        close(pdf, fig, page)

        # 14 — LINKS
        fig = new(pdf); page += 1
        header(fig, "Links & references", "reproduce everything")
        bullets(fig, [
            (f"Code (all experiments, one command):  {L['repo']}", 0),
            (f"Demo video:  {L['youtube']}", 0),
            ("Reproduce:  python src/run_experiments.py   (figures + results.json)", 0),
            ("Live demo:  python src/demo.py --image zebra --attack pgd --eps 8/255", 0),
            ("", 0),
            ("Key references: Goodfellow 2015 (FGSM, 1412.6572); Madry 2018 (PGD, 1706.06083); "
             "Selvaraju 2017 (Grad-CAM, 1610.02391); Sundararajan 2017 (IG, 1703.01365); "
             "Ghorbani 2019 (fragile interpretations, 1710.10547); Xu 2018 (feature squeezing, "
             "1704.01155).", 0),
            ("Course lectures: github.com/fatheral/ai-intro-course (L07 Robustness, "
             "L09 Interpretability).", 0),
        ], y0=0.79, size=15)
        fig.text(0.5, 0.10, "Thank you - questions?", fontsize=18, color=NAVY,
                 ha="center", fontweight="bold")
        close(pdf, fig, page)

    print(f"Wrote {out}  ({page} slides)")


if __name__ == "__main__":
    build()

# Demo / presentation script (~10 minutes)

A narration + action plan for the recorded YouTube walkthrough. Open
`MRP_presentation.pdf` full-screen and keep a terminal ready in the project
folder (`source .venv/bin/activate` first). Timings are a guide.

> Tip for recording: QuickTime (macOS) → File → New Screen Recording, or OBS.
> Show the slides, and cut to the terminal for the live demo around 3:30.

---

### 0:00 — Title + motivation  *(Slide 1–2)*
> "Hi, I'm Jagrat. My MRP is *Do Explanations Survive Attacks?* — it connects two
> topics from the course: adversarial robustness from Lecture 7, and
> interpretability from Lecture 9.
> Modern classifiers are often shipped with saliency-map explanations — heatmaps
> that say *this* part of the image is why I decided X. My question: if I fool the
> model with a perturbation a human can't even see, does the explanation still
> make sense? And can I turn the answer into a way to *detect* the attack?"

Read the TL;DR table. Promise three results: attacks work, explanations break,
fragility detects.

### 1:30 — Background  *(Slide 3)*
> "Two definitions. An *adversarial example* adds a tiny, L-infinity-bounded
> perturbation that's invisible but flips the label. A *saliency explanation*,
> like Grad-CAM or Integrated Gradients, is a heatmap of which pixels mattered."

### 2:15 — Related work + approach  *(Slides 4–6)*
> "This builds on Goodfellow's FGSM and Madry's PGD for attacks; Grad-CAM and
> Integrated Gradients for explanations; and Ghorbani et al., who first showed
> explanations are fragile. My setup: a pretrained ResNet-50, FGSM and PGD
> implemented from scratch, Grad-CAM from scratch, IG via Captum, all in about 40
> seconds on a laptop."

### 3:30 — LIVE DEMO  *(cut to terminal)*
Run, and narrate as the report prints:
```bash
python src/demo.py --image giant_panda --attack pgd --eps 4/255
```
> "Clean, the model is confident it's a giant panda. After PGD with a 4/255
> budget — invisible — it's now sure it's something else entirely. Notice the
> Grad-CAM IoU: the explanation barely overlaps the original. And the detector
> already flags this input as adversarial."

Run a second one for effect:
```bash
python src/demo.py --image zebra --attack pgd --eps 8/255
```
> "Zebra becomes a horned viper. Same story."

(Optional) open `figures/demo.png` to show the side-by-side Grad-CAM overlays.

### 5:30 — Result A: attacks succeed  *(Slide 7)*
> "Across all images, PGD's success climbs to 100% as the budget grows, while
> single-step FGSM stays weak. Even tiny budgets flip most predictions."

### 6:15 — Result B: explanations break  *(Slides 8–10)*
> "Here's the headline picture — identical pandas, different labels, and the
> perturbation amplified 10× so you can see it's just noise.
> Now the explanations: Grad-CAM sits on the animal in the clean image and jumps
> somewhere else after the attack. Quantitatively, only about 22% of the
> top-salient region survives — for both Grad-CAM and Integrated Gradients."

### 8:00 — Result C: detection  *(Slide 11)*
> "But there's a silver lining. Adversarial inputs live right next to a decision
> boundary, so if I add a little random noise, their prediction is unstable —
> while clean inputs don't budge. That single signal separates clean from
> adversarial with an AUC of 0.91."

### 8:45 — Discussion + conclusion  *(Slides 12–13)*
> "So: attacks succeed, explanations break, and the break is itself detectable.
> Caveats — it's a small sample, white-box, and a determined adversary can target
> the detector, as Carlini and Wagner warned. Future work: more architectures,
> ViT vs CNN, and adversarial training."

### 9:30 — Links  *(Slide 14)*
> "All the code reproduces with one command; the repo and these slides are
> linked here. Thanks — happy to take questions."

---

### Exact commands used
```bash
source .venv/bin/activate
python src/run_experiments.py                              # regenerate everything
python src/demo.py --image giant_panda --attack pgd --eps 4/255
python src/demo.py --image zebra       --attack pgd --eps 8/255
python src/demo.py --image golden_retriever --attack fgsm --eps 16/255   # FGSM example
```

### After recording
1. Upload to YouTube (Unlisted is fine for grading).
2. Put the link in `links.json` → `"youtube"`, then rerun `python src/make_slides.py`
   so the deck shows the real link, and update the README badge.
3. Commit & push.

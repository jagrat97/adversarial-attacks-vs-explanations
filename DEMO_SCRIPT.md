# Demo video script — target ≤ 1:50 (keep it under 2 minutes)

A short screen-capture that **shows the built system working**. The 10-minute
explanation is the live slide deck (`index.html`); this video just proves the
code does what the deck claims.

### Pre-flight (do this off-camera, before you hit record)
- `source .venv/bin/activate`
- **Run the demo once now** so the model weights are cached — otherwise the
  first run downloads ResNet-50 mid-video:
  `python src/demo.py --image giant_panda --attack pgd --eps 4/255`
- Have a terminal full-screen (big font) and a file viewer ready for
  `figures/demo.png`. Optionally keep the live deck open in a browser tab.

> Recording: macOS QuickTime → New Screen Recording, or OBS. Speak at a normal
> pace — the script below is ~230 words, which lands around 1:40.

---

| Time | Screen | Say (concise) |
|------|--------|---------------|
| **0:00–0:12** | terminal (or deck title) | "I'm Jagrat. My MRP asks: if an *invisible* attack fools an image classifier, does its **explanation** still hold up — and can we **detect** the attack? Here's the system." |
| **0:12–0:50** | run command ①, let the report print | "Pretrained ResNet-50. Clean, it's confident this is a **giant panda**. Now I add a PGD perturbation capped at 4/255 — invisible to the eye. The label **flips** to something unrelated, with high confidence. The Grad-CAM overlap with the original collapses, and the **detector flags it as adversarial**." |
| **0:50–1:05** | open `figures/demo.png` | "Left: the explanation on the clean panda. Right: after the attack — the heatmap has **slid off** the animal." |
| **1:05–1:25** | run command ② | "Any image, same story — here's a flamingo, gone." |
| **1:25–1:45** | terminal / deck final slide | "Across all 12 images: PGD flips up to **100%** of predictions, explanations keep only **~22%** of their salient region, and the detector hits **AUC 0.91**. Code and the full deck are linked below — thanks!" |

### The two commands
```bash
python src/demo.py --image giant_panda --attack pgd --eps 4/255   # ①
python src/demo.py --image flamingo     --attack pgd --eps 8/255   # ②
```
Each prints clean vs. adversarial prediction, the Grad-CAM IoU drop, the
instability detector verdict, and writes `figures/demo.png`.

> The exact flipped label is non-deterministic (PGD uses a random start), so
> react to whatever it outputs — don't promise a specific class on camera.

### If you're running long (60-second cut)
Drop command ② and the `demo.png` step. Command ① alone shows the whole
pipeline: attack → flipped label → broken explanation → detector verdict.

### After recording
1. Upload to YouTube (Unlisted is fine for grading).
2. Add the YouTube URL in two spots: the `DEMO` line in `index.html` and the
   "Demo video" badge in `README.md`.
3. Commit & push — the live deck on GitHub Pages updates automatically.

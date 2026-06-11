# Demo video — record silent, add AI voiceover (≤ 2 min, ~1:40 target)

Workflow: **(1)** generate the voiceover from the script below (ElevenLabs),
**(2)** screen-record the shots silently, **(3)** lay the audio over the video in
any free editor (iMovie / CapCut / DaVinci Resolve) and trim to fit.

### Before you record (off-camera)
- `source .venv/bin/activate`
- Run it once so weights are cached (no download mid-take):
  `python src/demo.py --image giant_panda --attack pgd --eps 4/255`
- `demo.py` is now **deterministic** (fixed seed) — the labels below are exactly
  what will appear, so the voiceover matches the screen.
- Big terminal font. Open the live deck in a browser tab for the title/result shots:
  `https://jagrat97.github.io/adversarial-attacks-vs-explanations/`

---

## What to record (the shot list)

| # | ~Time | On screen (record this) | Voiceover |
|---|-------|-------------------------|-----------|
| 1 | 0:00–0:14 | **Deck title slide** in the browser (fullscreen, press `F`) | ¶1 |
| 2 | 0:14–0:40 | **Terminal** — type & run command ①. The report prints after ~3 s (model load); hold it on screen so it's readable. | ¶2 |
| 3 | 0:40–0:52 | **Open `figures/demo.png`** (double-click / Quick Look) — clean vs. adversarial Grad-CAM | ¶3 |
| 4 | 0:52–1:08 | **Terminal** — run command ②; hold the report | ¶4 |
| 5 | 1:08–1:40 | **Browser** — arrow through 2–3 deck slides: *Result A* (attack-success chart) → *Result C* (detection, the 0.91) → end on the **title/links** slide | ¶5 |

```bash
python src/demo.py --image giant_panda --attack pgd --eps 4/255   # ①  panda -> pug
python src/demo.py --image flamingo     --attack pgd --eps 8/255   # ②  flamingo -> spider web (100%)
```
What each run prints: clean prediction, the flipped adversarial prediction, the
Grad-CAM **top-10% IoU** drop, and the **detector verdict**. Both are flagged
`ADVERSARIAL` — good to show on camera.

---

## Voiceover — paste this into ElevenLabs (≈205 words, ~1:40)

> ¶1 — Models today ship with visual explanations: heatmaps that claim to show why the model decided what it did. My project asks a simple question. If I fool the model with noise too small for a human to see, does that explanation still hold up — and can its failure warn us that an attack happened? Here's the system.

> ¶2 — This is a pretrained ResNet-50. On the clean image it's confident: a giant panda. Now I add a tiny perturbation, capped at just four out of 255 — invisible to us. The prediction collapses to "pug." The explanation's overlap with the original drops to five percent, and the detector immediately flags the input as adversarial.

> ¶3 — On the left, the clean explanation sits right on the panda. On the right, after the attack, the heatmap has slid off the animal entirely. The model's stated reason is gone.

> ¶4 — And it's not a fluke. A flamingo, with the same trick, becomes a "spider web" — at a hundred percent confidence — and again, the detector catches it.

> ¶5 — Across all twelve test images the pattern holds: attacks flip up to a hundred percent of predictions, explanations keep only about twenty-two percent of their salient region, yet that very fragility detects the attack with an AUC of point nine one. Code and the full deck are linked below. Thanks for watching.

---

## Assembling it
1. Paste ¶1–¶5 into ElevenLabs, generate, download the MP3 (~1:40).
2. Screen-record the 5 shots (QuickTime → New Screen Recording, or OBS). Either
   play the VO in headphones and pace your actions to it, or record loosely.
3. In a free editor, drop the video + the VO audio on the timeline, align each
   shot to its paragraph, and **trim the dead air** while the model loads. Export.
4. Upload to YouTube (Unlisted is fine), then put the URL in the `DEMO` line of
   `index.html` and the badge in `README.md`; commit & push.

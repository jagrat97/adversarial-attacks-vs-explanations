"""Download a small, curated set of ImageNet sample images with known labels.

Source: EliSchwartz/imagenet-sample-images (one representative image per
ImageNet-1k class, filenames encode the synset + human-readable label).
We pull ~12 visually distinct, confidently-classified categories so the
adversarial-attack / explanation demo is clear and reproducible.

Run once:  python src/fetch_images.py
Outputs:   images/*.jpg  and  images/labels.json  (file -> true label)
"""
import json
import os
import urllib.request

REPO = "EliSchwartz/imagenet-sample-images"
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "images")

# Keywords we want (matched against filenames). Chosen to be unambiguous,
# diverse, and reliably top-1 correct for an ImageNet-pretrained ResNet-50.
WANTED = [
    "giant_panda", "golden_retriever", "zebra", "African_elephant",
    "tabby", "goldfish", "sports_car", "school_bus",
    "airliner", "lemon", "broccoli", "flamingo",
]


def get_tree():
    for branch in ("master", "main"):
        url = f"https://api.github.com/repos/{REPO}/git/trees/{branch}?recursive=1"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mrp-fetch"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            paths = [t["path"] for t in data.get("tree", [])
                     if t["path"].lower().endswith((".jpeg", ".jpg", ".png"))]
            if paths:
                return branch, paths
        except Exception as e:  # noqa: BLE001
            print(f"  branch {branch}: {e}")
    raise SystemExit("Could not list repository tree.")


def label_from(path):
    """n02510455_giant_panda.JPEG -> 'giant panda'"""
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    if parts and parts[0].startswith("n") and parts[0][1:].isdigit():
        parts = parts[1:]
    return " ".join(parts)


def main():
    os.makedirs(OUT, exist_ok=True)
    branch, paths = get_tree()
    print(f"Listed {len(paths)} images on branch '{branch}'.")
    labels = {}
    for kw in WANTED:
        match = next((p for p in paths if kw.lower() in p.lower()), None)
        if not match:
            print(f"  [skip] no match for '{kw}'")
            continue
        dst = os.path.join(OUT, kw + ".jpg")
        url = RAW.format(repo=REPO, branch=branch, path=match)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mrp-fetch"})
            with urllib.request.urlopen(req, timeout=60) as r:
                blob = r.read()
            with open(dst, "wb") as f:
                f.write(blob)
            labels[os.path.basename(dst)] = label_from(match)
            print(f"  [ok]  {kw:18s} <- {match}  ({len(blob)//1024} KB)")
        except Exception as e:  # noqa: BLE001
            print(f"  [err] {kw}: {e}")
    with open(os.path.join(OUT, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\nSaved {len(labels)} images to {OUT}")


if __name__ == "__main__":
    main()

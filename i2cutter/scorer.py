"""Brief parsing and CLIP scoring of shots against each line of the brief."""
import os
import re

import numpy as np

from . import shots as shotlib

CLIP_MODELS = [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
]
TEMPLATES = ["{}", "a film still of {}", "a cinematic shot of {}"]


def parse_brief(text):
    """One story beat per line.

    plain line            -> a chapter (weight 1)
    line | 2              -> a chapter with relative length 2
    TITLE: WORDS | 2      -> a black title card lasting 2 bars (default 1)
    # anything            -> comment
    """
    chapters = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        weight = None
        m = re.match(r"^(.*?)\s*\|\s*([0-9]*\.?[0-9]+)\s*$", line)
        if m:
            line, weight = m.group(1).strip(), float(m.group(2))
        if line.upper().startswith("TITLE:"):
            chapters.append({"kind": "title", "text": line[6:].strip(), "bars": weight or 1.0})
        else:
            chapters.append({"kind": "story", "text": line, "weight": weight or 1.0})
    if not any(c["kind"] == "story" for c in chapters):
        chapters.append({"kind": "story", "text": "", "weight": 1.0})
    return chapters


_MODEL_CACHE = {}


def _device(pref):
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip(name, device, models_dir):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    key = (name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    local = name if os.path.isdir(name) else os.path.join(models_dir, name.replace("/", "--"))
    src = local if os.path.isdir(local) else name
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(src, dtype=dtype, cache_dir=os.path.join(models_dir, "hf_cache"))
    proc = CLIPProcessor.from_pretrained(src, cache_dir=os.path.join(models_dir, "hf_cache"))
    model = model.to(device).eval()
    if src == name and not os.path.isdir(local):
        try:  # keep a plain copy in ComfyUI/models/i2_cutter for offline use
            model.save_pretrained(local)
            proc.save_pretrained(local)
        except Exception:
            pass
    _MODEL_CACHE[key] = (model, proc)
    return model, proc


def unload_all():
    _MODEL_CACHE.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _as_tensor(out):
    if hasattr(out, "shape"):
        return out
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, k, None)
        if v is not None:
            return v
    return out[0]


def embed_texts(model, proc, texts, device):
    import torch
    with torch.no_grad():
        tok = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        e = _as_tensor(model.get_text_features(**tok)).float()
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def embed_images(model, proc, images, device, batch=32, progress=None):
    import torch
    out = []
    for i in range(0, len(images), batch):
        with torch.no_grad():
            px = proc(images=images[i:i + batch], return_tensors="pt")["pixel_values"]
            px = px.to(device, dtype=next(model.parameters()).dtype)
            e = _as_tensor(model.get_image_features(pixel_values=px)).float()
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        if progress:
            progress(min(len(images), i + batch), len(images))
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def score(bin_, brief, negative="", use_clip=True, model_name=CLIP_MODELS[0], device="auto",
          negative_strength=0.5, models_dir="", keep_loaded=False, progress=None):
    chapters = parse_brief(brief)
    story = [c for c in chapters if c["kind"] == "story"]
    shots = bin_["shots"]
    n, k = len(shots), len(story)
    res = {"chapters": chapters, "z": np.zeros((n, k), np.float32), "raw": np.zeros((n, k), np.float32),
           "neg": np.zeros(n, np.float32), "emb": None, "model": None}
    if not use_clip or n == 0 or all(not c["text"] for c in story):
        return res
    dev = _device(device)
    model, proc = load_clip(model_name, dev, models_dir)
    imgs = []
    for s in shots:
        for j in range(len(s["thumbs"])):
            imgs.append(shotlib.shot_thumb(bin_, s, j))
    per = [len(s["thumbs"]) for s in shots]
    ie = embed_images(model, proc, imgs, dev, progress=progress)
    shot_emb, o = [], 0
    for p in per:
        e = ie[o:o + p].mean(axis=0)
        shot_emb.append(e / (np.linalg.norm(e) + 1e-8))
        o += p
    shot_emb = np.stack(shot_emb)
    te = []
    for c in story:
        t = embed_texts(model, proc, [tpl.format(c["text"]) for tpl in TEMPLATES], dev).mean(axis=0)
        te.append(t / (np.linalg.norm(t) + 1e-8))
    te = np.stack(te)
    raw = shot_emb @ te.T
    z = (raw - raw.mean(axis=0, keepdims=True)) / (raw.std(axis=0, keepdims=True) + 1e-6)
    negs = [x.strip() for x in re.split(r"[\n,]", negative or "") if x.strip()]
    neg = np.zeros(n, np.float32)
    if negs:
        ne = embed_texts(model, proc, [f"a film still of {x}" for x in negs], dev)
        nraw = (shot_emb @ ne.T).max(axis=1)
        nz = (nraw - nraw.mean()) / (nraw.std() + 1e-6)
        neg = np.maximum(0, nz) * negative_strength
    res.update(z=(z - neg[:, None]).astype(np.float32), raw=raw.astype(np.float32), neg=neg,
               emb=shot_emb.astype(np.float32), model=model_name)
    if not keep_loaded:
        unload_all()
    return res


def report(bin_, res, top=3):
    story = [c for c in res["chapters"] if c["kind"] == "story"]
    lines = []
    if res["model"] is None:
        lines.append("CLIP off: the brief sets the structure only; shots are chosen by motion and variety.")
    for ci, c in enumerate(story):
        lines.append(f"[{ci + 1}] {c['text'] or '(no text)'}")
        if res["model"] is None:
            continue
        order = np.argsort(-res["z"][:, ci])[:top]
        for r in order:
            s = bin_["shots"][r]
            lines.append(f"     {res['z'][r, ci]:+.2f}  {s['name']}  {s['start']:.2f}-{s['end']:.2f}s")
    return "\n".join(lines)

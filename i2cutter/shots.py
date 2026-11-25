"""Footage analysis: shot detection, motion and brightness per shot."""
import hashlib
import json
import os

import numpy as np
from PIL import Image

from . import media

FEATURE_VERSION = 2
THUMB_W = 256


def _cache_path(cache_dir, path, analysis_fps):
    st = os.stat(path)
    key = f"{FEATURE_VERSION}|{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{analysis_fps:.3f}"
    return os.path.join(cache_dir, hashlib.sha1(key.encode()).hexdigest()[:20] + ".npz")


def analyze_file(path, analysis_fps, cache_dir=None):
    """Per-sample features for one file. Cached on disk so re-runs are instant."""
    cp = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cp = _cache_path(cache_dir, path, analysis_fps)
        if os.path.exists(cp):
            try:
                z = np.load(cp, allow_pickle=False)
                blob, offs = z["thumbs"].tobytes(), z["offsets"]
                thumbs = [blob[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
                info = json.loads(str(z["info"]))
                return info, {k: z[k] for k in ("times", "d_hist", "d_pix", "luma", "flat")}, thumbs
            except Exception:
                pass
    info = media.probe(path)
    if not info.get("has_video"):
        return info, None, []
    times, d_hist, d_pix, luma, flat, thumbs = [], [], [], [], [], []
    prev_g = prev_h = None
    for t, img in media.iter_samples(path, 1.0 / analysis_fps, THUMB_W):
        small = np.asarray(Image.fromarray(img).resize((64, 36), Image.BILINEAR), np.float32)
        g = small @ np.array([0.299, 0.587, 0.114], np.float32)
        q = (small // 16).astype(np.int32)  # 16 levels per channel
        hist = np.concatenate([np.bincount(q[..., c].ravel(), minlength=16) for c in range(3)])
        hist = hist / hist.sum()
        times.append(t)
        luma.append(g.mean() / 255.0)
        flat.append(g.std() / 255.0)
        d_hist.append(0.0 if prev_h is None else 0.5 * np.abs(hist - prev_h).sum())
        d_pix.append(0.0 if prev_g is None else float(np.abs(g - prev_g).mean() / 255.0))
        prev_g, prev_h = g, hist
        thumbs.append(media.jpeg_bytes(img))
    feat = {"times": np.array(times, np.float64), "d_hist": np.array(d_hist, np.float32),
            "d_pix": np.array(d_pix, np.float32), "luma": np.array(luma, np.float32),
            "flat": np.array(flat, np.float32)}
    if cp and times:
        offs = np.cumsum([0] + [len(b) for b in thumbs]).astype(np.int64)
        np.savez(cp, thumbs=np.frombuffer(b"".join(thumbs), np.uint8), offsets=offs,
                 info=json.dumps(info), **feat)
    return info, feat, thumbs


def detect_cuts(feat, sensitivity, min_shot_sec):
    """Return sample indices where a new shot starts (always includes 0)."""
    n = len(feat["times"])
    if n < 3:
        return [0]
    c = 0.65 * feat["d_hist"] + 0.35 * np.clip(feat["d_pix"] * 4.0, 0, 1)
    c[0] = 0
    med = float(np.median(c[1:]))
    mad = float(np.median(np.abs(c[1:] - med))) * 1.4826 + 1e-4
    k = 9.0 - 6.5 * sensitivity
    floor = 0.40 - 0.28 * sensitivity
    thr = max(med + k * mad, floor)
    cand = [i for i in range(1, n) if c[i] > thr and c[i] >= c[i - 1] and (i + 1 >= n or c[i] >= c[i + 1])]
    cuts = [0]
    t = feat["times"]
    for i in sorted(cand, key=lambda i: -c[i]):
        if all(abs(t[i] - t[j]) >= min_shot_sec for j in cuts):
            cuts.append(i)
    return sorted(cuts)


def build_shots(paths, analysis_fps=4.0, sensitivity=0.5, min_shot_sec=0.8, handle_sec=0.1,
                exclude_black=True, cache_dir=None, progress=None):
    files, shots = {}, []
    for fi, path in enumerate(paths):
        try:
            info, feat, thumbs = analyze_file(path, analysis_fps, cache_dir)
        except Exception as e:  # unreadable file: skip, but say so
            print(f"[i2 Cutter] skipped {path}: {e}")
            if progress:
                progress(fi + 1, len(paths))
            continue
        if progress:
            progress(fi + 1, len(paths))
        if feat is None or len(feat["times"]) == 0:
            continue
        files[path] = {"info": info, "feat": feat, "thumbs": thumbs}
        cuts = detect_cuts(feat, sensitivity, min_shot_sec) + [len(feat["times"])]
        t = feat["times"]
        step = 1.0 / analysis_fps
        dur = info.get("duration") or (t[-1] + step)
        for a, b in zip(cuts[:-1], cuts[1:]):
            start = 0.0 if a == 0 else t[a]
            end = dur if b >= len(t) else t[b - 1] + step * 0.5
            start_u, end_u = start + handle_sec, end - handle_sec
            if end_u - start_u < 0.2:
                continue
            idx = np.arange(a, b)
            motion = feat["d_pix"][idx].copy()
            if len(motion):
                motion[0] = motion[1] if len(motion) > 1 else 0.0
            lum = float(feat["luma"][idx].mean())
            flat = float(feat["flat"][idx].mean())
            black = lum < 0.035 or (flat < 0.012 and lum < 0.15)
            if exclude_black and black:
                continue
            q = [idx[int(len(idx) * f)] for f in (0.25, 0.5, 0.75)] if len(idx) else []
            shots.append({
                "id": len(shots), "file": path, "name": info["name"],
                "start": float(start_u), "end": float(end_u), "dur": float(end_u - start_u),
                "s0": int(a), "s1": int(b), "motion_mean": float(motion.mean()) if len(motion) else 0.0,
                "luma": lum, "thumbs": [int(i) for i in q],
            })
    # global motion normalisation (rank 0..1 across the bin)
    if shots:
        allm = np.concatenate([files[s["file"]]["feat"]["d_pix"][s["s0"]:s["s1"]] for s in shots])
        scale = float(np.percentile(allm, 95)) + 1e-6
        means = np.array([s["motion_mean"] for s in shots])
        ranks = means.argsort().argsort() / max(1, len(shots) - 1)
        for s, r in zip(shots, ranks):
            s["motion_rank"] = float(r) if len(shots) > 1 else 0.5
    else:
        scale = 1.0
    return {"files": files, "shots": shots, "analysis_fps": analysis_fps, "motion_scale": scale}


def shot_thumb(bin_, shot, which=1):
    f = bin_["files"][shot["file"]]
    idx = shot["thumbs"][min(which, len(shot["thumbs"]) - 1)]
    return media.jpeg_array(f["thumbs"][idx])


def thumb_at(bin_, shot, t):
    f = bin_["files"][shot["file"]]
    times = f["feat"]["times"]
    i = int(np.clip(np.searchsorted(times, t), shot["s0"], shot["s1"] - 1))
    return media.jpeg_array(f["thumbs"][i])

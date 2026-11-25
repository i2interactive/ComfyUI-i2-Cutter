"""The editor: lays shots onto the music's beat grid, chapter by chapter."""
from collections import Counter

import numpy as np

from . import media

STYLES = ["trailer", "music video", "montage", "slow burn"]


def _grid(music):
    """Cut grid: timeline start, every beat inside the segment, timeline end.

    If the first beat lands right after the start it becomes the start, so the
    grid stays in step with the bars instead of gaining a stray pre-roll beat.
    """
    L = music["length"]
    period = 60.0 / max(1e-3, music["bpm"])
    pts, down, en, hit = [], [], [], []
    for t, d, ee, h in zip(music["beats"], music["downbeat"], music["energy"], music["hits"]):
        if 0.0 <= t < L - 0.12:
            pts.append(t)
            down.append(bool(d))
            en.append(float(ee))
            hit.append(bool(h))
    if pts and pts[0] < period:
        pts[0] = 0.0
    else:
        pts.insert(0, 0.0)
        down.insert(0, True)
        en.insert(0, en[0] if en else 0.5)
        hit.insert(0, False)
    pts.append(L)
    down.append(True)
    en.append(en[-1])
    hit.append(False)
    return np.array(pts), np.array(down), np.array(en), np.array(hit)


def _regions(chapters, down, bpb):
    """Split grid points into chapter / title regions, snapped to bar lines."""
    n = len(down) - 1  # number of grid intervals
    titles = [c for c in chapters if c["kind"] == "title"]
    title_beats = sum(int(round(c["bars"] * bpb)) for c in titles)
    story_beats = max(1, n - title_beats)
    wsum = sum(c["weight"] for c in chapters if c["kind"] == "story") or 1.0
    downs = np.flatnonzero(down)
    regions, i, story_i = [], 0, 0
    acc = 0.0
    for ci, c in enumerate(chapters):
        if c["kind"] == "title":
            j = i + int(round(c["bars"] * bpb))
        else:
            acc += c["weight"] / wsum * story_beats
            j = int(round(acc + sum(int(round(t["bars"] * bpb)) for t in chapters[:ci] if t["kind"] == "title")))
        is_last = ci == len(chapters) - 1
        if is_last:
            j = n
        else:
            near = downs[(downs > i) & (downs < n)]
            if len(near):
                j = int(near[np.argmin(np.abs(near - j))])
            j = int(np.clip(j, i + 1, n - (len(chapters) - ci - 1)))
        regions.append({"chapter": ci, "story": story_i if c["kind"] == "story" else None,
                        "kind": c["kind"], "text": c["text"], "a": i, "b": j})
        if c["kind"] == "story":
            story_i += 1
        i = j
    return [r for r in regions if r["b"] > r["a"]]


def _beats_for(e, lo, hi):
    allowed = [b for b in (1, 2, 4, 8, 16, 32) if lo <= b <= hi] or [lo]
    x = hi * (lo / hi) ** e  # e=0 -> hi, e=1 -> lo
    return min(allowed, key=lambda b: abs(np.log2(b) - np.log2(x)))


def _effective_energy(e, progress, style, response):
    if style == "trailer":
        v = 0.65 * e + 0.35 * progress ** 1.5
    elif style == "music video":
        v = e
    elif style == "montage":
        v = 0.5 * e + 0.25
    else:  # slow burn
        v = 0.55 * e * (0.4 + 0.6 * progress)
    return float(np.clip(0.5 + (v - 0.5) * (0.4 + 1.6 * response), 0, 1))


def plan(bin_, music, scores=None, style="trailer", min_beats=1, max_beats=8, energy_response=0.6,
         story_weight=1.0, motion_match=0.5, variety=0.15, hit_cuts=True, allow_reuse=False,
         min_cut_sec=0.4, seq_rate="auto", seed=0):
    rng = np.random.default_rng(seed)
    shots = bin_["shots"]
    if not shots:
        raise ValueError("i2 Cutter: the footage bin has no usable shots")
    bpb = music["beats_per_bar"]
    chapters = scores["chapters"] if scores else [{"kind": "story", "text": "", "weight": 1.0}]
    z = scores["z"] if scores is not None else np.zeros((len(shots), 1), np.float32)
    emb = scores.get("emb") if scores else None
    g, down, en, hit = _grid(music)
    regions = _regions(chapters, down, bpb)
    ngrid = len(g) - 1

    # timeline rate
    if seq_rate == "auto":
        rates = Counter(bin_["files"][s["file"]]["info"]["rate"]["label"] for s in shots)
        rate = media.rate_from_label(rates.most_common(1)[0][0])
    else:
        rate = media.rate_from_label(seq_rate)
    fps = rate["fps"]
    frame = lambda t: int(round(t * fps))
    files = bin_["files"]
    w = Counter((files[s["file"]]["info"]["width"], files[s["file"]]["info"]["height"]) for s in shots)
    width, height = w.most_common(1)[0][0]

    used = {s["id"]: [] for s in shots}
    events, markers, log = [], [], []
    prev = None
    mscale = bin_["motion_scale"]
    afps = bin_["analysis_fps"]

    def window(s, dur, target):
        lo, hi = s["start"], s["end"] - dur
        if hi < lo - 1e-6:
            return None
        f = files[s["file"]]["feat"]
        t, m = f["times"], np.clip(f["d_pix"] / mscale, 0, 1)
        starts = np.unique(np.clip(np.concatenate([[lo, (lo + hi) / 2, hi], t[(t >= lo) & (t <= hi)]]), lo, hi))
        best, best_v = None, -1e9
        for st in starts:
            en_ = st + dur
            if any(st < b + 0.25 and en_ > a - 0.25 for a, b in used[s["id"]]):
                continue
            sel = (t >= st) & (t <= en_)
            wm = float(m[sel].mean()) if sel.any() else float(np.interp(st, t, m))
            centre = 1 - abs((st - lo) / max(1e-6, hi - lo) - 0.5) if hi > lo else 1
            v = -abs(wm - target) + 0.15 * centre
            if v > best_v:
                best, best_v = st, v
        return best

    def choose(story_i, dur, target):
        cands = []
        for s in shots:
            if used[s["id"]] and not allow_reuse:
                continue
            if s["dur"] + 1e-6 < dur:
                continue
            st = window(s, dur, target)
            if st is None:
                continue
            sc = 0.0
            reasons = []
            if story_i is not None and scores is not None and scores.get("model"):
                zz = float(z[s["id"], story_i])
                sc += story_weight * np.tanh(zz / 2)
                reasons.append(f"brief {zz:+.2f}")
            mm = 1 - abs(s["motion_rank"] - target)
            sc += motion_match * mm
            reasons.append(f"motion {s['motion_rank']:.2f}/{target:.2f}")
            if used[s["id"]]:
                sc -= 0.35 * len(used[s["id"]])
                reasons.append("reused")
            if prev is not None:
                if prev["file"] == s["file"]:
                    sc -= 0.4
                if emb is not None:
                    sim = float(emb[prev["id"]] @ emb[s["id"]])
                    sc -= 0.8 * max(0.0, sim - 0.85) / 0.15
            sc += variety * rng.standard_normal()
            cands.append((sc, s, st, reasons))
        if not cands:
            return None
        cands.sort(key=lambda c: -c[0])
        return cands[0]

    for r in regions:
        a, b = r["a"], r["b"]
        markers.append({"frame": frame(g[a]), "name": (r["text"] or "cut")[:60] if r["kind"] == "story"
                        else f"TITLE: {r['text']}", "color": "title" if r["kind"] == "title" else "chapter"})
        if r["kind"] == "title":
            events.append({"type": "gap", "title": r["text"], "rec_in": frame(g[a]), "rec_out": frame(g[b]),
                           "chapter": r["chapter"]})
            log.append(f"{media.frames_to_tc(frame(g[a]), rate['timebase'])}  TITLE  {r['text']}")
            prev = None
            continue
        i = a
        weak = 0
        while i < b:
            prog = i / max(1, ngrid)
            target = _effective_energy(float(en[i:min(b, i + 2)].mean()), prog, style, energy_response)
            L = _beats_for(target, min_beats, max_beats)
            if L >= bpb and not down[i]:
                nxt = np.flatnonzero(down[i + 1:b + 1])
                if len(nxt) and nxt[0] + 1 >= min_beats:
                    L = int(nxt[0] + 1)
            j = min(i + L, b)
            if hit_cuts:
                hs = [h for h in range(i + min_beats, j) if hit[h]]
                if hs:
                    j = hs[0]
            if b - j < max(min_beats, (j - i) // 2):
                j = b  # fold a short leftover into this cut rather than leave an orphan
            while j < b and g[j] - g[i] < min_cut_sec:
                j += 1
            dur = g[j] - g[i]
            pick = choose(r["story"], dur, target)
            if pick is None:  # nothing long enough: take the longest unused piece and shorten the slot
                pool = [s for s in shots if allow_reuse or not used[s["id"]]] or shots
                s = max(pool, key=lambda s: s["dur"])
                jj = i + 1
                while jj < j and g[jj + 1] - g[i] <= s["dur"]:
                    jj += 1
                j, dur = jj, min(g[jj] - g[i], s["dur"])
                if allow_reuse or not used[s["id"]]:
                    st = window(s, dur, target)
                else:
                    st = None
                st = s["start"] if st is None else st
                pick = (0.0, s, st, ["fallback: longest left"])
            sc, s, st, reasons = pick
            used[s["id"]].append((st, st + dur))
            if scores is not None and scores.get("model") and r["story"] is not None \
                    and z[s["id"], r["story"]] < 0.5:
                weak += 1
            rec_in, rec_out = frame(g[i]), frame(g[j])
            events.append({"type": "clip", "shot": s["id"], "file": s["file"], "name": s["name"],
                           "src_in": float(st), "src_out": float(st + (rec_out - rec_in) / fps),
                           "rec_in": rec_in, "rec_out": rec_out, "chapter": r["chapter"],
                           "story": r["story"], "beats": int(j - i), "energy": round(target, 2),
                           "score": round(float(sc), 3), "why": ", ".join(reasons)})
            log.append(f"{media.frames_to_tc(rec_in, rate['timebase'])}  {j - i:>2} beats  "
                       f"{s['name']} @ {st:.2f}s  ({', '.join(reasons)})")
            prev = s
            i = j
        nslots = sum(1 for e in events if e.get("chapter") == r["chapter"])
        if scores is not None and scores.get("model") and nslots and weak / nslots > 0.5:
            markers.append({"frame": frame(g[a]) + 1, "name": f"NEEDS SHOTS: {r['text']}"[:60],
                            "color": "warning",
                            "comment": f"{weak} of {nslots} cuts here match this line weakly. Generate or shoot more."})

    total = frame(g[-1])
    return {"rate": rate, "width": int(width), "height": int(height), "duration": total,
            "events": events, "markers": markers, "chapters": chapters, "music": {
                "path": music["path"], "start": music["start"], "length": music["length"],
                "bpm": music["bpm"], "info": music["info"]},
            "style": style, "log": log,
            "files_info": {p: files[p]["info"] for p in {e["file"] for e in events if e["type"] == "clip"}}}


def summary(cut):
    tb = cut["rate"]["timebase"]
    clips = [e for e in cut["events"] if e["type"] == "clip"]
    lines = [f"{len(clips)} cuts, {len(set(e['shot'] for e in clips))} shots, "
             f"{media.frames_to_tc(cut['duration'], tb)} at {cut['rate']['label']} fps, "
             f"music {cut['music']['bpm']:.1f} BPM, style {cut['style']}"]
    lines += [m["name"] for m in cut["markers"] if m["color"] == "warning"]
    lines.append("")
    lines += cut["log"]
    return "\n".join(lines)

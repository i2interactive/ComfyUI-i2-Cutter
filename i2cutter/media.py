"""Media helpers built on PyAV (ships with ComfyUI), so no ffmpeg CLI is needed."""
import io
import os
from fractions import Fraction

import av
import numpy as np
from PIL import Image

VIDEO_EXTS = ("mp4", "mov", "mxf", "mkv", "avi", "m4v", "webm", "mts", "m2ts", "mpg", "mpeg")

# Standard timeline rates: (fps, timebase, ntsc, rational)
STANDARD_RATES = [
    (23.976, 24, True, Fraction(24000, 1001)),
    (24.0, 24, False, Fraction(24, 1)),
    (25.0, 25, False, Fraction(25, 1)),
    (29.97, 30, True, Fraction(30000, 1001)),
    (30.0, 30, False, Fraction(30, 1)),
    (48.0, 48, False, Fraction(48, 1)),
    (50.0, 50, False, Fraction(50, 1)),
    (59.94, 60, True, Fraction(60000, 1001)),
    (60.0, 60, False, Fraction(60, 1)),
]


def snap_rate(fps):
    """Snap a measured frame rate to the nearest standard timeline rate."""
    best = min(STANDARD_RATES, key=lambda r: abs(r[0] - fps))
    return {"fps": float(best[3]), "timebase": best[1], "ntsc": best[2],
            "num": best[3].numerator, "den": best[3].denominator, "label": f"{best[0]:g}"}


def rate_from_label(label):
    for r in STANDARD_RATES:
        if f"{r[0]:g}" == label:
            return snap_rate(r[0])
    return snap_rate(float(label))


# ---------------------------------------------------------------- timecode
def tc_to_frames(tc, timebase):
    """Parse 'HH:MM:SS:FF' or drop-frame 'HH:MM:SS;FF' into a frame count."""
    if not tc:
        return 0, False
    drop = ";" in tc or "." in tc
    parts = tc.replace(";", ":").replace(".", ":").split(":")
    try:
        h, m, s, f = [int(p) for p in parts[-4:]]
    except ValueError:
        return 0, False
    total = ((h * 60 + m) * 60 + s) * timebase + f
    if drop and timebase in (30, 60):
        d = 2 if timebase == 30 else 4
        mins = h * 60 + m
        total -= d * (mins - mins // 10)
    return total, drop


def frames_to_tc(frames, timebase, drop=False):
    frames = int(round(frames))
    sep = ":"
    if drop and timebase in (30, 60):
        sep = ";"
        d = 2 if timebase == 30 else 4
        per10 = timebase * 600 - d * 9
        per1 = timebase * 60 - d
        tens, rem = divmod(frames, per10)
        if rem > d:
            frames += d * 9 * tens + d * ((rem - d) // per1)
        else:
            frames += d * 9 * tens
    f = frames % timebase
    s = (frames // timebase) % 60
    m = (frames // (timebase * 60)) % 60
    h = frames // (timebase * 3600)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{f:02d}"


# ---------------------------------------------------------------- probing
def resolve_input(path):
    """A relative path that isn't there as given is looked for in ComfyUI's input
    folder (where the Creative Hub and the ComfyUI upload button put files)."""
    p = os.path.expanduser(str(path or "").strip().strip('"').strip("'"))
    if not p or os.path.isabs(p) or os.path.exists(p):
        return os.path.abspath(p) if p else p
    try:
        import folder_paths  # noqa: WPS433
        base = folder_paths.get_input_directory()
    except Exception:
        return os.path.abspath(p)
    rel = p.replace("\\", "/")
    if rel.lower().startswith("input/"):
        rel = rel[6:]
    cand = os.path.normpath(os.path.join(base, rel))
    return cand if os.path.exists(cand) else os.path.abspath(p)


def list_media(folder, exts, recursive):
    folder = resolve_input(folder)
    if os.path.isfile(folder):
        return [folder]
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"i2 Cutter: folder not found: {folder}")
    exts = tuple("." + e.strip().lower().lstrip(".") for e in exts.split(",") if e.strip())
    out = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            for f in sorted(files):
                if f.lower().endswith(exts) and not f.startswith("._"):
                    out.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(folder)):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and f.lower().endswith(exts) and not f.startswith("._"):
                out.append(p)
    return out


def probe(path):
    with av.open(path) as c:
        v = next((s for s in c.streams if s.type == "video"), None)
        a = next((s for s in c.streams if s.type == "audio"), None)
        info = {"path": path, "name": os.path.basename(path), "has_video": v is not None,
                "has_audio": a is not None}
        tc = c.metadata.get("timecode")
        for s in c.streams:
            tc = tc or s.metadata.get("timecode")
        if v is not None:
            rate = v.average_rate or v.guessed_rate or v.base_rate or Fraction(24, 1)
            fps = float(rate)
            if v.duration is not None and v.time_base is not None:
                dur = float(v.duration * v.time_base)
            elif c.duration:
                dur = c.duration / 1_000_000
            else:
                dur = 0.0
            info.update(fps=fps, rate=snap_rate(fps), duration=dur,
                        frames=int(v.frames or round(dur * fps)),
                        width=v.codec_context.width, height=v.codec_context.height)
        else:
            dur = (c.duration or 0) / 1_000_000
            info.update(duration=dur)
        if a is not None:
            info.update(audio_rate=a.codec_context.sample_rate or 48000,
                        audio_channels=a.codec_context.channels or 2)
            if not info.get("duration") and a.duration is not None and a.time_base is not None:
                info["duration"] = float(a.duration * a.time_base)
        tb = info.get("rate", snap_rate(24))["timebase"]
        info["tc_frames"], info["tc_drop"] = tc_to_frames(tc, tb)
        info["timecode"] = tc or frames_to_tc(0, tb)
        return info


# ---------------------------------------------------------------- decoding
def _start_offset(stream):
    if stream.start_time is not None and stream.time_base is not None:
        return float(stream.start_time * stream.time_base)
    return 0.0


def iter_samples(path, step_sec, width):
    """Yield (t, rgb uint8 array) roughly every step_sec, scaled to `width` px wide."""
    with av.open(path) as c:
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        t0 = _start_offset(v)
        next_t = -1.0
        h = None
        for frame in c.decode(v):
            if frame.pts is None:
                continue
            t = float(frame.pts * v.time_base) - t0
            if t + 1e-6 < next_t:
                continue
            if h is None:
                h = max(2, int(round(width * frame.height / max(1, frame.width) / 2)) * 2)
            img = frame.reformat(width=width, height=h, format="rgb24").to_ndarray()
            yield t, img
            if next_t < 0 or t - next_t > step_sec:
                next_t = t + step_sec
            else:
                next_t += step_sec


class FrameReader:
    """Pull frames at a fixed output rate from a time range of a clip."""

    def __init__(self, path):
        self.c = av.open(path)
        self.v = self.c.streams.video[0]
        self.v.thread_type = "AUTO"
        self.t0 = _start_offset(self.v)

    def frames(self, start, count, fps, size):
        w, h = size
        tb = self.v.time_base
        self.c.seek(int((start + self.t0) / tb), stream=self.v, backward=True, any_frame=False)
        decoder = self.c.decode(self.v)
        last = None
        pending = None
        for k in range(count):
            target = start + k / fps
            while True:
                if pending is None:
                    try:
                        fr = next(decoder)
                    except StopIteration:
                        break
                    if fr.pts is None:
                        continue
                    pending = (float(fr.pts * tb) - self.t0, fr)
                if pending[0] <= target + 0.5 / fps:
                    last = pending[1]
                    pending = None
                else:
                    break
            if last is None and pending is not None:
                last = pending[1]
            if last is None:
                yield np.zeros((h, w, 3), np.uint8)
            else:
                yield letterbox(last, w, h)

    def close(self):
        self.c.close()


def letterbox(frame, w, h):
    fw, fh = frame.width, frame.height
    scale = min(w / fw, h / fh)
    nw = max(2, int(fw * scale) // 2 * 2)
    nh = max(2, int(fh * scale) // 2 * 2)
    img = frame.reformat(width=nw, height=nh, format="rgb24").to_ndarray()
    if nw == w and nh == h:
        return img
    canvas = np.zeros((h, w, 3), np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    canvas[y:y + nh, x:x + nw] = img
    return canvas


def letterbox_array(img, w, h):
    pil = Image.fromarray(img)
    scale = min(w / pil.width, h / pil.height)
    nw, nh = max(1, int(pil.width * scale)), max(1, int(pil.height * scale))
    pil = pil.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (w, h))
    canvas.paste(pil, ((w - nw) // 2, (h - nh) // 2))
    return np.asarray(canvas)


def load_audio_mono(path, sr=22050, start=0.0, length=0.0):
    """Decode an audio file (or a video's audio) to mono float32 at `sr`."""
    chunks = []
    with av.open(path) as c:
        a = c.streams.audio[0]
        res = av.AudioResampler(format="flt", layout="mono", rate=sr)
        if start > 0:
            c.seek(int(max(0.0, start - 1.0) / a.time_base) + (a.start_time or 0),
                   stream=a, backward=True)
        first_t = None
        for frame in c.decode(a):
            if first_t is None and frame.pts is not None:
                first_t = float(frame.pts * a.time_base) - _start_offset(a)
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
            if length > 0 and first_t is not None:
                got = sum(len(x) for x in chunks) / sr
                if first_t + got > start + length + 1.0:
                    break
        for f in res.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    y = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, np.float32)
    first_t = first_t or 0.0
    i0 = max(0, int(round((start - first_t) * sr)))
    y = y[i0:]
    if length > 0:
        y = y[:int(round(length * sr))]
    return y


def jpeg_bytes(img, quality=85):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def jpeg_array(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))

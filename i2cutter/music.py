"""Music analysis: tempo, beats, bars, accents and an energy curve (numpy only)."""
import numpy as np

from . import media

SR = 22050
HOP = 512
NFFT = 2048


def _stft_mag(y):
    y = np.pad(y, (NFFT // 2, NFFT // 2))  # centred frames: frame i is at i * HOP
    if len(y) < NFFT:
        y = np.pad(y, (0, NFFT - len(y)))
    n = 1 + (len(y) - NFFT) // HOP
    win = np.hanning(NFFT).astype(np.float32)
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = y[idx] * win
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)  # [n, bins]


def _band_matrix(n_bins, n_bands=40, fmin=30.0, fmax=11000.0):
    """Log-spaced triangular bands (mel-like) for a cheaper, more musical flux."""
    freqs = np.linspace(0, SR / 2, n_bins)
    edges = np.geomspace(fmin, fmax, n_bands + 2)
    m = np.zeros((n_bins, n_bands), np.float32)
    for b in range(n_bands):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        m[:, b] = np.clip(np.minimum(up, down), 0, None)
    return m, edges


def onset_envelope(y):
    mag = _stft_mag(y)
    bands, edges = _band_matrix(mag.shape[1])
    s = np.log1p(100.0 * (mag @ bands))
    flux = np.maximum(0.0, np.diff(s, axis=0, prepend=s[:1]))
    env = flux.mean(axis=1)
    low_bands = (edges[1:-1] < 160).sum()
    low = flux[:, :max(1, low_bands)].mean(axis=1)
    # remove slow trend, keep transients
    k = int(0.4 * SR / HOP) | 1
    trend = np.convolve(env, np.ones(k) / k, mode="same")
    env = np.maximum(0.0, env - trend)
    env = env / (env.std() + 1e-8)
    rms = np.sqrt((mag ** 2).mean(axis=1))
    return env.astype(np.float32), low.astype(np.float32), rms.astype(np.float32)


def estimate_tempo(env, bpm_hint=0.0, lo=60.0, hi=200.0):
    fps = SR / HOP
    e = env - env.mean()
    n = len(e)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, nfft)) ** 2)[:n]
    ac = ac / (ac[0] + 1e-8)
    lags = np.arange(1, n)
    bpms = 60.0 * fps / lags
    ok = (bpms >= lo) & (bpms <= hi)
    center = bpm_hint if bpm_hint > 0 else 115.0
    width = 0.35 if bpm_hint > 0 else 1.0
    prior = np.exp(-0.5 * (np.log2(bpms / center) / width) ** 2)
    score = np.where(ok, ac[1:] * prior, -np.inf)
    # also reward lags whose double is strong (helps half/double confusion)
    best = int(np.argmax(score)) + 1
    # parabolic refinement
    if 1 < best < n - 1:
        a, b, c = ac[best - 1], ac[best], ac[best + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        best = best + float(np.clip(d, -0.5, 0.5))
    return 60.0 * fps / best


def track_beats(env, bpm, tightness=120.0):
    """Ellis (2007) dynamic-programming beat tracker."""
    fps = SR / HOP
    period = 60.0 * fps / bpm
    n = len(env)
    # smooth onset with a gaussian matched to the beat period
    w = np.exp(-0.5 * (np.arange(-period, period + 1) * 32.0 / period) ** 2)
    local = np.convolve(env, w, "same")
    score = local.copy()
    back = -np.ones(n, np.int64)
    lo, hi = int(round(period / 2)), int(round(2 * period))
    offsets = np.arange(-hi, -lo + 1)
    txcost = -tightness * np.log(-offsets / period) ** 2
    for i in range(n):
        cand = i + offsets
        valid = cand >= 0
        if not valid.any():
            continue
        vals = np.where(valid, score[np.clip(cand, 0, None)] + txcost, -np.inf)
        j = int(np.argmax(vals))
        if vals[j] > 0:
            score[i] = local[i] + vals[j]
            back[i] = cand[j]
    # start from the best score in the last period
    tail = max(1, int(period))
    i = n - tail + int(np.argmax(score[-tail:]))
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    beats = np.array(beats[::-1])
    # drop weak beats at the very edges (silence)
    thr = 0.5 * np.median(local[beats]) if len(beats) else 0
    while len(beats) > 2 and local[beats[0]] < thr:
        beats = beats[1:]
    while len(beats) > 2 and local[beats[-1]] < thr:
        beats = beats[:-1]
    # spectral flux fires half a hop before the attack lands
    return beats / fps + 0.5 * HOP / SR


def analyze(path, start=0.0, length=0.0, bpm_hint=0.0, beats_per_bar=4):
    y = media.load_audio_mono(path, SR, start, length)
    dur = len(y) / SR
    if dur < 2.0:
        raise ValueError("i2 Cutter: the music is shorter than 2 seconds")
    env, low, rms = onset_envelope(y)
    bpm = estimate_tempo(env, bpm_hint)
    beats = track_beats(env, bpm)
    for _ in range(2):  # refine the period from the tracked beats, then track again
        if len(beats) < 8:
            break
        k = np.arange(len(beats))
        bpm = 60.0 / float(np.polyfit(k, beats, 1)[0])
        beats = track_beats(env, bpm)
    fps = SR / HOP

    def at(arr, t, win=0.05):
        a, b = int(max(0, (t - win) * fps)), int(min(len(arr), (t + win) * fps + 1))
        return float(arr[a:b].max()) if b > a else 0.0

    strength = np.array([at(env, t) for t in beats])
    lowk = np.array([at(low, t) for t in beats])
    # downbeat phase: the phase whose beats carry the most kick
    phase = 0
    if len(beats) >= beats_per_bar * 2:
        phase = int(np.argmax([lowk[p::beats_per_bar].mean() + 0.5 * strength[p::beats_per_bar].mean()
                               for p in range(beats_per_bar)]))
    downbeat = np.zeros(len(beats), bool)
    downbeat[phase::beats_per_bar] = True

    # loudness per beat, normalised 0..1 then smoothed over ~a bar
    rms_db = 20 * np.log10(rms + 1e-6)
    seg = [rms_db[int(a * fps):max(int(a * fps) + 1, int(b * fps))].mean()
           for a, b in zip(beats, np.append(beats[1:], dur))]
    seg = np.array(seg) if len(seg) else np.zeros(0)
    density = np.array([env[int(a * fps):max(int(a * fps) + 1, int(b * fps))].mean()
                        for a, b in zip(beats, np.append(beats[1:], dur))]) if len(beats) else np.zeros(0)

    def norm(x):
        if len(x) == 0:
            return x
        lo_, hi_ = np.percentile(x, 5), np.percentile(x, 95)
        return np.clip((x - lo_) / (hi_ - lo_ + 1e-6), 0, 1)

    energy = 0.6 * norm(seg) + 0.4 * norm(density)
    k = max(1, beats_per_bar)
    if len(energy) > k:
        ker = np.hanning(2 * k + 1)
        ker /= ker.sum()
        energy = np.convolve(np.pad(energy, k, mode="edge"), ker, "valid")
    hits = strength > (strength.mean() + 1.6 * strength.std()) if len(strength) else np.zeros(0, bool)

    # whole-track energy curve for drawing (10 Hz)
    curve_t = np.arange(0, dur, 0.1)
    curve = np.interp(curve_t, beats, energy) if len(beats) > 1 else np.zeros_like(curve_t)
    wave = np.array([np.abs(y[int(t * SR):int((t + 0.1) * SR)]).max() if int(t * SR) < len(y) else 0
                     for t in curve_t])
    return {
        "path": path, "start": float(start), "length": float(dur), "bpm": float(bpm),
        "beats_per_bar": int(beats_per_bar), "beats": beats.tolist(), "downbeat": downbeat.tolist(),
        "energy": energy.tolist(), "strength": strength.tolist(), "hits": hits.tolist(),
        "curve_t": curve_t.tolist(), "curve": curve.tolist(), "wave": wave.tolist(),
        "info": media.probe(path),
    }


def draw(music, w=1200, h=260):
    img = np.full((h, w, 3), 18, np.uint8)
    dur = music["length"]
    X = lambda t: int(np.clip(t / dur * (w - 1), 0, w - 1))
    wave = np.array(music["wave"])
    if len(wave):
        wave = wave / (wave.max() + 1e-6)
        for i, t in enumerate(music["curve_t"]):
            x = X(t)
            a = int(wave[i] * (h * 0.42))
            img[h // 2 - a:h // 2 + a + 1, x] = (70, 70, 80)
    for t, d, hit in zip(music["beats"], music["downbeat"], music["hits"]):
        x = X(t)
        if hit:
            img[:, x] = (230, 60, 50)
        elif d:
            img[: h // 3, x] = (240, 240, 240)
        else:
            img[: h // 8, x] = (130, 130, 130)
    prev = None
    for t, e in zip(music["curve_t"], music["curve"]):
        x, y_ = X(t), int((1 - e) * (h - 20)) + 10
        if prev:
            x0, y0 = prev
            for xx in range(min(x0, x), max(x0, x) + 1):
                yy = y0 if x == x0 else int(y0 + (y_ - y0) * (xx - x0) / (x - x0))
                img[max(0, yy - 1):yy + 2, xx] = (255, 200, 40)
        prev = (x, y_)
    return img

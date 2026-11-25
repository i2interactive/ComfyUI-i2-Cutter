"""Write the cut out: FCP 7 XML (Premiere + Resolve), CMX 3600 EDL, preview MP4, storyboard."""
import json
import os
import re
from fractions import Fraction
from urllib.parse import quote
from xml.sax.saxutils import escape

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import media
from . import shots as shotlib

RECORD_START_HOURS = 1  # timelines start at 01:00:00:00


def pathurl(path):
    p = os.path.abspath(path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", p):
        return "file://localhost/" + quote(p, safe="/:")
    return "file://localhost" + quote(p, safe="/")


def _rate_xml(rate, ind):
    return (f"{ind}<rate>\n{ind}  <timebase>{rate['timebase']}</timebase>\n"
            f"{ind}  <ntsc>{'TRUE' if rate['ntsc'] else 'FALSE'}</ntsc>\n{ind}</rate>\n")


def _tc_xml(rate, frames, ind, drop=False):
    tb = rate["timebase"]
    return (f"{ind}<timecode>\n{_rate_xml(rate, ind + '  ')}"
            f"{ind}  <string>{media.frames_to_tc(frames, tb, drop)}</string>\n"
            f"{ind}  <frame>{frames}</frame>\n"
            f"{ind}  <displayformat>{'DF' if drop else 'NDF'}</displayformat>\n{ind}</timecode>\n")


def write_fcp7_xml(cut, out_path, seq_name, source_audio=False):
    rate = cut["rate"]
    fps = rate["fps"]
    files_written = {}
    fid = lambda p: files_written.setdefault(p, f"file-{len(files_written) + 1}")
    infos = {}
    music = cut["music"]
    events = cut["events"]

    def file_xml(path, info, ind, kind="video"):
        new = path not in files_written
        id_ = fid(path)
        if not new:
            return f'{ind}<file id="{id_}"/>\n'
        mrate = info.get("rate", rate)
        dur = int(round((info.get("duration") or 0) * fps))
        s = f'{ind}<file id="{id_}">\n'
        s += f"{ind}  <name>{escape(info['name'])}</name>\n"
        s += f"{ind}  <pathurl>{escape(pathurl(path))}</pathurl>\n"
        s += _rate_xml(mrate, ind + "  ")
        s += f"{ind}  <duration>{dur}</duration>\n"
        tcf = info.get("tc_frames", 0)
        s += _tc_xml(mrate, tcf, ind + "  ", info.get("tc_drop", False))
        s += f"{ind}  <media>\n"
        if info.get("has_video"):
            s += (f"{ind}    <video>\n{ind}      <samplecharacteristics>\n"
                  f"{_rate_xml(mrate, ind + '        ')}"
                  f"{ind}        <width>{info.get('width', cut['width'])}</width>\n"
                  f"{ind}        <height>{info.get('height', cut['height'])}</height>\n"
                  f"{ind}        <anamorphic>FALSE</anamorphic>\n"
                  f"{ind}        <pixelaspectratio>square</pixelaspectratio>\n"
                  f"{ind}        <fielddominance>none</fielddominance>\n"
                  f"{ind}      </samplecharacteristics>\n{ind}    </video>\n")
        if info.get("has_audio"):
            s += (f"{ind}    <audio>\n{ind}      <samplecharacteristics>\n"
                  f"{ind}        <depth>16</depth>\n"
                  f"{ind}        <samplerate>{info.get('audio_rate', 48000)}</samplerate>\n"
                  f"{ind}      </samplecharacteristics>\n"
                  f"{ind}      <channelcount>{info.get('audio_channels', 2)}</channelcount>\n"
                  f"{ind}    </audio>\n")
        s += f"{ind}  </media>\n{ind}</file>\n"
        return s

    clip_n = [0]

    def clipitem(name, path, info, start, end, in_, ind, kind="video", track_index=1, links=None, cid=None):
        clip_n[0] += 1
        cid = cid or f"clipitem-{clip_n[0]}"
        dur = int(round((info.get("duration") or 0) * fps))
        s = f'{ind}<clipitem id="{cid}">\n'
        s += f"{ind}  <name>{escape(name)}</name>\n{ind}  <enabled>TRUE</enabled>\n"
        s += f"{ind}  <duration>{dur}</duration>\n"
        s += _rate_xml(rate, ind + "  ")
        s += (f"{ind}  <start>{start}</start>\n{ind}  <end>{end}</end>\n"
              f"{ind}  <in>{in_}</in>\n{ind}  <out>{in_ + (end - start)}</out>\n")
        s += file_xml(path, info, ind + "  ", kind)
        if kind == "audio":
            s += (f"{ind}  <sourcetrack>\n{ind}    <mediatype>audio</mediatype>\n"
                  f"{ind}    <trackindex>{track_index}</trackindex>\n{ind}  </sourcetrack>\n")
        for l in links or []:
            s += (f"{ind}  <link>\n{ind}    <linkclipref>{l[0]}</linkclipref>\n"
                  f"{ind}    <mediatype>{l[1]}</mediatype>\n{ind}    <trackindex>{l[2]}</trackindex>\n"
                  f"{ind}    <clipindex>{l[3]}</clipindex>\n{ind}  </link>\n")
        s += f"{ind}</clipitem>\n"
        return s, cid

    I = "      "
    clips = [e for e in events if e["type"] == "clip"]
    # pre-assign ids so video and source-audio can link to each other
    for k, e in enumerate(clips):
        e["_vid"] = f"clipitem-v{k + 1}"
        e["_a1"] = f"clipitem-a{k + 1}L"
        e["_a2"] = f"clipitem-a{k + 1}R"
    vtrack = f"{I}<track>\n"
    for k, e in enumerate(clips):
        info = _info(cut, e["file"])
        x, _ = clipitem(e["name"], e["file"], info, e["rec_in"], e["rec_out"],
                        int(round(e["src_in"] * fps)), I + "  ", "video", 1, None, e["_vid"])
        vtrack += x
    vtrack += (f"{I}  <enabled>TRUE</enabled>\n{I}  <locked>FALSE</locked>\n{I}</track>\n")

    # music on A1/A2 (FCP 7 style: one mono-channel clipitem per track, linked)
    minfo = dict(music["info"])
    minfo["has_video"] = False
    m_in = int(round(music["start"] * fps))
    chans = min(2, max(1, minfo.get("audio_channels", 2)))
    atracks = ""
    mids = [f"clipitem-music-{c + 1}" for c in range(chans)]
    for c in range(chans):
        links = [(mids[o], "audio", o + 1, 1) for o in range(chans)] if chans > 1 else None
        x, _ = clipitem(minfo["name"], music["path"], minfo, 0, cut["duration"], m_in, I + "  ",
                        "audio", c + 1, links, mids[c])
        atracks += f"{I}<track>\n{x}{I}  <enabled>TRUE</enabled>\n{I}  <locked>FALSE</locked>\n{I}</track>\n"
    if source_audio:
        for ch, key in ((1, "_a1"), (2, "_a2")):
            t = f"{I}<track>\n"
            for e in clips:
                info = _info(cut, e["file"])
                if not info.get("has_audio"):
                    continue
                ainfo = dict(info)
                x, _ = clipitem(e["name"], e["file"], ainfo, e["rec_in"], e["rec_out"],
                                int(round(e["src_in"] * fps)), I + "  ", "audio", ch, None, e[key])
                t += x
            t += f"{I}  <enabled>TRUE</enabled>\n{I}  <locked>FALSE</locked>\n{I}</track>\n"
            atracks += t

    mk = ""
    for m in cut["markers"]:
        mk += (f"    <marker>\n      <name>{escape(m['name'])}</name>\n"
               f"      <comment>{escape(m.get('comment', ''))}</comment>\n"
               f"      <in>{m['frame']}</in>\n      <out>-1</out>\n    </marker>\n")

    start_tc = RECORD_START_HOURS * 3600 * rate["timebase"]
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">\n'
           f'  <sequence id="sequence-1">\n    <name>{escape(seq_name)}</name>\n'
           f"    <duration>{cut['duration']}</duration>\n{_rate_xml(rate, '    ')}"
           f"{_tc_xml(rate, start_tc, '    ')}"
           "    <in>-1</in>\n    <out>-1</out>\n"
           "    <media>\n      <video>\n        <format>\n          <samplecharacteristics>\n"
           f"{_rate_xml(rate, '            ')}"
           f"            <width>{cut['width']}</width>\n            <height>{cut['height']}</height>\n"
           "            <anamorphic>FALSE</anamorphic>\n            <pixelaspectratio>square</pixelaspectratio>\n"
           "            <fielddominance>none</fielddominance>\n"
           "          </samplecharacteristics>\n        </format>\n"
           f"{_indent(vtrack, 2)}      </video>\n      <audio>\n"
           "        <numOutputChannels>2</numOutputChannels>\n"
           "        <format>\n          <samplecharacteristics>\n            <depth>16</depth>\n"
           "            <samplerate>48000</samplerate>\n          </samplecharacteristics>\n        </format>\n"
           f"{_indent(atracks, 2)}      </audio>\n    </media>\n{mk}  </sequence>\n</xmeml>\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    for e in clips:
        for k in ("_vid", "_a1", "_a2"):
            e.pop(k, None)
    return out_path


def _indent(s, n):
    pad = "  " * n
    return "".join(pad + l if l.strip() else l for l in s.splitlines(True))


def _info(cut, path):
    return cut["files_info"][path]


def write_edl(cut, out_path, title):
    rate = cut["rate"]
    tb = rate["timebase"]
    drop = False
    start = RECORD_START_HOURS * 3600 * tb
    lines = [f"TITLE: {title[:70]}", "FCM: NON-DROP FRAME", ""]
    n = 0
    for e in cut["events"]:
        n += 1
        r_in, r_out = start + e["rec_in"], start + e["rec_out"]
        if e["type"] == "gap":
            lines.append(f"{n:03d}  BL       V     C        00:00:00:00 "
                         f"{media.frames_to_tc(r_out - r_in, tb)} {media.frames_to_tc(r_in, tb)} "
                         f"{media.frames_to_tc(r_out, tb)}")
            lines.append(f"* TITLE CARD: {e['title']}")
        else:
            info = _info(cut, e["file"])
            mtb = info.get("rate", rate)["timebase"]
            s_in = info.get("tc_frames", 0) + int(round(e["src_in"] * info.get("rate", rate)["fps"]))
            s_out = s_in + int(round((e["rec_out"] - e["rec_in"]) * mtb / tb))
            reel = re.sub(r"[^A-Za-z0-9_]", "", os.path.splitext(e["name"])[0])[:8] or "AX"
            lines.append(f"{n:03d}  {reel:<8} V     C        {media.frames_to_tc(s_in, mtb)} "
                         f"{media.frames_to_tc(s_out, mtb)} {media.frames_to_tc(r_in, tb)} "
                         f"{media.frames_to_tc(r_out, tb)}")
            lines.append(f"* FROM CLIP NAME: {e['name']}")
            lines.append(f"* SOURCE FILE: {e['file']}")
        lines.append("")
    m = cut["music"]
    lines.append(f"* MUSIC: {os.path.basename(m['path'])} from {m['start']:.2f}s, {m['length']:.2f}s, "
                 f"{m['bpm']:.1f} BPM (on A1/A2 in the XML)")
    with open(out_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


# ------------------------------------------------------------------ pictures
def _font(size):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def title_frame(text, w, h):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    f = _font(max(14, h // 12))
    bbox = d.textbbox((0, 0), text, font=f)
    d.text(((w - (bbox[2] - bbox[0])) / 2, (h - (bbox[3] - bbox[1])) / 2), text, fill=(235, 235, 235), font=f)
    return np.asarray(img)


def storyboard(cut, bin_, cols=6, tw=320, th=180):
    tb = cut["rate"]["timebase"]
    evs = cut["events"]
    if not evs:
        return np.zeros((th, tw, 3), np.uint8)
    rows = (len(evs) + cols - 1) // cols
    band = 40
    sheet = Image.new("RGB", (cols * tw, rows * (th + band)), (14, 14, 16))
    d = ImageDraw.Draw(sheet)
    f = _font(13)
    story = [c for c in cut["chapters"]]
    by_id = {s["id"]: s for s in bin_["shots"]}
    for k, e in enumerate(evs):
        x, y = (k % cols) * tw, (k // cols) * (th + band)
        if e["type"] == "gap":
            img = title_frame(e["title"] or "TITLE", tw, th)
        else:
            s = by_id[e["shot"]]
            img = media.letterbox_array(shotlib.thumb_at(bin_, s, (e["src_in"] + e["src_out"]) / 2), tw, th)
        sheet.paste(Image.fromarray(img), (x, y))
        tc = media.frames_to_tc(e["rec_in"], tb)[3:]
        ch = story[e["chapter"]]["text"] if e["chapter"] < len(story) else ""
        label1 = f"{k + 1:02d}  {tc}  {e.get('beats', '')}{'b' if e['type'] == 'clip' else ''}"
        label2 = (e["name"] if e["type"] == "clip" else "title card")[:34]
        d.text((x + 6, y + th + 3), label1, fill=(255, 200, 40), font=f)
        d.text((x + 6, y + th + 20), label2, fill=(200, 200, 200), font=f)
        if ch and (k == 0 or evs[k - 1]["chapter"] != e["chapter"]):
            d.rectangle([x, y, x + tw - 1, y + 18], fill=(200, 30, 30))
            d.text((x + 5, y + 2), ch[:40], fill=(255, 255, 255), font=f)
    return np.asarray(sheet)


def render_preview(cut, out_path, height=540, burn_titles=True, progress=None):
    rate = cut["rate"]
    fps_r = Fraction(rate["num"], rate["den"])
    fps = float(fps_r)
    W = int(round(height * cut["width"] / cut["height"] / 2)) * 2
    H = int(height) // 2 * 2
    out = av.open(out_path, "w")
    codec = "libx264" if "libx264" in av.codecs_available else (
        "h264" if "h264" in av.codecs_available else "mpeg4")
    vs = out.add_stream(codec, rate=fps_r)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    if codec == "libx264":
        vs.options = {"crf": "22", "preset": "veryfast"}
    else:
        vs.bit_rate = 4_000_000
    a_s = out.add_stream("aac", rate=48000)
    try:
        a_s.layout = "stereo"
    except Exception:
        pass
    total = cut["duration"]
    n = 0
    readers = {}
    try:
        for e in cut["events"]:
            count = e["rec_out"] - e["rec_in"]
            if e["type"] == "gap":
                fr = title_frame(e["title"], W, H) if burn_titles and e["title"] else np.zeros((H, W, 3), np.uint8)
                it = (fr for _ in range(count))
            else:
                if e["file"] not in readers:
                    for r in readers.values():
                        r.close()
                    readers = {e["file"]: media.FrameReader(e["file"])}
                it = readers[e["file"]].frames(e["src_in"], count, fps, (W, H))
            for img in it:
                vf = av.VideoFrame.from_ndarray(img, format="rgb24")
                vf.pts = n
                vf.time_base = Fraction(1, 1) / fps_r
                for p in vs.encode(vf):
                    out.mux(p)
                n += 1
                if progress and n % 24 == 0:
                    progress(n, total)
        for p in vs.encode(None):
            out.mux(p)
        # music
        m = cut["music"]
        with av.open(m["path"]) as ac:
            ast = ac.streams.audio[0]
            res = av.AudioResampler(format="fltp", layout="stereo", rate=48000)
            fifo = av.AudioFifo()
            need = int(round(total / fps * 48000))
            skip = int(round(m["start"] * 48000))
            got = 0
            apts = 0
            for frame in ac.decode(ast):
                for rf in res.resample(frame):
                    rf.pts = None
                    fifo.write(rf)
            for rf in res.resample(None):
                rf.pts = None
                fifo.write(rf)
            while skip > 0 and fifo.samples > 0:
                take = min(skip, fifo.samples, 48000)
                fifo.read(take)
                skip -= take
            fs = a_s.codec_context.frame_size or 1024
            while got < need and fifo.samples > 0:
                chunk = fifo.read(min(fs, fifo.samples, need - got))
                if chunk is None:
                    break
                if chunk.samples < fs:
                    arr = chunk.to_ndarray()
                    arr = np.pad(arr, ((0, 0), (0, fs - chunk.samples)))
                    chunk = av.AudioFrame.from_ndarray(arr.astype(np.float32), format="fltp", layout="stereo")
                    chunk.sample_rate = 48000
                chunk.pts = apts
                apts += chunk.samples
                got += chunk.samples
                for p in a_s.encode(chunk):
                    out.mux(p)
            for p in a_s.encode(None):
                out.mux(p)
    finally:
        for r in readers.values():
            r.close()
        out.close()
    return out_path


def write_report(cut, out_path, summary_text):
    data = {k: v for k, v in cut.items() if not k.startswith("_")}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.splitext(out_path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

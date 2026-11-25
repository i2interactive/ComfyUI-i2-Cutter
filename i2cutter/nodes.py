import os
import time

import numpy as np
import torch

import folder_paths

from . import export, media, music, planner, scorer
from . import shots as shotlib

CAT = "i2/cutter"


def _bar(total):
    try:
        import comfy.utils
        pb = comfy.utils.ProgressBar(total)
        return lambda done, tot: pb.update_absolute(done, tot)
    except Exception:
        return None


def _img(arr):
    return torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32) / 255.0)[None]


def _models_dir():
    d = os.path.join(folder_paths.models_dir, "i2_cutter")
    os.makedirs(d, exist_ok=True)
    return d


class I2FootageBin:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": ("STRING", {"default": "", "tooltip": "Folder of clips (or one long file). Full-res originals are fine: they are only read, never changed."}),
            "recursive": ("BOOLEAN", {"default": True}),
            "extensions": ("STRING", {"default": ",".join(media.VIDEO_EXTS)}),
            "cut_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "How eagerly to split a file into shots. 0 = only hard, obvious cuts."}),
            "min_shot_sec": ("FLOAT", {"default": 0.8, "min": 0.2, "max": 10.0, "step": 0.1}),
            "analysis_fps": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 12.0, "step": 0.5,
                                       "tooltip": "Frames looked at per second of footage. Results are cached per file."}),
            "handle_sec": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.02,
                                     "tooltip": "Trimmed off both ends of every shot so a cut never lands on a flash frame or the original cut."}),
            "exclude_black": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("I2_SHOTS", "IMAGE", "STRING")
    RETURN_NAMES = ("shots", "contact_sheet", "summary")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, folder, recursive, extensions, cut_sensitivity, min_shot_sec, analysis_fps,
            handle_sec, exclude_black):
        paths = media.list_media(folder, extensions, recursive)
        if not paths:
            raise ValueError(f"i2 Cutter: no video files found in {folder}")
        t = time.time()
        cache = os.path.join(_models_dir(), "analysis_cache")
        b = shotlib.build_shots(paths, analysis_fps, cut_sensitivity, min_shot_sec, handle_sec,
                                exclude_black, cache, _bar(len(paths)))
        sh = b["shots"]
        tiles = [media.letterbox_array(shotlib.shot_thumb(b, s), 256, 144) for s in sh[:240]]
        if tiles:
            cols = 8
            rows = (len(tiles) + cols - 1) // cols
            sheet = np.zeros((rows * 144, cols * 256, 3), np.uint8)
            for k, im in enumerate(tiles):
                sheet[(k // cols) * 144:(k // cols + 1) * 144, (k % cols) * 256:(k % cols + 1) * 256] = im
        else:
            sheet = np.zeros((144, 256, 3), np.uint8)
        total = sum(s["dur"] for s in sh)
        msg = (f"{len(sh)} shots from {len(b['files'])} of {len(paths)} files, "
               f"{total / 60:.1f} min usable, analysed in {time.time() - t:.1f}s")
        print(f"[i2 Cutter] {msg}")
        return (b, _img(sheet), msg)


class I2MusicMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio_path": ("STRING", {"default": "", "tooltip": "Music file (wav, aif, mp3, m4a...). The XML links to this file."}),
            "start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1}),
            "length_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1,
                                     "tooltip": "Length of the cut. 0 = to the end of the track."}),
            "bpm_hint": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.5,
                                   "tooltip": "0 = detect. Set it if the detected tempo is half or double."}),
            "beats_per_bar": ("INT", {"default": 4, "min": 2, "max": 12}),
        }}

    RETURN_TYPES = ("I2_MUSIC", "IMAGE", "STRING")
    RETURN_NAMES = ("music", "beat_map", "summary")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, audio_path, start_sec, length_sec, bpm_hint, beats_per_bar):
        p = os.path.abspath(os.path.expanduser(audio_path.strip().strip('"')))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"i2 Cutter: music not found: {p}")
        m = music.analyze(p, start_sec, length_sec, bpm_hint, beats_per_bar)
        msg = (f"{m['bpm']:.1f} BPM, {len(m['beats'])} beats, {sum(m['downbeat'])} bars, "
               f"{sum(m['hits'])} accents, {m['length']:.1f}s")
        return (m, _img(music.draw(m)), msg)


class I2ShotScorer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shots": ("I2_SHOTS",),
            "brief": ("STRING", {"multiline": True, "default":
                "# one story beat per line, in order. '| 2' = twice as long.\n"
                "a quiet stable at dawn, a horse breathing\n"
                "a man alone, starving himself, a mirror | 1.5\n"
                "TITLE: THIS NOVEMBER\n"
                "the race, hooves, speed, a crowd roaring | 2\n"
                "collapse, aftermath, an empty track\n"
                "TITLE: WASTING | 2"}),
            "negative": ("STRING", {"multiline": True, "default": "text, watermark, logo, black screen, blurry"}),
            "use_clip": ("BOOLEAN", {"default": True, "tooltip": "Off = the brief only sets the structure; shots are picked by motion and variety."}),
            "clip_model": (scorer.CLIP_MODELS, {"default": scorer.CLIP_MODELS[0]}),
            "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
            "negative_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05}),
            "keep_model_loaded": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "clip_model_folder": ("STRING", {"default": "", "tooltip": "Optional: a local CLIP folder (transformers format). Overrides clip_model."}),
        }}

    RETURN_TYPES = ("I2_SCORES", "STRING")
    RETURN_NAMES = ("scores", "report")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, shots, brief, negative, use_clip, clip_model, device, negative_strength, keep_model_loaded,
            clip_model_folder=""):
        n = sum(len(s["thumbs"]) for s in shots["shots"])
        name = clip_model_folder.strip().strip('"') or clip_model
        r = scorer.score(shots, brief, negative, use_clip, name, device, negative_strength,
                         _models_dir(), keep_model_loaded, _bar(max(1, n)))
        return (r, scorer.report(shots, r))


class I2CutPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shots": ("I2_SHOTS",),
            "music": ("I2_MUSIC",),
            "style": (planner.STYLES, {"default": "trailer",
                                       "tooltip": "trailer: cuts speed up as it builds. music video: follows the music. montage: even pace. slow burn: long holds."}),
            "min_beats_per_cut": ("INT", {"default": 1, "min": 1, "max": 16}),
            "max_beats_per_cut": ("INT", {"default": 8, "min": 1, "max": 32}),
            "energy_response": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "How hard the music's loudness drives the cut rate."}),
            "story_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                                       "tooltip": "How strongly shots must match their line of the brief."}),
            "motion_match": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                                       "tooltip": "Put busy shots on loud passages and still shots on quiet ones."}),
            "variety": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "Randomness. Change the seed for another cut."}),
            "cut_on_accents": ("BOOLEAN", {"default": True}),
            "allow_reuse": ("BOOLEAN", {"default": False,
                                        "tooltip": "Let a long shot supply more than one cut (never overlapping)."}),
            "min_cut_sec": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 5.0, "step": 0.05}),
            "timeline_fps": (["auto", "23.976", "24", "25", "29.97", "30", "50", "59.94", "60"], {"default": "auto"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1, "control_after_generate": True}),
        }, "optional": {"scores": ("I2_SCORES",)}}

    RETURN_TYPES = ("I2_CUT", "IMAGE", "STRING")
    RETURN_NAMES = ("cut", "storyboard", "cut_list")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, shots, music, style, min_beats_per_cut, max_beats_per_cut, energy_response, story_weight,
            motion_match, variety, cut_on_accents, allow_reuse, min_cut_sec, timeline_fps, seed, scores=None):
        lo, hi = sorted((min_beats_per_cut, max_beats_per_cut))
        cut = planner.plan(shots, music, scores, style, lo, hi, energy_response, story_weight, motion_match,
                           variety, cut_on_accents, allow_reuse, min_cut_sec, timeline_fps, seed)
        cut["_bin"] = shots
        return (cut, _img(export.storyboard(cut, shots)), planner.summary(cut))


class I2TimelineExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cut": ("I2_CUT",),
            "filename_prefix": ("STRING", {"default": "i2_cutter/trailer"}),
            "sequence_name": ("STRING", {"default": "i2 Cutter"}),
            "fcp_xml": ("BOOLEAN", {"default": True, "tooltip": "FCP 7 XML: File > Import in Premiere, File > Import > Timeline in Resolve."}),
            "edl": ("BOOLEAN", {"default": True}),
            "source_audio": ("BOOLEAN", {"default": False, "tooltip": "Also lay each clip's own sound on A3/A4."}),
            "preview_mp4": ("BOOLEAN", {"default": True}),
            "preview_height": ([360, 540, 720, 1080], {"default": 540}),
            "burn_titles": ("BOOLEAN", {"default": True, "tooltip": "Show TITLE lines as text in the preview (the XML gets markers)."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("xml_path", "folder")
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True

    def run(self, cut, filename_prefix, sequence_name, fcp_xml, edl, source_audio, preview_mp4,
            preview_height, burn_titles):
        out_dir = folder_paths.get_output_directory()
        full, base, counter, sub, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        stem = f"{base}_{counter:05}"
        paths = []
        xml = os.path.join(full, stem + ".xml")
        if fcp_xml:
            export.write_fcp7_xml(cut, xml, sequence_name, source_audio)
            paths.append(xml)
        if edl:
            paths.append(export.write_edl(cut, os.path.join(full, stem + ".edl"), sequence_name))
        export.write_report(cut, os.path.join(full, stem + "_cutlist.json"), planner.summary(cut))
        ui = {}
        if preview_mp4:
            mp4 = stem + "_preview.mp4"
            export.render_preview(cut, os.path.join(full, mp4), preview_height, burn_titles,
                                  _bar(max(1, cut["duration"])))
            paths.append(os.path.join(full, mp4))
            ui = {"images": [{"filename": mp4, "subfolder": sub, "type": "output"}], "animated": (True,)}
        print("[i2 Cutter] wrote:\n  " + "\n  ".join(paths))
        return {"ui": ui, "result": (xml if fcp_xml else "", full)}


NODE_CLASS_MAPPINGS = {
    "i2FootageBin": I2FootageBin,
    "i2MusicMap": I2MusicMap,
    "i2ShotScorer": I2ShotScorer,
    "i2CutPlanner": I2CutPlanner,
    "i2TimelineExport": I2TimelineExport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "i2FootageBin": "i2 Footage Bin",
    "i2MusicMap": "i2 Music Map",
    "i2ShotScorer": "i2 Shot Scorer (CLIP)",
    "i2CutPlanner": "i2 Cut Planner",
    "i2TimelineExport": "i2 Timeline Export (Premiere / Resolve)",
}

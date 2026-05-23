"""Stage 1: phrase extraction from a monophonic vocal stem (v2).

Improvements over v1:
  * adaptive per-song gap threshold (using the song's own inter-note-gap distribution)
  * octave-jump correction on the pitch contour (pyin's most common error)
  * voicing-confidence drops as an *additional* phrase-cut signal
  * optional cut_clips() helper to dump each phrase as its own WAV for listening

Pipeline:
  vocal WAV (24 kHz mono)
   -> librosa.pyin pitch track (10 ms hop) + voicing confidence
   -> octave-jump correction
   -> note grouping (runs of stable pitch, dur >= 60 ms)
   -> phrase grouping (gap > adaptive_thresh OR voicing-confidence collapse)
   -> JSON-serialisable list of phrases per song
"""
from __future__ import annotations
import json
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np

C2_HZ = 65.41
C6_HZ = 1046.5


@dataclass
class Note:
    pitch: float
    start_s: float
    duration_s: float
    amp: float

    def to_dict(self) -> Dict[str, float]:
        return {"pitch": round(self.pitch, 2),
                "start_s": round(self.start_s, 3),
                "duration_s": round(self.duration_s, 3),
                "amp": round(self.amp, 4)}


@dataclass
class Phrase:
    phrase_idx: int
    notes: List[Note]

    @property
    def start_s(self) -> float: return self.notes[0].start_s
    @property
    def end_s(self) -> float:   return self.notes[-1].start_s + self.notes[-1].duration_s
    @property
    def n_notes(self) -> int:   return len(self.notes)
    @property
    def pitch_range_st(self) -> float:
        ps = [n.pitch for n in self.notes]
        return max(ps) - min(ps)

    def to_dict(self) -> Dict[str, Any]:
        return {"phrase_idx": self.phrase_idx,
                "start_s": round(self.start_s, 3),
                "end_s":   round(self.end_s, 3),
                "n_notes": self.n_notes,
                "pitch_range_st": round(self.pitch_range_st, 2),
                "notes":   [n.to_dict() for n in self.notes]}


# ---------- pitch utils ----------

def hz_to_midi(f_hz: np.ndarray) -> np.ndarray:
    out = np.full_like(f_hz, np.nan, dtype=np.float64)
    m = np.isfinite(f_hz) & (f_hz > 0)
    out[m] = 12.0 * np.log2(f_hz[m] / 440.0) + 69.0
    return out


def median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x
    half = k // 2; n = len(x); out = x.copy()
    for i in range(n):
        win = x[max(0, i - half):min(n, i + half + 1)]
        win = win[np.isfinite(win)]
        if len(win) > 0: out[i] = float(np.median(win))
    return out


def correct_octave_jumps(midi: np.ndarray, ctx: int = 5,
                         jump_threshold: float = 8.0) -> np.ndarray:
    """If a single frame deviates from its ±ctx-frame median by > jump_threshold
    semitones, snap it by the nearest multiple of 12 semitones (octave). Repeat
    once to catch double-octave errors."""
    out = midi.copy()
    n = len(out)
    for _ in range(2):
        for i in range(n):
            if not np.isfinite(out[i]): continue
            win = out[max(0, i-ctx):min(n, i+ctx+1)]
            win = win[np.isfinite(win)]
            if len(win) < 3: continue
            med = float(np.median(win))
            diff = out[i] - med
            if abs(diff) > jump_threshold:
                # snap to nearest octave of med
                shift = round(diff / 12.0) * 12.0
                out[i] = out[i] - shift
    return out


# ---------- segmentation ----------

def group_into_notes(pitch_midi: np.ndarray,
                     amp: np.ndarray,
                     hop_s: float,
                     min_note_s: float = 0.06,
                     band_st: float = 0.5) -> List[Note]:
    notes: List[Note] = []
    n = len(pitch_midi); i = 0
    while i < n:
        if not np.isfinite(pitch_midi[i]): i += 1; continue
        j = i; vals = [pitch_midi[i]]; amps = [amp[i]]
        while j + 1 < n and np.isfinite(pitch_midi[j + 1]):
            running_med = float(np.median(vals))
            if abs(pitch_midi[j + 1] - running_med) > band_st: break
            j += 1; vals.append(pitch_midi[j]); amps.append(amp[j])
        dur_s = (j - i + 1) * hop_s
        if dur_s >= min_note_s:
            notes.append(Note(pitch=float(np.median(vals)), start_s=i * hop_s,
                              duration_s=dur_s, amp=float(np.mean(amps))))
        i = j + 1
    return notes


def adaptive_gap_threshold(notes: List[Note],
                           pct: float = 90.0,
                           min_thresh_s: float = 0.12,
                           max_thresh_s: float = 0.50) -> float:
    """Pick the cut threshold as the p`pct` of this song's inter-note gaps,
    clipped into a sane range."""
    if len(notes) < 5: return 0.20
    gaps = [notes[i+1].start_s - (notes[i].start_s + notes[i].duration_s)
            for i in range(len(notes)-1)]
    gaps = [g for g in gaps if g >= 0]
    if not gaps: return 0.20
    t = float(np.percentile(gaps, pct))
    return max(min_thresh_s, min(max_thresh_s, t))


def group_into_phrases(notes: List[Note],
                       phrase_gap_thresh_s: float,
                       voicing_drop_cuts_s: Optional[List[float]] = None,
                       min_phrase_notes: int = 3) -> List[Phrase]:
    """Boundary if (gap > thresh) OR (a voicing-confidence drop falls inside the gap).
    Drops shorter phrases (likely artefacts)."""
    if not notes: return []
    cuts_after = [False] * (len(notes) - 1)
    for i in range(len(notes) - 1):
        gap_start = notes[i].start_s + notes[i].duration_s
        gap_end = notes[i+1].start_s
        gap = gap_end - gap_start
        if gap > phrase_gap_thresh_s: cuts_after[i] = True; continue
        if voicing_drop_cuts_s:
            for vd in voicing_drop_cuts_s:
                if gap_start <= vd <= gap_end:
                    cuts_after[i] = True; break

    groups: List[List[Note]] = [[notes[0]]]
    for i, n in enumerate(notes[1:]):
        if cuts_after[i]: groups.append([n])
        else: groups[-1].append(n)
    phrases = [Phrase(i, g) for i, g in enumerate(groups) if len(g) >= min_phrase_notes]
    for i, p in enumerate(phrases): p.phrase_idx = i
    return phrases


def detect_voicing_drops(voiced_prob: np.ndarray, hop_s: float,
                         min_drop_s: float = 0.10,
                         conf_thresh: float = 0.5) -> List[float]:
    """Find runs of low voicing-confidence longer than min_drop_s — emit the
    center timestamp of each as a candidate phrase boundary."""
    low = voiced_prob < conf_thresh
    cuts: List[float] = []
    n = len(low); i = 0
    while i < n:
        if low[i]:
            j = i
            while j + 1 < n and low[j+1]: j += 1
            run = (j - i + 1) * hop_s
            if run >= min_drop_s:
                cuts.append((i + j) / 2 * hop_s)
            i = j + 1
        else: i += 1
    return cuts


# ---------- top-level API ----------

def extract_phrases_from_wav(wav: np.ndarray,
                             sr: int,
                             *,
                             fmin_hz: float = C2_HZ,
                             fmax_hz: float = C6_HZ,
                             hop_length: int = 256,
                             frame_length: int = 2048,
                             median_filter_frames: int = 3,
                             gap_percentile: float = 90.0,
                             phrase_gap_thresh_s: Optional[float] = None,
                             use_voicing_drops: bool = True,
                             ) -> Dict[str, Any]:
    """Returns dict with phrases plus diagnostics."""
    import librosa
    f0, voiced, voiced_prob = librosa.pyin(
        wav, sr=sr, fmin=fmin_hz, fmax=fmax_hz,
        frame_length=frame_length, hop_length=hop_length,
    )
    rms = librosa.feature.rms(y=wav, frame_length=frame_length,
                              hop_length=hop_length, center=True)[0]
    L = min(len(f0), len(rms), len(voiced_prob))
    f0 = f0[:L]; rms = rms[:L]; vp = voiced_prob[:L]
    midi = hz_to_midi(f0)
    midi = median_filter_1d(midi, median_filter_frames)
    midi = correct_octave_jumps(midi)
    hop_s = hop_length / sr
    notes = group_into_notes(midi, rms, hop_s=hop_s)
    if phrase_gap_thresh_s is None:
        phrase_gap_thresh_s = adaptive_gap_threshold(notes, pct=gap_percentile)
    voicing_cuts = detect_voicing_drops(vp, hop_s=hop_s) if use_voicing_drops else None
    phrases = group_into_phrases(notes, phrase_gap_thresh_s, voicing_cuts)
    return {
        "phrases": phrases,
        "n_notes": len(notes),
        "n_phrases": len(phrases),
        "gap_thresh_used_s": phrase_gap_thresh_s,
        "n_voicing_cuts": len(voicing_cuts or []),
        "voicing_pct": float(np.isfinite(midi).mean()),
    }


def phrases_to_dict(song_id: str, sr: int, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": song_id, "sr": sr,
        "n_phrases": result["n_phrases"],
        "n_notes": result["n_notes"],
        "gap_thresh_used_s": round(result["gap_thresh_used_s"], 3),
        "n_voicing_cuts": result["n_voicing_cuts"],
        "voicing_pct": round(result["voicing_pct"], 3),
        "phrases": [p.to_dict() for p in result["phrases"]],
    }


# ---------- WAV clip extraction ----------

def cut_phrase_clips(wav: np.ndarray, sr: int, phrases: List[Phrase],
                     out_dir: str, song_id: str,
                     pad_ms: int = 50) -> List[pathlib.Path]:
    """Slice the input wav into one WAV per phrase, with small padding for clean
    starts/ends. Returns list of written paths."""
    import soundfile as sf
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pad = int(pad_ms / 1000 * sr)
    paths = []
    for p in phrases:
        a = max(0, int(p.start_s * sr) - pad)
        b = min(len(wav), int(p.end_s * sr) + pad)
        clip = wav[a:b]
        fp = out / f"{song_id}_phrase{p.phrase_idx:03d}_{p.start_s:06.2f}s.wav"
        sf.write(str(fp), clip, sr)
        paths.append(fp)
    return paths

"""
tools/tts_fa_mms.py
===================
Native Farsi TTS via Meta MMS (Massively Multilingual Speech).

Model : facebook/mms-tts-fas  (~30 MB, CC-BY-NC 4.0)
SR    : 16 000 Hz (upsampled to 24 000 Hz for ChatTTS pipeline compatibility)
Quality: native Persian — far better than Finglish fallback

Install (already in speech env):
    pip install transformers     # VitsModel, AutoTokenizer
    pip install scipy            # resample_poly (already installed)

Usage:
    from tools.tts_fa_mms import fa_tts_to_pcm, is_available
    if is_available():
        wav24k = fa_tts_to_pcm("سلام! امیدوارم روز خوبی داشته باشید.")
        # wav24k: float32 numpy array at 24 000 Hz
"""

from __future__ import annotations
import numpy as np

_model     = None
_tokenizer = None
_SR_NATIVE = 16_000
_SR_OUT    = 24_000          # resample target — matches ChatTTS / audio tools

# ── availability ───────────────────────────────────────────────────────────────
def is_available() -> bool:
    """Return True if transformers VitsModel can be imported."""
    try:
        from transformers import VitsModel, AutoTokenizer   # noqa: F401
        return True
    except ImportError:
        return False


# ── lazy loader ───────────────────────────────────────────────────────────────
def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    from transformers import VitsModel, AutoTokenizer
    import logging
    logging.getLogger("transformers").setLevel(logging.WARNING)

    print("[MMS-FA] Loading facebook/mms-tts-fas …")
    _tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fas")
    _model     = VitsModel.from_pretrained("facebook/mms-tts-fas")
    _model.eval()
    print(f"[MMS-FA] Model ready  (native SR={_SR_NATIVE} Hz)")


# ── resampler ─────────────────────────────────────────────────────────────────
def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality polyphase resample without librosa."""
    if orig_sr == target_sr:
        return wav.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g  = gcd(orig_sr, target_sr)
    up = target_sr // g
    dn = orig_sr   // g
    return resample_poly(wav.astype(np.float32), up, dn).astype(np.float32)


# ── public API ────────────────────────────────────────────────────────────────
def fa_tts_to_pcm(text: str, target_sr: int = _SR_OUT) -> np.ndarray:
    """
    Synthesize Persian text with MMS-TTS.

    Parameters
    ----------
    text      : Persian (Farsi) text — Unicode, no transliteration needed.
    target_sr : Output sample rate. Default 24 000 Hz.

    Returns
    -------
    numpy float32 array, shape (N,), range ≈ [-1, 1].
    """
    import torch
    _load()

    inputs = _tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        wav_tensor = _model(**inputs).waveform   # (1, T) float32
    wav = wav_tensor.squeeze().numpy().astype(np.float32)

    if _SR_NATIVE != target_sr:
        wav = _resample(wav, _SR_NATIVE, target_sr)

    return wav


def fa_tts_sr() -> int:
    """Return the output sample rate (24 000 Hz)."""
    return _SR_OUT

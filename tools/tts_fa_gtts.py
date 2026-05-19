"""
tools/tts_fa_gtts.py
====================
Native Persian TTS via Google Text-to-Speech (gTTS).

Quality : Native Persian — Google's TTS engine
Latency : ~0.5–1s per request (internet required)
Output  : MP3 bytes directly (no conversion needed)
Size    : No local model — uses Google's servers

Install:
    pip install gtts   (already installed)

Usage:
    from tools.tts_fa_gtts import fa_gtts_mp3, is_available
    mp3_bytes = fa_gtts_mp3("سلام! امیدوارم روز خوبی داشته باشید.")
"""

from __future__ import annotations

def is_available() -> bool:
    try:
        import gtts  # noqa: F401
        return True
    except ImportError:
        return False


def fa_gtts_mp3(text: str, slow: bool = False) -> bytes:
    """
    Synthesize Persian text with Google TTS.

    Parameters
    ----------
    text : Persian (Farsi) text — no transliteration needed.
    slow : If True, speak slower (useful for learners).

    Returns
    -------
    MP3 bytes ready to serve or save.
    """
    from gtts import gTTS
    import io

    tts = gTTS(text=text, lang="fa", slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()


def fa_gtts_to_wav(text: str, target_sr: int = 24_000) -> "np.ndarray":
    """
    Synthesize Persian text and return float32 PCM at target_sr Hz.
    Useful for feeding into existing audio pipeline tools.
    """
    import io
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    mp3_bytes = fa_gtts_mp3(text)

    # Decode MP3 → PCM using soundfile (needs libsndfile with mp3 support)
    # Fallback: return raw mp3 bytes and let caller handle
    try:
        buf = io.BytesIO(mp3_bytes)
        wav, orig_sr = sf.read(buf, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)   # stereo → mono
        if orig_sr != target_sr:
            g  = gcd(orig_sr, target_sr)
            up = target_sr // g
            dn = orig_sr   // g
            wav = resample_poly(wav, up, dn).astype(np.float32)
        return wav
    except Exception:
        # Can't decode — caller should use mp3 bytes directly
        raise RuntimeError(
            "soundfile could not decode MP3. "
            "Use fa_gtts_mp3() to get raw MP3 bytes instead."
        )

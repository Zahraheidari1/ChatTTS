"""
tools/tts_fa_edge.py
====================
Native Persian TTS via Microsoft Edge TTS (Neural voices).

Voices  : fa-IR-DilaraNeural (female) · fa-IR-FaridNeural (male)
Quality : Neural — Microsoft Azure cognitive TTS (same engine as Edge browser)
Output  : MP3 bytes directly
Requires: pip install edge-tts   (no local model, ~0 disk space)
Internet: Yes (streams audio from Microsoft's servers)

Usage:
    from tools.tts_fa_edge import fa_edge_mp3, is_available
    mp3 = fa_edge_mp3("سلام! امیدوارم روز خوبی داشته باشید.")
    # or choose voice:
    mp3 = fa_edge_mp3("سلام", voice="fa-IR-FaridNeural")   # male
"""

from __future__ import annotations
import asyncio
import io

VOICE_FEMALE = "fa-IR-DilaraNeural"
VOICE_MALE   = "fa-IR-FaridNeural"
DEFAULT_VOICE = VOICE_FEMALE


def is_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


async def _synthesize(text: str, voice: str) -> bytes:
    """Async synthesis — returns MP3 bytes."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def fa_edge_mp3(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """
    Synthesize Persian text with Microsoft Edge TTS.

    Parameters
    ----------
    text  : Persian (Farsi) text.
    voice : fa-IR-DilaraNeural (female, default) or fa-IR-FaridNeural (male).

    Returns
    -------
    MP3 bytes.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside async context (FastAPI) — use new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                return ex.submit(asyncio.run, _synthesize(text, voice)).result()
        else:
            return loop.run_until_complete(_synthesize(text, voice))
    except RuntimeError:
        return asyncio.run(_synthesize(text, voice))

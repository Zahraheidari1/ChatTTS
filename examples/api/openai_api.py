"""
openai_api.py
=============
FastAPI server compatible with OpenAI's TTS interface.

Language routing
----------------
- Farsi  → Meta MMS-TTS  (facebook/mms-tts-fas, ~30 MB)  native Persian ✅
           Falls back to Finglish pipeline if transformers unavailable.
- Chinese / English → ChatTTS  (GPT-based, 40 k-hr training)

Run
---
    conda activate speech
    cd ChatTTS-main
    python run_api.py
        or
    uvicorn examples.api.openai_api:app --host 0.0.0.0 --port 8000
"""

import io
import os
import sys
import asyncio
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import torch
import numpy as np

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import ChatTTS
from tools.audio import pcm_arr_to_mp3_view, pcm_arr_to_ogg_view, pcm_arr_to_wav_view
from tools.logger import get_logger
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn
from tools.normalizer.fa import normalizer_fa_finglish, normalizer_fa_hazm

logger = get_logger("Command")
app    = FastAPI()

# ── Voice map ─────────────────────────────────────────────────────────────────
VOICE_MAP: dict[str, str] = {
    "default":   "1528.pt",
    "alloy":     "1384.pt",
    "echo":      "2443.pt",
    "fa_female": "voices/fa_female.pt",
    "fa_male":   "voices/fa_male.pt",
}

ALLOWED_FORMATS = {"mp3", "wav", "ogg"}
ALLOWED_PARAMS  = {"model", "input", "voice", "response_format",
                   "speed", "stream", "output_format"}


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    app.state.chat       = ChatTTS.Chat(get_logger("ChatTTS"))
    app.state.model_lock = asyncio.Lock()

    # ── Text normalizers ──────────────────────────────────────────────────
    try:
        app.state.chat.normalizer.register("en", normalizer_en_nemo_text())
        logger.info("English normalizer (nemo) loaded.")
    except Exception as e:
        logger.warning(f"English normalizer unavailable: {e}")

    try:
        app.state.chat.normalizer.register("zh", normalizer_zh_tn())
        logger.info("Chinese normalizer (WeText) loaded.")
    except Exception as e:
        logger.warning(f"Chinese normalizer unavailable: {e}")

    try:
        _hz = normalizer_fa_hazm()
        _fi = normalizer_fa_finglish()
        app.state.chat.normalizer.register("fa", lambda t: _fi(_hz(t)))
        logger.info("Farsi normalizer (hazm + Finglish) loaded.")
    except Exception:
        try:
            app.state.chat.normalizer.register("fa", normalizer_fa_finglish())
            logger.info("Farsi normalizer (Finglish) loaded.")
        except Exception as e2:
            logger.warning(f"Farsi normalizer unavailable: {e2}")

    # Keep Finglish converter as fallback
    app.state.fa_finglish = normalizer_fa_finglish()

    # ── Farsi TTS engine: MMS (primary) → Finglish fallback ─────────────
    app.state.fa_engine    = "finglish"
    app.state.mms_fa_infer = None

    try:
        from tools.tts_fa_mms import is_available as mms_ok, fa_tts_to_pcm, fa_tts_sr
        if mms_ok():
            logger.info("Loading MMS-FA (facebook/mms-tts-fas) …")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: fa_tts_to_pcm("سلام"))
            app.state.mms_fa_infer = fa_tts_to_pcm
            app.state.mms_fa_sr    = fa_tts_sr()
            app.state.fa_engine    = "mms"
            logger.info("Farsi engine: MMS-FA native Persian ✅")
        else:
            logger.warning("MMS-FA: transformers not installed.")
    except Exception as e:
        logger.warning(f"MMS-FA load failed ({e}) — falling back to Finglish.")

    if app.state.fa_engine == "finglish":
        logger.warning("Farsi engine: Finglish (ChatTTS) — accented output.")

    # ── Load ChatTTS model ────────────────────────────────────────────────
    logger.info("Initializing ChatTTS model…")
    if not app.state.chat.load(source="huggingface"):
        raise RuntimeError("ChatTTS model failed to load.")
    logger.info("ChatTTS model loaded.")

    # ── Pre-load speaker embeddings ───────────────────────────────────────
    app.state.spk_emb_map: dict = {}
    for voice, path in VOICE_MAP.items():
        if os.path.exists(path):
            app.state.spk_emb_map[voice] = torch.load(
                path, map_location="cpu", weights_only=True
            )
            logger.info(f"Speaker embedding loaded: {voice} ← {path}")
        else:
            logger.warning(f"Speaker file not found (skipped): {path}")

    if "default" not in app.state.spk_emb_map:
        logger.warning("'default' voice file missing – using random speaker.")
        app.state.spk_emb_map["default"] = app.state.chat.sample_random_speaker()

    app.state.default_spk = app.state.spk_emb_map["default"]


# ── Exception handler ─────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def _exc(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={"error": {"message": str(exc), "type": type(exc).__name__}},
    )


# ── Request model ─────────────────────────────────────────────────────────────
class OpenAITTSRequest(BaseModel):
    model:           str            = Field(..., description="TTS model, always 'tts-1'")
    input:           str            = Field(..., max_length=4096)
    voice:           Optional[str]  = Field("default",
        description="Voice: default | alloy | echo | fa_female | fa_male")
    response_format: Optional[str]  = Field("mp3",
        description="Audio format: mp3 | wav | ogg")
    speed:           Optional[float]= Field(1.0, ge=0.5, le=2.0)
    stream:          Optional[bool] = Field(False)
    output_format:   Optional[str]  = "mp3"
    extra_params:    Dict[str, Optional[str]] = Field(default_factory=dict)

    @classmethod
    def validate_request(cls, data: Dict) -> Dict:
        data["model"] = "tts-1"
        unsupported = set(data) - ALLOWED_PARAMS
        if unsupported:
            logger.warning(f"Ignoring unsupported params: {unsupported}")
        return {k: data[k] for k in ALLOWED_PARAMS if k in data}


# ── Audio helpers ─────────────────────────────────────────────────────────────
def _wav_header(sr=24000, bits=16, ch=1) -> bytes:
    hdr = bytearray()
    hdr += b"RIFF" + b"\xff\xff\xff\xff" + b"WAVEfmt "
    hdr += (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
    hdr += ch.to_bytes(2, "little")   + sr.to_bytes(4, "little")
    hdr += (sr * ch * bits // 8).to_bytes(4, "little")
    hdr += (ch * bits // 8).to_bytes(2, "little")
    hdr += bits.to_bytes(2, "little") + b"data" + b"\xff\xff\xff\xff"
    return bytes(hdr)

def _to_audio(wav: np.ndarray, fmt: str) -> bytes:
    if fmt == "mp3":
        return bytes(pcm_arr_to_mp3_view(wav))
    if fmt == "wav":
        return bytes(pcm_arr_to_wav_view(wav))
    if fmt == "ogg":
        return bytes(pcm_arr_to_ogg_view(wav))
    return bytes(pcm_arr_to_mp3_view(wav))


# ── Main TTS endpoint ─────────────────────────────────────────────────────────
@app.post("/v1/audio/speech")
async def generate_voice(request_data: Dict):
    request_data = OpenAITTSRequest.validate_request(request_data)
    req = OpenAITTSRequest(**request_data)

    logger.info(f"TTS request: voice={req.voice} fmt={req.response_format} "
                f"stream={req.stream} text={req.input[:60]!r}")

    if req.response_format not in ALLOWED_FORMATS:
        raise HTTPException(400, detail=(
            f"Unsupported format '{req.response_format}'. "
            f"Choose from: {', '.join(sorted(ALLOWED_FORMATS))}"
        ))

    text = req.input
    is_farsi_voice  = (req.voice or "").startswith("fa_")
    has_persian     = any("؀" <= c <= "ۿ" for c in text)
    use_mms         = (is_farsi_voice or has_persian) and app.state.mms_fa_infer is not None

    # ── Route: Farsi → MMS native TTS ────────────────────────────────────
    fa_engine = getattr(app.state, "fa_engine", "finglish")
    if (is_farsi_voice or has_persian) and fa_engine == "mms":
        logger.info(f"Farsi → MMS-TTS: {text[:60]!r}")
        try:
            loop = asyncio.get_event_loop()
            wav  = await loop.run_in_executor(
                None,
                lambda: app.state.mms_fa_infer(text)
            )
            data = _to_audio(wav, req.response_format)
            logger.info(f"MMS-FA done: {len(data):,} bytes")
            return StreamingResponse(
                io.BytesIO(data),
                media_type="audio/mpeg" if req.response_format != "wav" else "audio/wav",
                headers={"Content-Disposition":
                         f"attachment; filename=output.{req.response_format}"},
            )
        except Exception as e:
            logger.warning(f"MMS-FA failed ({e}), falling back to Finglish.")

    # ── Route: Farsi fallback → Finglish → ChatTTS ────────────────────────
    if is_farsi_voice or has_persian:
        text = app.state.fa_finglish(text)
        logger.info(f"Finglish fallback: {req.input[:40]!r} → {text[:40]!r}")

    # ── Route: English / Chinese / Finglish → ChatTTS ────────────────────
    spk_emb = app.state.spk_emb_map.get(req.voice)
    if spk_emb is None:
        logger.warning(f"Voice '{req.voice}' not found, using default.")
        spk_emb = app.state.default_spk

    params_infer_code = app.state.chat.InferCodeParams(
        prompt="[speed_5]",
        top_P=0.5,
        top_K=10,
        temperature=0.3,
        repetition_penalty=1.05,
        max_new_token=2048,
        min_new_token=0,
        show_tqdm=True,
        ensure_non_empty=True,
        manual_seed=42,
        spk_emb=spk_emb,
        spk_smp=None,
        txt_smp=None,
        stream_batch=24,
        stream_speed=12000,
        pass_first_n_batches=2,
    )

    try:
        async with app.state.model_lock:
            wavs = app.state.chat.infer(
                text=[text],
                stream=req.stream,
                lang=None,
                skip_refine_text=True,
                use_decoder=True,
                do_text_normalization=False,
                do_homophone_replacement=False,
                params_infer_code=params_infer_code,
            )
    except Exception as e:
        raise HTTPException(500, detail=f"Synthesis failed: {e}")

    # ── Streaming ─────────────────────────────────────────────────────────
    if req.stream:
        first = True
        async def _stream():
            nonlocal first
            for wav in wavs:
                if req.response_format == "wav" and first:
                    yield _wav_header()
                    first = False
                yield _to_audio(wav, req.response_format)
        mt = "audio/wav" if req.response_format == "wav" else "audio/mpeg"
        return StreamingResponse(_stream(), media_type=mt)

    # ── Non-streaming ─────────────────────────────────────────────────────
    data = _to_audio(wavs[0], req.response_format)
    return StreamingResponse(
        io.BytesIO(data),
        media_type="audio/mpeg" if req.response_format != "wav" else "audio/wav",
        headers={"Content-Disposition":
                 f"attachment; filename=output.{req.response_format}"},
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    loaded = hasattr(app.state, "chat") and bool(app.state.chat)
    return {
        "status":        "healthy",
        "model_loaded":  loaded,
        "farsi_engine":  getattr(app.state, "fa_engine", "unknown"),
    }

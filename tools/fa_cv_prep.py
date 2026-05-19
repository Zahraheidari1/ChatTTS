"""
tools/fa_cv_prep.py
===================
Generate Farsi speaker embeddings from Mozilla Common Voice for ChatTTS.

What it does
------------
1. Downloads / loads the Mozilla Common Voice Farsi (fa) subset from
   Hugging Face  (mozilla-foundation/common_voice_17_0).
2. Selects clean, validated clips and groups them by gender.
3. Resamples each clip to 24 kHz (ChatTTS native sample rate).
4. Uses ChatTTS zero-shot speaker encoder to extract per-clip embeddings.
5. Averages embeddings per gender and saves:
       voices/fa_female.pt
       voices/fa_male.pt
   Drop these files in ChatTTS-main/ and they will be picked up
   automatically by run_api.py as  voice="fa_female" / voice="fa_male".

Usage
-----
    conda activate speech
    cd ChatTTS-main
    pip install datasets soundfile librosa   # one-time
    python tools/fa_cv_prep.py

Requirements
------------
    pip install datasets soundfile librosa
    ChatTTS model must already be downloaded (run the server once first).

Dataset
-------
    Mozilla Common Voice  https://commonvoice.mozilla.org/en/datasets
    HuggingFace mirror    mozilla-foundation/common_voice_17_0
    Language code         fa  (Persian / Farsi)
    License               CC-0  (public domain)

Note
----
If the Common Voice 17 dataset requires a HuggingFace access token, run:
    huggingface-cli login
before executing this script.
"""

import os
import sys
import io
import base64

import torch
import numpy as np

# Make sure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ── Configuration ─────────────────────────────────────────────────────────────
CV_DATASET     = "mozilla-foundation/common_voice_17_0"
CV_LANG        = "fa"          # Persian / Farsi
MAX_PER_GENDER = 50            # clips averaged per gender (more = smoother)
TARGET_SR      = 24000         # ChatTTS sample rate
OUT_DIR        = "voices"
# ─────────────────────────────────────────────────────────────────────────────


def resample_audio(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample a mono float32 array to TARGET_SR."""
    if orig_sr == TARGET_SR:
        return audio.astype(np.float32)
    try:
        import librosa
        return librosa.resample(audio.astype(np.float32),
                                orig_sr=orig_sr, target_sr=TARGET_SR)
    except ImportError:
        # Basic linear interpolation fallback (lower quality)
        ratio  = TARGET_SR / orig_sr
        n_out  = int(len(audio) * ratio)
        idx    = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(idx, np.arange(len(audio)),
                         audio.astype(np.float32))


def load_chat_model():
    import ChatTTS
    from tools.logger import get_logger
    chat = ChatTTS.Chat(get_logger("fa_cv_prep"))
    print("Loading ChatTTS model …")
    if not chat.load(source="huggingface"):
        raise RuntimeError("ChatTTS model failed to load.")
    print("Model loaded.")
    return chat


def extract_embedding(chat, wav: np.ndarray) -> torch.Tensor:
    """
    Encode a waveform into a speaker embedding using ChatTTS zero-shot encoder.
    Returns a CPU tensor.
    """
    spk_str = chat.sample_audio_speaker(wav.astype(np.float32))
    buf = base64.b64decode(spk_str)
    return torch.load(io.BytesIO(buf), map_location="cpu")


def main():
    from datasets import load_dataset, Audio as HFAudio

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading Common Voice '{CV_LANG}' from {CV_DATASET} …")
    ds = load_dataset(CV_DATASET, CV_LANG, split="validated",
                      trust_remote_code=True)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=TARGET_SR))
    print(f"Total validated clips: {len(ds)}")

    # Collect up to MAX_PER_GENDER clips per gender
    groups: dict[str, list[np.ndarray]] = {"female": [], "male": []}
    for sample in ds:
        gender = (sample.get("gender") or "").lower().strip()
        if gender not in groups:
            continue
        if len(groups[gender]) >= MAX_PER_GENDER:
            continue
        groups[gender].append(sample["audio"]["array"])
        if all(len(v) >= MAX_PER_GENDER for v in groups.values()):
            break

    print("Clips collected:", {g: len(v) for g, v in groups.items()})

    chat = load_chat_model()

    for gender, clips in groups.items():
        if not clips:
            print(f"No '{gender}' clips found — skipping.")
            continue

        print(f"\nExtracting embeddings for {len(clips)} {gender} clips …")
        embeddings = []
        for i, wav in enumerate(clips):
            try:
                emb = extract_embedding(chat, wav)
                embeddings.append(emb)
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(clips)}")
            except Exception as e:
                print(f"  clip {i} failed: {e}")

        if not embeddings:
            print(f"No embeddings for {gender} — skipping.")
            continue

        avg = torch.stack(embeddings).mean(dim=0)
        path = os.path.join(OUT_DIR, f"fa_{gender}.pt")
        torch.save(avg, path)
        print(f"Saved: {path}")

    print("\nDone!")
    print("Place the .pt files in ChatTTS-main/ and restart the API server.")
    print("Then use:  voice='fa_female'  or  voice='fa_male'  in requests.")


if __name__ == "__main__":
    main()

# ChatTTS OpenAI-Compatible API

A FastAPI server that exposes ChatTTS via the OpenAI `/v1/audio/speech` interface.
Supports **Chinese**, **English**, and **Farsi (فارسی)** out of the box.

---

## Quick Start

### 1. Install dependencies

```bash
conda activate speech
cd ChatTTS-main
pip install fastapi uvicorn
pip install hazm          # optional — improves Farsi text cleanup
```

### 2. Start the server

```bash
python run_api.py
# Server: http://localhost:8000
# Health: http://localhost:8000/health
```

### 3. Send a request

```bash
# English
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Good morning! Have a wonderful day.","voice":"default","response_format":"mp3"}' \
  --output english.mp3

# Farsi (auto-converted to Finglish internally)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"سلام! امیدوارم روز خوبی داشته باشید.","voice":"default","response_format":"mp3"}' \
  --output farsi.mp3
```

```python
import requests

r = requests.post("http://localhost:8000/v1/audio/speech", json={
    "model": "tts-1",
    "input": "سلام! امیدوارم روز خوبی داشته باشید.",
    "voice": "default",
    "response_format": "mp3",
})
open("output.mp3", "wb").write(r.content)
```

---

## Language Comparison Results

Measured on a live server (RTX GPU, `voice="default"`):

| Language    | Sample Text                              | Audio Size | Time  | Quality         |
|-------------|------------------------------------------|-----------|-------|-----------------|
| **Chinese** | 早上好！希望你有美好的一天。             | ~38.5 KB  | ~4.7s | ⭐⭐⭐⭐⭐ Native  |
| **English** | Good morning! Have a wonderful day.      | ~30.2 KB  | ~4.0s | ⭐⭐⭐⭐ Good     |
| **Farsi**   | سلام! امیدوارم روز خوبی داشته باشید.   | ~40.0 KB  | ~4.6s | ⭐⭐⭐ Accented   |

> **How Farsi works:** ChatTTS was trained on Chinese + English only. Farsi characters map
> to `[UNK]` tokens in the BERT tokenizer, producing near-silent output. This server
> automatically converts Farsi to **Finglish** (phonetic Latin) before synthesis:
>
> ```
> سلام! امیدوارم روز خوبی داشته باشید.
>   ↓  (automatic Finglish conversion)
> salam! omidvaram rooz khoobi dasht bashid.
>   ↓  (ChatTTS synthesizes)
> 🔊  audio output
> ```
>
> The result sounds like an English/Chinese speaker reading Persian words phonetically —
> clear and intelligible, with a slight foreign accent.

---

## Available Voices

| Voice       | Description                    | Required file         |
|-------------|--------------------------------|-----------------------|
| `default`   | Random speaker (ready to use)  | None ✅               |
| `alloy`     | Alternative voice              | `1384.pt`             |
| `echo`      | Alternative voice              | `2443.pt`             |
| `fa_female` | Female Persian voice           | `voices/fa_female.pt` |
| `fa_male`   | Male Persian voice             | `voices/fa_male.pt`   |

**Generate Farsi voice embeddings (optional):**

```bash
pip install datasets soundfile librosa
python tools/fa_cv_prep.py
# Saves: voices/fa_female.pt  and  voices/fa_male.pt
```

Then use:
```python
{"voice": "fa_female"}   # or "fa_male"
```

---

## API Reference

### `POST /v1/audio/speech`

```json
{
  "model": "tts-1",
  "input": "Your text here (Chinese, English, or Farsi)",
  "voice": "default",
  "response_format": "mp3",
  "speed": 1.0,
  "stream": false
}
```

| Field             | Type    | Default     | Description                          |
|-------------------|---------|-------------|--------------------------------------|
| `model`           | string  | required    | Always `"tts-1"`                     |
| `input`           | string  | required    | Text to synthesize (max 2048 chars)  |
| `voice`           | string  | `"default"` | See voice table above                |
| `response_format` | string  | `"mp3"`     | `mp3` / `wav` / `ogg`               |
| `speed`           | float   | `1.0`       | 0.5–2.0                              |
| `stream`          | bool    | `false`     | Stream audio chunks                  |

### `GET /health`

```json
{"status": "healthy", "model_loaded": true}
```

---

## Run the Comparison Yourself

```bash
python compare_test.py
```

Calls the server with Chinese, English, and Farsi text and saves MP3 files to `output_compare/`.

# ChatTTS — Farsi (Persian) Support Guide 🇮🇷

> **Language:** English | [فارسی راهنما](#farsi-quick-reference)

---

## How Farsi Works

ChatTTS was trained on **Chinese + English only**.  
Its BERT tokenizer maps most Farsi characters to `[UNK]`, producing near-silent output.

This project solves that with a **Finglish pipeline**:

```
Farsi text  ──►  Finglish (phonetic Latin)  ──►  ChatTTS  ──►  Audio
سلام        ──►  salam                       ──►  ChatTTS  ──►  🔊
خداحافظ    ──►  khodahafez                  ──►  ChatTTS  ──►  🔊
ایران       ──►  iran                        ──►  ChatTTS  ──►  🔊
```

The conversion is **automatic** — just send plain Farsi text, the server handles the rest.

---

## Quick Start

### 1. Install dependencies

```bash
conda activate speech
pip install hazm           # optional but recommended (better text cleanup)
```

### 2. Start the server

```bash
cd ChatTTS-main
conda activate speech
python run_api.py
# Server running at http://localhost:8000
```

### 3. Send a Farsi request

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"سلام! امیدوارم روز خوبی داشته باشید.","voice":"default","response_format":"mp3"}' \
  --output salam.mp3
```

```python
import requests

r = requests.post("http://localhost:8000/v1/audio/speech", json={
    "model": "tts-1",
    "input": "ایران کشوری با تاریخ و تمدن کهن است.",
    "voice": "default",          # no .pt file required
    "response_format": "mp3",
})
open("output.mp3", "wb").write(r.content)
print(f"Saved {len(r.content):,} bytes")
```

---

## Finglish Conversion Table

Common Farsi phrases and their automatic Finglish output:

| Farsi | Finglish (auto) | Meaning |
|-------|----------------|---------|
| سلام | salam | Hello |
| خداحافظ | khodahafez | Goodbye |
| مرسی ممنون | mersi mamnoon | Thank you |
| روز خوب | rooz khoob | Good day |
| ایران | iran | Iran |
| فارسی | farsi | Persian |
| سلام! امیدوارم روز خوبی داشته باشید. | salam! omidvaram rooz khoobi dashth bashid. | Hello! I hope you have a good day. |
| هوش مصنوعی می‌تواند متن فارسی را به گفتار تبدیل کند. | hosh masnu-i mitavanad matn farsi ra bh goftar tabdil knd. | AI can convert Persian text to speech. |
| ایران کشوری با تاریخ و تمدن کهن است. | iran keshvari ba tarikh v tamaddon kohan ast. | Iran is a country with ancient history. |
| سپاسگزارم که از این سیستم استفاده می‌کنید. | sepasgozaram kh az in sistem estefade mikonid. | Thank you for using this system. |
| فناوری تبدیل متن به گفتار روز به روز بهتر می‌شود. | fanavari tabdil matn bh goftar rooz bh rooz behtar mishavad. | TTS technology improves day by day. |

---

## Available Voices

| Voice name | Description | Required file |
|------------|-------------|---------------|
| `default` | Random speaker (works immediately) | None ✅ |
| `fa_female` | Female Persian voice (from Common Voice) | `voices/fa_female.pt` |
| `fa_male` | Male Persian voice (from Common Voice) | `voices/fa_male.pt` |

### Generate Farsi voice embeddings (optional)

```bash
pip install datasets soundfile librosa
python tools/fa_cv_prep.py
# Saves: voices/fa_female.pt  and  voices/fa_male.pt
```

Then restart the server and use:
```python
{"voice": "fa_female"}   # or "fa_male"
```

---

## Language Comparison Results

Synthesis quality comparison across all three supported languages:

| Language | Example text | Audio size | Quality |
|----------|-------------|-----------|---------|
| **Chinese** | 早上好！希望你有美好的一天。 | ~60–80 KB | ⭐⭐⭐⭐⭐ Native |
| **English** | Good morning! Have a wonderful day. | ~40–60 KB | ⭐⭐⭐⭐ Good |
| **Farsi** | سلام! امیدوارم روز خوبی داشته باشید. | ~40 KB | ⭐⭐⭐ Accented |

> Farsi sounds like an English/Chinese speaker reading Persian words phonetically.
> Quality improves significantly with Farsi-specific voice embeddings (`fa_female.pt`).

---

## Audio Samples / نمونه‌های صوتی

<table>
<tr><th>متن فارسی</th><th>صوت</th></tr>
<tr><td>خداحافظ</td><td><audio controls src="../../samples/mms_02_bye.mp3"></audio></td></tr>
<tr><td>سلام! امیدوارم روز خوبی داشته باشید.</td><td><audio controls src="../../samples/mms_05_hello.mp3"></audio></td></tr>
<tr><td>امیدوارم از این سیستم لذت ببرید.</td><td><audio controls src="../../samples/mms_10_enjoy.mp3"></audio></td></tr>
</table>

---

## Normalizer API

```python
from tools.normalizer.fa import normalizer_fa_finglish, normalizer_fa_hazm, normalizer_fa_basic

# Recommended for ChatTTS input — converts Farsi to phonetic Latin
fn = normalizer_fa_finglish()
print(fn("سلام خداحافظ ایران"))
# → salam khodahafez iran

# Unicode cleanup only (does NOT transliterate)
clean = normalizer_fa_hazm()   # requires: pip install hazm
clean = normalizer_fa_basic()  # built-in fallback, no deps
```

File location: `tools/normalizer/fa.py`  
To add a word: edit the `_FA_WORD_DICT` dictionary in that file.

---

## Data Source

| Source | URL | License |
|--------|-----|---------|
| Mozilla Common Voice | https://commonvoice.mozilla.org/fa | CC-0 (public domain) |
| HuggingFace mirror | https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 | CC-0 |

---

## Farsi Quick Reference

| English | فارسی |
|---------|-------|
| Hello | سلام |
| Goodbye | خداحافظ |
| Thank you | ممنون / مرسی |
| Good morning | صبح بخیر |
| Good night | شب بخیر |
| Iran | ایران |
| Persian | فارسی |
| Artificial Intelligence | هوش مصنوعی |
| Text to Speech | تبدیل متن به گفتار |

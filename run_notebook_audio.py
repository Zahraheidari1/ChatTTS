"""
run_notebook_audio.py
Generates all audio files that the notebook would produce.
Farsi uses MMS (Meta native Persian TTS) — much better quality than Finglish.
Run: python run_notebook_audio.py
"""
import sys, io, os, time, requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = "http://localhost:8000"
os.makedirs("output_notebook", exist_ok=True)

def tts(text, voice="default", fmt="mp3", stream=False):
    t0 = time.time()
    r = requests.post(
        f"{BASE}/v1/audio/speech",
        json={"model": "tts-1", "input": text,
              "voice": voice, "response_format": fmt, "stream": stream},
        timeout=180, stream=stream,
    )
    r.raise_for_status()
    if stream:
        buf = io.BytesIO()
        for chunk in r.iter_content(8192):
            if chunk:
                buf.write(chunk)
        data = buf.getvalue()
    else:
        data = r.content
    return data, time.time() - t0

def save(data, name, fmt="mp3"):
    path = f"output_notebook/{name}.{fmt}"
    with open(path, "wb") as f:
        f.write(data)
    return path

# Health check
try:
    h = requests.get(f"{BASE}/health", timeout=5).json()
    print(f"Server: {h}")
    print(f"Farsi engine: {h.get('farsi_engine','?')}\n")
except Exception as e:
    print(f"ERROR: Server not reachable: {e}")
    sys.exit(1)

results = []

def run(label, text, name, fmt="mp3", stream=False):
    print(f"  [{name}]  {text[:45]}")
    try:
        data, t = tts(text, fmt=fmt, stream=stream)
        path = save(data, name, fmt)
        print(f"    ✅ {len(data):,} bytes  ({t:.1f}s)")
        results.append((label, name, fmt, len(data), t, "OK"))
    except Exception as e:
        print(f"    ❌ FAIL: {e}")
        results.append((label, name, fmt, 0, 0, f"FAIL:{e}"))

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 2 — English  (ChatTTS)")
print("=" * 60)
run("EN single",  "Good morning! Have a wonderful day ahead.",                   "en_single")
run("EN hello",   "Hello! How are you doing today?",                              "en_01_hello")
run("EN ai",      "Artificial intelligence is changing the world.",               "en_02_ai")
run("EN tts",     "Text to speech technology keeps improving.",                   "en_03_tts")
run("EN thanks",  "Thank you very much for using this system.",                   "en_04_thanks")

print()
print("=" * 60)
print("SECTION 3 — Chinese  (ChatTTS)")
print("=" * 60)
run("ZH single",  "早上好！希望你有美好的一天。",   "zh_single")
run("ZH hello",   "你好！今天过得怎么样？",         "zh_01_hello")
run("ZH ai",      "人工智能正在改变世界。",         "zh_02_ai")
run("ZH tts",     "语音合成技术越来越好。",         "zh_03_tts")
run("ZH thanks",  "感谢您使用这个系统。",           "zh_04_thanks")

print()
print("=" * 60)
print("SECTION 4 — Farsi / فارسی  (MMS native Persian)")
print("=" * 60)
run("FA single",  "سلام! امیدوارم روز خوبی داشته باشید.",                        "fa_single")
run("FA salam",   "سلام",                                                          "fa_01_salam")
run("FA bye",     "خداحافظ",                                                       "fa_02_bye")
run("FA thanks",  "مرسی ممنون",                                                    "fa_03_thanks")
run("FA morning", "صبح بخیر",                                                      "fa_04_morning")
run("FA night",   "شب بخیر",                                                       "fa_05_night")
run("FA iran",    "ایران کشوری با تاریخ و تمدن کهن است.",                         "fa_06_iran")
run("FA ai",      "هوش مصنوعی می‌تواند متن فارسی را به گفتار تبدیل کند.",        "fa_07_ai")
run("FA tts",     "فناوری تبدیل متن به گفتار روز به روز بهتر می‌شود.",           "fa_08_tts")
run("FA thanks2", "سپاسگزارم که از این سیستم استفاده می‌کنید.",                  "fa_09_thanks2")
run("FA enjoy",   "امیدوارم از این سیستم لذت ببرید.",                             "fa_10_enjoy")
run("FA quality", "صدای این سیستم چطور است؟ آیا کیفیت خوبی دارد؟",               "fa_11_quality")
run("FA wav",     "صبح بخیر، امیدوارم روز خوبی داشته باشید.",                    "fa_morning", fmt="wav")

print()
print("=" * 60)
print("SECTION 5 — 3-Language Comparison")
print("=" * 60)
run("Compare ZH", "早上好！希望你有美好的一天。",               "compare_chinese")
run("Compare EN", "Good morning! Have a wonderful day.",         "compare_english")
run("Compare FA", "سلام! امیدوارم روز خوبی داشته باشید.",      "compare_farsi")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
ok  = [r for r in results if r[5]=="OK"]
err = [r for r in results if r[5]!="OK"]
print(f"  TOTAL: {len(results)}  |  OK: {len(ok)}  |  Failed: {len(err)}")
print()
print(f"  {'Label':<14} {'File':<30} {'Size':>10}  {'Time':>6}")
print("  " + "-"*65)
for label, name, fmt, sz, t, status in results:
    fname = f"{name}.{fmt}"
    if status == "OK":
        print(f"  {label:<14} {fname:<30} {sz/1024:>8.1f}KB  {t:>5.1f}s")
    else:
        print(f"  {label:<14} FAILED: {status}")
print()
print(f"  Audio files: output_notebook/  ({len(ok)} files)")

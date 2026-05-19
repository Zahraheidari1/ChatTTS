"""
test_mms.py — Download and test Meta MMS Farsi TTS
Run: python test_mms.py
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Meta MMS-TTS (facebook/mms-tts-fas)...")
print("Downloading model if not cached (~30 MB)...\n")

from tools.tts_fa_mms import fa_tts_to_pcm, fa_tts_sr, is_available
from tools.audio import pcm_arr_to_mp3_view, pcm_arr_to_wav_view
import time

if not is_available():
    print("ERROR: transformers not available. Run: pip install transformers")
    sys.exit(1)

os.makedirs("output_mms", exist_ok=True)

TESTS = [
    ("سلام",                                                        "mms_01_salam"),
    ("خداحافظ",                                                     "mms_02_bye"),
    ("مرسی ممنون",                                                  "mms_03_thanks"),
    ("صبح بخیر",                                                    "mms_04_morning"),
    ("سلام! امیدوارم روز خوبی داشته باشید.",                        "mms_05_hello"),
    ("ایران کشوری با تاریخ و تمدن کهن است.",                        "mms_06_iran"),
    ("هوش مصنوعی می‌تواند متن فارسی را به گفتار تبدیل کند.",       "mms_07_ai"),
    ("فناوری تبدیل متن به گفتار روز به روز بهتر می‌شود.",          "mms_08_tts"),
    ("سپاسگزارم که از این سیستم استفاده می‌کنید.",                  "mms_09_thanks2"),
    ("امیدوارم از این سیستم لذت ببرید.",                            "mms_10_enjoy"),
]

sr = fa_tts_sr()
print(f"Output sample rate: {sr} Hz\n")

results = []
for text, name in TESTS:
    t0 = time.time()
    wav = fa_tts_to_pcm(text)
    t1 = time.time()

    # Save as MP3
    mp3 = bytes(pcm_arr_to_mp3_view(wav))
    mp3_path = f"output_mms/{name}.mp3"
    with open(mp3_path, "wb") as f:
        f.write(mp3)

    # Save as WAV (for comparison)
    wavfile = bytes(pcm_arr_to_wav_view(wav))
    wav_path = f"output_mms/{name}.wav"
    with open(wav_path, "wb") as f:
        f.write(wavfile)

    results.append((name, text, len(mp3), len(wavfile), t1-t0))
    print(f"  [{name}]  {len(mp3):,}B mp3 | {len(wavfile):,}B wav  ({t1-t0:.1f}s)")
    print(f"    {text}")

print()
print("=" * 65)
print("  MMS-FA RESULTS  (native Persian, no Finglish)")
print("=" * 65)
print(f"  {'File':<25} {'MP3':>10}  {'WAV':>10}  {'Time':>6}")
print("  " + "-" * 60)
for name, text, mp3s, wavs, t in results:
    print(f"  {name+'.mp3':<25} {mp3s/1024:>8.1f}KB  {wavs/1024:>8.1f}KB  {t:>5.1f}s")
print()
print(f"  Files saved to: output_mms/  ({len(results)*2} files: MP3 + WAV)")
print("  Open any .mp3 file to listen — native Persian quality!")

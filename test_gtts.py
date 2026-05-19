"""
test_gtts.py — Test Google TTS native Persian
Run: python test_gtts.py
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.tts_fa_gtts import fa_gtts_mp3, is_available

if not is_available():
    print("ERROR: gtts not installed. Run: pip install gtts")
    sys.exit(1)

os.makedirs("output_gtts", exist_ok=True)

TESTS = [
    ("سلام",                                                          "fa_01_salam"),
    ("خداحافظ",                                                       "fa_02_bye"),
    ("مرسی ممنون",                                                    "fa_03_thanks"),
    ("صبح بخیر",                                                      "fa_04_morning"),
    ("شب بخیر",                                                       "fa_05_night"),
    ("سلام! امیدوارم روز خوبی داشته باشید.",                          "fa_06_hello"),
    ("ایران کشوری با تاریخ و تمدن کهن است.",                          "fa_07_iran"),
    ("هوش مصنوعی می‌تواند متن فارسی را به گفتار تبدیل کند.",         "fa_08_ai"),
    ("فناوری تبدیل متن به گفتار روز به روز بهتر می‌شود.",            "fa_09_tts"),
    ("سپاسگزارم که از این سیستم استفاده می‌کنید.",                    "fa_10_thanks2"),
    ("امیدوارم از این سیستم لذت ببرید.",                              "fa_11_enjoy"),
    ("صدای این سیستم چطور است؟ آیا کیفیت خوبی دارد؟",                "fa_12_quality"),
]

print("Testing gTTS — Native Persian (Google TTS)\n")

results = []
for text, name in TESTS:
    t0 = time.time()
    try:
        mp3 = fa_gtts_mp3(text)
        t  = time.time() - t0
        path = f"output_gtts/{name}.mp3"
        with open(path, "wb") as f:
            f.write(mp3)
        results.append((name, text, len(mp3), t, "OK"))
        print(f"  [{name}]  {len(mp3):,} bytes  ({t:.1f}s)")
        print(f"    {text}")
    except Exception as e:
        t = time.time() - t0
        results.append((name, text, 0, t, f"FAIL: {e}"))
        print(f"  [{name}]  FAILED: {e}")

print()
print("=" * 65)
print("  gTTS NATIVE PERSIAN RESULTS")
print("=" * 65)
ok  = [r for r in results if r[4]=="OK"]
err = [r for r in results if r[4]!="OK"]
print(f"  OK: {len(ok)}/{len(results)}  |  Failed: {len(err)}")
print()
print(f"  {'File':<25} {'Size':>10}  {'Time':>6}")
print("  " + "-"*50)
for name, text, sz, t, status in results:
    if status == "OK":
        print(f"  {name+'.mp3':<25} {sz/1024:>8.1f}KB  {t:>5.1f}s")
    else:
        print(f"  {name:<25} FAILED: {status}")

print()
print("  Files: output_gtts/")
print("  Open any .mp3 to hear native Persian speech!")

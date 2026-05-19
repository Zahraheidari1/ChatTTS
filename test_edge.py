"""test_edge.py — Test Microsoft Edge TTS Persian voices"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.tts_fa_edge import fa_edge_mp3, VOICE_FEMALE, VOICE_MALE

os.makedirs("output_edge", exist_ok=True)

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

print("Microsoft Edge TTS — Native Persian Neural Voices\n")

for voice_name, voice_code in [("Female (Dilara)", VOICE_FEMALE), ("Male (Farid)", VOICE_MALE)]:
    print(f"=== {voice_name} ===")
    for text, name in TESTS:
        t0 = time.time()
        try:
            mp3 = fa_edge_mp3(text, voice=voice_code)
            t   = time.time() - t0
            suffix = "f" if "Dilara" in voice_name else "m"
            path = f"output_edge/{name}_{suffix}.mp3"
            with open(path, "wb") as f:
                f.write(mp3)
            print(f"  OK  {len(mp3):>8,}B  {t:.1f}s  {text[:40]}")
        except Exception as e:
            print(f"  FAIL  {text[:30]}  -> {e}")
    print()

print("Files saved to output_edge/")
print("  _f.mp3 = female (Dilara Neural)")
print("  _m.mp3 = male   (Farid Neural)")

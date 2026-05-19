"""
tools/normalizer/fa.py
======================
Farsi (Persian) text normalizers for ChatTTS.

Three modes
-----------
normalizer_fa_hazm()      – Unicode cleanup via hazm (pip install hazm).
normalizer_fa_basic()     – Built-in cleanup, no external deps.
normalizer_fa_finglish()  – Converts Farsi → phonetic Latin (Finglish).

Why Finglish?
-------------
ChatTTS was trained on Chinese + English only.  Its BertTokenizer maps most
Farsi characters to [UNK], producing near-silent output.  Converting to
Finglish gives ChatTTS real English tokens that approximate Persian phonemes,
resulting in audible (accented) Farsi-sounding speech with no fine-tuning.

Example
-------
    >>> from tools.normalizer.fa import normalizer_fa_finglish
    >>> fn = normalizer_fa_finglish()
    >>> fn("سلام! امیدوارم روز خوبی داشته باشید.")
    'salam! omidvaram rooz khoobi dashth bashid.'
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable

# ──────────────────────────────────────────────────────────────────────────────
# Word-level dictionary — correct Finglish for common Farsi words
# Applied before the character map so key words sound right.
# ──────────────────────────────────────────────────────────────────────────────
_FA_WORD_DICT: dict[str, str] = {
    # Greetings
    "سلام": "salam",
    "خداحافظ": "khodahafez",
    "درود": "dorood",
    "صبح بخیر": "sobh bekhair",
    "شب بخیر": "shab bekhair",
    "روز بخیر": "rooz bekhair",
    # Courtesy
    "ممنون": "mamnoon",
    "ممنونم": "mamnoonam",
    "مرسی": "mersi",
    "سپاسگزارم": "sepasgozaram",
    "متشکرم": "motashakeram",
    "خواهش می‌کنم": "khahesh mikonam",
    "خواهش میکنم": "khahesh mikonam",
    "ببخشید": "bebakhshid",
    # Common words
    "خوب": "khoob",
    "خوبم": "khoobam",
    "بله": "bale",
    "نه": "na",
    "باشه": "bashe",
    "باشد": "bashad",
    "است": "ast",
    "روز": "rooz",
    "شب": "shab",
    "صبح": "sobh",
    "امروز": "emrooz",
    "فردا": "farda",
    "دیروز": "diruz",
    # Places / language
    "ایران": "iran",
    "تهران": "tehran",
    "فارسی": "farsi",
    "فارس": "fars",
    "ایرانی": "irani",
    # Technology
    "هوش مصنوعی": "hosh masnu-i",
    "گفتار": "goftar",
    "متن": "matn",
    "تبدیل": "tabdil",
    "سیستم": "sistem",
    "فناوری": "fanavari",
    # Grammar / common
    "چطوری": "chetori",
    "چطور": "chetor",
    "کجا": "koja",
    "چرا": "chera",
    "چی": "chi",
    "کی": "ki",
    "من": "man",
    "تو": "to",
    "ما": "ma",
    "شما": "shoma",
    "آن": "an",
    "این": "in",
    "زیبا": "ziba",
    "زیباست": "zibast",
    "کهن": "kohan",
    "تاریخ": "tarikh",
    "تمدن": "tamaddon",
    "کشور": "keshvar",
    "امیدوارم": "omidvaram",
    "استفاده": "estefade",
    "بهتر": "behtar",
    "می‌کنید": "mikonid",
    "می‌شود": "mishavad",
    "می‌تواند": "mitavanad",
    "می‌کنم": "mikonam",
}

# ──────────────────────────────────────────────────────────────────────────────
# Character-level map: Farsi/Arabic character → Latin phoneme
# ──────────────────────────────────────────────────────────────────────────────
_FA_TO_LATIN: dict[str, str] = {
    "آ": "a",  "ا": "a",  "أ": "a",  "إ": "e",  "ء": "",
    "ب": "b",  "پ": "p",  "ت": "t",  "ث": "s",
    "ج": "j",  "چ": "ch", "ح": "h",  "خ": "kh",
    "د": "d",  "ذ": "z",  "ر": "r",  "ز": "z",  "ژ": "zh",
    "س": "s",  "ش": "sh", "ص": "s",  "ض": "z",
    "ط": "t",  "ظ": "z",  "ع": "",   "غ": "gh",
    "ف": "f",  "ق": "gh", "ک": "k",  "ك": "k",  "گ": "g",
    "ل": "l",  "م": "m",  "ن": "n",
    "و": "v",  "ه": "h",  "ی": "i",  "ي": "i",  "ى": "i",
    # Numerals (Eastern Arabic → Western)
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    # Punctuation
    "،": ",",  "؟": "?",  "؛": ";",  "«": '"',  "»": '"',
    # Cleanup
    "ـ": "",   "ّ": "",   "‌": " ",  "‍": "",
}

# Unicode diacritics (harakat) U+064B–U+065F
_DIACRITICS = re.compile(r"[ً-ٟ]")
_SPACES     = re.compile(r" {2,}")
# Residual Arabic/Persian block characters
_RESIDUAL   = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def _unicode_normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return _DIACRITICS.sub("", text)


# ── Public normalizers ─────────────────────────────────────────────────────────

def normalizer_fa_finglish() -> Callable[[str], str]:
    """
    Farsi → Finglish (phonetic Latin) converter.
    Uses a word-level dictionary first, then character-by-character mapping.
    This is the recommended normalizer to use with ChatTTS for Farsi input.
    """
    digraph_map = {k: v for k, v in _FA_TO_LATIN.items() if len(k) > 1}
    char_map    = {k: v for k, v in _FA_TO_LATIN.items() if len(k) == 1}
    trans_table = str.maketrans(char_map)

    def _convert(text: str) -> str:
        text = _unicode_normalize(text)
        # ① Word dictionary (longest match first)
        for fa in sorted(_FA_WORD_DICT, key=len, reverse=True):
            if fa in text:
                text = text.replace(fa, _FA_WORD_DICT[fa])
        # ② Digraph substitutions
        for fa, lat in digraph_map.items():
            text = text.replace(fa, lat)
        # ③ Character map
        text = text.translate(trans_table)
        # ④ Strip residual Persian/Arabic Unicode
        text = _RESIDUAL.sub("", text)
        return _SPACES.sub(" ", text).strip()

    return _convert


def normalizer_fa_hazm() -> Callable[[str], str]:
    """
    Farsi normalizer using the hazm library (pip install hazm).
    Cleans Unicode; does NOT transliterate.
    Combine with normalizer_fa_finglish() for ChatTTS input.
    """
    from hazm import Normalizer as HazmNorm
    _n = HazmNorm(
        correct_spacing=True,
        remove_diacritics=True,
        persian_numbers=True,
        unicodes_replacement=True,
    )
    return _n.normalize


def normalizer_fa_basic() -> Callable[[str], str]:
    """
    Lightweight built-in Farsi normalizer — no external deps.
    Maps Arabic chars to Persian equivalents and strips diacritics.
    Does NOT transliterate.
    """
    _ar_to_fa = str.maketrans({
        "ك": "ک", "ي": "ی", "ى": "ی",
        "ۀ": "ی", "ة": "ه", "ٱ": "ا",
    })

    def _clean(text: str) -> str:
        text = _unicode_normalize(text)
        text = text.translate(_ar_to_fa)
        return _SPACES.sub(" ", text).strip()

    return _clean

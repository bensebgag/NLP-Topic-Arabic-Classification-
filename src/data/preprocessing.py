import re , unicodedata
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.loader import load_dataset 

df = load_dataset()  


_CONTROL = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")      
_MENTION = re.compile(r"[@#]\w+")
_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")
_TATWEEL = "\u0640"
_DEC_SEP = re.compile(r"(?<=\d)[،,](?=\d)")
_NOT_ALLOWED = re.compile(r"[^\u0621-\u064Aa-z0-9.\s]")
_LONE_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_SPACES = re.compile(r"\s+")
 
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_LETTERS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ة": "ه",
    "ی": "ي", "ک": "ك",
})
 
def clean_arabic_text(text) -> str:
    if not isinstance(text, str) or not text:
        return ""
 
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text)
 
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _MENTION.sub(" ", text)
 
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_DIGITS).translate(_LETTERS)
    text = _DEC_SEP.sub(".", text)
 
    text = text.lower()
    text = _NOT_ALLOWED.sub(" ", text)
    text = _LONE_DOT.sub(" ", text)
 
    return _SPACES.sub(" ", text).strip()
"""Caption filtering and the evaluation normalisation shared by every model and the baseline."""
import re
import unicodedata

_JUNK = re.compile(r"^(blank|dash|\[music\]|music|\W*)\.?$", re.IGNORECASE)


def exclusion_reason(text):
    """Same rule as the ResNet baseline: empty strings, subtitle placeholders, pure punctuation."""
    t = " ".join(str(text).split())
    if not t:
        return "empty"
    if _JUNK.fullmatch(t):
        return "placeholder"
    return None


def normalize(text):
    """Evaluation normalisation: NFKC, lowercase, drop every Unicode punctuation char, collapse spaces.

    Apostrophes are deleted ("it's" -> "its"); other punctuation becomes a space ("56.Circle" -> "56 circle").
    """
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"['‘’`]", "", t)
    t = "".join(" " if unicodedata.category(c).startswith("P") or c in "`^~|<>=+$" else c for c in t)
    return " ".join(t.split())

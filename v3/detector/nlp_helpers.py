"""
nlp_helpers.py — shared lightweight NLP utilities.

Tries to use NLTK / spaCy when available; falls back to pure-Python
implementations so the detector works even in minimal environments.
"""

import re
import math
from collections import Counter
from typing import List, Optional

# ── NLTK ────────────────────────────────────────────────────────────────────
try:
    import nltk
    for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet",
                 "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg
                           else f"corpora/{_pkg}")
        except LookupError:
            nltk.download(_pkg, quiet=True)
    from nltk.tokenize import sent_tokenize as _nltk_sent, word_tokenize as _nltk_word
    NLTK_OK = True
except Exception:
    NLTK_OK = False

# ── spaCy ────────────────────────────────────────────────────────────────────
try:
    import spacy as _spacy
    _nlp = _spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    _nlp = None


# ── Public helpers ────────────────────────────────────────────────────────────

def sent_tokenize(text: str) -> List[str]:
    """Split text into sentences."""
    if NLTK_OK:
        try:
            return _nltk_sent(text)
        except Exception:
            pass
    # Fallback
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def word_tokenize(text: str) -> List[str]:
    """Split text into tokens (alphanumeric)."""
    if NLTK_OK:
        try:
            return [w for w in _nltk_word(text) if w.isalnum()]
        except Exception:
            pass
    return re.findall(r"\b\w+\b", text)


def nlp_doc(text: str):
    """Return a spaCy Doc or None."""
    if SPACY_OK and _nlp is not None:
        try:
            return _nlp(text[:10_000])
        except Exception:
            pass
    return None


def get_entities(text: str):
    """Return list of (text, label) entity tuples."""
    doc = nlp_doc(text)
    if doc is not None:
        return [(ent.text, ent.label_) for ent in doc.ents]
    # Fallback: capitalised words as rough proper-noun proxy
    return [(m.group(), "PROPN") for m in re.finditer(r"\b[A-Z][a-z]+\b", text)]


def count_tokens(text: str):
    """Return (word_tokens, numeric_tokens, proper_noun_tokens) counts."""
    doc = nlp_doc(text)
    if doc is not None:
        words   = [t for t in doc if t.is_alpha]
        nums    = [t for t in doc if t.like_num]
        propns  = [t for t in doc if t.pos_ == "PROPN"]
        return len(words), len(nums), len(propns)
    # Fallback
    tokens = re.findall(r"\b\w+\b", text)
    nums   = re.findall(r"\b\d[\d,\.]*\b", text)
    propns = re.findall(r"\b[A-Z][a-z]+\b", text)
    return len(tokens), len(nums), len(propns)

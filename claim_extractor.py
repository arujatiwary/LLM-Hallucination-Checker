# claim_extractor.py
"""
Lazy-loading claim extractor using spaCy.
This file avoids importing or loading the spaCy model at module import time,
so the app can start on Streamlit Cloud even if the spaCy model isn't preinstalled.
"""

import spacy

_nlp = None

def _ensure_spacy_model():
    """
    Load en_core_web_sm if available, otherwise download it at runtime then load.
    Safe to call multiple times.
    """
    global _nlp
    if _nlp is not None:
        return _nlp

    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Try to download the model at runtime (Streamlit Cloud allows this during startup)
        from spacy.cli import download
        download("en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_claims(text):
    """
    Extracts factual-sounding declarative sentences from text.

    - Ensures the spaCy model is available (downloads if needed).
    - Returns a list of sentence strings that look like factual claims.
    """
    nlp = _ensure_spacy_model()
    doc = nlp(text)
    claims = []
    for sent in doc.sents:
        s = sent.text.strip()
        # preserve your original heuristic: starts with uppercase and ends with a period
        if s and s[0].isupper() and s.endswith("."):
            claims.append(s)
    return claims

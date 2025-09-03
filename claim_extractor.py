import re

def extract_claims(text):
    """
    Extract factual claims from text using regex-based sentence splitting.
    Avoids spaCy dependency so it's lightweight and Streamlit Cloud friendly.
    Returns a list of candidate claims.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    claims = []
    for sent in sentences:
        claim = sent.strip()
        if len(claim.split()) > 3:  # ignore fragments
            claims.append(claim)

    return claims

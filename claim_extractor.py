import spacy

# Load English model (make sure to install: pip install spacy && python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

def extract_claims(text):
    """
    Extracts factual claims (sentences) from text.
    Uses spaCy sentence segmentation to avoid skipping claims.
    Returns a list of candidate claims.
    """
    doc = nlp(text)
    claims = []

    for sent in doc.sents:
        claim = sent.text.strip()

        # Filter out very short fragments that aren't claims
        if len(claim.split()) > 3:  # at least 4 words
            claims.append(claim)

    return claims

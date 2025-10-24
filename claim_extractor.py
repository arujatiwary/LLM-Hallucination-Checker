import spacy

nlp = spacy.load("en_core_web_sm")

def extract_claims(text):
    """
    Extracts factual-sounding declarative sentences from text.
    """
    doc = nlp(text)
    claims = []
    for sent in doc.sents:
        if sent.text.strip() and sent.text[0].isupper() and sent.text.endswith("."):
            claims.append(sent.text.strip())
    return claims

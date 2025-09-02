import spacy

nlp = spacy.load("en_core_web_sm")

def extract_claims(text):
    """
    Extract simple factual claims (sentences with entities/numbers).
    """
    doc = nlp(text)
    claims = []
    for sent in doc.sents:
        if any(ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "TIME", "QUANTITY", "PERCENT", "MONEY", "CARDINAL"]
               for ent in sent.ents):
            claims.append(sent.text.strip())
    return claims

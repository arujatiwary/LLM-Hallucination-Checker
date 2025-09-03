from transformers import pipeline

# Load NER pipeline once
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def extract_claims(text):
    """
    Extracts factual-looking claims from text using NER as a proxy.
    Right now, it just extracts sentences containing named entities.
    """
    sentences = text.split(".")
    claims = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        entities = ner_pipeline(sentence)
        if entities:  # only keep sentences with named entities
            claims.append(sentence.strip())

    return claims

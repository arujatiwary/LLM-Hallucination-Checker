# --- START OF FILE claim_extractor.py ---
import streamlit as st # Assuming this is the top of the file, though it's not relevant to this snippet
import re
from transformers import pipeline
import spacy # Import spacy

# Load spaCy model (assuming it's installed via requirements.txt)
try:
    nlp = spacy.load("en_core_web_sm")
    print("Claim Extractor: SpaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:
    print(f"Claim Extractor: Error loading spaCy model 'en_core_web_sm': {e}")
    print("Please ensure 'en_core_web_sm' is correctly installed via requirements.txt for your Python version.")
    raise SystemExit("SpaCy model 'en_core_web_sm' not found or failed to load in Claim Extractor.")

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def extract_claims(text):
    """
    Extracts factual-looking claims from text using NER as a proxy.
    Right now, it just extracts sentences containing named entities.
    """
    # Using spaCy's sentence segmentation for better accuracy
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    claims = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        entities = ner_pipeline(sentence)
        if entities:  # only keep sentences with named entities
            claims.append(sentence.strip())

    return claims

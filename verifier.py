import os
import re
import streamlit as st
import logging

# NLP / ML imports
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from duckduckgo_search import DDGS
import spacy
from sentence_transformers import SentenceTransformer, util

# --- logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- label map (identical mapping) ---
label_map = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

# ----------------- Device -----------------
def get_device():
    """Return torch device (cuda if available else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Cached model loaders -----------------
@st.cache_resource(show_spinner=False)
def load_nli_model(model_name="roberta-large-mnli"):
    """
    Load NLI tokenizer and model and move model to device.
    Cached for Streamlit so models load once per server process.
    """
    device = get_device()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        logger.info(f"Verifier: Successfully loaded NLI model '{model_name}'.")
    except Exception as e:
        logger.warning(f"Verifier: Error loading NLI model '{model_name}': {e}")
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        logger.info(f"Verifier: Successfully loaded NLI model '{model_name}' using cache directory: {cache_dir}.")
    return tokenizer, model, device

@st.cache_resource(show_spinner=False)
def load_ner_pipeline(model_name="dslim/bert-base-NER"):
    """
    Load NER pipeline for query generation. Returns a HuggingFace pipeline.
    """
    device = get_device()
    device_idx = device.index if getattr(device, "type", None) == "cuda" else -1
    try:
        ner = pipeline("ner", model=model_name, aggregation_strategy="simple",
                       device=device_idx)
        logger.info(f"Verifier: Successfully loaded NER pipeline '{model_name}' for query generation.")
    except Exception as e:
        logger.warning(f"Verifier: Error loading NER pipeline '{model_name}': {e}")
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        ner = pipeline("ner", model=model_name, aggregation_strategy="simple",
                       device=device_idx, cache_dir=cache_dir)
        logger.info(f"Verifier: Successfully loaded NER pipeline '{model_name}' using cache directory: {cache_dir}.")
    return ner

@st.cache_resource(show_spinner=False)
def load_spacy_model(model_name="en_core_web_sm"):
    """Load spaCy model (cached)."""
    try:
        nlp = spacy.load(model_name)
        logger.info(f"Loaded spaCy model '{model_name}'.")
    except Exception as e:
        logger.error(f"Error loading spaCy model '{model_name}': {e}")
        raise
    return nlp

@st.cache_resource(show_spinner=False)
def load_embedder(model_name="all-MiniLM-L6-v2"):
    """Load SentenceTransformer embedder (cached)."""
    embedder = SentenceTransformer(model_name)
    logger.info(f"Loaded embedder '{model_name}'.")
    return embedder

# ----------------- Utilities & Core functions -----------------

def generate_search_query(claim):
    """
    Generate a refined search query from a claim using NER and noun-chunk heuristics.
    Mirrors the notebook logic exactly.
    """
    ner_pipeline = load_ner_pipeline()
    nlp = load_spacy_model()
    claim_lower = claim.lower()

    # Step 1: Extract NER entities and merge subwords
    entities = ner_pipeline(claim)
    merged_tokens = []
    current = ""
    for e in entities:
        word = e.get("word", "")
        if isinstance(word, str) and word.startswith("##"):
            current += word[2:]
        else:
            if current:
                merged_tokens.append(current)
            current = word
    if current:
        merged_tokens.append(current)

    ner_phrases = []
    for token in merged_tokens:
        if " " in token:
            ner_phrases.append(f'"{token}"')
        else:
            ner_phrases.append(token)

    # Step 2: Extract noun phrases with spaCy
    doc = nlp(claim)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        if chunk.text not in merged_tokens:
            if len(chunk.text.split()) > 1:
                noun_phrases.append(f'"{chunk.text}"')
            else:
                noun_phrases.append(chunk.text)

    # Step 3: Heuristic for "discovered"
    if "discovered" in claim_lower:
        match = re.search(r'discovered ([\w\s]+)', claim_lower)
        if match:
            noun_phrases.append(f'who discovered "{match.group(1).strip()}"')
        else:
            noun_phrases.append("who discovered")

    # Step 4: Combine NER + noun phrases, dedupe while keeping order
    all_phrases = ner_phrases + noun_phrases
    query_parts = list(dict.fromkeys(all_phrases))

    if query_parts:
        q = " AND ".join(query_parts) + " facts"
        logger.debug(f"Generated refined query: {q}")
        return q
    fallback = f"{claim} facts"
    logger.debug(f"No query parts found; using fallback query: {fallback}")
    return fallback

def search_snippets(claim_original, num_results=10):
    """
    Search web snippets using DDGS.
    Returns list of snippet bodies (strings).
    (Behavior matches the notebook exactly.)
    """
    results = []
    search_query = generate_search_query(claim_original)

    logger.debug(f"Original claim for search: '{claim_original}'")
    logger.debug(f"Generated search query: '{search_query}' with {num_results} results...")

    try:
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(search_query, max_results=num_results))

            if not ddgs_results:
                logger.debug("No results for generated query; falling back to original claim search.")
                ddgs_results = list(ddgs.text(claim_original, max_results=num_results))

            for i, r in enumerate(ddgs_results):
                if "body" in r and r["body"].strip():
                    results.append(r["body"])
                    logger.debug(f"Snippet {i+1} (first 100 chars): {r['body'][:100]}...")
                else:
                    logger.debug(f"Snippet {i+1} had no or empty 'body' key: {r}")
            if not results:
                logger.debug("No substantial 'body' content found in any search results.")
    except Exception as e:
        logger.error(f"DuckDuckGo search failed for query '{search_query}' or '{claim_original}': {e}")
        results = []
    return results

# Note: We keep the filter function available in case other code uses it,
# but we do NOT call it in verify_claim so behavior matches the notebook.
def filter_snippets_by_similarity(claim, snippets, threshold=0.7):
    """
    Filter snippets by cosine similarity using sentence-transformers embedder.
    Present for compatibility but not used in the verification path to match notebook behavior.
    """
    if not snippets:
        logger.debug("No snippets provided to filter by similarity.")
        return []
    embedder = load_embedder()
    claim_emb = embedder.encode(claim, convert_to_tensor=True)
    snippet_embs = embedder.encode(snippets, convert_to_tensor=True)
    cosine_scores = util.cos_sim(claim_emb, snippet_embs)[0]
    filtered = [snippet for snippet, score in zip(snippets, cosine_scores) if float(score) >= float(threshold)]
    logger.debug(f"Filtered {len(filtered)}/{len(snippets)} snippets above similarity threshold {threshold}")
    return filtered

def classify_nli(premise, hypothesis):
    """
    Classify premise/hypothesis pair using NLI model.
    Returns (label, score) where label is one of CONTRADICTION/NEUTRAL/ENTAILMENT.
    """
    tokenizer_nli, model_nli, device = load_nli_model()
    inputs = tokenizer_nli(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model_nli(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    label_id = int(probs.argmax())
    logger.debug(f"NLI classify: label_id={label_id}, score={probs[label_id]:.4f}")
    return label_map[label_id], float(probs[label_id])

def verify_claim(claim, top_k=10):
    """
    Full claim verification pipeline matching the notebook logic:
    1. search_snippets (top_k default 10)
    2. run NLI classification on each snippet
    3. aggregate max entailment and contradiction scores
    4. final decision:
       - hallucination if max_contradiction >= CONFIDENCE_THRESHOLD + 0.05
       - verified if max_entailment >= CONFIDENCE_THRESHOLD
       - hallucination if max_contradiction >= CONFIDENCE_THRESHOLD
       - otherwise uncertain
    """
    logger.info(f"Verifying claim: '{claim}' (top_k={top_k})")
    snippets = search_snippets(claim, num_results=top_k)

    if not snippets:
        logger.info(f"No snippets available to verify the claim '{claim}'. Returning 'uncertain'.")
        return {"claim": claim, "status": "uncertain", "evidence": []}

    logger.info(f"Number of snippets received: {len(snippets)}")
    if snippets:
        logger.debug(f"First snippet (first 100 chars): {snippets[0][:100]}...")

    best_status = "uncertain"
    all_evidence_snippets = []

    CONFIDENCE_THRESHOLD = 0.75
    max_entailment_score = 0.0
    max_contradiction_score = 0.0
    best_entailment_snippet = None
    best_contradiction_snippet = None

    logger.info("Starting NLI classification for each snippet...")
    for i, snippet in enumerate(snippets):
        all_evidence_snippets.append(snippet)
        try:
            label, score = classify_nli(snippet, claim)
            logger.info(f"Snippet {i+1} vs Claim - NLI Result: Label='{label}', Score={score:.4f}")
            logger.debug(f"  Premise (Snippet start): {snippet[:150]}...")
            logger.debug(f"  Hypothesis (Claim): {claim}")

            if label == "ENTAILMENT":
                if score > max_entailment_score:
                    max_entailment_score = score
                    best_entailment_snippet = snippet
            elif label == "CONTRADICTION":
                if score > max_contradiction_score:
                    max_contradiction_score = score
                    best_contradiction_snippet = snippet
        except Exception as e:
            logger.error(f"Error classifying snippet {i+1}: {e}")
            continue

    # Decision logic identical to notebook
    if max_contradiction_score >= CONFIDENCE_THRESHOLD + 0.05:
        best_status = "hallucination"
    elif max_entailment_score >= CONFIDENCE_THRESHOLD:
        best_status = "verified"
    elif max_contradiction_score >= CONFIDENCE_THRESHOLD:
        best_status = "hallucination"
    else:
        best_status = "uncertain"

    # Choose best snippet to show in UI
    if best_status == "verified" and best_entailment_snippet:
        final_context_snippet = best_entailment_snippet
    elif best_status == "hallucination" and best_contradiction_snippet:
        final_context_snippet = best_contradiction_snippet
    elif all_evidence_snippets:
        final_context_snippet = all_evidence_snippets[0]
    else:
        final_context_snippet = None

    logger.info(f"Final decision for claim '{claim}': Status='{best_status}', "
                f"Max Entailment={max_entailment_score:.4f}, Max Contradiction={max_contradiction_score:.4f}")

    return {
        "claim": claim,
        "status": best_status,
        "evidence": all_evidence_snippets,
        "best_snippet": final_context_snippet
    }
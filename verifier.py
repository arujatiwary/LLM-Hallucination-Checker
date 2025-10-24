# verifier.py
import os
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from duckduckgo_search import DDGS
import spacy
from sentence_transformers import SentenceTransformer, util

# Device (works locally; on Streamlit Cloud this will be CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verifier Device (lazy) set to use {device}")

# Globals to be filled by ensure_models()
_tokenizer_nli = None
_model_nli = None
_label_map = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
_ner_pipeline = None
_nlp = None
_embedder = None

def ensure_models():
    """Load heavy models on-demand. Safe to call multiple times."""
    global _tokenizer_nli, _model_nli, _ner_pipeline, _nlp, _embedder

    # spaCy model (download at runtime if missing)
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")

    # NLI tokenizer + model (roberta-large-mnli)
    if _tokenizer_nli is None or _model_nli is None:
        MODEL_NAME_NLI = "roberta-large-mnli"
        _tokenizer_nli = AutoTokenizer.from_pretrained(MODEL_NAME_NLI)
        _model_nli = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_NLI).to(device)

    # NER pipeline
    if _ner_pipeline is None:
        MODEL_NAME_NER = "dslim/bert-base-NER"
        _ner_pipeline = pipeline("ner", model=MODEL_NAME_NER, aggregation_strategy="simple",
                                device=device.index if device.type == 'cuda' else -1)

    # Sentence-transformer embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')

    return {
        "nlp": _nlp,
        "tokenizer_nli": _tokenizer_nli,
        "model_nli": _model_nli,
        "ner_pipeline": _ner_pipeline,
        "embedder": _embedder,
    }

# The rest of your functions remain the same, but reference objects via ensure_models()
def generate_search_query(claim, ner_pipeline=None, nlp=None):
    ner_pipeline = ner_pipeline or ensure_models()["ner_pipeline"]
    nlp = nlp or ensure_models()["nlp"]
    claim_lower = claim.lower()
    entities = ner_pipeline(claim)
    merged_tokens = []
    current = ""
    for e in entities:
        word = e["word"]
        if word.startswith("##"):
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

    doc = nlp(claim)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        if chunk.text not in merged_tokens:
            if len(chunk.text.split()) > 1:
                noun_phrases.append(f'"{chunk.text}"')
            else:
                noun_phrases.append(chunk.text)

    if "discovered" in claim_lower:
        match = re.search(r'discovered ([\w\s]+)', claim_lower)
        if match:
            noun_phrases.append(f'who discovered "{match.group(1).strip()}"')
        else:
            noun_phrases.append("who discovered")

    all_phrases = ner_phrases + noun_phrases
    query_parts = list(dict.fromkeys(all_phrases))
    if query_parts:
        return " AND ".join(query_parts) + " facts"
    return f"{claim} facts"


def search_snippets(claim_original, num_results=10):
    results = []
    search_query = generate_search_query(claim_original)
    print(f"\n[DEBUG verifier.py] Generated search query: '{search_query}'")

    try:
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(search_query, max_results=num_results))
            if not ddgs_results:
                ddgs_results = list(ddgs.text(claim_original, max_results=num_results))
            for r in ddgs_results:
                if "body" in r and r["body"].strip():
                    results.append(r["body"])
    except Exception as e:
        print(f"[ERROR verifier.py] DuckDuckGo search failed: {e}")
        results = []
    return results


def filter_snippets_by_similarity(claim, snippets, threshold=0.7):
    embedder = ensure_models()["embedder"]
    if not snippets:
        return []
    claim_emb = embedder.encode(claim, convert_to_tensor=True)
    snippet_embs = embedder.encode(snippets, convert_to_tensor=True)
    cosine_scores = util.cos_sim(claim_emb, snippet_embs)[0]
    return [snippet for snippet, score in zip(snippets, cosine_scores) if float(score) >= threshold]


def classify_nli(premise, hypothesis):
    tokenizer_nli = ensure_models()["tokenizer_nli"]
    model_nli = ensure_models()["model_nli"]
    inputs = tokenizer_nli(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model_nli(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    label_id = int(probs.argmax())
    label_map = _label_map
    return label_map[label_id], float(probs[label_id])


def verify_claim(claim, top_k=10, sim_threshold=0.7):
    # load models lazily
    ensure_models()

    snippets = search_snippets(claim, num_results=top_k)
    snippets = filter_snippets_by_similarity(claim, snippets, threshold=sim_threshold)

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": []}

    CONFIDENCE_THRESHOLD = 0.75
    max_entailment_score = 0.0
    max_contradiction_score = 0.0
    best_entailment_snippet = None
    best_contradiction_snippet = None

    for snippet in snippets:
        try:
            label, score = classify_nli(snippet, claim)
            if label == "ENTAILMENT" and score > max_entailment_score:
                max_entailment_score = score
                best_entailment_snippet = snippet
            elif label == "CONTRADICTION" and score > max_contradiction_score:
                max_contradiction_score = score
                best_contradiction_snippet = snippet
        except Exception as e:
            continue

    if max_contradiction_score >= CONFIDENCE_THRESHOLD + 0.05:
        best_status = "hallucination"
    elif max_entailment_score >= CONFIDENCE_THRESHOLD:
        best_status = "verified"
    elif max_contradiction_score >= CONFIDENCE_THRESHOLD:
        best_status = "hallucination"
    else:
        best_status = "uncertain"

    best_snippet = best_entailment_snippet or best_contradiction_snippet or (snippets[0] if snippets else None)

    return {"claim": claim, "status": best_status, "evidence": snippets, "best_snippet": best_snippet}

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F
from duckduckgo_search import DDGS
import os
import re

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verifier Device set to use {device}")

# --- NLI Model Loading ---
MODEL_NAME_NLI = "roberta-large-mnli"
try:
    tokenizer_nli = AutoTokenizer.from_pretrained(MODEL_NAME_NLI)
    model_nli = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_NLI).to(device)
    print(f"Verifier: Successfully loaded NLI model '{MODEL_NAME_NLI}'.")
except Exception as e:
    print(f"Verifier: Error loading NLI model '{MODEL_NAME_NLI}': {e}")
    print("Verifier: Attempting to load NLI model with an explicit cache directory.")
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer_nli = AutoTokenizer.from_pretrained(MODEL_NAME_NLI, cache_dir=cache_dir)
    model_nli = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_NLI, cache_dir=cache_dir).to(device)
    print(f"Verifier: Successfully loaded NLI model '{MODEL_NAME_NLI}' using cache directory: {cache_dir}.")

label_map = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

# --- NER Model for Query Generation ---
MODEL_NAME_NER = "dslim/bert-base-NER"
try:
    ner_pipeline = pipeline("ner", model=MODEL_NAME_NER, aggregation_strategy="simple",
                            device=device.index if device.type == 'cuda' else -1)
    print(f"Verifier: Successfully loaded NER pipeline '{MODEL_NAME_NER}' for query generation.")
except Exception as e:
    print(f"Verifier: Error loading NER pipeline '{MODEL_NAME_NER}': {e}")
    print("Verifier: Attempting to load NER pipeline with an explicit cache directory.")
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    ner_pipeline = pipeline("ner", model=MODEL_NAME_NER, aggregation_strategy="simple",
                            device=device.index if device.type == 'cuda' else -1, cache_dir=cache_dir)
    print(f"Verifier: Successfully loaded NER pipeline '{MODEL_NAME_NER}' using cache directory: {cache_dir}.")

def generate_search_query(claim):
    """
    Generates a more refined search query based on the claim using NER and keyword extraction.
    Prioritizes proper nouns and specific keywords.
    """
    original_claim_lower = claim.lower()
    query_parts = []
    
    # 1. Use NER to extract key entities
    entities = ner_pipeline(claim)
    for entity in entities:
        # We want to be specific, so use quotes for multi-word entities
        if len(entity['word'].split()) > 1:
            query_parts.append(f'"{entity["word"]}"')
        else:
            query_parts.append(entity['word'])

    # 2. Add significant verbs and nouns (simple heuristic, can be improved with POS tagging)
    # This is a very basic approach; for more robustness, consider spaCy for POS tagging.
    words = re.findall(r'\b\w+\b', original_claim_lower)
    stop_words = set(["is", "was", "are", "were", "has", "have", "had", "a", "an", "the", "in", "on", "at", "for", "with", "and", "but", "or", "of", "to", "from", "by"])
    
    significant_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Add significant single words if they are not already covered by NER entities
    for s_word in significant_words:
        if s_word not in original_claim_lower: # Avoid re-adding parts of NER phrases
            query_parts.append(s_word)

    # 3. Add context words like "facts" or "information" if the query is too short
    if len(query_parts) < 3:
        query_parts.append("facts")
    
    # Use set to remove duplicates and join with "AND" for precise searching
    final_query = ' AND '.join(list(set(query_parts)))
    
    # Fallback to original claim + facts if nothing substantial was extracted
    if not final_query:
        return f"{claim} facts"
    
    return final_query

def search_snippets(claim_original, num_results=15): # Increased num_results for a wider net
    """
    Searches DuckDuckGo for snippets relevant to the query.
    Takes the original claim, generates a search query from it, then searches.
    Returns a list of snippet bodies.
    """
    results = []
    search_query = generate_search_query(claim_original)
    
    print(f"\n[DEBUG verifier.py] Original claim for search: '{claim_original}'")
    print(f"  [DEBUG verifier.py] Generated search query: '{search_query}' with {num_results} results...")

    try:
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(search_query, max_results=num_results))
            
            if not ddgs_results:
                print(f"  [DEBUG verifier.py] No results for generated query, falling back to original claim search.")
                ddgs_results = list(ddgs.text(claim_original, max_results=num_results)) # Fallback
                if not ddgs_results: # If still no results, try just the keywords from the original claim
                    print(f"  [DEBUG verifier.py] No results for original claim, trying keyword-only search.")
                    keyword_query = ' '.join(re.findall(r'\b\w{3,}\b', claim_original.lower())) # Words >= 3 chars
                    if keyword_query:
                        ddgs_results = list(ddgs.text(keyword_query, max_results=num_results))

            for i, r in enumerate(ddgs_results):
                if "body" in r and r["body"].strip():
                    results.append(r["body"])
                    print(f"  [DEBUG verifier.py] Snippet {i+1} (first 100 chars): {r['body'][:100]}...")
                else:
                    print(f"  [DEBUG verifier.py] Snippet {i+1} had no or empty 'body' key: {r}")
            if not results:
                print("[DEBUG verifier.py] No substantial 'body' content found in any search results.")
    except Exception as e:
        print(f"[ERROR verifier.py] DuckDuckGo search failed: {e}")
        results = []
    return results

def classify_nli(premise, hypothesis):
    """
    Runs NLI on (premise, hypothesis) and returns the best label and its score.
    Uses the pre-loaded tokenizer and model.
    """
    inputs = tokenizer_nli(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model_nli(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    label_id = int(probs.argmax())
    return label_map[label_id], float(probs[label_id])

def verify_claim(claim, top_k=15): # Adjusted top_k to match search_snippets
    """
    Verifies a factual claim against search snippets using an NLI model.
    Returns dict: {claim, status, evidence: list of snippets, best_snippet}
    """
    print(f"\n[DEBUG verifier.py] Verifying claim: '{claim}' (top_k={top_k})")
    snippets = search_snippets(claim, num_results=top_k)

    if not snippets:
        print(f"[DEBUG verifier.py] No snippets available to verify the claim '{claim}'. Returning 'uncertain'.")
        return {"claim": claim, "status": "uncertain", "evidence": [], "best_snippet": None}

    best_status = "uncertain"
    all_evidence_snippets = []

    # Tunable confidence thresholds
    STRONG_ENTAILMENT_THRESHOLD = 0.80 # High confidence for verified
    WEAK_ENTAILMENT_THRESHOLD = 0.60  # Minimum for any entailment consideration
    STRONG_CONTRADICTION_THRESHOLD = 0.85 # Very high confidence for hallucination
    WEAK_CONTRADICTION_THRESHOLD = 0.65 # Minimum for any contradiction consideration

    # Track highest scores and associated snippets
    max_entailment_score = 0.0
    max_contradiction_score = 0.0
    
    best_entailment_snippet = None
    best_contradiction_snippet = None

    # Track count of strong signals
    strong_entailment_count = 0
    strong_contradiction_count = 0

    print("\n--- NLI Classification Results for Each Snippet ---")
    for i, snippet in enumerate(snippets):
        all_evidence_snippets.append(snippet)
        try:
            label, score = classify_nli(snippet, claim)
            print(f"  [DEBUG verifier.py] Snippet {i+1} vs Claim:")
            print(f"    Premise (Snippet): {snippet[:150]}...")
            print(f"    Hypothesis (Claim): {claim}")
            print(f"    NLI Result: Label='{label}', Score={score:.4f}")

            if label == "ENTAILMENT":
                if score > max_entailment_score:
                    max_entailment_score = score
                    best_entailment_snippet = snippet
                if score >= STRONG_ENTAILMENT_THRESHOLD:
                    strong_entailment_count += 1
            elif label == "CONTRADICTION":
                if score > max_contradiction_score:
                    max_contradiction_score = score
                    best_contradiction_snippet = snippet
                if score >= STRONG_CONTRADICTION_THRESHOLD:
                    strong_contradiction_count += 1

        except Exception as e:
            print(f"  [ERROR verifier.py] Error classifying snippet {i+1}: {e}")
            continue
    
    # --- Advanced Decision Logic ---
    # Prioritize strong contradiction if it exists from multiple sources or a very high single source
    if strong_contradiction_count >= 1 and max_contradiction_score >= STRONG_CONTRADICTION_THRESHOLD:
        best_status = "hallucination"
    elif strong_entailment_count >= 1 and max_entailment_score >= STRONG_ENTAILMENT_THRESHOLD:
        best_status = "verified"
    # If no strong signals, check for weaker but consistent signals
    elif max_entailment_score >= WEAK_ENTAILMENT_THRESHOLD and strong_entailment_count >= (top_k / 3): # If a third of snippets weakly entail
        best_status = "verified"
    elif max_contradiction_score >= WEAK_CONTRADICTION_THRESHOLD and strong_contradiction_count >= (top_k / 3):
        best_status = "hallucination"
    # Final fallback if still uncertain but some weak signals exist
    elif max_entailment_score > max_contradiction_score + 0.1 and max_entailment_score >= WEAK_ENTAILMENT_THRESHOLD - 0.1: # Entailment moderately stronger
        best_status = "verified"
    elif max_contradiction_score > max_entailment_score + 0.1 and max_contradiction_score >= WEAK_CONTRADICTION_THRESHOLD - 0.1: # Contradiction moderately stronger
        best_status = "hallucination"
    else:
        best_status = "uncertain"

    # Decide which snippet to return as 'best_snippet' for context in UI
    final_context_snippet = None
    if best_status == "verified" and best_entailment_snippet:
        final_context_snippet = best_entailment_snippet
    elif best_status == "hallucination" and best_contradiction_snippet:
        final_context_snippet = best_contradiction_snippet
    elif all_evidence_snippets: 
        # If uncertain, pick the snippet with the highest absolute score (either entailment or contradiction)
        # or simply the first one if scores are too close
        if max(max_entailment_score, max_contradiction_score) > WEAK_ENTAILMENT_THRESHOLD - 0.1: # only if there's *some* decent signal
            if max_entailment_score >= max_contradiction_score:
                final_context_snippet = best_entailment_snippet if best_entailment_snippet else all_evidence_snippets[0]
            else:
                final_context_snippet = best_contradiction_snippet if best_contradiction_snippet else all_evidence_snippets[0]
        else:
            final_context_snippet = all_evidence_snippets[0] # Fallback to first if no strong lead

    print(f"\n[DEBUG verifier.py] Final decision for claim '{claim}': Status='{best_status}', Max Entailment={max_entailment_score:.4f}, Max Contradiction={max_contradiction_score:.4f}")
    
    return {"claim": claim, "status": best_status, "evidence": all_evidence_snippets, "best_snippet": final_context_snippet}

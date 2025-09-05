import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F
from duckduckgo_search import DDGS
import os
import re # Import regex for query refinement

# --- Device Setup ---
# Prioritize CUDA if available, otherwise use CPU
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
MODEL_NAME_NER = "dslim/bert-base-NER" # Using the same NER as claim_extractor for consistency
try:
    # Use device.index for pipeline if CUDA is available, otherwise -1 for CPU
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
    Generates a more refined search query based on the claim using NER.
    Prioritizes proper nouns and specific keywords.
    """
    original_claim_lower = claim.lower()
    
    # 1. Use NER to extract key entities
    entities = ner_pipeline(claim)
    key_phrases = []
    
    # Filter for relevant entity types (Person, Location, Organization, Miscellaneous)
    for entity in entities:
        # We want to be specific, so use quotes for multi-word entities
        if len(entity['word'].split()) > 1: # Multi-word entity
            key_phrases.append(f'"{entity["word"]}"')
        else: # Single-word entity, less strict matching
            key_phrases.append(entity['word'])

    # 2. Add specific heuristics for common patterns if NER alone isn't enough
    if "discovered" in original_claim_lower:
        # If NER found who discovered it, great. Otherwise, try to infer.
        # Example: "Albert Einstein discovered penicillin." -> "who discovered penicillin"
        match = re.search(r'discovered ([\w\s]+)', original_claim_lower)
        if match:
            key_phrases.append(f"who discovered \"{match.group(1).strip()}\"")
        else:
            key_phrases.append("who discovered")
    
    # Prioritize key phrases identified by NER or specific heuristics
    if key_phrases:
        # Join with "AND" to ensure all terms are present in search
        query_parts = list(set(key_phrases)) # Use set to remove duplicates
        return ' AND '.join(query_parts) + " facts"
    
    # 3. Fallback: Original claim with "facts" or "information"
    return f"{claim} facts"

def search_snippets(claim_original, num_results=10):
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
            
            # Fallback if the generated query yields no results, try original claim
            if not ddgs_results:
                print(f"  [DEBUG verifier.py] No results for generated query, falling back to original claim search.")
                ddgs_results = list(ddgs.text(claim_original, max_results=num_results))

            for i, r in enumerate(ddgs_results):
                if "body" in r and r["body"].strip(): # Ensure snippet body is not empty
                    results.append(r["body"])
                    print(f"  [DEBUG verifier.py] Snippet {i+1} (first 100 chars): {r['body'][:100]}...")
                else:
                    print(f"  [DEBUG verifier.py] Snippet {i+1} had no or empty 'body' key: {r}")
            if not results:
                print("[DEBUG verifier.py] No substantial 'body' content found in any search results.")
    except Exception as e:
        print(f"[ERROR verifier.py] DuckDuckGo search failed for query '{search_query}' or '{claim_original}': {e}")
        results = []
    return results

def classify_nli(premise, hypothesis):
    """
    Runs NLI on (premise, hypothesis) and returns the best label and its score.
    Uses the pre-loaded tokenizer and model.
    """
    # max_length for RoBERTa is typically 512
    inputs = tokenizer_nli(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model_nli(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    label_id = int(probs.argmax())
    return label_map[label_id], float(probs[label_id])

def verify_claim(claim, top_k=10):
    """
    Verifies a factual claim against search snippets using an NLI model.
    Returns dict: {claim, status, evidence: list of snippets}
    """
    print(f"\n[DEBUG verifier.py] Verifying claim: '{claim}' (top_k={top_k})")
    snippets = search_snippets(claim, num_results=top_k)

    if not snippets:
        print(f"[DEBUG verifier.py] No snippets available to verify the claim '{claim}'. Returning 'uncertain'.")
        return {"claim": claim, "status": "uncertain", "evidence": []}

    print(f"[DEBUG verifier.py] Type of 'snippets' before iteration: {type(snippets)}")
    print(f"[DEBUG verifier.py] Number of snippets received: {len(snippets)}")
    if snippets:
        print(f"[DEBUG verifier.py] First snippet (first 100 chars): {snippets[0][:100]}...")

    best_status = "uncertain"
    all_evidence_snippets = []

    # Tunable confidence threshold for strong classifications
    CONFIDENCE_THRESHOLD = 0.75 # Good balance, adjust if needed

    # Track highest entailment and contradiction scores separately
    max_entailment_score = 0.0
    max_contradiction_score = 0.0
    
    # Store the snippet that gave the highest entailment/contradiction score
    best_entailment_snippet = None
    best_contradiction_snippet = None

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
            elif label == "CONTRADICTION":
                if score > max_contradiction_score:
                    max_contradiction_score = score
                    best_contradiction_snippet = snippet

        except Exception as e:
            print(f"  [ERROR verifier.py] Error classifying snippet {i+1}: {e}")
            continue
    
    # --- Final Decision Logic after reviewing all snippets ---
    # Prioritize contradiction if it's very strong, to highlight potential hallucinations more aggressively
    if max_contradiction_score >= CONFIDENCE_THRESHOLD + 0.05: # Give contradiction a slight edge
        best_status = "hallucination"
    elif max_entailment_score >= CONFIDENCE_THRESHOLD:
        best_status = "verified"
    elif max_contradiction_score >= CONFIDENCE_THRESHOLD: # If contradiction is just at threshold
        best_status = "hallucination"
    else:
        best_status = "uncertain"

    # Decide which snippet to return as 'best_snippet' for context in UI
    if best_status == "verified" and best_entailment_snippet:
        final_context_snippet = best_entailment_snippet
    elif best_status == "hallucination" and best_contradiction_snippet:
        final_context_snippet = best_contradiction_snippet
    elif all_evidence_snippets: # If uncertain, just return the first snippet found for some context
        final_context_snippet = all_evidence_snippets[0]
    else:
        final_context_snippet = None # Should not happen if all_evidence_snippets is empty

    print(f"\n[DEBUG verifier.py] Final decision for claim '{claim}': Status='{best_status}', Max Entailment={max_entailment_score:.4f}, Max Contradiction={max_contradiction_score:.4f}")
    
    # Return all snippets for detailed view, but the best_snippet for the main 'evidence' in UI
    return {"claim": claim, "status": best_status, "evidence": all_evidence_snippets, "best_snippet": final_context_snippet}

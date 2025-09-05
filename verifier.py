import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F
from duckduckgo_search import DDGS
import os
import re
import spacy # Import spacy for more robust keyword extraction

# Load spaCy model for English (run `python -m spacy download en_core_web_sm` once)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

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
    Generates a more refined search query based on the claim using NER and spaCy for keyword extraction.
    Prioritizes proper nouns and specific keywords, aiming for direct answers.
    """
    doc = nlp(claim)
    query_parts = []
    
    # 1. Extract Named Entities using NER (from HuggingFace pipeline)
    entities = ner_pipeline(claim)
    for entity in entities:
        if len(entity['word'].split()) > 1:
            query_parts.append(f'"{entity["word"]}"') # Phrase match for multi-word entities
        else:
            query_parts.append(entity['word'])

    # 2. Extract significant nouns and adjectives using spaCy POS tagging
    # Filter for Nouns (NOUN, PROPN), Adjectives (ADJ), and Verbs (VERB)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"] and not token.is_stop and not token.is_punct and len(token.text) > 2:
            # Only add if not already part of an NER entity (to avoid redundancy for proper nouns)
            is_ner_entity = False
            for entity in entities:
                if token.text.lower() in entity['word'].lower(): # Simple check
                    is_ner_entity = True
                    break
            if not is_ner_entity:
                query_parts.append(token.lemma_) # Use lemma for better matching (e.g., "discovers" -> "discover")

    # 3. Add context words if the query is too short or seems too generic
    # Try to make the query sound like a factual inquiry
    if len(query_parts) < 3 and "is" in claim.lower() or "was" in claim.lower():
        query_parts.insert(0, "what is") # E.g., "what is Taj Mahal location"
    elif len(query_parts) < 3:
        query_parts.append("information")
    
    # Remove duplicates and construct the query
    final_query_components = list(set(query_parts))
    
    # Prioritize question-like structure if it makes sense, otherwise join with spaces
    if any(q.startswith("who") or q.startswith("what") or q.startswith("where") for q in final_query_components):
        final_query = ' '.join(final_query_components)
    else:
        final_query = ' AND '.join(final_query_components) # Use AND for more precise keyword matching

    # Add "facts" or "details" explicitly if not too many components already
    if "facts" not in final_query.lower() and "details" not in final_query.lower() and len(final_query_components) < 4:
        final_query += " facts"
    
    # Fallback: if somehow nothing useful extracted, use the whole claim
    if not final_query.strip():
        return f"{claim} facts"
    
    print(f"  [DEBUG verifier.py] Raw query parts: {query_parts}")
    print(f"  [DEBUG verifier.py] Final query string: {final_query}")
    return final_query


def search_snippets(claim_original, num_results=20): # Increased num_results to 20 to get more raw data
    """
    Searches DuckDuckGo for snippets relevant to the query.
    Takes the original claim, generates a search query from it, then searches.
    Returns a list of filtered snippet bodies.
    """
    all_raw_snippets = []
    search_query = generate_search_query(claim_original)
    
    print(f"\n[DEBUG verifier.py] Original claim for search: '{claim_original}'")
    print(f"  [DEBUG verifier.py] Generated search query: '{search_query}' with {num_results} results...")

    try:
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(search_query, max_results=num_results))
            
            # Fallback if the generated query yields no results, try original claim, then keyword-only
            if not ddgs_results:
                print(f"  [DEBUG verifier.py] No results for generated query '{search_query}', falling back to original claim search.")
                ddgs_results = list(ddgs.text(claim_original, max_results=num_results))
                if not ddgs_results:
                    print(f"  [DEBUG verifier.py] No results for original claim, trying keyword-only search.")
                    # Simple keyword extraction for a very broad fallback
                    keyword_query_fallback = ' '.join(re.findall(r'\b\w{3,}\b', claim_original.lower()))
                    if keyword_query_fallback:
                        ddgs_results = list(ddgs.text(keyword_query_fallback, max_results=num_results))

            for i, r in enumerate(ddgs_results):
                if "body" in r and r["body"].strip():
                    all_raw_snippets.append(r["body"])
                    print(f"  [DEBUG verifier.py] Raw Snippet {i+1} (first 100 chars): {r['body'][:100]}...")
                else:
                    print(f"  [DEBUG verifier.py] Raw Snippet {i+1} had no or empty 'body' key: {r}")
            if not all_raw_snippets:
                print("[DEBUG verifier.py] No substantial 'body' content found in any search results.")
    except Exception as e:
        print(f"[ERROR verifier.py] DuckDuckGo search failed: {e}")
        all_raw_snippets = []
    
    # --- Post-search Snippet Filtering and Ranking ---
    if not all_raw_snippets:
        return []

    # Keywords from the claim for filtering
    claim_keywords = set([token.lemma_.lower() for token in nlp(claim) if not token.is_stop and not token.is_punct and len(token.text) > 2])
    
    filtered_snippets = []
    for snippet in all_raw_snippets:
        snippet_doc = nlp(snippet)
        snippet_words = set([token.lemma_.lower() for token in snippet_doc if not token.is_stop and not token.is_punct and len(token.text) > 2])
        
        # Calculate overlap score: count of claim keywords present in snippet
        overlap_score = len(claim_keywords.intersection(snippet_words))
        
        # Filter out snippets with very low overlap or known generic "facts" phrases
        if overlap_score > 0 and not any(phrase in snippet.lower() for phrase in ["wtf fun facts", "interesting facts that will change", "random & interesting facts", "collection of fun facts"]):
            filtered_snippets.append((overlap_score, snippet))
        else:
            print(f"  [DEBUG verifier.py] Filtering out generic snippet (first 50 chars): {snippet[:50]}...")

    # Sort snippets by overlap score (highest first) and return only the body
    filtered_snippets.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in filtered_snippets[:num_results]] # Return up to num_results of the best ones


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

def verify_claim(claim, top_k=15): # Adjusted top_k to match search_snippets and be for NLI processing
    """
    Verifies a factual claim against search snippets using an NLI model.
    Returns dict: {claim, status, evidence: list of snippets, best_snippet}
    """
    print(f"\n[DEBUG verifier.py] Verifying claim: '{claim}' (top_k={top_k})")
    snippets = search_snippets(claim, num_results=top_k) # Pass num_results here for filtered snippets

    if not snippets:
        print(f"[DEBUG verifier.py] No *relevant* snippets available to verify the claim '{claim}'. Returning 'uncertain'.")
        return {"claim": claim, "status": "uncertain", "evidence": [], "best_snippet": None}

    best_status = "uncertain"
    all_evidence_snippets = snippets # These are now the filtered ones

    # Tunable confidence thresholds
    STRONG_ENTAILMENT_THRESHOLD = 0.85 # Higher confidence for verified
    WEAK_ENTAILMENT_THRESHOLD = 0.65  # Minimum for any entailment consideration
    STRONG_CONTRADICTION_THRESHOLD = 0.90 # Very high confidence for hallucination
    WEAK_CONTRADICTION_THRESHOLD = 0.70 # Minimum for any contradiction consideration

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
                if score >= WEAK_ENTAILMENT_THRESHOLD: # Count even weaker ones if they contribute
                    strong_entailment_count += 1
            elif label == "CONTRADICTION":
                if score > max_contradiction_score:
                    max_contradiction_score = score
                    best_contradiction_snippet = snippet
                if score >= WEAK_CONTRADICTION_THRESHOLD:
                    strong_contradiction_count += 1

        except Exception as e:
            print(f"  [ERROR verifier.py] Error classifying snippet {i+1}: {e}")
            continue
    
    # --- Advanced Decision Logic ---
    # Give priority to very strong single signals or multiple strong signals
    if max_contradiction_score >= STRONG_CONTRADICTION_THRESHOLD:
        best_status = "hallucination"
    elif max_entailment_score >= STRONG_ENTAILMENT_THRESHOLD:
        best_status = "verified"
    # If no single strong signal, look for a majority of weaker, but still significant, signals
    elif strong_entailment_count > strong_contradiction_count and max_entailment_score >= WEAK_ENTAILMENT_THRESHOLD:
        best_status = "verified"
    elif strong_contradiction_count > strong_entailment_count and max_contradiction_score >= WEAK_CONTRADICTION_THRESHOLD:
        best_status = "hallucination"
    # If still ambiguous, compare the absolute max scores with a slight margin
    elif max_entailment_score > max_contradiction_score + 0.1 and max_entailment_score >= WEAK_ENTAILMENT_THRESHOLD:
        best_status = "verified"
    elif max_contradiction_score > max_entailment_score + 0.1 and max_contradiction_score >= WEAK_CONTRADICTION_THRESHOLD:
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
        # For uncertain, or if best_X_snippet is None, try to provide *some* relevant snippet.
        # Pick the one with the highest NLI score (entailment or contradiction) if above a minimal threshold
        if max(max_entailment_score, max_contradiction_score) >= WEAK_ENTAILMENT_THRESHOLD - 0.1:
            if max_entailment_score >= max_contradiction_score:
                final_context_snippet = best_entailment_snippet if best_entailment_snippet else all_evidence_snippets[0]
            else:
                final_context_snippet = best_contradiction_snippet if best_contradiction_snippet else all_evidence_snippets[0]
        else:
            final_context_snippet = all_evidence_snippets[0] # Fallback to first if all NLI scores are very low

    print(f"\n[DEBUG verifier.py] Final decision for claim '{claim}': Status='{best_status}', Max Entailment={max_entailment_score:.4f}, Max Contradiction={max_contradiction_score:.4f}")
    
    return {"claim": claim, "status": best_status, "evidence": all_evidence_snippets, "best_snippet": final_context_snippet}

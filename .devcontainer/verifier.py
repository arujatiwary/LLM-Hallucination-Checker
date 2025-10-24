import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F
from duckduckgo_search import DDGS
import os
import re
import spacy
from sentence_transformers import SentenceTransformer, util

# --- Device Setup ---
# The device variable will be defined and passed from app.py
# For standalone testing, you can uncomment these:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Verifier Device set to use {device}")

# --- NLI Label Map ---
label_map = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

def generate_search_query(claim, ner_pipeline, nlp):
    """
    Generates a refined search query from a claim, accepting pre-loaded models.
    (Contains the logic previously split between claim_extractor.py and search.py)
    """
    if ner_pipeline is None or nlp is None:
        print("[ERROR verifier.py] NER/NLP models not provided. Falling back to simple query.")
        return f"{claim} facts"

    claim_lower = claim.lower()

    # --- Step 1: Extract NER entities and merge subwords ---
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

    # --- Step 2: Extract noun phrases with spaCy ---
    doc = nlp(claim)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        if chunk.text not in merged_tokens:
            if len(chunk.text.split()) > 1:
                noun_phrases.append(f'"{chunk.text}"')
            else:
                noun_phrases.append(chunk.text)

    # --- Step 3: Heuristic for "discovered" ---
    if "discovered" in claim_lower:
        match = re.search(r'discovered ([\w\s]+)', claim_lower)
        if match:
            noun_phrases.append(f'who discovered \"{match.group(1).strip()}\"')
        else:
            noun_phrases.append("who discovered")

    # --- Step 4: Combine, Dedupe, and Refine (Improved for "Earth" scenario) ---
    all_phrases = ner_phrases + noun_phrases
    
    stop_words = {'the', 'a', 'is', 'of', 'and', 'it', 'was'}
    
    query_parts = []
    seen = set()

    # Priority 1: Multi-word noun/NER phrases (already quoted)
    for phrase in [p for p in all_phrases if len(p.split()) > 1]:
        if phrase not in seen:
            query_parts.append(phrase)
            seen.add(phrase)

    # Priority 2: Important single words that aren't stop words
    for word in [p for p in all_phrases if len(p.split()) == 1]:
        if word.lower() not in stop_words and word not in seen:
            query_parts.append(word)
            seen.add(word)

    final_query_parts = query_parts[:4]

    if final_query_parts:
        return " AND ".join(p.strip('"') for p in final_query_parts if p.strip('"')) + " facts"

    # --- Step 5: Fallback ---
    return f"{claim} facts"


def search_snippets(claim_original, num_results, ner_pipeline, nlp):
    """
    Searches DuckDuckGo for snippets relevant to the query.
    """
    results = []
    search_query = generate_search_query(claim_original, ner_pipeline, nlp)
    
    print(f"\n[DEBUG verifier.py] Original claim for search: '{claim_original}'")
    print(f"  [DEBUG verifier.py] Generated search query: '{search_query}' with {num_results} results...")

    try:
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(search_query, max_results=num_results))
            
            # Fallback 
            if not ddgs_results:
                print(f"  [DEBUG verifier.py] No results for generated query, falling back to original claim search.")
                ddgs_results = list(ddgs.text(claim_original, max_results=num_results))

            for i, r in enumerate(ddgs_results):
                if "body" in r and r["body"].strip():
                    results.append(r["body"])
                    print(f"  [DEBUG verifier.py] Snippet {i+1} (first 100 chars): {r['body'][:100]}...")
            if not results:
                print("[DEBUG verifier.py] No substantial 'body' content found in any search results.")
    except Exception as e:
        print(f"[ERROR verifier.py] DuckDuckGo search failed for query: {e}")
        results = []
    return results

def filter_snippets_by_similarity(claim, snippets, embedder, threshold=0.7):
    """
    Filters snippets by cosine similarity to the claim using Sentence-Transformers.
    """
    if not snippets or embedder is None:
        return []
        
    try:
        claim_emb = embedder.encode(claim, convert_to_tensor=True)
        snippet_embs = embedder.encode(snippets, convert_to_tensor=True)
        cosine_scores = util.cos_sim(claim_emb, snippet_embs)[0]
        
        filtered = [snippet for snippet, score in zip(snippets, cosine_scores) if score >= threshold]
        print(f"[DEBUG verifier.py] Filtered {len(filtered)}/{len(snippets)} snippets above similarity threshold {threshold}")
        return filtered
    except Exception as e:
        print(f"[ERROR verifier.py] Similarity filtering failed: {e}. Returning all snippets.")
        return snippets

def classify_nli(premise, hypothesis, tokenizer_nli, model_nli, device):
    """
    Runs NLI on (premise, hypothesis) and returns the best label and its score.
    """
    if tokenizer_nli is None or model_nli is None:
        return "ERROR_NLI_MODEL_MISSING", 0.0
        
    inputs = tokenizer_nli(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model_nli(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    label_id = int(probs.argmax())
    return label_map[label_id], float(probs[label_id])


def verify_claim(claim, top_k, sim_threshold, nli_confidence_threshold, ner_pipeline, nlp, embedder, tokenizer_nli, model_nli, device):
    """
    Main verification function.
    """
    print(f"\n[DEBUG verifier.py] Verifying claim: '{claim}' (top_k={top_k}, sim_threshold={sim_threshold})")
    
    # 1. Search for snippets
    snippets = search_snippets(claim, num_results=top_k, ner_pipeline=ner_pipeline, nlp=nlp)

    # 2. Filter by cosine similarity
    snippets = filter_snippets_by_similarity(claim, snippets, embedder, threshold=sim_threshold)

    if not snippets:
        print(f"[DEBUG verifier.py] No snippets available after filtering for the claim '{claim}'. Returning 'uncertain'.")
        # FIX: Ensure consistent keys are returned to prevent KeyError in app.py
        return {
            "claim": claim, 
            "status": "uncertain", 
            "evidence": [], 
            "best_snippet": None,
            "max_entailment": 0.0,
            "max_contradiction": 0.0
        }

    best_status = "uncertain"
    all_evidence_snippets = []

    CONFIDENCE_THRESHOLD = nli_confidence_threshold
    max_entailment_score = 0.0
    max_contradiction_score = 0.0
    best_entailment_snippet = None
    best_contradiction_snippet = None
  
    print("\n--- NLI Classification Results for Each Snippet ---")
    # 3. NLI Classification
    for i, snippet in enumerate(snippets):
        all_evidence_snippets.append(snippet)
        try:
            label, score = classify_nli(snippet, claim, tokenizer_nli, model_nli, device)
            print(f"  [DEBUG verifier.py] Snippet {i+1} vs Claim: NLI Result: Label='{label}', Score={score:.4f}")

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
    
    # 4. Final Decision Logic
    if max_contradiction_score >= CONFIDENCE_THRESHOLD + 0.05:
        best_status = "hallucination"
    elif max_entailment_score >= CONFIDENCE_THRESHOLD:
        best_status = "verified"
    elif max_contradiction_score >= CONFIDENCE_THRESHOLD:
        best_status = "hallucination"
    else:
        best_status = "uncertain"

    # Decide which snippet to return as 'best_snippet'
    if best_status == "verified" and best_entailment_snippet:
        final_context_snippet = best_entailment_snippet
    elif best_status == "hallucination" and best_contradiction_snippet:
        final_context_snippet = best_contradiction_snippet
    elif all_evidence_snippets:
        final_context_snippet = all_evidence_snippets[0]
    else:
        final_context_snippet = None

    print(f"\n[DEBUG verifier.py] Final decision for claim '{claim}': Status='{best_status}', Max Entailment={max_entailment_score:.4f}, Max Contradiction={max_contradiction_score:.4f}")
    
    return {
        "claim": claim,
        "status": best_status,
        "max_entailment": max_entailment_score,
        "max_contradiction": max_contradiction_score,
        "evidence": all_evidence_snippets,
        "best_snippet": final_context_snippet
    }
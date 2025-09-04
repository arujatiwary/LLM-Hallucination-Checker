import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from duckduckgo_search import DDGS # Import DDGS directly here
import os # For potential cache directory setup

# --- Device Setup ---
# Prioritize CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verifier Device set to use {device}")

# --- NLI Model Loading ---
MODEL_NAME = "roberta-large-mnli"
try:
    # Attempt to load from local cache first
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    print(f"Verifier: Successfully loaded NLI model '{MODEL_NAME}'.")
except Exception as e:
    print(f"Verifier: Error loading NLI model '{MODEL_NAME}': {e}")
    print("Verifier: Attempting to load with an explicit cache directory.")
    # Fallback to a specified cache directory if there are permission issues
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=cache_dir).to(device)
    print(f"Verifier: Successfully loaded NLI model '{MODEL_NAME}' using cache directory: {cache_dir}.")

label_map = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

def search_snippets(query, num_results=5):
    """
    Searches DuckDuckGo for snippets relevant to the query.
    Returns a list of snippet bodies.
    """
    results = []
    print(f"\n[DEBUG verifier.py] Searching for: '{query}' with {num_results} results...")
    try:
        # Using DDGS context manager for cleaner resource handling
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(query, max_results=num_results))
            for i, r in enumerate(ddgs_results):
                if "body" in r:
                    results.append(r["body"])
                    print(f"  [DEBUG verifier.py] Snippet {i+1} (first 100 chars): {r['body'][:100]}...")
                else:
                    print(f"  [DEBUG verifier.py] Snippet {i+1} had no 'body' key: {r}")
            if not results:
                print("[DEBUG verifier.py] No 'body' content found in any search results for the query.")
    except Exception as e:
        print(f"[ERROR verifier.py] DuckDuckGo search failed for query '{query}': {e}")
        # Ensure results is an empty list if search fails
        results = []
    return results

def classify_nli(premise, hypothesis):
    """
    Runs NLI on (premise, hypothesis) and returns the best label and its score.
    Uses the pre-loaded tokenizer and model.
    """
    # Truncate inputs to prevent tokenization errors with very long texts
    # max_length for RoBERTa is typically 512
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    label_id = int(probs.argmax())
    return label_map[label_id], float(probs[label_id])

def verify_claim(claim, top_k=5):
    """
    Verifies a factual claim against search snippets using an NLI model.
    Returns dict: {claim, status, evidence: list of snippets}
    """
    print(f"\n[DEBUG verifier.py] Verifying claim: '{claim}' (top_k={top_k})")
    snippets = search_snippets(claim, num_results=top_k)

    if not snippets:
        print(f"[DEBUG verifier.py] No snippets available to verify the claim '{claim}'. Returning 'uncertain'.")
        return {"claim": claim, "status": "uncertain", "evidence": []} # Return empty list for consistency

    # Debugging lines for 'snippets' before iteration
    print(f"[DEBUG verifier.py] Type of 'snippets' before iteration: {type(snippets)}")
    print(f"[DEBUG verifier.py] Number of snippets received: {len(snippets)}")
    if snippets:
        print(f"[DEBUG verifier.py] First snippet (first 100 chars): {snippets[0][:100]}...")
    # END Debugging lines

    best_status = "uncertain"
    best_conf = 0.0
    best_snippet = None # Will store the single best snippet
    all_evidence_snippets = [] # Store all snippets used for verification

    # Tunable confidence threshold for strong classifications
    CONFIDENCE_THRESHOLD = 0.75 # Adjust as needed (e.g., 0.7 to 0.9)

    print("\n--- NLI Classification Results for Each Snippet ---")
    for i, snippet in enumerate(snippets):
        all_evidence_snippets.append(snippet) # Add every snippet to the evidence list
        try:
            label, score = classify_nli(snippet, claim)
            print(f"  [DEBUG verifier.py] Snippet {i+1} vs Claim:")
            print(f"    Premise (Snippet): {snippet[:150]}...") # Truncate for display
            print(f"    Hypothesis (Claim): {claim}")
            print(f"    NLI Result: Label='{label}', Score={score:.4f}")

            # --- Logic for updating best_status based on confidence and hierarchy ---
            # Prioritize strong entailment or contradiction
            if label == "ENTAILMENT" and score >= CONFIDENCE_THRESHOLD:
                if best_status != "verified" or score > best_conf:
                    best_status = "verified"
                    best_conf = score
                    best_snippet = snippet
            elif label == "CONTRADICTION" and score >= CONFIDENCE_THRESHOLD:
                # Contradiction gets priority if strong enough
                # If a strong contradiction is found, it immediately becomes the best status
                if best_status != "hallucination" or score > best_conf:
                    best_status = "hallucination"
                    best_conf = score
                    best_snippet = snippet
            elif label == "NEUTRAL":
                # Only update to 'uncertain' if no strong 'verified' or 'hallucination'
                # has been established yet, AND this NEUTRAL score is higher than current best_conf.
                if best_status not in ["verified", "hallucination"]:
                    if score > best_conf:
                        best_status = "uncertain"
                        best_conf = score
                        best_snippet = snippet # Assign snippet if this NEUTRAL is the highest confidence
            # --- End Logic for updating best_status ---

        except Exception as e:
            print(f"  [ERROR verifier.py] Error classifying snippet {i+1}: {e}")
            continue

    print(f"\n[DEBUG verifier.py] Final decision for claim '{claim}': Status='{best_status}', Confidence={best_conf:.4f}")
    
    # Return all snippets as evidence for the UI
    return {"claim": claim, "status": best_status, "evidence": all_evidence_snippets}

# Example Usage (for testing verifier.py directly)
if __name__ == "__main__":
    print("\n--- Running verifier.py in standalone mode for testing ---")
    claims_to_test = [
        "The capital of France is Paris.",          # Expected: verified
        "Humans can breathe underwater unaided.",   # Expected: hallucination
        "The Eiffel Tower is located in Rome.",     # Expected: hallucination
        "The sun is a star.",                       # Expected: verified
        "The moon is made of cheese.",              # Expected: hallucination
        "The Earth revolves around the Sun.",       # Expected: verified
        "Dogs lay eggs.",                           # Expected: hallucination
        "The Amazon rainforest is in Africa."       # Expected: hallucination
    ]

    for claim_to_verify in claims_to_test:
        result = verify_claim(claim_to_verify, top_k=10) # Increased top_k for better chances
        print(f"\n--- Result for: \"{result['claim']}\" ---")
        print(f"Status: {result['status'].upper()}")
        if result['evidence']:
            print(f"Number of evidence snippets: {len(result['evidence'])}")
            print(f"First evidence (first 150 chars): \"{result['evidence'][0][:150]}...\"")
        else:
            print("No specific evidence found.")
        print("-" * 70)

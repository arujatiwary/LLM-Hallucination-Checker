from duckduckgo_search import DDGS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os

# Device setup
# Prioritize CUDA if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Load NLI model (roberta-large-mnli)
MODEL_NAME = "roberta-large-mnli"
try:
    # Attempt to load from local cache first
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    print(f"Successfully loaded NLI model '{MODEL_NAME}'.")
except Exception as e:
    print(f"Error loading NLI model '{MODEL_NAME}': {e}")
    print("Attempting to load with an explicit cache directory.")
    # Fallback to a specified cache directory if there are permission issues
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=cache_dir).to(device)
    print(f"Successfully loaded NLI model '{MODEL_NAME}' using cache directory: {cache_dir}.")


label_map = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

def get_retriever_and_llm():
    """Initializes and returns the DuckDuckGo Search retriever and the NLI model."""
    retriever = DDGS()
    return retriever, model

def search_snippets(query, retriever, num_results=5):
    """
    Searches DuckDuckGo for snippets relevant to the query.
    Returns a list of snippet bodies.
    """
    results = []
    print(f"\n[DEBUG] Searching for: '{query}' with {num_results} results...")
    try:
        # Using a list comprehension to ensure all results are collected
        # and to handle potential exceptions from the retriever itself
        ddgs_results = list(retriever.text(query, max_results=num_results))
        for i, r in enumerate(ddgs_results):
            if "body" in r:
                results.append(r["body"])
                print(f"  [DEBUG] Snippet {i+1} (first 100 chars): {r['body'][:100]}...")
            else:
                print(f"  [DEBUG] Snippet {i+1} had no 'body' key: {r}")
        if not results:
            print("[DEBUG] No 'body' content found in any search results for the query.")
    except Exception as e:
        print(f"[ERROR] DuckDuckGo search failed for query '{query}': {e}")
        # Ensure results is an empty list if search fails
        results = []
    return results

def classify_nli(premise, hypothesis):
    """
    Runs NLI on (premise, hypothesis) and returns the best label and its score.
    """
    # Truncate inputs to prevent tokenization errors with very long texts
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    label_id = int(probs.argmax())
    return label_map[label_id], float(probs[label_id])

def verify_claim(claim, retriever, llm=None, top_k=5):
    """
    Verifies a factual claim against search snippets using an NLI model.
    """
    print(f"\n[DEBUG] Verifying claim: '{claim}' (top_k={top_k})")
    snippets = search_snippets(claim, retriever, num_results=top_k)

    if not snippets:
        print(f"[DEBUG] No snippets available to verify the claim '{claim}'. Returning 'uncertain'.")
        return {"claim": claim, "status": "uncertain", "evidence": None}

    # Debugging lines for 'snippets' before iteration
    print(f"[DEBUG] Type of 'snippets' before iteration: {type(snippets)}")
    print(f"[DEBUG] Number of snippets received: {len(snippets)}")
    if snippets:
        print(f"[DEBUG] First snippet (first 100 chars): {snippets[0][:100]}...")
    # END Debugging lines

    best_status = "uncertain"
    best_conf = 0.0 # Initialize with float
    best_snippet = None

    # Tunable confidence threshold for strong classifications
    CONFIDENCE_THRESHOLD = 0.75 # Recommended starting point, adjust as needed (e.g., 0.7 to 0.9)

    print("\n--- NLI Classification Results for Each Snippet ---")
    # LINE 52 (approximately)
    for i, snippet in enumerate(snippets):
        try:
            label, score = classify_nli(snippet, claim)
            print(f"  [DEBUG] Snippet {i+1} vs Claim:")
            print(f"    Premise (Snippet): {snippet[:150]}...") # Truncate for display
            print(f"    Hypothesis (Claim): {claim}")
            print(f"    NLI Result: Label='{label}', Score={score:.4f}")

            # --- Logic for updating best_status based on confidence and hierarchy ---
            # Prioritize strong entailment or contradiction
            if label == "ENTAILMENT" and score >= CONFIDENCE_THRESHOLD:
                # If we find a strong entailment, it becomes the best status,
                # or if it's more confident than a previously found entailment.
                if best_status != "verified" or score > best_conf:
                    best_status = "verified"
                    best_conf = score
                    best_snippet = snippet
            elif label == "CONTRADICTION" and score >= CONFIDENCE_THRESHOLD:
                # If we find a strong contradiction, it becomes the best status,
                # overriding even a less confident 'verified' if found earlier,
                # or a more confident 'contradiction'.
                if best_status != "hallucination" or score > best_conf:
                    best_status = "hallucination"
                    best_conf = score
                    best_snippet = snippet
            elif label == "NEUTRAL":
                # Only update to 'uncertain' if no strong 'verified' or 'hallucination'
                # has been established yet, or if this 'NEUTRAL' is more confident than
                # the current 'uncertain' best_conf.
                # Crucially, a strong ENTAILMENT/CONTRADICTION should not be easily
                # overridden by a NEUTRAL, even if the NEUTRAL's score is slightly higher,
                # if its score is below the CONFIDENCE_THRESHOLD.
                if best_status not in ["verified", "hallucination"]: # Only consider NEUTRAL if no strong E/C found
                    if score > best_conf: # And if this NEUTRAL is more confident than current best_conf
                        best_status = "uncertain"
                        best_conf = score
                        best_snippet = snippet
            # --- End Logic for updating best_status ---

        except Exception as e:
            print(f"  [ERROR] Error classifying snippet {i+1}: {e}")
            continue

    print(f"\n[DEBUG] Final decision for claim '{claim}': Status='{best_status}', Confidence={best_conf:.4f}")
    return {"claim": claim, "status": best_status, "evidence": best_snippet}

# Example Usage (for testing verifier.py directly)
if __name__ == "__main__":
    retriever, llm_model = get_retriever_and_llm()

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
        result = verify_claim(claim_to_verify, retriever, top_k=10) # Increased top_k for better chances
        print(f"\n--- Result for: \"{result['claim']}\" ---")
        print(f"Status: {result['status'].upper()}")
        if result['evidence']:
            print(f"Evidence (first 150 chars): \"{result['evidence'][:150]}...\"")
        else:
            print("No specific evidence found.")
        print("-" * 70)

print("Let me generate an image of a detective examining a complex case file, representing the detailed debugging and verification process.") 

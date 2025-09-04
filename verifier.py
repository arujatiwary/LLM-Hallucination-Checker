from duckduckgo_search import DDGS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Load NLI model (roberta-large-mnli)
MODEL_NAME = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
label_map = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

def get_retriever_and_llm():
    retriever = DDGS()
    return retriever, model

def search_snippets(query, retriever, num_results=5):
    results = []
    print(f"\nSearching for: '{query}' with {num_results} results...")
    for i, r in enumerate(retriever.text(query, max_results=num_results)):
        if "body" in r:
            results.append(r["body"])
            print(f"  Snippet {i+1} (first 100 chars): {r['body'][:100]}...")
    if not results:
        print("No snippets found for the query.")
    return results

def classify_nli(premise, hypothesis):
    """Run NLI on (premise, hypothesis) and return best label + score."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    label_id = int(probs.argmax())
    return label_map[label_id], float(probs[label_id])

def verify_claim(claim, retriever, llm=None, top_k=5):
    print(f"\nVerifying claim: '{claim}'")
    snippets = search_snippets(claim, retriever, num_results=top_k)

    if not snippets:
        print("No snippets available to verify the claim.")
        return {"claim": claim, "status": "uncertain", "evidence": None}

    best_status = "uncertain"
    best_conf = 0
    best_snippet = None

    print("\n--- NLI Classification Results for Each Snippet ---")
    for i, snippet in enumerate(snippets):
        try:
            label, score = classify_nli(snippet, claim)
            print(f"  Snippet {i+1}:")
            print(f"    Premise (Snippet): {snippet[:150]}...") # Truncate for display
            print(f"    Hypothesis (Claim): {claim}")
            print(f"    NLI Result: Label='{label}', Score={score:.4f}")

            if score > best_conf:
                best_conf = score
                best_snippet = snippet
                if label == "ENTAILMENT":
                    best_status = "verified"
                elif label == "CONTRADICTION":
                    best_status = "hallucination"
                # If it's NEUTRAL and it's the highest confidence so far, it keeps best_status as uncertain
                elif label == "NEUTRAL":
                    if best_status != "verified" and best_status != "hallucination": # Only update if no stronger status found
                        best_status = "uncertain"
        except Exception as e:
            print(f"  Error classifying snippet {i+1}: {e}")
            continue

    print("\n--- Final Verification Result ---")
    return {"claim": claim, "status": best_status, "evidence": best_snippet}

# Example Usage:
if __name__ == "__main__":
    retriever, llm_model = get_retriever_and_llm()

    claims_to_test = [
        "The capital of France is Paris.",
        "Humans can breathe underwater unaided.",
        "The Eiffel Tower is located in Rome.",
        "The sun is a star.",
        "The moon is made of cheese."
    ]

    for claim_to_verify in claims_to_test:
        result = verify_claim(claim_to_verify, retriever, top_k=10) # Increased top_k for better chances
        print(f"Claim: \"{result['claim']}\"")
        print(f"Status: {result['status']}")
        if result['evidence']:
            print(f"Evidence (first 100 chars): \"{result['evidence'][:100]}...\"")
        else:
            print("No specific evidence found.")
        print("-" * 50)

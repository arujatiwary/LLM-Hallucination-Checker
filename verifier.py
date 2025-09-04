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

# Modified verify_claim function snippet
def verify_claim(claim, retriever, llm=None, top_k=5):
    # ... (previous code) ...

    best_status = "uncertain"
    best_conf = 0.0 # Initialize with float
    best_snippet = None

    CONFIDENCE_THRESHOLD = 0.70 # Adjust as needed. Start with a moderate value.

    print("\n--- NLI Classification Results for Each Snippet ---")
    for i, snippet in enumerate(snippets):
        try:
            label, score = classify_nli(snippet, claim)
            print(f"  Snippet {i+1}:")
            print(f"    Premise (Snippet): {snippet[:150]}...") # Truncate for display
            print(f"    Hypothesis (Claim): {claim}")
            print(f"    NLI Result: Label='{label}', Score={score:.4f}")

            # Only consider updating if the current score is higher,
            # or if the current status is weaker and this new classification is strong enough.

            # Prioritize stronger classifications
            if label == "ENTAILMENT" and score >= CONFIDENCE_THRESHOLD:
                if best_status != "verified" or score > best_conf: # Update if first verified, or more confident verified
                    best_status = "verified"
                    best_conf = score
                    best_snippet = snippet
            elif label == "CONTRADICTION" and score >= CONFIDENCE_THRESHOLD:
                # Contradiction might override uncertain or even less confident verified
                # Decision: if a strong contradiction is found, it's a hallucination
                if best_status != "hallucination" or score > best_conf: # Update if first hallucination, or more confident hallucination
                    best_status = "hallucination"
                    best_conf = score
                    best_snippet = snippet
            elif label == "NEUTRAL":
                # Only update to uncertain if it's the highest score AND no strong entailment/contradiction has been found yet
                if best_conf < CONFIDENCE_THRESHOLD: # If we haven't found a strong E/C yet
                    if score > best_conf:
                        best_status = "uncertain"
                        best_conf = score
                        best_snippet = snippet

        except Exception as e:
            print(f"  Error classifying snippet {i+1}: {e}")
            continue

    print("\n--- Final Verification Result ---")
    return {"claim": claim, "status": best_status, "evidence": best_snippet}

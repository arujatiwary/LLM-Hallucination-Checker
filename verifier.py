import torch
from transformers import pipeline
from google_search import google_search

# Device setup
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

# Load NLI model
nli_model = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    tokenizer="roberta-large-mnli",
    device=device
)

def verify_claim(claim, num_results=5):
    """
    Verifies a claim using Google search + NLI model.
    Returns dict: {claim, status, evidence (list of snippets)}.
    """
    snippets = [snippet for _, snippet in google_search(claim, num_results=num_results)]

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": []}

    try:
        # Compare claim vs snippet
        inputs = [f"{claim} </s></s> {snippet}" for snippet in snippets]
        results = nli_model(inputs, truncation=True, padding=True)
    except Exception as e:
        return {"claim": claim, "status": "uncertain", "evidence": [f"NLI error: {e}"]}

    status_counts = {"verified": 0, "hallucination": 0, "uncertain": 0}
    evidence = []

    for snippet, result in zip(snippets, results):
        label = result["label"].upper()
        score = result["score"]

        if label == "ENTAILMENT" and score > 0.7:
            status_counts["verified"] += 1
            evidence.append(snippet)
        elif label == "CONTRADICTION" and score > 0.7:
            status_counts["hallucination"] += 1
            evidence.append(snippet)
        else:
            status_counts["uncertain"] += 1
            evidence.append(snippet)

    # Pick the majority label
    final_status = max(status_counts, key=status_counts.get)

    return {"claim": claim, "status": final_status, "evidence": evidence}

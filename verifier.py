import torch
from transformers import pipeline
from google_search import google_search

# Device setup
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

# Load NLI model (zero-shot classification)
nli_model = pipeline(
    "zero-shot-classification",
    model="roberta-large-mnli",
    tokenizer="roberta-large-mnli",
    device=device
)

def verify_claim(claim, num_results=5):
    """
    Verifies a claim using Google search + NLI model.
    Returns dict: {claim, status, evidence}
    """
    snippets = [snippet for _, snippet in google_search(claim, num_results=num_results)]

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    entailments, contradictions, neutrals = [], [], []

    for snippet in snippets:
        try:
            result = nli_model(
                sequences=snippet,
                candidate_labels=["true", "false"],
                hypothesis_template=f"This statement is {claim}."
            )
            label = result["labels"][0]
            
            if label == "true":
                entailments.append(snippet)
            elif label == "false":
                contradictions.append(snippet)
            else:
                neutrals.append(snippet)
        except Exception as e:
            return {"claim": claim, "status": "uncertain", "evidence": f"NLI error: {e}"}

    if entailments:
        return {"claim": claim, "status": "verified", "evidence": entailments[0]}
    elif contradictions:
        return {"claim": claim, "status": "hallucination", "evidence": contradictions[0]}
    else:
        return {"claim": claim, "status": "uncertain", "evidence": neutrals[0] if neutrals else None}

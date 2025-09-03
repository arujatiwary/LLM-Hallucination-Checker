import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from google_search import google_search

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device.upper()}")

# Load MNLI model
model_name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Label mapping for MNLI
id2label = {
    0: "CONTRADICTION",
    1: "NEUTRAL",
    2: "ENTAILMENT"
}

def nli_infer(premise, hypothesis):
    """Run NLI inference on a (premise, hypothesis) pair."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    label_id = probs.argmax()
    return id2label[label_id], probs[label_id]

def verify_claim(claim, num_results=5):
    """
    Verifies a claim using Google search + MNLI.
    Returns dict: {claim, status, evidence}
    """
    snippets = [snippet for _, snippet in google_search(claim, num_results=num_results)]

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    best_evidence = None
    best_confidence = 0.0
    final_status = "uncertain"

    for snippet in snippets:
        try:
            label, score = nli_infer(snippet, claim)

            if label == "ENTAILMENT" and score > best_confidence:
                final_status = "verified"
                best_confidence = score
                best_evidence = snippet
            elif label == "CONTRADICTION" and score > best_confidence:
                final_status = "hallucination"
                best_confidence = score
                best_evidence = snippet
        except Exception as e:
            return {"claim": claim, "status": "uncertain", "evidence": f"NLI error: {e}"}

    return {"claim": claim, "status": final_status, "evidence": best_evidence}

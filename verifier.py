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
    for r in retriever.text(query, max_results=num_results):
        if "body" in r:
            results.append(r["body"])
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
    snippets = search_snippets(claim, retriever, num_results=top_k)

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    best_status = "uncertain"
    best_conf = 0
    best_snippet = None

    for snippet in snippets:
        try:
            label, score = classify_nli(snippet, claim)
            if score > best_conf:
                best_conf = score
                best_snippet = snippet
                if label == "ENTAILMENT":
                    best_status = "verified"
                elif label == "CONTRADICTION":
                    best_status = "hallucination"
                else:
                    best_status = "uncertain"
        except Exception:
            continue

    return {"claim": claim, "status": best_status, "evidence": best_snippet}

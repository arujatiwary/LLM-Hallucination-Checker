from duckduckgo_search import DDGS
from transformers import pipeline
import torch

# Device setup
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

# Load NLI model
nli_model = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    tokenizer="roberta-large-mnli",
    device=device,
    top_k=None  # get all labels with scores
)

def get_retriever_and_llm():
    """Return retriever (DuckDuckGo) and the NLI model."""
    retriever = DDGS()
    return retriever, nli_model

def search_snippets(query, retriever, num_results=5):
    """Fetch top snippets from DuckDuckGo search."""
    results = []
    try:
        for r in retriever.text(query, max_results=num_results):
            if "body" in r:
                results.append(r["body"])
    except Exception as e:
        print("Retriever error:", e)
    return results

def _normalize_label(label: str) -> str:
    """Map model labels to ENTAILMENT/CONTRADICTION/NEUTRAL."""
    lab = label.upper()
    mapping = {
        "LABEL_0": "CONTRADICTION",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "ENTAILMENT",
        "CONTRADICTION": "CONTRADICTION",
        "NEUTRAL": "NEUTRAL",
        "ENTAILMENT": "ENTAILMENT",
    }
    return mapping.get(lab, "NEUTRAL")

def verify_claim(claim, retriever, llm, top_k=5):
    """
    Verify a claim using DuckDuckGo + RoBERTa-MNLI.
    Returns dict: {claim, status, evidence}
    """
    snippets = search_snippets(claim, retriever, num_results=top_k)
    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    best_label = "NEUTRAL"
    best_score = -1
    best_snippet = None

    for snip in snippets:
        try:
            out = llm({"text": snip, "text_pair": claim})
            out = out[0]  # list of dicts
            top = max(out, key=lambda d: d["score"])
            label = _normalize_label(top["label"])

            print(f"\nClaim: {claim}")
            print(f"Snippet: {snip}")
            print(f"Model raw: {out}")
            print(f"Chosen label: {label} (score {top['score']:.4f})")

            if top["score"] > best_score:
                best_score = top["score"]
                best_label = label
                best_snippet = snip
        except Exception as e:
            print("NLI error:", e)
            continue

    if best_label == "ENTAILMENT":
        status = "verified"
    elif best_label == "CONTRADICTION":
        status = "hallucination"
    else:
        status = "uncertain"

    return {"claim": claim, "status": status, "evidence": best_snippet}

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
    return_all_scores=True
)

def get_retriever_and_llm():
    """Return retriever and the NLI model."""
    retriever = DDGS()
    return retriever, nli_model

def search_snippets(query, retriever, num_results=5):
    """Fetch top snippets from DuckDuckGo search."""
    results = []
    for r in retriever.text(query, max_results=num_results):
        if "body" in r:
            results.append(r["body"])
    return results

def verify_claim(claim, retriever, llm, runs=1, top_k=5):
    """
    Verifies a claim using search + NLI model.
    """
    snippets = search_snippets(claim, retriever, num_results=top_k)

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    best_label = "NEUTRAL"
    best_score = -1
    best_snippet = None

    for snippet in snippets:
        try:
            # Format input for MNLI
            input_text = f"{snippet} </s></s> {claim}"
            outputs = llm(input_text)

            if not outputs or not isinstance(outputs, list):
                continue

            for res in outputs[0]:
                if res["score"] > best_score:
                    best_score = res["score"]
                    best_label = res["label"].upper()
                    best_snippet = snippet

        except Exception as e:
            continue

    # Map labels to statuses
    if best_label == "ENTAILMENT":
        status = "verified"
    elif best_label == "CONTRADICTION":
        status = "hallucination"
    else:
        status = "uncertain"

    return {
        "claim": claim,
        "status": status,
        "evidence": best_snippet,
        "confidence": round(best_score, 3)
    }

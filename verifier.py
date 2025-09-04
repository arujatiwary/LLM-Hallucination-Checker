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
    """Return retriever (DuckDuckGo) and the NLI model."""
    retriever = DDGS()
    return retriever, nli_model

def search_snippets(query, retriever, num_results=5):
    """Fetch top snippets from DuckDuckGo search."""
    results = []
    for r in retriever.text(query, max_results=num_results):
        if "body" in r:
            results.append(r["body"])
    return results

def verify_claim(claim, retriever, llm, top_k=5):
    """
    Verifies a claim using DuckDuckGo search + NLI model.
    Picks the label with the highest score across snippets.
    """
    snippets = search_snippets(claim, retriever, num_results=top_k)

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    best_label = "NEUTRAL"
    best_score = -1
    best_snippet = None

    for snippet in snippets:
        try:
            result = llm(f"{snippet}</s></s>{claim}", truncation=True)
            if not result or not isinstance(result[0], list):
                continue

            for score_dict in result[0]:
                label = score_dict["label"].upper()
                score = score_dict["score"]

                if score > best_score:
                    best_score = score
                    best_label = label
                    best_snippet = snippet
        except Exception:
            continue

    # Map to statuses
    if best_label == "ENTAILMENT":
        status = "verified"
    elif best_label == "CONTRADICTION":
        status = "hallucination"
    else:
        status = "uncertain"

    return {"claim": claim, "status": status, "evidence": best_snippet}

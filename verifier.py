from duckduckgo_search import DDGS
from transformers import pipeline
import torch

# Device setup
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

# Load NLI model (RoBERTa trained on MNLI)
nli_model = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    tokenizer="roberta-large-mnli",
    device=device,
    return_all_scores=True   # get all 3 scores: entailment / contradiction / neutral
)

def get_retriever_and_llm():
    """Return DuckDuckGo retriever and the NLI model."""
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
    Verifies a claim using search + NLI.
    Picks the highest-confidence result from the snippets.
    """
    try:
        snippets = search_snippets(claim, retriever, num_results=top_k)
    except Exception as e:
        return {"claim": claim, "status": "uncertain", "evidence": f"Retriever error: {e}"}

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    best_label = "NEUTRAL"
    best_score = -1
    best_snippet = None

    for snippet in snippets:
        try:
            # Compare snippet (premise) vs claim (hypothesis)
            results = llm((snippet, claim), truncation=True)
            # results looks like: [[{'label': 'CONTRADICTION', 'score': 0.01}, ... ]]
            for r in results[0]:
                if r["score"] > best_score:
                    best_score = r["score"]
                    best_label = r["label"].upper()
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

    return {"claim": claim, "status": status, "evidence": best_snippet}

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
    return_all_scores=True   # so we see entailment/contradiction/neutral scores
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

def verify_claim(claim, retriever, llm, runs=3, top_k=5, threshold=0.6):
    """
    Verifies a factual claim using web search + NLI model.
    Returns dict: {claim, status, evidence}
    """
    snippets = search_snippets(claim, retriever, num_results=top_k)

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    try:
        # IMPORTANT: premise = snippet, hypothesis = claim
        inputs = [(snippet, claim) for snippet in snippets]
        results = llm(inputs, truncation=True, padding=True)
    except Exception as e:
        return {"claim": claim, "status": "uncertain", "evidence": f"NLI error: {e}"}

    status = "uncertain"
    evidence = None

    for snippet, scores in zip(snippets, results):
        # convert list of dicts -> {label: score}
        score_map = {s["label"].upper(): s["score"] for s in scores}

        if score_map.get("ENTAILMENT", 0) > threshold:
            status = "verified"
            evidence = snippet
            break
        elif score_map.get("CONTRADICTION", 0) > threshold:
            status = "hallucination"
            evidence = snippet
            break

    return {"claim": claim, "status": status, "evidence": evidence}

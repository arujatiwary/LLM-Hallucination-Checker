from duckduckgo_search import DDGS
from transformers import pipeline
import torch

# Load NLI model (for entailment / contradiction check)
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

nli_model = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    tokenizer="roberta-large-mnli",
    device=device
)

def get_retriever_and_llm():
    """
    Returns retriever (DuckDuckGo search) and the NLI model as LLM.
    """
    retriever = DDGS()
    return retriever, nli_model

def search_snippets(query, retriever, num_results=5):
    """Fetch top snippets from DuckDuckGo search."""
    results = []
    for r in retriever.text(query, max_results=num_results):
        if "body" in r:
            results.append(r["body"])
    return results

def verify_claim(claim, retriever, llm, runs=3, top_k=5):
    """
    Verifies a factual claim using web search + NLI model.
    Returns dict: {claim, status, evidence}
    """
    snippets = search_snippets(claim, retriever, num_results=top_k)

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    try:
        inputs = [(claim, snippet) for snippet in snippets]
        results = llm(inputs, truncation=True, padding=True)
    except Exception as e:
        return {"claim": claim, "status": "uncertain", "evidence": f"NLI error: {e}"}

    status = "uncertain"
    evidence = None
    for snippet, result in zip(snippets, results):
        label = result["label"].upper()
        if label == "ENTAILMENT":
            status = "verified"
            evidence = snippet
            break
        elif label == "CONTRADICTION":
            status = "hallucination"
            evidence = snippet
            break

    return {"claim": claim, "status": status, "evidence": evidence}

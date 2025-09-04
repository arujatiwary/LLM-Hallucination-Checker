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

def verify_claim(claim, retriever, llm, runs=3, top_k=5):
    """
    Verify claim with multiple search runs and majority vote.
    """
    labels = []
    evidence_map = {}

    for _ in range(runs):
        snippets = search_snippets(claim, retriever, num_results=top_k)
        if not snippets:
            continue

        for snippet in snippets:
            prompt = f"Claim: \"{claim}\"\nEvidence: \"{snippet}\"\nAnswer with ENTAILMENT, CONTRADICTION, or NEUTRAL."
            try:
                result = llm.invoke(prompt).content.strip().upper()
                labels.append(result)
                if result in ["ENTAILMENT", "CONTRADICTION"]:
                    evidence_map[result] = snippet
            except Exception:
                continue

    if not labels:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    # majority vote
    final = max(set(labels), key=labels.count)
    if final == "ENTAILMENT":
        return {"claim": claim, "status": "verified", "evidence": evidence_map.get("ENTAILMENT")}
    elif final == "CONTRADICTION":
        return {"claim": claim, "status": "hallucination", "evidence": evidence_map.get("CONTRADICTION")}
    else:
        return {"claim": claim, "status": "uncertain", "evidence": None}

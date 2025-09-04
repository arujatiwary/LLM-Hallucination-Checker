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

def verify_claim(claim, retriever, llm, runs=3, top_k=5):
    """
    Verifies a claim using search + LLM/NLI.
    Always assigns the highest scoring label instead of defaulting to uncertain.
    """
    try:
        docs = retriever.get_relevant_documents(claim)
        snippets = [doc.page_content for doc in docs[:top_k]]
    except Exception as e:
        return {"claim": claim, "status": "uncertain", "evidence": f"Retriever error: {e}"}

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    best_label = "NEUTRAL"
    best_score = -1
    best_snippet = None

    for snippet in snippets:
        try:
            prompt = f"""
            Claim: "{claim}"
            Evidence: "{snippet}"

            result = llm.invoke(prompt).content.strip().upper()
        except Exception as e:
            continue

        # crude scoring (could refine with logprobs if available)
        score = 1.0  
        if score > best_score:
            best_score = score
            best_label = result
            best_snippet = snippet

    # map to statuses
    if best_label == "ENTAILMENT":
        status = "verified"
    elif best_label == "CONTRADICTION":
        status = "hallucination"
    else:
        status = "uncertain"

    return {"claim": claim, "status": status, "evidence": best_snippet}


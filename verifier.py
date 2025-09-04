# verifier.py
import re
import torch
from transformers import pipeline
try:
    # Preferred (new name)
    from ddgs import DDGS
except ImportError:
    # Backward-compat (older package name still works but warns)
    from duckduckgo_search import DDGS

# -----------------------
# Device & NLI pipeline
# -----------------------
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

nli_model = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    tokenizer="roberta-large-mnli",
    device=device,
    top_k=None  # returns all labels with scores (replacement for return_all_scores=True)
)

# Robust label normalization for different HF configs
def _norm(label: str) -> str:
    lab = label.upper()
    mapping = {
        "LABEL_0": "CONTRADICTION",
        "LABEL_1": "NEUTRAL",
        "LABEL_2": "ENTAILMENT",
        "CONTRADICTION": "CONTRADICTION",
        "NEUTRAL": "NEUTRAL",
        "ENTAILMENT": "ENTAILMENT",
    }
    return mapping.get(lab, lab)

def get_retriever_and_llm():
    """Return (retriever, nli_model) to match your app imports."""
    # Force English, avoid random-language snippets
    retriever = DDGS()
    return retriever, nli_model

# -----------------------
# Retrieval
# -----------------------
def search_snippets(query: str, retriever, num_results: int = 6):
    """
    Use DuckDuckGo to fetch snippet bodies (fallback to title).
    We enforce English via region where possible.
    """
    snippets = []
    # DDGS().text(...) is a generator of dicts: {title, href, body}
    try:
        with retriever as ddg:
            for r in ddg.text(
                query,
                region="us-en",     # enforce English results
                safesearch="off",
                timelimit=None,
                max_results=num_results
            ):
                body = r.get("body") or ""
                title = r.get("title") or ""
                snippet = body.strip() or title.strip()
                if not snippet:
                    continue
                # light clean-up
                snippet = re.sub(r"\s+", " ", snippet)
                snippets.append(snippet)
    except TypeError:
        # Some versions of DDGS are context-manager-less; fallback
        for r in retriever.text(
            query,
            region="us-en",
            safesearch="off",
            timelimit=None,
            max_results=num_results
        ):
            body = r.get("body") or ""
            title = r.get("title") or ""
            snippet = body.strip() or title.strip()
            if not snippet:
                continue
            snippet = re.sub(r"\s+", " ", snippet)
            snippets.append(snippet)
    return snippets

# -----------------------
# Verification
# -----------------------
def verify_claim(claim: str, retriever, llm, top_k: int = 6):
    """
    Verify a claim with web snippets + RoBERTa MNLI.
    - premise = snippet
    - hypothesis = claim
    Majority vote across snippets; ties broken by summed scores.
    """
    snippets = search_snippets(claim, retriever, num_results=top_k)

    if not snippets:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    vote_counts = {"ENTAILMENT": 0, "CONTRADICTION": 0, "NEUTRAL": 0}
    score_sums  = {"ENTAILMENT": 0.0, "CONTRADICTION": 0.0, "NEUTRAL": 0.0}

    per_snippet_best = []

    for snip in snippets:
        try:
            # Call NLI with (premise, hypothesis)
            # Using dict style ensures correct pairing for HF pipelines
            out = llm({"text": snip, "text_pair": claim}, truncation=True)
            # top_k=None => out is a list [ {label, score}, {label, score}, {label, score} ]
            if not out or not isinstance(out, list):
                continue
            # Pick highest scoring label for this snippet
            best = max(out, key=lambda d: d["score"])
            best_label = _norm(best["label"])
            best_score = float(best["score"])

            vote_counts[best_label] += 1
            score_sums[best_label]  += best_score
            per_snippet_best.append((snip, best_label, best_score))
        except Exception:
            continue

    if not per_snippet_best:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    # Majority vote first
    winner = max(vote_counts.items(), key=lambda kv: kv[1])[0]
    # Tie-break on summed scores if needed
    tied = [k for k, v in vote_counts.items() if v == vote_counts[winner]]
    if len(tied) > 1:
        winner = max(tied, key=lambda k: score_sums[k])

    # Pick strongest evidence among snippets that voted for the winner
    candidate_snips = [x for x in per_snippet_best if x[1] == winner]
    evidence_snip = max(candidate_snips, key=lambda x: x[2])[0] if candidate_snips else per_snippet_best[0][0]

    if winner == "ENTAILMENT":
        status = "verified"
    elif winner == "CONTRADICTION":
        status = "hallucination"
    else:
        status = "uncertain"

    return {"claim": claim, "status": status, "evidence": evidence_snip}

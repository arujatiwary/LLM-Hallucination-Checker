import wikipediaapi
from sentence_transformers import SentenceTransformer, util

# Add a descriptive User-Agent (your name/project is fine)
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='HallucinationChecker/1.0 (arujatiwary)'
)

model = SentenceTransformer('all-MiniLM-L6-v2')

def verify_claim(claim, top_k=3):
    """
    Check claim against Wikipedia.
    Returns: dict with status and evidence snippet.
    """
    keyword = " ".join(claim.split()[:5])
    page = wiki.page(keyword)

    if not page.exists():
        return {"claim": claim, "status": "uncertain", "evidence": None}

    summary_sentences = page.summary.split(". ")
    claim_emb = model.encode(claim, convert_to_tensor=True)
    evidences = [
        (sent, util.cos_sim(claim_emb, model.encode(sent, convert_to_tensor=True)).item())
        for sent in summary_sentences if sent.strip()
    ]

    if not evidences:
        return {"claim": claim, "status": "uncertain", "evidence": None}

    best_sentence, score = max(evidences, key=lambda x: x[1])

    if score > 0.6:
        return {"claim": claim, "status": "verified", "evidence": best_sentence}
    elif score > 0.4:
        return {"claim": claim, "status": "uncertain", "evidence": best_sentence}
    else:
        return {"claim": claim, "status": "hallucination", "evidence": best_sentence}

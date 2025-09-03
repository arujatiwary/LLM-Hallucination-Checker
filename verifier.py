import random
import numpy as np
import torch

# Fix seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def verify_claim(claim, retriever, llm, runs=3, top_k=5):
    """
    Verify a claim against retrieved evidence with majority voting.
    """
    decisions = []

    for _ in range(runs):
        # Retrieve top-k evidence deterministically
        retrieved_docs = retriever.get_relevant_documents(claim)
        evidence_texts = [doc.page_content for doc in retrieved_docs[:top_k]]

        # Build prompt for LLM
        context = "\n".join(evidence_texts)
        prompt = f"""
        Claim: {claim}
        Evidence:
        {context}

        Based on the evidence, classify the claim as one of:
        - VERIFIED (evidence strongly supports it)
        - HALLUCINATION (evidence contradicts or disproves it)
        - UNCERTAIN (not enough clear evidence either way)

        Answer with only one word: VERIFIED, HALLUCINATION, or UNCERTAIN.
        """

        response = llm.predict(prompt).strip().upper()

        if "VERIFIED" in response:
            decisions.append("VERIFIED")
        elif "HALLUCINATION" in response:
            decisions.append("HALLUCINATION")
        else:
            decisions.append("UNCERTAIN")

    # Majority vote
    final = max(set(decisions), key=decisions.count)

    return {
        "claim": claim,
        "status": final,
        "evidence": evidence_texts,
        "votes": decisions
    }

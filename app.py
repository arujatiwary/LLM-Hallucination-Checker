# app.py
import streamlit as st
import re
from claim_extractor import extract_claims
from verifier import verify_claim, ensure_models  # ensure_models may or may not exist depending on your file

st.set_page_config(page_title="Hallucination Detector", layout="wide")
st.title("🕵️ Hallucination Detector for LLM Outputs")

# Optional: attempt to warm models on first run but keep errors visible in UI
@st.cache_resource(ttl=3600)
def _warm_models():
    try:
        # call ensure_models if provided in verifier (non-invasive; does nothing if not present)
        if "ensure_models" in globals() and callable(ensure_models):
            ensure_models()
        return True, None
    except Exception as e:
        return False, str(e)

input_text = st.text_area("Paste the LLM-generated text here:", height=220)

if st.button("Check for Hallucinations"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Preparing models and running checks..."):
            ok, err = _warm_models()
            if not ok:
                st.error("Model warmup failed. See details below and check logs.")
                st.write(err)

            # Extract claims using your logic
            try:
                claims = extract_claims(input_text)
            except Exception as e:
                st.error("Error extracting claims. See logs for details.")
                st.exception(e)
                st.stop()

            if not claims:
                st.info("No factual claims detected by the extractor.")
            else:
                # Run verify_claim on each claim using your verifier (keeps your logic unchanged)
                results = []
                for claim in claims:
                    try:
                        # verify_claim might expect extra args; we call with single arg as your verifier originally did
                        res = verify_claim(claim)
                        results.append(res)
                    except TypeError:
                        # if your verify_claim signature needs retriever/llm, try to call defensively:
                        try:
                            res = verify_claim(claim, top_k=5)
                            results.append(res)
                        except Exception as e:
                            results.append({"claim": claim, "status": "uncertain", "evidence": f"Verifier error: {e}"})
                    except Exception as e:
                        results.append({"claim": claim, "status": "uncertain", "evidence": f"Verifier error: {e}"})

                # Highlight the original text with colors
                highlighted_text = input_text
                for result in results:
                    claim_text = result.get("claim", "")
                    if not claim_text:
                        continue
                    # basic status -> color mapping (unchanged labels considered)
                    status = result.get("status", "").lower()
                    color = "#f8c316"  # default yellow = uncertain
                    if status in ("verified", "supported", "entailment"):
                        color = "#6cc644"  # green
                    elif status in ("hallucination", "refuted", "contradiction"):
                        color = "#e74c3c"  # red
                    # replace first occurrence only to preserve rest
                    safe_claim = re.escape(claim_text)
                    replacement = f"<span style='background-color:{color}; padding:2px 6px; border-radius:4px;'>{claim_text}</span>"
                    try:
                        highlighted_text = re.sub(safe_claim, replacement, highlighted_text, count=1)
                    except re.error:
                        # fallback: simple replacement
                        highlighted_text = highlighted_text.replace(claim_text, replacement, 1)

                st.subheader("Annotated Output")
                st.markdown(highlighted_text, unsafe_allow_html=True)

                st.subheader("Details")
                for result in results:
                    status = result.get("status", "uncertain")
                    evidence = result.get("evidence", None)
                    st.markdown(f"- **{result.get('claim','(no claim)')}** → **{status.upper()}**")
                    if evidence:
                        # If evidence is a list, print a short version
                        if isinstance(evidence, (list, tuple)):
                            for i, e in enumerate(evidence[:3]):
                                st.caption(f"Evidence {i+1}: {e[:300]}{'...' if len(e)>300 else ''}")
                        else:
                            st.caption(f"Evidence: {str(evidence)[:500]}{'...' if len(str(evidence))>500 else ''}")

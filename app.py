import streamlit as st
from claim_extractor import extract_claims
from verifier import verify_claim
import re

st.set_page_config(page_title="Hallucination Detector", layout="wide")

st.title("🕵️ Hallucination Detector for LLM Outputs")

input_text = st.text_area("Paste the LLM-generated text here:", height=200)

if st.button("Check for Hallucinations"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        claims = extract_claims(input_text)

        if not claims:
            st.write("No factual claims detected.")
        else:
            # Verify all claims
            results = [verify_claim(claim) for claim in claims]

            # Highlight text
            highlighted_text = input_text
            for result in results:
                claim = re.escape(result["claim"])  # escape regex chars
                color = {
                    "verified": "#6cc644",      # green
                    "uncertain": "#ffb347",     # orange
                    "hallucination": "#e74c3c"  # red
                }[result["status"]]
                replacement = f"<span style='background-color:{color}; padding:2px 4px; border-radius:4px;'>{result['claim']}</span>"
                highlighted_text = re.sub(result["claim"], replacement, highlighted_text, 1)

            st.subheader("Annotated Output")
            st.markdown(highlighted_text, unsafe_allow_html=True)

            st.subheader("Details")
            for result in results:
                st.markdown(
                    f"- **{result['claim']}** → "
                    f"<span style='color:{color};font-weight:bold'>{result['status'].upper()}</span>",
                    unsafe_allow_html=True
                )
                if result["evidence"]:
                    st.caption(f"Evidence: {result['evidence']}")

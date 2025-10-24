import streamlit as st
import re
from claim_extractor import extract_claims
from verifier import verify_claim

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
            results = [verify_claim(claim) for claim in claims]

            highlighted_text = input_text
            for result in results:
                claim = re.escape(result["claim"])
                color = {
                    "verified": "#6cc644",
                    "uncertain": "#f8c316",
                    "hallucination": "#e74c3c"
                }[result["status"]]
                replacement = f"<span style='background-color:{color}; padding:2px 4px; border-radius:4px;'>{result['claim']}</span>"
                highlighted_text = re.sub(result["claim"], replacement, highlighted_text, 1)

            st.subheader("Annotated Output")
            st.markdown(highlighted_text, unsafe_allow_html=True)

            st.subheader("Details")
            for result in results:
                color = {
                    "verified": "#6cc644",
                    "uncertain": "#f8c316",
                    "hallucination": "#e74c3c"
                }[result["status"]]
                st.markdown(
                    f"- **{result['claim']}** → "
                    f"<span style='color:{color};font-weight:bold'>{result['status'].upper()}</span>",
                    unsafe_allow_html=True
                )
                if result["best_snippet"]:
                    st.caption(f"Evidence: {result['best_snippet']}")

import streamlit as st
import re
from claim_extractor import extract_claims
from verifier import verify_claim, get_retriever_and_llm

# Streamlit page config
st.set_page_config(page_title="Hallucination Detector", layout="wide")

st.title("🕵️ Hallucination Detector for LLM Outputs")

# Initialize retriever + LLM once
retriever, llm = get_retriever_and_llm()

# Input text
input_text = st.text_area("Paste the LLM-generated text here:", height=200)

if st.button("Check for Hallucinations"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        claims = extract_claims(input_text)

        if not claims:
            st.write("No factual claims detected.")
        else:
            st.subheader("Processing claims...")
            results = [verify_claim(claim, retriever, llm, top_k=5) for claim in claims]

            # Highlighted version of text
            highlighted_text = input_text
            for result in results:
                claim = re.escape(result["claim"])
                color = {
                    "verified": "#6cc644",      # green
                    "uncertain": "#f8c316",     # yellow
                    "hallucination": "#e74c3c"  # red
                }[result["status"]]
                replacement = (
                    f"<span style='background-color:{color}; "
                    f"padding:2px 4px; border-radius:4px;'>{result['claim']}</span>"
                )
                # Replace only first occurrence
                highlighted_text = re.sub(result["claim"], replacement, highlighted_text, 1)

            # Annotated output
            st.subheader("Annotated Output")
            st.markdown(highlighted_text, unsafe_allow_html=True)

            # Detailed results
            st.subheader("Details")
            for result in results:
                color = {
                    "verified": "green",
                    "uncertain": "orange",
                    "hallucination": "red"
                }[result["status"]]

                st.markdown(
                    f"- **{result['claim']}** → "
                    f"<span style='color:{color}; font-weight:bold'>{result['status'].upper()}</span>",
                    unsafe_allow_html=True
                )
                if result["evidence"]:
                    st.caption(f"Evidence: {result['evidence']}")

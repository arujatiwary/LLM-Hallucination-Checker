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
            results = [verify_claim(claim, retriever, llm, runs=3, top_k=5) for claim in claims]

            # Highlight text
            highlighted_text = input_text
            for result in results:
                claim = re.escape(result["claim"])  # escape regex chars
                color = {
                    "verified": "#6cc644",      # green
                    "uncertain": "#f8c316",     # yellow
                    "hallucination": "#e74c3c"  # red
                }[result["status"]]
                replacement = (
                    f"<span style='background-color:{color}; padding:2px 4px; border-radius:4px;'>"
                    f"{result['claim']}</span>"
                )
                highlighted_text = re.sub(result["claim"], replacement, highlighted_text, 1)

            # Summary
            st.subheader("📊 Summary")
            total = len(results)
            verified = sum(1 for r in results if r["status"] == "verified")
            uncertain = sum(1 for r in results if r["status"] == "uncertain")
            hallucinations = sum(1 for r in results if r["status"] == "hallucination")

            col1, col2, col3 = st.columns(3)
            col1.metric("✔️ Verified", f"{verified}/{total}", f"{(verified/total)*100:.1f}%")
            col2.metric("⚠️ Uncertain", f"{uncertain}/{total}", f"{(uncertain/total)*100:.1f}%")
            col3.metric("❌ Hallucinations", f"{hallucinations}/{total}", f"{(hallucinations/total)*100:.1f}%")

            # Annotated text
            st.subheader("📝 Annotated Output")
            st.markdown(highlighted_text, unsafe_allow_html=True)

            # Claim details
            st.subheader("🔎 Claim Details")
            for result in results:
                color_map = {
                    "verified": "green",
                    "uncertain": "orange",
                    "hallucination": "red"
                }
                st.markdown(
                    f"- **{result['claim']}** → "
                    f"<span style='color:{color_map[result['status']]};font-weight:bold'>{result['status'].upper()}</span>",
                    unsafe_allow_html=True
                )
                if result["evidence"]:
                    with st.expander("See evidence"):
                        for ev in result["evidence"]:
                            st.markdown(f"- {ev}")

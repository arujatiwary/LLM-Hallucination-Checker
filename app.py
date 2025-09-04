import streamlit as st
import re
from claim_extractor import extract_claims
from verifier import verify_claim # Only import verify_claim now

# Streamlit page config
st.set_set_page_config(page_title="Hallucination Detector", layout="wide")

st.title("🕵️ Hallucination Detector for LLM Outputs")

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
            # Call verify_claim correctly with only the claim and top_k
            results = [verify_claim(claim, top_k=5) for claim in claims]

            # --- Summary Metrics ---
            total = len(results)
            verified_count = sum(1 for r in results if r["status"] == "verified")
            uncertain_count = sum(1 for r in results if r["status"] == "uncertain")
            hallucination_count = sum(1 for r in results if r["status"] == "hallucination")

            st.subheader("📊 Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("✅ Verified", f"{verified_count}/{total}", f"{(verified_count/total)*100:.1f}%")
            with col2:
                st.metric("⚠️ Uncertain", f"{uncertain_count}/{total}", f"{(uncertain_count/total)*100:.1f}%")
            with col3:
                st.metric("❌ Hallucinations", f"{hallucination_count}/{total}", f"{(hallucination_count/total)*100:.1f}%")

            # --- Annotated Output ---
            highlighted_text = input_text
            for result in results:
                escaped_claim = re.escape(result["claim"]) # Escape special regex characters
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
                highlighted_text = re.sub(escaped_claim, replacement, highlighted_text, 1)

            st.subheader("📝 Annotated Output")
            st.markdown(highlighted_text, unsafe_allow_html=True)

            # --- Detailed Results ---
            st.subheader("🔍 Claim Details")
            for result in results:
                claim_color = {
                    "verified": "#6cc644",
                    "uncertain": "#f8c316",
                    "hallucination": "#e74c3c"
                }[result["status"]]

                st.markdown(
                    f"- **{result['claim']}** → "
                    f"<span style='color:{claim_color}; font-weight:bold'>{result['status'].upper()}</span>",
                    unsafe_allow_html=True
                )

                if result["evidence"]:
                    with st.expander("See evidence snippets"):
                        # Iterate through the list of evidence snippets
                        for i, ev_snippet in enumerate(result["evidence"]):
                            st.caption(f"• Snippet {i+1}: {ev_snippet}")
                else:
                    st.caption("No evidence snippets found.")

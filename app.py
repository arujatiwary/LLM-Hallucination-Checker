import streamlit as st
import html
from verifier import verify_claim, load_spacy_model

st.set_page_config(page_title="LLM Hallucination Detector", layout="wide")
st.title("LLM Hallucination Detection System")
st.caption("Paste an LLM-generated paragraph to check each sentence for factual support.")

# ----------------- User controls -----------------
paragraph = st.text_area(
    "Enter an LLM-generated paragraph:",
    height=260,
    placeholder="Paste your paragraph here..."
)

col1, col2 = st.columns([2, 1])
with col1:
    top_k = st.slider("DuckDuckGo search results per claim (top_k)", 1, 100, 10, 1,
                      help="How many search results to fetch per claim. Default 10 to match notebook behavior.")
with col2:
    run_button = st.button("Run Verification")

# ----------------- Helpers -----------------
@st.cache_resource(show_spinner=False)
def get_spacy():
    return load_spacy_model()

def split_sentences(text):
    nlp = get_spacy()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def color_for_status(status):
    if status == "verified":
        return "#b6f5b6"  # green
    if status == "uncertain":
        return "#fff2a6"  # yellow
    if status == "hallucination":
        return "#ffb6b6"  # red
    return "#ffd1d1"      # error / fallback

# ----------------- Main logic -----------------
if run_button:
    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")
        st.stop()

    sentences = split_sentences(paragraph)
    if not sentences:
        st.warning("No valid sentences detected.")
        st.stop()

    st.info(f"Found {len(sentences)} sentence(s). Verifying each claim (this may take a few minutes on first run).")
    progress = st.progress(0.0)
    results = []

    for i, sent in enumerate(sentences, 1):
        with st.spinner(f"Verifying claim {i}/{len(sentences)}..."):
            try:
                # verify_claim default top_k matches notebook (10) unless overridden by slider
                res = verify_claim(sent, top_k=top_k)
            except Exception as e:
                res = {"claim": sent, "status": "error", "best_snippet": None, "evidence": [], "error": str(e)}
            results.append(res)
            progress.progress(i / len(sentences))

    # ----------------- Annotated paragraph -----------------
    st.subheader("Annotated Paragraph")
    escaped_text = html.escape(paragraph)

    # Replace longer sentences first to avoid partial collisions
    for res in sorted(results, key=lambda x: len(x["claim"]), reverse=True):
        claim = html.escape(res["claim"])
        status = res.get("status", "uncertain")
        color = color_for_status(status)
        badge = f"<span style='font-weight:600; font-size:0.80em; margin-left:6px; background:#fff; border:1px solid #ccc; padding:2px 6px; border-radius:5px;'>{status.upper()}</span>"
        mark = f"<mark style='background:{color}; padding:0.15em 0.25em; border-radius:3px;'>{claim}{badge}</mark>"
        escaped_text = escaped_text.replace(claim, mark, 1)

    st.markdown(escaped_text, unsafe_allow_html=True)

    # ----------------- Detailed results -----------------
    st.subheader("Detailed Claim Analysis")
    for idx, r in enumerate(results, 1):
        st.markdown(f"### Claim {idx}: {r.get('status','unknown').upper()}")
        st.write(r.get("claim", ""))

        if r.get("status") == "error":
            st.error(f"Error verifying claim: {r.get('error', 'Unknown error')}")
            st.markdown("---")
            continue

        best_snippet = r.get("best_snippet")
        evidence = r.get("evidence", []) or []

        if best_snippet:
            st.markdown("**Best supporting snippet:**")
            snippet_short = best_snippet if len(best_snippet) < 700 else best_snippet[:700] + " ...[truncated]"
            st.write(snippet_short)
        elif evidence:
            st.markdown("**Top evidence snippets (first few):**")
            for s in evidence[:3]:
                s_short = s if len(s) < 500 else s[:500] + " ...[truncated]"
                st.write("-", s_short)
        else:
            st.info("No relevant snippets found for this claim.")

        st.markdown("---")

    st.success("Verification complete ✅")


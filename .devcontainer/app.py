import streamlit as st
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
# Import the consolidated logic file
import verifier 

# --- Centralized Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"App Device set to use {DEVICE}")

# --- Model Loading with Streamlit Caching ---
# @st.cache_resource decorator ensures models are loaded only once and cached.

@st.cache_resource
def load_nli_model():
    """Loads the RoBERTa NLI model."""
    MODEL_NAME_NLI = "roberta-large-mnli"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_NLI)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_NLI).to(DEVICE)
        print(f"[APP] Successfully loaded NLI model '{MODEL_NAME_NLI}'.")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading NLI model: {e}")
        return None, None

@st.cache_resource
def load_ner_pipeline():
    """Loads the BERT NER pipeline for query generation."""
    MODEL_NAME_NER = "dslim/bert-base-NER"
    try:
        ner_device = DEVICE.index if DEVICE.type == 'cuda' else -1
        ner_pipeline = pipeline("ner", model=MODEL_NAME_NER, aggregation_strategy="simple", device=ner_device)
        print(f"[APP] Successfully loaded NER pipeline '{MODEL_NAME_NER}'.")
        return ner_pipeline
    except Exception as e:
        st.error(f"Error loading NER pipeline: {e}")
        return None

@st.cache_resource
def load_spacy_model():
    """Loads the small English spaCy model for noun phrase extraction."""
    try:
        # Load the model (assumes successful installation from requirements.txt)
        nlp = spacy.load("en_core_web_sm")
        print("[APP] Successfully loaded spaCy model 'en_core_web_sm'.")
        return nlp
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        return None

@st.cache_resource
def load_embedder_model():
    """Loads the Sentence-Transformer model for semantic filtering."""
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("[APP] Successfully loaded SentenceTransformer 'all-MiniLM-L6-v2'.")
        return embedder
    except Exception as e:
        st.error(f"Error loading SentenceTransformer: {e}")
        return None

# --- Main Streamlit App ---

st.set_page_config(
    page_title="LLM Hallucination Detector",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("💡 LLM Hallucination Detector")
st.markdown("This tool uses **Search**, **Semantic Similarity Filtering**, and **Natural Language Inference (NLI)** to detect potential factual hallucinations in claims.")

# Load all cached resources (runs only once)
tokenizer_nli, model_nli = load_nli_model()
ner_pipeline = load_ner_pipeline()
nlp = load_spacy_model()
embedder = load_embedder_model()

# Check for model availability before proceeding
if None in [model_nli, ner_pipeline, nlp, embedder]:
    st.error("One or more core NLP models failed to load. Please check your environment and dependency installation.")
else:
    # --- Input and Settings ---
    claim_input = st.text_area(
        "Enter the factual claim to verify:",
        "The Earth’s atmosphere is composed mostly of nitrogen and oxygen." # Changed to the claim that previously failed to test the new logic
    )

    with st.expander("Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_k = st.slider(
                "Search Snippets (Top K):",
                min_value=10, max_value=200, value=100, step=10,
                help="The number of initial search results to fetch."
            )
        with col2:
            # Keep sim_threshold lower (e.g., 0.6) for better results on diverse snippets
            sim_threshold = st.slider(
                "Semantic Similarity Filter Threshold:",
                min_value=0.5, max_value=1.0, value=0.6, step=0.05,
                help="Snippets must be this similar to the claim to proceed to NLI."
            )
        with col3:
            nli_confidence_threshold = st.slider(
                "NLI Confidence Threshold:",
                min_value=0.5, max_value=1.0, value=0.75, step=0.05,
                help="Min. NLI score (Entailment/Contradiction) for a definitive status."
            )


    # Verification Button
    if st.button("Verify Claim", type="primary"):
        if not claim_input.strip():
            st.error("Please enter a claim to verify.")
        else:
            with st.spinner("🔍 Searching for evidence and classifying claim..."):
                # Pass all loaded models and parameters to the verifier
                result = verifier.verify_claim(
                    claim=claim_input, 
                    top_k=top_k, 
                    sim_threshold=sim_threshold,
                    nli_confidence_threshold=nli_confidence_threshold,
                    ner_pipeline=ner_pipeline, 
                    nlp=nlp, 
                    embedder=embedder, 
                    tokenizer_nli=tokenizer_nli, 
                    model_nli=model_nli, 
                    device=DEVICE
                )

            # --- Display Results ---
            st.subheader("Verification Result")
            
            status = result['status']
            best_snippet = result['best_snippet']
            
            # Map status to color/emoji (now safe from KeyError)
            if status == 'verified':
                st.success(f"✅ Status: **{status.upper()}** (Entailment Score: {result['max_entailment']:.4f})")
                status_message = "The claim is strongly supported by the evidence found."
            elif status == 'hallucination':
                st.error(f"❌ Status: **{status.upper()}** (Contradiction Score: {result['max_contradiction']:.4f})")
                status_message = "The claim is strongly contradicted by the evidence found, suggesting a potential hallucination."
            else: # uncertain
                st.warning(f"⚠️ Status: **{status.upper()}** (Max Entailment: {result['max_entailment']:.4f}, Max Contradiction: {result['max_contradiction']:.4f})")
                status_message = "The evidence found was inconclusive or lacked strong confidence for a definitive status."

            st.markdown(status_message)
            
            if best_snippet:
                st.markdown("---")
                st.caption("Most Relevant Evidence Snippet")
                st.code(best_snippet, language='text')

            # Debug/Detailed Output
            with st.expander("Show Detailed Log and All Evidence"):
                st.markdown("### Full Result Dictionary")
                st.json(result)
                
                if result['evidence']:
                    st.markdown("### All Snippets Used in NLI (Post-Filtering)")
                    for i, snippet in enumerate(result['evidence']):
                        st.code(f"Snippet {i+1}: {snippet}", language='text')
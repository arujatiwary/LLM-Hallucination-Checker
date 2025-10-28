# LLM Hallucination Detection System

This project detects and classifies hallucinations in LLM-generated text by verifying factual claims against real-world web evidence.  
It uses Natural Language Inference (NLI), Named Entity Recognition (NER) and semantic search to evaluate each sentence for factual consistency.



## Overview

Large Language Models (LLMs) can produce fluent text that sounds factual but isn’t backed by real information.  
This system analyzes each sentence (or “claim”) in a generated paragraph and classifies it as one of:

- 🟩 Verified — Supported by online evidence  
- 🟥 Hallucination — Contradicted by evidence or likely false  
- 🟨 Uncertain — No sufficient supporting data found  

The verification process is powered by RoBERTa-Large-MNLI for entailment/contradiction detection and web snippets retrieved from DuckDuckGo.



## Core Pipeline

Each claim passes through the following stages:

1. Sentence Segmentation – The paragraph is split into claims using spaCy.  
2. Search Query Generation – Each claim is expanded into a refined search query using:
   - Named entities (from a BERT-based NER model)
   - Noun phrases (from spaCy)
   - Heuristics for keywords like “discovered”, “invented”, etc.
3. Web Search – Queries are sent to DuckDuckGo using the `duckduckgo_search` library to retrieve contextual snippets.
4. NLI Evaluation – Each snippet is compared to the claim using `roberta-large-mnli`:
   - Label **ENTAILMENT** → supports the claim  
   - Label **CONTRADICTION** → refutes the claim  
   - Label **NEUTRAL** → irrelevant or insufficient evidence
5. Decision Logic 
   Based on maximum entailment and contradiction scores:
if contradiction >= 0.80 → hallucination
elif entailment >= 0.75 → verified
elif contradiction >= 0.75 → hallucination
else → uncertain
6. Visualization (Streamlit) – The paragraph is color-coded:
- 🟩 Green: Verified  
- 🟨 Yellow: Uncertain  
- 🟥 Red: Hallucination  



## Models Used
1. RoBERTa-large-MNLI (Natural Language Inference Model)
2. dslim/bert-base-NER (Named Entity Recognition Model)
3. all-MiniLM-L6-v2 (Sentence Transformer Model)
4. spaCy’s en_core_web_sm Model



## Project Structure

LLM-Hallucination-Detection/
│
├── app.py # Streamlit web interface
├── verifier.py # Core verification logic
├── requirements.txt # All Python dependencies
└── README.md # This documentation file

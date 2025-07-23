# DOMAIN AWARE RAG FOR MEDICAL NLP XAI

This project aims to develop a Retrieval-Augmented Generation (RAG) system for medical information, with a focus on explainable AI (XAI) techniques. The system will leverage large language models (LLMs) and medical knowledge bases to provide accurate and interpretable responses to medical queries.

## Project Structure

- `.env`: For storing API keys and other secrets (DO NOT COMMIT TO GIT)
- `.gitignore`: Specifies intentionally untracked files to ignore
- `README.md`: High-level project documentation (your project's front page)
- `requirements.txt`: Lists all Python package dependencies
- `data/`:
  - `raw/`: Unprocessed, original data (e.g., PubMed XMLs, NICE PDFs)
  - `processed/`: Cleaned, chunked, and domain-tagged data (e.g., JSONL files)
- `notebooks/`:
  - `01_data_exploration.ipynb`: Initial data analysis and EDA
  - `02_model_prototyping.ipynb`: Quick experiments with embeddings and models
  - `03_rag_pipeline_test.ipynb`: End-to-end pipeline testing before scripting
- `src/`:
  - `core/`:
    - `data_ingestion.py`: Scripts for loading, cleaning, and chunking data
    - `domain_classifier.py`: Logic for training and using the domain tagger
    - `embeddings.py`: Handles creation of BioClinical-BERT embeddings
    - `vector_db.py`: Manages ChromaDB indexing and retrieval
  - `pipeline/`:
    - `main_pipeline.py`: The main LangChain RAG orchestration logic
    - `explainability.py`: SHAP and attention extraction logic
  - `utils/`:
    - `config.py`: Loads configurations and paths
- `models/`:
  - `domain_classifier/`: Saved fine-tuned domain classification model
  - `llama_gguf/`: Location for your downloaded Llama 3 GGUF file
- `app/`:
  - `main.py`: The main Streamlit application file
  - `ui_components.py`: Custom Streamlit components (like the treemap)
- `tests/`:
  - `test_data_ingestion.py`: Unit tests for the data processing functions
  - `test_rag_pipeline.py`: Unit tests for the RAG pipeline components
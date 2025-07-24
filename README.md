# Domain-Aware, Explainable Medical RAG System

An advanced Retrieval-Augmented Generation (RAG) system designed to answer medical questions with a focus on domain-awareness and explainable AI (XAI). The system uses a hybrid retrieval strategy and provides SHAP-based explanations to show why information was considered relevant for the response.

---

## Core Features

- **Hybrid Retrieval**  
  Combines a domain-filtered search with a global semantic search to ensure comprehensive and relevant information retrieval from the knowledge base.

- **Domain-Awareness**  
  A fine-tuned transformer model predicts the medical domain of a user's query to focus the initial search, improving the relevance of retrieved documents.

- **Explainable AI (XAI) with SHAP**  
  Utilizes the SHAP (SHapley Additive exPlanations) framework on a cross-encoder model to provide word-level explanations for document relevance.

- **Local & Open-Source**  
  Runs entirely on local hardware using open-source models (e.g., Microsoft's Phi-3) and libraries, ensuring privacy and control.

- **Interactive UI**  
  A user-friendly web interface built with Streamlit for easy interaction and clear visualization of results and explanations.

---

## System Architecture

The pipeline processes a user query through the following steps:

1. **Query Input**: The user submits a question through the Streamlit interface.

2. **Domain Classification**: The query is analyzed by a fine-tuned BioClinical-BERT model to predict the primary medical domain (e.g., Cardiology, Oncology).

3. **Hybrid Retrieval**: The system performs two parallel searches on the ChromaDB vector store:
   - A targeted search filtered by the predicted domain.
   - A global semantic search across all domains.
   - The results are merged and de-duplicated to create a comprehensive context.

4. **SHAP Explanation**: The retrieved documents are passed to the SHAP Explainer, which uses a Cross-Encoder model to generate word-level importance scores, explaining why each document is relevant to the query.

5. **LLM Response Generation**: The original retrieved documents and the user's query are formatted into a prompt and sent to a local LLM (Phi-3) to generate a final, synthesized answer.

6. **UI Display**: The final answer, domain contribution analysis, and the SHAP-highlighted source documents are presented to the user.

---

## Technology Stack

- **Backend**: Python 3.10
- **ML/NLP**: PyTorch, Hugging Face Transformers, LangChain, sentence-transformers
- **Vector Database**: ChromaDB
- **Explainable AI**: SHAP
- **LLM Inference**: LlamaCpp
- **Language Model**: Microsoft Phi-3 (GGUF)
- **Frontend**: Streamlit
- **Data Visualization**: Plotly

---

## Project Structure

```
med_rag_xai/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── domain_classifier/
│   └── phi3mini/
│
├── notebooks/
│
├── src/
│   ├── core/
│   │   ├── data_ingestion.py
│   │   ├── domain_classifier.py
│   │   ├── embeddings.py
│   │   ├── pubmed_downloader.py
│   │   ├── pubmed_selector.py
│   │   └── vector_db.py
│   │
│   ├── pipeline/
│   │   ├── explainability.py
│   │   └── main_pipeline.py
│   └── utils/
│
├── app/
│   ├── main.py
│   └── ui_components.py
│
└── tests/
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TharaneshA/Med_RAG_XAI
cd med_rag_xai
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# For Linux/macOS:
source venv/bin/activate
# For Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download LLM

Download a GGUF-compatible language model (e.g., phi-3-mini-4k-instruct.Q4_K_M.gguf) and place it in the `models/phi3mini/` directory.

### Data Ingestion and Model Training (One-Time Setup)

Run the following scripts in order to ingest data, train the domain classifier, and index the knowledge base.

```bash
# 1. Select and prepare PubMed metadata
python src/core/pubmed_selector.py

# 2. Download the selected PubMed articles
python src/core/pubmed_downloader.py

# 3. Process all raw data sources into a unified format
python src/core/data_ingestion.py

# 4. Prepare the labeled dataset for the classifier
python scripts/prepare_classifier_data.py

# 5. Train the domain classification model
python src/core/domain_classifier.py

# 6. Index all processed data into ChromaDB (this may take time)
python -m src.core.vector_db
```

## Usage

After the setup is complete, launch the Streamlit web interface:

```bash
streamlit run app/main.py
```

This starts the application with full domain-aware RAG and explainability features.

## The Explainability (XAI) Feature

The explainability in this project is powered by the SHAP (SHapley Additive exPlanations) framework.

After document retrieval, a Cross-Encoder model scores each document's relevance to the user query.

SHAP analyzes these scores and provides token-level attributions.

In the UI:

- Red-highlighted words positively influence relevance.

- Blue-highlighted words negatively influence relevance.

This visualization helps users understand why specific documents were selected to generate the answer.

## License

This project is licensed under the MIT License.
See the LICENSE file for full details.
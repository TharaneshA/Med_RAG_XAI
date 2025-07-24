# src/pipeline/main_pipeline.py

import logging
from pathlib import Path
import chromadb
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from .explainability import get_shap_explainer 

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLASSIFIER_MODEL_PATH = BASE_DIR / "models" / "domain_classifier"
# Updated to use the recommended Phi-3-mini model
LLM_MODEL_PATH = BASE_DIR / "models" / "llama_gguf" / "phi-3-mini-4k-instruct.Q4_K_M.gguf"
DB_PATH = str(BASE_DIR / "chromadb")
COLLECTION_NAME = "medical_knowledge_base"

class MedicalRAGPipeline:
    def __init__(self):
        """Initializes all components of the RAG pipeline."""
        logging.info("Initializing Medical RAG Pipeline...")
        self.explainer = get_shap_explainer() # Get the SHAP explainer instance
        logging.info("Medical RAG Pipeline initialized successfully.")
        
        # --- 1. Load Domain Classifier ---
        logging.info("Loading domain classifier...")
        self.domain_classifier = pipeline(
            "text-classification",
            model=str(CLASSIFIER_MODEL_PATH),
            tokenizer=str(CLASSIFIER_MODEL_PATH)
        )

        # --- 2. Initialize VectorDB Client ---
        logging.info("Initializing ChromaDB client...")
        self.db_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.db_client.get_collection(name=COLLECTION_NAME)

        # --- 3. Load the Language Model (LLM) ---
        logging.info(f"Loading LLM: {LLM_MODEL_PATH.name}...")
        self.llm = LlamaCpp(
            model_path=str(LLM_MODEL_PATH),
            n_gpu_layers=32,  # Adjust based on your GPU. -1 to offload all layers. 32 is a good starting point for RTX 4060.
            n_batch=512,
            n_ctx=4096,    # Context window size
            verbose=False,
        )
        
        # --- 4. Define the Prompt Template ---
        logging.info("Defining prompt template...")
        # This template is specifically formatted for Phi-3 instruct models
        prompt_template_str = """<|user|>
You are a helpful medical assistant. Answer the user's question based ONLY on the provided context.
If the information is not in the context, state that you cannot answer based on the provided information.

Context:
{context}

Question: {question}<|end|>
<|assistant|>
"""
        self.prompt_template = PromptTemplate(
            template=prompt_template_str,
            input_variables=["context", "question"]
        )
        
        # --- 5. Create the LangChain Chain ---
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        
        logging.info("Medical RAG Pipeline initialized successfully.")

    def retrieve_documents(self, query: str, predicted_domain: str, n_results: int = 5) -> tuple[list[str], list[dict]]:
        """Retrieves relevant documents from ChromaDB based on the query and predicted domain."""
        logging.info(f"Retrieving documents for domain: '{predicted_domain}'")
        
        # Perform the search with a metadata filter
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"domain": predicted_domain},
            include=["documents", "metadatas"] # Ensure we get both text and metadata
        )
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        return documents, metadatas

    def run(self, query: str):
        """
        Executes the full RAG pipeline for a given query.
        
        Returns:
            A dictionary containing the answer, source documents, and their metadata.
        """
        # --- 1. Classify the query's domain ---
        prediction = self.domain_classifier(query)
        predicted_domain = prediction[0]['label']
        logging.info(f"Predicted query domain: {predicted_domain}")

        retrieved_docs, retrieved_metadatas = self.retrieve_documents(query, predicted_domain)
        if not retrieved_docs:
            return {"answer": "I could not find any relevant information...", "sources": [], "contributions": {}}
            
        # --- NEW: Use the SHAP explainer ---
        explained_sources, domain_contributions = self.explainer.explain(
            query, retrieved_docs, retrieved_metadatas
        )
        
        context_str = "\n\n---\n\n".join(retrieved_docs)

        logging.info("Generating answer with the LLM...")
        response = self.llm_chain.invoke({
            "context": context_str,
            "question": query
        })

        return {
            "answer": response['text'], 
            "sources": explained_sources, # Pass the explained sources to the UI
            "contributions": domain_contributions
        }

if __name__ == '__main__':
    # This is for testing the pipeline directly
    if not LLM_MODEL_PATH.exists():
        print(f"Error: LLM model not found at {LLM_MODEL_PATH}")
        print("Please download the Phi-3-mini GGUF model and place it in the models/llama_gguf directory.")
    else:
        rag_pipeline = MedicalRAGPipeline()
        
        test_query = "What are the first-line treatments for newly diagnosed type 2 diabetes?"
        print(f"--- Running test query: '{test_query}' ---")
        
        result = rag_pipeline.run(test_query)
        
        print("\n--- Generated Answer ---")
        print(result['answer'])
        
        print("\n--- Sources Used ---")
        for i, source in enumerate(result['sources']):
            print(f"Source {i+1} (from file: {result['metadatas'][i].get('file_name', 'N/A')}):\n{source}\n")
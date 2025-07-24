import logging
from pathlib import Path
import chromadb
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain

# --- ADD THIS IMPORT ---
# We need access to our embedding model
from src.core.embeddings import get_embedding_model
from .explainability import get_shap_explainer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLASSIFIER_MODEL_PATH = BASE_DIR / "models" / "domain_classifier"
LLM_MODEL_PATH = BASE_DIR / "models" / "llama_gguf" / "phi-3-mini-4k-instruct.Q4_K_M.gguf"
DB_PATH = str(BASE_DIR / "chromadb")
COLLECTION_NAME = "medical_knowledge_base"

class MedicalRAGPipeline:
    def __init__(self):
        """Initializes all components of the RAG pipeline."""
        logging.info("Initializing Medical RAG Pipeline...")
        
        self.domain_classifier = pipeline(
            "text-classification",
            model=str(CLASSIFIER_MODEL_PATH),
            tokenizer=str(CLASSIFIER_MODEL_PATH)
        )

        self.db_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.db_client.get_collection(name=COLLECTION_NAME)

        # --- ADD THIS LINE ---
        # Load the same embedding model we used for indexing
        self.embedding_model = get_embedding_model()

        self.llm = LlamaCpp(
            model_path=str(LLM_MODEL_PATH),
            n_gpu_layers=32,
            n_batch=512,
            n_ctx=4096,
            verbose=False,
        )
        
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
        
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.explainer = get_shap_explainer()
        logging.info("Medical RAG Pipeline initialized successfully.")

    def retrieve_documents(self, query: str, predicted_domain: str, n_results: int = 5) -> tuple[list[str], list[dict]]:
        """Retrieves relevant documents from ChromaDB using the correct embedding model."""
        logging.info(f"Retrieving documents for domain: '{predicted_domain}'")
        
        # --- MODIFIED SECTION ---
        # 1. Manually create the embedding for the query string
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # 2. Query the collection using the embedding vector, not the text
        results = self.collection.query(
            query_embeddings=[query_embedding], # Use query_embeddings instead of query_texts
            n_results=n_results,
            where={"domain": predicted_domain},
            include=["documents", "metadatas"]
        )
        # --- END MODIFIED SECTION ---
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        return documents, metadatas

    def run(self, query: str):
        """Executes the full RAG pipeline for a given query."""
        prediction = self.domain_classifier(query)
        predicted_domain = prediction[0]['label']
        logging.info(f"Predicted query domain: {predicted_domain}")

        retrieved_docs, retrieved_metadatas = self.retrieve_documents(query, predicted_domain)
        if not retrieved_docs:
            return {"answer": "I could not find any relevant information...", "sources": [], "contributions": {}}
            
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
            "sources": explained_sources,
            "contributions": domain_contributions
        }

if __name__ == '__main__':
    # This is for testing the pipeline directly
    if not LLM_MODEL_PATH.exists():
        print(f"Error: LLM model not found at {LLM_MODEL_PATH}")
        print("Please download the Phi-3-mini GGUF model and place it in the models/phi3mini directory.")
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
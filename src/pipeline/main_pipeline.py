import logging
from pathlib import Path
import chromadb
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain

# Import our custom modules
from src.core.embeddings import get_embedding_model
from .explainability import get_shap_explainer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLASSIFIER_MODEL_PATH = BASE_DIR / "models" / "domain_classifier"
LLM_MODEL_PATH = BASE_DIR / "models" / "phi3mini" / "phi-3-mini-4k-instruct-q4.gguf"
DB_PATH = str(BASE_DIR / "chromadb")
COLLECTION_NAME = "medical_knowledge_base"

class MedicalRAGPipeline:
    def __init__(self):
        """Initializes all components of the RAG pipeline."""
        logging.info("Initializing Medical RAG Pipeline...")
        
        # Load Domain Classifier
        self.domain_classifier = pipeline(
            "text-classification",
            model=str(CLASSIFIER_MODEL_PATH),
            tokenizer=str(CLASSIFIER_MODEL_PATH)
        )

        # Initialize VectorDB Client
        self.db_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.db_client.get_collection(name=COLLECTION_NAME)

        # Load the same embedding model we used for indexing
        self.embedding_model = get_embedding_model()

        # Load the Language Model (LLM)
        logging.info(f"Loading LLM: {LLM_MODEL_PATH.name}...")
        self.llm = LlamaCpp(
            model_path=str(LLM_MODEL_PATH),
            n_gpu_layers=32,
            n_batch=512,
            n_ctx=4096,
            verbose=False,
        )
        
        logging.info("Defining prompt template...") 
        system_prompt = """You are Med-AI, an advanced AI medical assistant. Your goal is to provide comprehensive and accurate answers to medical and scientific questions. 
        
        **Primary Directive:** Answer the user's question using your own extensive knowledge. 
        
        **Using Provided Context:** You will be given a `[CONTEXT]` section containing retrieved documents. Use this context ONLY for the following purposes: 
        1. To verify factual information, such as specific statistics, drug dosages, or study results. 
        2. To provide direct quotes or citations to support your answer. 
        3. If your own knowledge is insufficient, you may use the context to formulate the answer. 
        
        If you use information from the context, cite it by mentioning it (e.g., "According to a retrieved document..."). 
        
        **Mandatory Rules:** 1. **Disclaimer First:** Do NOT wait until the end. Start EVERY response with the following disclaimer: "**Disclaimer: This is for informational purposes only. Consult a qualified healthcare professional for medical advice.**" 
        2. **No Medical Advice:** Never provide direct medical advice, diagnoses, or treatment plans. You are an informational tool, not a medical provider. 
        3. **Tone:** Maintain a professional, clear, and empathetic tone. 
        """ 
        
        prompt_template_str = f"""<|system|> 
        {system_prompt}<|end|> 
        <|user|> 
        [CONTEXT] 
        {{context}} 
        
        [QUESTION] 
        {{question}}<|end|> 
        <|assistant|> 
        """ 
        
        self.prompt_template = PromptTemplate( 
            template=prompt_template_str, 
            input_variables=["context", "question"] 
        )
        
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.explainer = get_shap_explainer()
        logging.info("Medical RAG Pipeline initialized successfully.")

    # --- MODIFIED FUNCTION WITH HYBRID SEARCH ---
    def retrieve_documents(self, query: str, predicted_domain: str, n_results: int = 5) -> tuple[list[str], list[dict]]:
        """
        Retrieves documents using a hybrid search strategy: one targeted and one global search.
        """
        logging.info(f"Performing Hybrid Search. Primary domain: '{predicted_domain}'")
        query_embedding = self.embedding_model.encode(query).tolist()

        # 1. Targeted search within the predicted domain (gets n_results, e.g., 5)
        targeted_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"domain": predicted_domain},
            include=["documents", "metadatas", "ids"]
        )

        # 2. Global search across all domains (gets n_results, e.g., 5)
        global_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            # No 'where' filter for the global search
            include=["documents", "metadatas", "ids"]
        )
        
        # 3. Merge and de-duplicate results
        all_results = {}
        # Use document IDs to handle duplicates, giving priority to targeted results
        for i, doc_id in enumerate(targeted_results['ids'][0]):
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "document": targeted_results['documents'][0][i],
                    "metadata": targeted_results['metadatas'][0][i]
                }
        
        for i, doc_id in enumerate(global_results['ids'][0]):
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "document": global_results['documents'][0][i],
                    "metadata": global_results['metadatas'][0][i]
                }
        
        final_docs = [res['document'] for res in all_results.values()]
        final_metadatas = [res['metadata'] for res in all_results.values()]

        logging.info(f"Retrieved {len(final_docs)} unique documents after hybrid search.")
        return final_docs, final_metadatas
    # --- END MODIFIED FUNCTION ---

    def run(self, query: str):
        """Executes the full RAG pipeline for a given query."""
        prediction = self.domain_classifier(query)
        predicted_domain = prediction[0]['label']
        logging.info(f"Predicted query domain: {predicted_domain}")

        retrieved_docs, retrieved_metadatas = self.retrieve_documents(query, predicted_domain)
        if not retrieved_docs:
            return {
                "answer": "I could not find any relevant information in the knowledge base to answer your question.", 
                "sources": [], 
                "contributions": {}
            }
            
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
    if not LLM_MODEL_PATH.exists():
        print(f"Error: LLM model not found at {LLM_MODEL_PATH}")
    else:
        rag_pipeline = MedicalRAGPipeline()
        test_query = "What are the first-line treatments for newly diagnosed type 2 diabetes?"
        print(f"--- Running test query: '{test_query}' ---")
        
        result = rag_pipeline.run(test_query)
        
        print("\n--- Generated Answer ---")
        print(result['answer'])
        
        print("\n--- Sources Used ---")
        for i, source in enumerate(result['sources']):
            print(f"Source {i+1} (Domain: {source['metadata'].get('domain', 'N/A')})")
        
        print("\n--- Domain Contributions ---")
        print(result['contributions'])
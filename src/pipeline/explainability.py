import logging
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter
import numpy as np
import scipy as sp

# We need to ensure the nltk sentence tokenizer is downloaded
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Changed from nltk.downloader.DownloadError
    print("Downloading nltk punkt tokenizer...")
    nltk.download('punkt')

class ShapRAGExplainer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logging.info("Initializing SHAP Explainer...")
            cls._instance = super(ShapRAGExplainer, cls).__new__(cls)
            
            model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._instance.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # --- THE FIX: Create a custom prediction function for the explainer ---
            # This function ensures the data passed to the model is in the correct batch format
            def prediction_function(x):
                try:
                    # Tokenize the input text array
                    inputs = cls._instance.tokenizer(list(x), padding=True, truncation=True, return_tensors="pt")
                    # Move tensors to the same device as the model
                    inputs = {key: val.to(cls._instance.model.device) for key, val in inputs.items()}
                    # Get model predictions
                    with torch.no_grad():
                        outputs = cls._instance.model(**inputs)
                    # We're interested in the raw score (logit) for the positive class
                    # For cross-encoders, this is typically the only output logit
                    scores = outputs.logits.detach().cpu().numpy()
                    return scores
                except Exception as e:
                    logging.error(f"Error in SHAP prediction function: {e}")
                    # Return a correctly shaped array of zeros on error
                    return np.zeros((len(x), cls._instance.model.config.num_labels))
            
            cls._instance.prediction_function = prediction_function
            
            # Pass the custom function to the explainer
            cls._instance.explainer = shap.Explainer(cls._instance.prediction_function, cls._instance.tokenizer)
            logging.info("SHAP Explainer initialized successfully.")
        return cls._instance

    def explain(self, query: str, retrieved_docs: list[str], retrieved_metadatas: list[dict]):
        if not retrieved_docs:
            return [], {}

        explained_sources = []
        for doc_text, metadata in zip(retrieved_docs, retrieved_metadatas):
            # The input for SHAP text explainer is a list of strings
            shap_values = self._instance.explainer([query + " " + doc_text])
            
            # The output has shape (batch, tokens). We take the first (and only) item.
            explained_sources.append({
                "full_text": doc_text,
                "metadata": metadata,
                "shap_values": shap_values.values[0].tolist(),
                "tokens": shap_values.data[0].tolist()
            })

        domain_contributions = self._calculate_domain_contributions(retrieved_metadatas)
        return explained_sources, domain_contributions
    
    def _calculate_domain_contributions(self, metadatas: list[dict]) -> dict:
        if not metadatas: return {}
        domain_counts = Counter(meta.get('domain', 'Unknown') for meta in metadatas)
        total_docs = len(metadatas)
        return {domain: (count / total_docs) * 100 for domain, count in domain_counts.items()}

# Singleton instance
shap_explainer_handler = ShapRAGExplainer()

def get_shap_explainer():
    return shap_explainer_handler
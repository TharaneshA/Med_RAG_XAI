import logging
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter

class ShapRAGExplainer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logging.info("Initializing SHAP Explainer...")
            cls._instance = super(ShapRAGExplainer, cls).__new__(cls)
            
            # --- Load Cross-Encoder Model and Tokenizer ---
            model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._instance.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # --- Create SHAP Explainer ---
            # We use a Partition explainer, which is suitable for transformer models.
            cls._instance.explainer = shap.Explainer(cls._instance.model, cls._instance.tokenizer)
            logging.info("SHAP Explainer initialized successfully.")
        return cls._instance

    def explain(self, query: str, retrieved_docs: list[str], retrieved_metadatas: list[dict]):
        if not retrieved_docs:
            return [], {}

        explained_sources = []
        for doc_text, metadata in zip(retrieved_docs, retrieved_metadatas):
            # Generate SHAP values for the [query, document] pair
            shap_values = self._instance.explainer([query, doc_text])
            
            # The output for a cross-encoder is a single score. We look at the explanation for that.
            # We use shap_values.values[1] because we want the explanation for the document part (index 1), not the query part (index 0).
            # We use shap_values.data[1] to get the tokenized document text.
            explained_sources.append({
                "full_text": doc_text,
                "metadata": metadata,
                "shap_values": shap_values.values[1].tolist(),
                "tokens": shap_values.data[1].tolist()
            })

        # Calculate domain contributions based on all retrieved documents
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
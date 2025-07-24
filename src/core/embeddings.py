# src/core/embeddings.py

from sentence_transformers import SentenceTransformer
import logging

# --- Configuration ---
# We use the same model for embeddings as our classifier's base
# for semantic consistency, though other sentence-specific models also work well.
EMBEDDING_MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"

# Use a singleton pattern to ensure we only load the model once.
class EmbeddingModel:
    """A singleton class to handle the sentence embedding model."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logging.info("Initializing and loading the embedding model...")
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            # Load the model
            cls._instance.model = SentenceTransformer(EMBEDDING_MODEL_CHECKPOINT)
            logging.info("Embedding model loaded successfully.")
        return cls._instance

    def get_model(self):
        """Returns the loaded SentenceTransformer model."""
        return self.model

# Instantiate the model so it's ready for import
embedding_model_handler = EmbeddingModel()

def get_embedding_model():
    """Convenience function to get the embedding model instance."""
    return embedding_model_handler.get_model()
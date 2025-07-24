# src/core/vector_db.py

import json
from pathlib import Path
import logging
import chromadb
from tqdm import tqdm
from transformers import pipeline
from .embeddings import get_embedding_model
import torch

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE_PATH = PROCESSED_DATA_DIR / "all_processed_chunks.jsonl"
CLASSIFIER_MODEL_PATH = BASE_DIR / "models" / "domain_classifier"
DB_PATH = str(BASE_DIR / "chromadb") # ChromaDB path must be a string
COLLECTION_NAME = "medical_knowledge_base"

def main():
    logging.info("--- Starting Vector Database Indexing ---")

    # --- 1. Load Models ---
    logging.info("Loading Domain Classifier and Embedding Model...")
    # Determine device: use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    
    # Load your fine-tuned classifier
    domain_classifier = pipeline(
        "text-classification",
        model=str(CLASSIFIER_MODEL_PATH),
        tokenizer=str(CLASSIFIER_MODEL_PATH),
        device=device
    )
    
    # Get the embedding model from our handler
    embedding_model = get_embedding_model()
    
    # --- 2. Load Data ---
    if not CHUNKS_FILE_PATH.exists():
        logging.error(f"Chunks file not found at {CHUNKS_FILE_PATH}. Aborting.")
        return
        
    logging.info("Loading processed chunks from file...")
    with open(CHUNKS_FILE_PATH, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f]

    # --- 3. Initialize ChromaDB ---
    logging.info(f"Initializing ChromaDB client at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    # Get or create the collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # --- 4. Process and Insert in Batches ---
    batch_size = 500 # Process 500 chunks at a time to manage memory
    logging.info(f"Processing {len(all_chunks)} chunks in batches of {batch_size}...")

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Indexing Chunks"):
        batch_chunks = all_chunks[i:i + batch_size]
        
        # Prepare data for the current batch
        chunk_texts = [chunk['text'] for chunk in batch_chunks]
        chunk_ids = [chunk['chunk_id'] for chunk in batch_chunks]

        # Predict domains for the batch
        # We pass only the texts to the classifier
        domain_predictions = domain_classifier(chunk_texts, padding=True, truncation=True, max_length=512)
        
        # Generate embeddings for the batch
        embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False).tolist()

        # Create metadata for the batch
        metadatas = []
        for j, chunk in enumerate(batch_chunks):
            metadata = chunk['metadata']
            # Add the predicted domain to the metadata
            metadata['domain'] = domain_predictions[j]['label']
            metadatas.append(metadata)

        # Insert the batch into ChromaDB
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=chunk_texts # Storing the text directly in the DB is useful
        )

    logging.info(f"Successfully indexed {len(all_chunks)} chunks into the '{COLLECTION_NAME}' collection.")
    logging.info("--- Vector Database Indexing Finished ---")


if __name__ == "__main__":
    main()
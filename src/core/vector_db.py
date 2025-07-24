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
DB_PATH = str(BASE_DIR / "chromadb") 
COLLECTION_NAME = "medical_knowledge_base"

def main():
    logging.info("--- Starting Vector Database Indexing (Optimized) ---")

    # --- 1. Load Models ---
    logging.info("Loading Domain Classifier and Embedding Model...")
    device = 0 if torch.cuda.is_available() else -1
    
    domain_classifier = pipeline(
        "text-classification",
        model=str(CLASSIFIER_MODEL_PATH),
        tokenizer=str(CLASSIFIER_MODEL_PATH),
        device=device
    )
    
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
    
    # Delete the old collection to start fresh if it exists
    try:
        if COLLECTION_NAME in [c.name for c in client.list_collections()]:
            logging.warning(f"Collection '{COLLECTION_NAME}' already exists. Deleting to re-index.")
            client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        logging.error(f"Could not delete old collection: {e}")

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # --- 4. Process and Insert in Batches (Optimized) ---
    main_batch_size = 2048  # Increased batch size for less loop overhead
    gpu_batch_size = 64    # Batch size for model inference to maximize GPU usage
    
    logging.info(f"Processing {len(all_chunks)} chunks in main batches of {main_batch_size}...")

    # Use tqdm to show progress in terms of chunks, not batches
    with tqdm(total=len(all_chunks), desc="Indexing Chunks") as progress_bar:
        for i in range(0, len(all_chunks), main_batch_size):
            batch_chunks = all_chunks[i:i + main_batch_size]
            
            chunk_texts = [chunk['text'] for chunk in batch_chunks]
            chunk_ids = [chunk['chunk_id'] for chunk in batch_chunks]

            # Predict domains for the batch with optimized batching
            domain_predictions = domain_classifier(
                chunk_texts, padding=True, truncation=True, max_length=512, batch_size=gpu_batch_size
            )
            
            # Generate embeddings for the batch with optimized batching
            embeddings = embedding_model.encode(
                chunk_texts, show_progress_bar=False, batch_size=gpu_batch_size
            ).tolist()

            metadatas = []
            for j, chunk in enumerate(batch_chunks):
                metadata = chunk['metadata']
                metadata['domain'] = domain_predictions[j]['label']
                
                if 'MeSH' in metadata and isinstance(metadata['MeSH'], list):
                    metadata['MeSH'] = ", ".join(metadata['MeSH'])

                metadatas.append(metadata)

            collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=chunk_texts
            )
            progress_bar.update(len(batch_chunks))

    logging.info(f"Successfully indexed {len(all_chunks)} chunks into the '{COLLECTION_NAME}' collection.")
    logging.info("--- Vector Database Indexing Finished ---")


if __name__ == "__main__":
    main()

import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from tqdm import tqdm
import json
import logging
import uuid
import numpy as np

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    """Performs basic text cleaning."""
    return ' '.join(text.split()).strip()

def process_pubmed_files(directory: Path) -> list[dict]:
    """
    Processes PubMed NXML files using a metadata CSV for MeSH term lookup.
    """
    metadata_path = directory.parent / "pubmed_metadata.csv"
    if not metadata_path.exists():
        logging.error(f"Metadata lookup file not found at {metadata_path}. Cannot process PubMed files.")
        return []

    metadata_df = pd.read_csv(metadata_path)
    
    # The key is the clean PMCID (e.g., 'PMC12345'), which matches the folder name.
    file_stem_to_mesh = {
        Path(row['File']).name.replace('.tar.gz', ''): json.loads(row['MeSH_JSON']) 
        for _, row in metadata_df.iterrows()
    }

    documents = []
    logging.info(f"Scanning for NXML files in {directory}...")
    nxml_files = list(directory.rglob("*.nxml"))
    
    if not nxml_files:
        logging.warning(f"No .nxml files found in {directory}.")
        return []

    logging.info(f"Found {len(nxml_files)} NXML files to process.")

    for file_path in tqdm(nxml_files, desc="Processing PubMed Articles"):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            title = ''.join(root.find('.//article-title').itertext()).strip() if root.find('.//article-title') is not None else ""
            abstract = ''.join(root.find('.//abstract').itertext()).strip() if root.find('.//abstract') is not None else ""
            body = ''.join(root.find('.//body').itertext()).strip() if root.find('.//body') is not None else ""
            full_text = f"Title: {title}\n\nAbstract: {abstract}\n\nBody: {body}"
            
            # The key is the PARENT FOLDER'S NAME (e.g., 'PMC12345')
            lookup_key = file_path.parent.name
            mesh_list = file_stem_to_mesh.get(lookup_key, [])

            documents.append({
                "doc_id": f"pubmed_{lookup_key}",
                "text": clean_text(full_text),
                "metadata": { "source": "PubMed", "file_name": file_path.name, "MeSH": mesh_list }
            })
        except Exception as e:
            logging.error(f"An unexpected error occurred with file {file_path}: {e}")
            
    return documents

def process_nice_guidelines(directory: Path) -> list[dict]:
    """Processes all NICE guideline HTML files in a given directory."""
    documents = []
    logging.info(f"Scanning for HTML files in {directory}...")
    html_files = list(directory.glob("*.html"))
    if not html_files:
        logging.warning(f"No .html files found in {directory}. Skipping NICE processing.")
        return []
    logging.info(f"Found {len(html_files)} HTML files to process.")
    for file_path in tqdm(html_files, desc="Processing NICE Guidelines"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
            content_node = soup.find('main') or soup.find('body')
            text = content_node.get_text(separator=' ', strip=True) if content_node else ""
            documents.append({
                "doc_id": f"nice_{file_path.stem}",
                "text": clean_text(text),
                "metadata": {
                    "source": "NICE Guidelines",
                    "file_name": file_path.name
                }
            })
        except Exception as e:
            logging.error(f"An error occurred while processing {file_path}: {e}")
    return documents

def process_synthea_notes(directory: Path) -> list[dict]:
    """Processes Synthea CSV files to extract relevant textual information."""
    logging.info("Processing Synthea data...")
    all_documents = []
    files_to_process = {
        "careplans.csv": ["DESCRIPTION", "REASONDESCRIPTION"],
        "conditions.csv": ["DESCRIPTION"],
        "encounters.csv": ["REASONDESCRIPTION", "DESCRIPTION"],
        "medications.csv": ["DESCRIPTION", "REASONDESCRIPTION"],
        "observations.csv": ["DESCRIPTION", "VALUE"],
        "procedures.csv": ["DESCRIPTION"]
    }
    patients_path = directory / "patients.csv"
    if not patients_path.exists():
        logging.warning("Synthea patients.csv not found. Context will be limited.")
        patients_df = pd.DataFrame()
    else:
        patients_df = pd.read_csv(patients_path, low_memory=False)
    for filename, text_columns in files_to_process.items():
        file_path = directory / filename
        if not file_path.exists():
            logging.warning(f"Synthea file {filename} not found. Skipping.")
            continue
        df = pd.read_csv(file_path, low_memory=False)
        if 'PATIENT' in df.columns and not patients_df.empty:
            df = pd.merge(df, patients_df[['Id', 'GENDER']], left_on='PATIENT', right_on='Id', how='left')
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {filename}"):
            full_text = ""
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    full_text += f"{col.replace('_', ' ').title()}: {row[col]}\n"
            if full_text:
                doc_id_prefix = f"synthea_{filename.split('.')[0]}"
                patient_id = row.get('PATIENT', 'UNKNOWN')
                context = f"Clinical entry for Patient ID {patient_id}."
                if 'GENDER' in row and pd.notna(row['GENDER']):
                    context += f" Gender: {row['GENDER']}."
                full_text = context + "\n\n" + full_text
                all_documents.append({
                    "doc_id": f"{doc_id_prefix}_{row.name}",
                    "text": clean_text(full_text),
                    "metadata": {
                        "source": "Synthea EHR",
                        "source_file": filename,
                        "patient_id": str(patient_id)
                    }
                })
    return all_documents

def main():
    """Main function to orchestrate the data ingestion and preprocessing pipeline."""
    logging.info("--- Starting Data Ingestion Pipeline ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, length_function=len)
    all_docs = []
    all_docs.extend(process_pubmed_files(RAW_DATA_DIR / "pubmed"))
    all_docs.extend(process_nice_guidelines(RAW_DATA_DIR / "nice_guidelines"))
    all_docs.extend(process_synthea_notes(RAW_DATA_DIR / "synthea"))
    if not all_docs:
        logging.info("No documents were processed. Exiting.")
        return
    logging.info(f"Total documents processed from all sources: {len(all_docs)}")
    all_chunks = []
    for doc in tqdm(all_docs, desc="Chunking All Documents"):
        if not isinstance(doc.get('text'), str) or not doc['text'].strip():
            continue
        chunks = text_splitter.split_text(doc['text'])
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = doc.get("metadata", {}).copy()
            for key, value in chunk_metadata.items():
                if isinstance(value, list):
                    continue
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    chunk_metadata[key] = value.item()
                elif pd.isna(value):
                    chunk_metadata[key] = None
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_id": doc['doc_id'],
                "chunk_number": i,
                "text": chunk_text,
                "metadata": chunk_metadata
            })
    logging.info(f"Total chunks created: {len(all_chunks)}")
    output_path = PROCESSED_DATA_DIR / "all_processed_chunks.jsonl"
    logging.info(f"Saving processed chunks to {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk) + '\n')
    except TypeError as e:
        logging.error(f"Serialization error when writing JSON: {e}")
        for i, chunk in enumerate(all_chunks):
            try:
                json.dumps(chunk)
            except TypeError as te:
                logging.error(f"Problematic chunk at index {i}: {chunk}")
                logging.error(f"Specific error: {te}")
                for k, v in chunk.get('metadata', {}).items():
                    logging.error(f"Metadata key '{k}' has type {type(v)}")
                break
    logging.info("--- Data Ingestion Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
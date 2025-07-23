# src/core/data_ingestion.py

import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from tqdm import tqdm
import json
import logging
import uuid

# --- Setup Logging ---
# Professional scripts should always have logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Use pathlib for robust path management
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Ensure the processed data directory exists
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning.

    Args:
        text: The input string to clean.

    Returns:
        The cleaned string.
    """
    # Replace multiple newlines/spaces with a single one
    text = ' '.join(text.split())
    return text.strip()

def process_pubmed_files(directory: Path) -> list[dict]:
    """
    Processes all PubMed NXML files in a given directory.

    Args:
        directory: The path to the directory containing PubMed NXML files.

    Returns:
        A list of dictionaries, where each dictionary contains the content
        and metadata of one article.
    """
    documents = []
    logging.info(f"Scanning for NXML files in {directory}...")
    # Using rglob to find all .nxml files recursively, which handles the nested folder structure
    nxml_files = list(directory.rglob("*.nxml"))
    
    if not nxml_files:
        logging.warning(f"No .nxml files found in {directory}. Skipping PubMed processing.")
        return []

    logging.info(f"Found {len(nxml_files)} NXML files to process.")

    for file_path in tqdm(nxml_files, desc="Processing PubMed Articles"):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extracting title
            title_node = root.find('.//article-title')
            title = ''.join(title_node.itertext()).strip() if title_node is not None else "No Title"
            
            # Extracting abstract
            abstract_node = root.find('.//abstract')
            abstract = ''.join(abstract_node.itertext()).strip() if abstract_node is not None else ""
            
            # Extracting body
            body_node = root.find('.//body')
            body = ''.join(body_node.itertext()).strip() if body_node is not None else ""
            
            full_text = f"Title: {title}\n\nAbstract: {abstract}\n\nBody: {body}"
            
            documents.append({
                "doc_id": f"pubmed_{file_path.stem}",
                "text": clean_text(full_text),
                "metadata": {
                    "source": "PubMed",
                    "file_name": file_path.name
                }
            })
        except ET.ParseError:
            logging.error(f"Could not parse XML for {file_path}. Skipping.")
        except Exception as e:
            logging.error(f"An unexpected error occurred with file {file_path}: {e}")
            
    return documents

def process_nice_guidelines(directory: Path) -> list[dict]:
    """
    Processes all NICE guideline HTML files in a given directory.

    Args:
        directory: The path to the directory containing NICE HTML files.

    Returns:
        A list of dictionaries for each guideline.
    """
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
                
            # A common tag for main content in modern websites is <main>
            # If not found, fall back to <body>
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
    """
    Processes Synthea CSV files to extract relevant textual information.

    Args:
        directory: The path to the directory containing Synthea CSV files.

    Returns:
        A list of dictionaries, one for each piece of clinical information.
    """
    logging.info("Processing Synthea data...")
    all_documents = []
    
    # Define which files and columns to process
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
        patients_df = pd.read_csv(patients_path)

    for filename, text_columns in files_to_process.items():
        file_path = directory / filename
        if not file_path.exists():
            logging.warning(f"Synthea file {filename} not found. Skipping.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Merge with patient data for context
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
                
                # Add patient context if available
                context = f"Clinical entry for Patient ID {patient_id}."
                if 'GENDER' in row:
                    context += f" Gender: {row['GENDER']}."
                
                full_text = context + "\n\n" + full_text

                all_documents.append({
                    "doc_id": f"{doc_id_prefix}_{row.name}",
                    "text": clean_text(full_text),
                    "metadata": {
                        "source": "Synthea EHR",
                        "source_file": filename,
                        "patient_id": patient_id
                    }
                })

    return all_documents

def main():
    """
    Main function to orchestrate the data ingestion and preprocessing pipeline.
    """
    logging.info("--- Starting Data Ingestion Pipeline ---")

    # Initialize the text splitter from LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )

    # Process all data sources
    all_docs = []
    all_docs.extend(process_pubmed_files(RAW_DATA_DIR / "pubmed"))
    all_docs.extend(process_nice_guidelines(RAW_DATA_DIR / "nice_guidelines"))
    all_docs.extend(process_synthea_notes(RAW_DATA_DIR / "synthea"))

    if not all_docs:
        logging.info("No documents were processed. Exiting.")
        return
        
    logging.info(f"Total documents processed from all sources: {len(all_docs)}")
    
    # Chunk all documents
    all_chunks = []
    for doc in tqdm(all_docs, desc="Chunking All Documents"):
        chunks = text_splitter.split_text(doc['text'])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_id": doc['doc_id'],
                "chunk_number": i,
                "text": chunk_text,
                "metadata": doc['metadata']
            })

    logging.info(f"Total chunks created: {len(all_chunks)}")

    # Save the processed chunks to a JSONL file
    output_path = PROCESSED_DATA_DIR / "all_processed_chunks.jsonl"
    logging.info(f"Saving processed chunks to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')
            
    logging.info("--- Data Ingestion Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
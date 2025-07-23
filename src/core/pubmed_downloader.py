import requests
import tarfile
from pathlib import Path
from tqdm import tqdm
import logging
import time
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
METADATA_INPUT_PATH = RAW_DATA_DIR / "pubmed_metadata.csv"
DOWNLOAD_DIR = RAW_DATA_DIR / "pubmed"

FTP_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def download_and_extract(file_path_on_server: str):
    full_url = FTP_BASE_URL + file_path_on_server
    local_tar_path = DOWNLOAD_DIR / Path(file_path_on_server).name
    
    # --- CORRECTED LOGIC TO GET ARTICLE ID ---
    article_pmcid = Path(file_path_on_server).name.replace('.tar.gz', '')
    
    # Check if the extracted directory already exists to avoid re-downloading
    if (DOWNLOAD_DIR / article_pmcid).exists():
        logging.info(f"Article {article_pmcid} already exists. Skipping download.")
        return

    try:
        logging.debug(f"Downloading {full_url}")
        with requests.get(full_url, stream=True) as r:
            r.raise_for_status()
            with open(local_tar_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        
        logging.debug(f"Extracting {local_tar_path}")
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(path=DOWNLOAD_DIR)

        local_tar_path.unlink()
        
    except Exception as e:
        logging.error(f"An error occurred for {file_path_on_server}: {e}")

def main():
    logging.info("--- Starting PubMed Downloader ---")
    if not METADATA_INPUT_PATH.exists():
        logging.error(f"Metadata file not found at {METADATA_INPUT_PATH}. Please run pubmed_selector.py first.")
        return
        
    df = pd.read_csv(METADATA_INPUT_PATH)
    target_files = df['File'].tolist()
        
    if not target_files:
        logging.warning("Target list is empty. Nothing to download.")
        return
        
    logging.info(f"Found {len(target_files)} articles to download.")
    
    for file_path in tqdm(target_files, desc="Downloading PubMed Articles"):
        download_and_extract(file_path)
        time.sleep(0.5)
        
    logging.info("--- PubMed Downloader Finished ---")

if __name__ == "__main__":
    main()
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm
import logging
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
TARGET_LIST_PATH = RAW_DATA_DIR / "pubmed_target_list.txt"
DOWNLOAD_DIR = RAW_DATA_DIR / "pubmed"

# Base URL for the FTP server
FTP_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/"

# Ensure the download directory exists
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_and_extract(file_path_on_server: str):
    """
    Downloads a single tar.gz file from the PubMed FTP server and extracts it.

    Args:
        file_path_on_server: The relative path of the file on the server
                             (e.g., 'oa_package/a5/3f/PMC1000001.tar.gz').
    """
    full_url = FTP_BASE_URL + file_path_on_server
    local_tar_path = DOWNLOAD_DIR / Path(file_path_on_server).name
    
    try:
        # --- Download ---
        logging.debug(f"Downloading {full_url}")
        with requests.get(full_url, stream=True) as r:
            r.raise_for_status()
            with open(local_tar_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # --- Extract ---
        logging.debug(f"Extracting {local_tar_path}")
        with tarfile.open(local_tar_path, "r:gz") as tar:
            # The tarball contains a directory, extract its contents into DOWNLOAD_DIR
            # Get the top-level directory name inside the tarball
            top_level_dir = tar.getnames()[0].split('/')[0]
            extract_target_dir = DOWNLOAD_DIR / top_level_dir
            extract_target_dir.mkdir(exist_ok=True, parents=True)
            tar.extractall(path=extract_target_dir)

        # --- Cleanup ---
        # We can remove the downloaded .tar.gz to save space
        local_tar_path.unlink()
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {full_url}: {e}")
    except tarfile.TarError as e:
        logging.error(f"Failed to extract {local_tar_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred for {file_path_on_server}: {e}")


def main():
    logging.info("--- Starting PubMed Downloader ---")
    if not TARGET_LIST_PATH.exists():
        logging.error(f"Target list not found at {TARGET_LIST_PATH}. Please run pubmed_selector.py first.")
        return 
        
    with open(TARGET_LIST_PATH, 'r') as f:
        target_files = [line.strip() for line in f if line.strip()]
        
    if not target_files:
        logging.warning("Target list is empty. Nothing to download.")
        return 
        
    logging.info(f"Found {len(target_files)} articles to download.")
    
    for file_path in tqdm(target_files, desc="Downloading PubMed Articles"):
        download_and_extract(file_path)
        time.sleep(0.5) # Be respectful to the server
        
    logging.info("--- PubMed Downloader Finished ---")


if __name__ == "__main__":
    main()
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import time
import logging
import random

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
OA_FILE_LIST_PATH = RAW_DATA_DIR / "oa_file_list.csv"  # The big index file you downloaded
TARGET_LIST_PATH = RAW_DATA_DIR / "pubmed_target_list.txt"  # Our output "shopping list"

# --- Parameters for Selection ---
# Let's take a random sample to avoid querying the entire massive file
SAMPLE_SIZE = 5000
# Define the high-level medical domains we want to ensure are in our dataset
# These are top-level MeSH (Medical Subject Headings) categories
TARGET_DOMAINS = [
    "Cardiovascular Diseases",
    "Neoplasms",  # Cancers
    "Nervous System Diseases",
    "Respiratory Tract Diseases",
    "Musculoskeletal Diseases",
    "Digestive System Diseases",
    "Endocrine System Diseases",
    "Virus Diseases",
    "Mental Disorders"
]
# How many articles we want per domain
ARTICLES_PER_DOMAIN = 10
# Your email for NCBI API. It's good practice to let them know who you are.
YOUR_EMAIL = "tharanesh2k5@gmail.com"


def get_mesh_terms(pmid_list: list[str]) -> dict:
    """
    Fetches MeSH terms for a list of PubMed IDs (PMIDs) using the NCBI E-utilities API.

    Args:
        pmid_list: A list of PMIDs as strings.

    Returns:
        A dictionary mapping each PMID to its list of MeSH terms.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmid_list),
        "retmode": "xml",
        "rettype": "abstract"
    }
    headers = {'User-Agent': f'MedRAGXAIProject/1.0 ({YOUR_EMAIL})'}

    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()  # Raises an exception for bad status codes

        root = ET.fromstring(response.content)
        pmid_to_mesh = {}
        for article in root.findall('.//PubmedArticle'):
            pmid_node = article.find('.//PMID')
            if pmid_node is None:
                continue
            pmid = pmid_node.text

            mesh_list = []
            mesh_heading_list = article.find('.//MeshHeadingList')
            if mesh_heading_list is not None:
                for descriptor in mesh_heading_list.findall('.//DescriptorName'):
                    if descriptor.text:
                        mesh_list.append(descriptor.text)
            
            pmid_to_mesh[pmid] = mesh_list
        return pmid_to_mesh
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return {}
    except ET.ParseError as e:
        logging.error(f"XML parsing failed: {e}")
        return {}

def main():
    logging.info("--- Starting PubMed Article Selection ---")
    
    if not OA_FILE_LIST_PATH.exists():
        logging.error(f"Cannot find oa_file_list.csv at {OA_FILE_LIST_PATH}. Please download it first.")
        return 

    # Read the large CSV file
    logging.info(f"Reading {OA_FILE_LIST_PATH}...")
    df = pd.read_csv(OA_FILE_LIST_PATH, comment='#') # The file can have comments
    df.dropna(subset=['PMID', 'File'], inplace=True) # Ensure we have the necessary IDs and paths

    # Convert PMID to integer, then to string for the API call
    df['PMID'] = df['PMID'].astype(int).astype(str)

    # Take a random sample for efficiency
    logging.info(f"Taking a random sample of {SAMPLE_SIZE} articles...")
    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    logging.info("Fetching MeSH terms from NCBI. This may take a while...")
    # Fetch data in batches to be nice to the API
    batch_size = 100
    pmids = sample_df['PMID'].tolist()
    all_mesh_data = {}
    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching MeSH Batches"):
        batch_pmids = pmids[i:i+batch_size]
        mesh_data = get_mesh_terms(batch_pmids)
        all_mesh_data.update(mesh_data)
        time.sleep(0.5) # Be respectful and don't spam the API

    sample_df['MeSH'] = sample_df['PMID'].map(all_mesh_data)
    sample_df.dropna(subset=['MeSH'], inplace=True)

    # Select articles based on our target domains
    logging.info("Selecting articles to ensure domain diversity...")
    target_files = set()
    for domain in TARGET_DOMAINS:
        domain_articles_added = 0
        # Iterate through the dataframe to find articles for the current domain
        for _, row in sample_df.iterrows():
            if domain_articles_added >= ARTICLES_PER_DOMAIN:
                break
            # Check if any MeSH term contains our domain string
            if any(domain.lower() in term.lower() for term in row['MeSH']):
                if row['File'] not in target_files:
                    target_files.add(row['File'])
                    domain_articles_added += 1
    
    # Save the list of target files
    logging.info(f"Selected a total of {len(target_files)} unique articles for download.")
    with open(TARGET_LIST_PATH, 'w') as f:
        for file_path in sorted(list(target_files)):
            f.write(f"{file_path}\n")
            
    logging.info(f"Target download list saved to {TARGET_LIST_PATH}")
    logging.info("--- PubMed Article Selection Finished ---")


if __name__ == "__main__":
    main()
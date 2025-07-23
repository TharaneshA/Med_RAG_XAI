
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
OA_FILE_LIST_PATH = RAW_DATA_DIR / "oa_file_list.csv"
METADATA_OUTPUT_PATH = RAW_DATA_DIR / "pubmed_metadata.csv" # The new, more informative output

SAMPLE_SIZE = 5000
TARGET_DOMAINS = [
    "Cardiovascular Diseases", "Neoplasms", "Nervous System Diseases",
    "Respiratory Tract Diseases", "Musculoskeletal Diseases", "Digestive System Diseases",
    "Endocrine System Diseases", "Virus Diseases", "Mental Disorders"
]
ARTICLES_PER_DOMAIN = 10
YOUR_EMAIL = "your.email@example.com"

def get_mesh_terms(pmid_list: list[str]) -> dict:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml", "rettype": "abstract"}
    headers = {'User-Agent': f'MedRAGXAIProject/1.0 ({YOUR_EMAIL})'}

    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        pmid_to_mesh = {}
        for article in root.findall('.//PubmedArticle'):
            pmid_node = article.find('.//PMID')
            if pmid_node is None: continue
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
    logging.info("--- Starting PubMed Article Selection & Metadata Generation ---")
    
    if not OA_FILE_LIST_PATH.exists():
        logging.error(f"Cannot find oa_file_list.csv at {OA_FILE_LIST_PATH}.")
        return

    df = pd.read_csv(OA_FILE_LIST_PATH, comment='#')
    df.dropna(subset=['PMID', 'File'], inplace=True)
    df['PMID'] = df['PMID'].astype(int).astype(str)
    sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    logging.info("Fetching MeSH terms from NCBI...")
    batch_size = 100
    pmids = sample_df['PMID'].tolist()
    all_mesh_data = {}
    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching MeSH Batches"):
        batch_pmids = pmids[i:i+batch_size]
        mesh_data = get_mesh_terms(batch_pmids)
        all_mesh_data.update(mesh_data)
        time.sleep(0.5)

    sample_df['MeSH'] = sample_df['PMID'].map(all_mesh_data)
    sample_df.dropna(subset=['MeSH'], inplace=True)
    sample_df = sample_df[sample_df['MeSH'].apply(len) > 0] # Only keep articles that have MeSH terms

    logging.info("Selecting articles to ensure domain diversity...")
    final_selection_df = pd.DataFrame()
    for domain in TARGET_DOMAINS:
        domain_articles_added = 0
        domain_df = sample_df[sample_df['MeSH'].apply(lambda terms: any(domain.lower() in term.lower() for term in terms))]
        
        # Add up to ARTICLES_PER_DOMAIN, avoiding duplicates
        for _, row in domain_df.iterrows():
            if domain_articles_added >= ARTICLES_PER_DOMAIN:
                break
            if row['File'] not in final_selection_df.get('File', pd.Series()).values:
                final_selection_df = pd.concat([final_selection_df, pd.DataFrame([row])], ignore_index=True)
                domain_articles_added += 1

    logging.info(f"Selected a total of {len(final_selection_df)} unique articles for download.")
    
    # Save the selected data with MeSH terms to a CSV
    # We convert the MeSH list to a JSON string to store it cleanly in one CSV cell
    final_selection_df['MeSH_JSON'] = final_selection_df['MeSH'].apply(json.dumps)
    output_df = final_selection_df[['File', 'PMID', 'MeSH_JSON']]
    output_df.to_csv(METADATA_OUTPUT_PATH, index=False)
            
    logging.info(f"Target metadata saved to {METADATA_OUTPUT_PATH}")
    logging.info("--- PubMed Selection Finished ---")

if __name__ == "__main__":
    main()
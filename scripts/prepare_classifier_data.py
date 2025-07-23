import json
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict
from sklearn.model_selection import train_test_split

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
INPUT_FILE = PROCESSED_DATA_DIR / "all_processed_chunks.jsonl"
OUTPUT_DIR = PROCESSED_DATA_DIR / "classifier_data"

# Ensure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define the mapping from our target high-level domains to keywords
# that might appear in the detailed MeSH terms. This allows us to
# map fine-grained MeSH terms to our broader categories.
DOMAIN_KEYWORD_MAP = {
    "Cardiology": ["Cardiovascular", "Heart", "Vascular", "Blood Circulation"],
    "Oncology": ["Neoplasms", "Cancer", "Tumor"],
    "Neurology": ["Nervous System", "Brain", "Neuro", "Spinal Cord"],
    "Respiratory": ["Respiratory", "Lung", "Pulmonary"],
    "Musculoskeletal": ["Musculoskeletal", "Bone", "Joint", "Muscle"],
    "Gastroenterology": ["Digestive System", "Gastrointestinal", "Stomach", "Intestine", "Liver"],
    "Endocrinology": ["Endocrine", "Hormones", "Diabetes Mellitus"],
    "Infectious Disease": ["Virus Diseases", "Bacterial Infections", "Parasitic Diseases"],
    "Psychiatry": ["Mental Disorders", "Psych", "Behavior"]
}

def get_domain_from_mesh(mesh_terms: list, domain_map: dict) -> str | None:
    """
    Maps a list of MeSH terms to a single high-level domain.
    It prioritizes the first match found based on the order in the domain_map.
    """
    if not mesh_terms:
        return None
    
    for domain, keywords in domain_map.items():
        for keyword in keywords:
            for mesh_term in mesh_terms:
                if keyword.lower() in mesh_term.lower():
                    return domain
    return None

def main():
    logging.info("--- Preparing Dataset for Domain Classifier ---")

    if not INPUT_FILE.exists():
        logging.error(f"Input file not found: {INPUT_FILE}. Please run the ingestion script first.")
        return 

    # Load the processed chunks, focusing on PubMed data which has MeSH terms
    chunks = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            # We can only use data that has MeSH terms for supervised training
            if chunk.get("metadata", {}).get("source") == "PubMed" and "MeSH" in chunk.get("metadata", {}):
                chunks.append(chunk)

    if not chunks:
        logging.error("No PubMed chunks with MeSH terms found. Cannot create a training dataset.")
        return 

    logging.info(f"Found {len(chunks)} PubMed chunks with MeSH metadata.")

    # Create a DataFrame for easier processing
    df = pd.DataFrame(chunks)
    df['text'] = df['text']
    df['mesh_terms'] = df['metadata'].apply(lambda x: x.get('MeSH', []))

    # Apply the mapping to get a single domain label for each chunk
    df['domain'] = df['mesh_terms'].apply(lambda x: get_domain_from_mesh(x, DOMAIN_KEYWORD_MAP))

    # We can only train on data that we could successfully label
    df.dropna(subset=['domain'], inplace=True)
    
    if df.empty:
        logging.error("Could not assign any domain labels based on the provided MeSH terms and mapping. Check your DOMAIN_KEYWORD_MAP.")
        return 

    logging.info("Domain counts in the dataset:")
    logging.info(f"\n{df['domain'].value_counts()}")

    # Select only the columns we need for training
    final_df = df[['text', 'domain']].copy()

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42, stratify=final_df['domain'])

    # Save the datasets as JSON files, which is easy to load with the Hugging Face datasets library
    train_df.to_json(OUTPUT_DIR / "train.json", orient="records", lines=True)
    test_df.to_json(OUTPUT_DIR / "test.json", orient="records", lines=True)

    # Save the label mapping
    labels = final_df['domain'].unique().tolist()
    label_map = {label: i for i, label in enumerate(labels)}
    with open(OUTPUT_DIR / "label_map.json", 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=4)

    logging.info(f"Successfully created training and testing datasets in {OUTPUT_DIR}")
    logging.info(f"Training set size: {len(train_df)}")
    logging.info(f"Testing set size: {len(test_df)}")
    logging.info("--- Dataset Preparation Finished ---")

if __name__ == "__main__":
    main()
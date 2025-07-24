# scripts/cache_metrics.py

import evaluate
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The four metrics our training script needs
METRICS_TO_CACHE = ["accuracy", "f1", "precision", "recall"]

def main():
    """
    Downloads and caches evaluation metric scripts from the Hugging Face Hub.
    """
    logging.info("--- Caching Evaluation Metrics from Hugging Face Hub ---")
    for metric_name in METRICS_TO_CACHE:
        try:
            logging.info(f"Downloading and caching '{metric_name}'...")
            # This line downloads the script and saves it to your local cache
            evaluate.load(metric_name)
            logging.info(f"Successfully cached '{metric_name}'.")
        except Exception as e:
            logging.error(f"Failed to cache '{metric_name}': {e}")
            logging.error("Please ensure you have a stable internet connection and can access huggingface.co.")
    logging.info("--- Caching complete. ---")

if __name__ == "__main__":
    main()
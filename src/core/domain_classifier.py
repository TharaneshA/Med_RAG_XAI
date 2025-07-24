
import json
from pathlib import Path
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
import evaluate

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "classifier_data"
MODEL_OUTPUT_DIR = BASE_DIR / "models" / "domain_classifier"
# --- CORRECTED MODEL CHECKPOINT ---
MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"

# Ensure the output directory exists
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(eval_pred):
    """Computes accuracy, F1, precision, and recall from predictions."""
    # --- MODIFIED SECTION ---
    # Load metrics from the local 'metrics' folder
    metric_acc = evaluate.load("./metrics/accuracy")
    metric_f1 = evaluate.load("./metrics/f1")
    metric_precision = evaluate.load("./metrics/precision")
    metric_recall = evaluate.load("./metrics/recall")
    # --- END MODIFIED SECTION ---
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric_acc.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels)
    precision = metric_precision.compute(predictions=predictions, references=labels)
    recall = metric_recall.compute(predictions=predictions, references=labels)

    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"], "precision": precision["precision"], "recall": recall["recall"]}

def main():
    logging.info("--- Training Domain Classifier ---")

    # Load the label map
    label_map_path = DATA_DIR / "label_map.json"
    if not label_map_path.exists():
        logging.error(f"Label map not found at {label_map_path}. Run the preparation script first.")
        return
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)

    # Load the datasets
    try:
        dataset = load_dataset('json', data_files={
            'train': str(DATA_DIR / "train.json"),
            'test': str(DATA_DIR / "test.json")
        })
    except FileNotFoundError:
        logging.error(f"Training/testing data not found in {DATA_DIR}. Run the preparation script first.")
        return

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Preprocess the dataset
    def preprocess_function(examples):
        # The Trainer expects a 'labels' column, so we rename 'domain'
        examples['labels'] = [label_map[domain] for domain in examples['domain']]
        return tokenizer(examples['text'], truncation=True, max_length=512, padding="max_length")

    logging.info("Tokenizing the dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['text', 'domain'])

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label_map
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save a checkpoint at the end of each epoch
        load_best_model_at_end=True, # Load the best model found during training
        metric_for_best_model="f1",
        greater_is_better=True
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    logging.info(f"Starting model training with {MODEL_CHECKPOINT}...")
    trainer.train()

    # Save the final model and tokenizer
    logging.info(f"Saving the best model to {MODEL_OUTPUT_DIR}")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    logging.info("--- Domain Classifier Training Finished ---")

if __name__ == "__main__":
    main()
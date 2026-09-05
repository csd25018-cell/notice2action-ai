"""
train.py – Baseline Token Classification Training Pipeline
==========================================================
Trains a Hugging Face token classification model (e.g., MuRIL) for NoticeIE-Gold.
Uses BIO tags parsed by convert_labels.py and datasets from data/processed/.

Usage:
    PYTHONPATH=. python -m src.training.train \
        --model_name google/muril-base-cased \
        --data_dir data/processed \
        --output_dir output/baseline_model \
        --epochs 3 \
        --batch_size 8 \
        --learning_rate 2e-5
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.training.dataset import TokenClassificationDataset
from src.training.evaluate import get_compute_metrics_fn
from src.training.model import TrustExtractMultiTaskModel

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_label_map(data_dir: Path) -> dict:
    label_map_path = data_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"Label map not found at {label_map_path}. "
            "Did you run convert_labels.py first?"
        )
    with open(label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train TrustExtract-N Baseline")
    parser.add_argument("--model_name", type=str, default="google/muril-base-cased")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="output/baseline_model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # 1. Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the training configuration
    config_path = Path(args.output_dir) / "training_args.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Saved training configuration to {config_path}")

    # 2. Load Label Map
    data_dir = Path(args.data_dir)
    label_map = load_label_map(data_dir)
    label2id = label_map["label2id"]
    id2label = label_map["id2label"]
    ignore_label_id = label_map.get("ignore_label_id", -100)
    num_labels = len(label2id)
    
    logger.info(f"Loaded {num_labels} labels. Ignored ID: {ignore_label_id}")

    # 3. Load Datasets
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"

    logger.info("Loading datasets...")
    train_dataset = TokenClassificationDataset(train_path)
    val_dataset = TokenClassificationDataset(val_path)
    
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")

    # 4. Load Model and Tokenizer
    logger.info(f"Loading model '{args.model_name}'...")
    
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Define document types
    doc_types = ["Circular", "Notification", "Ordinance", "Draft Rule"]
    num_doc_labels = len(doc_types)
    
    model = TrustExtractMultiTaskModel(
        config=config,
        num_token_labels=num_labels,
        num_doc_labels=num_doc_labels
    )

    # 5. Data Collator
    # This automatically pads input_ids, attention_mask, and labels to the batch max length
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        label_pad_token_id=ignore_label_id
    )

    # 6. Training Arguments
    # We want to select the best model based on F1
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        seed=args.seed,
    )

    # 7. Initialize Trainer
    compute_metrics_fn = get_compute_metrics_fn(id2label, ignore_label_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenClassificationDataset(
            "data/processed/train.jsonl"
        ),
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    # 8. Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # 9. Save best model & metrics
    logger.info("Training complete. Saving best model...")
    trainer.save_model(args.output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 10. Evaluate on Test set (Optional, to see final performance)
    if test_path.exists():
        logger.info("Evaluating on test dataset...")
        test_dataset = TokenClassificationDataset(test_path)
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    logger.info(f"Done! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

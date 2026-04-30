"""
Fine-tune Gemma 3 270M with LoRA for code comment generation.
Memory-efficient training on M1 MacBook (8GB RAM, CPU-only).

Tokenisation strategy: pack prompt + response into one sequence;
mask prompt tokens with -100 so loss is computed only on the comment.
"""

import torch
import pandas as pd
import json
import logging
from pathlib import Path
import time

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODEL_NAME = "google/gemma-3-270m-it"
TRAIN_FILE = ROOT / "data" / "processed" / "train_split.parquet"
EVAL_FILE = ROOT / "data" / "processed" / "eval_split.parquet"
CHECKPOINT_DIR = ROOT / "models" / "checkpoint_latest"
ADAPTER_DIR = ROOT / "models" / "lora_adapter"

# Keep training set small enough for M1 RAM
TRAIN_SAMPLE = 500
EVAL_SAMPLE = 100
MAX_LENGTH = 384   # prompt + response combined

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4   # effective batch = 8
LR = 5e-4


def build_prompt(code: str) -> str:
    return f"Generate a docstring for the following code. Reply with only the docstring text.\n\n{code}\n\nDocstring:"


def tokenize_dataset(df: pd.DataFrame, tokenizer, desc: str) -> Dataset:
    """Return a dataset with input_ids, attention_mask, labels (prompt masked)."""
    records = {"input_ids": [], "attention_mask": [], "labels": []}

    for _, row in df.iterrows():
        prompt = build_prompt(row["code"])
        response = row["comment"]

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        eos = [tokenizer.eos_token_id]

        ids = (prompt_ids + response_ids + eos)[:MAX_LENGTH]
        labels = ([-100] * len(prompt_ids) + response_ids + eos)[:MAX_LENGTH]
        pad_len = MAX_LENGTH - len(ids)

        records["input_ids"].append(ids + [tokenizer.pad_token_id] * pad_len)
        records["attention_mask"].append([1] * len(ids) + [0] * pad_len)
        records["labels"].append(labels + [-100] * pad_len)

    dataset = Dataset.from_dict(records)
    dataset.set_format("torch")
    logger.info(f"{desc}: {len(dataset)} examples")
    return dataset


def run():
    logger.info("=" * 60)
    logger.info("FINE-TUNING: Gemma 3 270M + LoRA")
    logger.info("=" * 60)

    # --- Tokenizer ---
    logger.info(f"Loading tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Data ---
    train_df = pd.read_parquet(TRAIN_FILE).sample(n=TRAIN_SAMPLE, random_state=42)
    eval_df = pd.read_parquet(EVAL_FILE).sample(n=EVAL_SAMPLE, random_state=42)
    train_dataset = tokenize_dataset(train_df, tokenizer, "Train")
    eval_dataset = tokenize_dataset(eval_df, tokenizer, "Eval")

    # --- Model ---
    logger.info(f"Loading base model {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # --- LoRA ---
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training args ---
    # adamw_torch works on M1 CPU; adamw_8bit requires bitsandbytes (CUDA)
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting training …")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Time:       {elapsed / 60:.1f} min")
    logger.info(f"Final loss: {result.training_loss:.4f}")
    logger.info("=" * 60)

    # Save LoRA adapter
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    logger.info(f"✓ LoRA adapter saved → {ADAPTER_DIR}")

    # Save training summary
    summary = {
        "model": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "train_samples": TRAIN_SAMPLE,
        "eval_samples": EVAL_SAMPLE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": LR,
        "training_time_min": round(elapsed / 60, 1),
        "final_loss": result.training_loss,
    }
    out = ROOT / "benchmarks" / "training_summary.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    return model, tokenizer


if __name__ == "__main__":
    run()
    print("\n✓ Fine-tuning complete — adapter saved to models/lora_adapter/")

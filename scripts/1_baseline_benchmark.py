"""
Baseline Benchmarking: Evaluate untuned Gemma 3 270M on code comment generation.
Samples 50 examples from the eval split for speed on M1 CPU.
"""

import torch
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODEL_NAME = "google/gemma-3-270m-it"
EVAL_FILE = ROOT / "data" / "processed" / "eval_split.parquet"
BENCHMARK_DIR = ROOT / "benchmarks"
BENCHMARK_DIR.mkdir(exist_ok=True)

SAMPLE_SIZE = 50


def load_model(model_name: str):
    logger.info(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info("✓ Model loaded on CPU")
    return model, tokenizer


def generate_comment(model, tokenizer, code: str, max_new_tokens: int = 80) -> str:
    prompt = f"Generate a docstring for the following code. Reply with only the docstring text.\n\n{code}\n\nDocstring:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_metrics(reference: str, hypothesis: str) -> Dict:
    ref_tok = word_tokenize(reference.lower())
    hyp_tok = word_tokenize(hypothesis.lower()) if hypothesis else [""]
    bleu = sentence_bleu(
        [ref_tok], hyp_tok,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(reference, hypothesis or "")
    return {
        "bleu_4": bleu,
        "rouge_l": rouge["rougeL"].fmeasure,
        "rouge_l_precision": rouge["rougeL"].precision,
        "rouge_l_recall": rouge["rougeL"].recall,
    }


def run():
    logger.info("=" * 60)
    logger.info("BASELINE BENCHMARK")
    logger.info("=" * 60)

    df = pd.read_parquet(EVAL_FILE)
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
    logger.info(f"Evaluating on {len(df)} examples")

    model, tokenizer = load_model(MODEL_NAME)

    predictions: List[str] = []
    references: List[str] = []
    metrics_list: List[Dict] = []
    gen_times: List[float] = []

    for i, row in df.iterrows():
        t0 = time.time()
        pred = generate_comment(model, tokenizer, row["code"])
        elapsed = time.time() - t0

        ref = row["comment"]
        m = compute_metrics(ref, pred)

        predictions.append(pred)
        references.append(ref)
        metrics_list.append(m)
        gen_times.append(elapsed)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  [{i+1}/{len(df)}]  BLEU-4: {m['bleu_4']:.4f}  ROUGE-L: {m['rouge_l']:.4f}  ({elapsed:.1f}s)"
            )

    avg_bleu = sum(m["bleu_4"] for m in metrics_list) / len(metrics_list)
    avg_rouge = sum(m["rouge_l"] for m in metrics_list) / len(metrics_list)
    avg_time = sum(gen_times) / len(gen_times)

    summary = {
        "model": MODEL_NAME,
        "num_examples": len(df),
        "avg_bleu_4": avg_bleu,
        "avg_rouge_l": avg_rouge,
        "avg_generation_time_sec": avg_time,
        "total_time_sec": sum(gen_times),
        "examples": [
            {
                "code": df.iloc[i]["code"][:200],
                "reference": references[i],
                "prediction": predictions[i],
                "bleu_4": metrics_list[i]["bleu_4"],
                "rouge_l": metrics_list[i]["rouge_l"],
            }
            for i in range(min(5, len(df)))
        ],
    }

    logger.info("\n" + "=" * 60)
    logger.info("BASELINE RESULTS")
    logger.info("=" * 60)
    logger.info(f"BLEU-4:      {avg_bleu:.4f}")
    logger.info(f"ROUGE-L:     {avg_rouge:.4f}")
    logger.info(f"Avg gen time:{avg_time:.2f}s")
    logger.info("=" * 60)

    out = BENCHMARK_DIR / "baseline_metrics.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved → {out}")
    return summary


if __name__ == "__main__":
    run()
    print("\n✓ Baseline benchmark complete — see benchmarks/baseline_metrics.json")

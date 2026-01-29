"""
Inspect tokenization for parquet rows using the training tokenizer.

Examples:
  python test/text_tokenizer_parquet.py \
    --parquet_path output/split_data_pretrain_rec/part-00000-of-00006.parquet \
    --dataset_config test/pretrain/pretrain.json \
    --max_rows 3
"""

import argparse
import json
import os
from typing import List

import pyarrow.parquet as pq
from transformers import AutoTokenizer


DEFAULT_PARQUET_PATH = "output/split_data_pretrain_rec/part-00000-of-00006.parquet"
DEFAULT_MAX_ROWS = 3
DEFAULT_DATASET_CONFIG = "test/pretrain/pretrain.json"
DEFAULT_MODEL_DIR = None


def load_model_dir(model_dir: str | None, dataset_config: str | None) -> str:
    if model_dir:
        return model_dir
    if dataset_config and os.path.isfile(dataset_config):
        with open(dataset_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_model_dir = cfg.get("base_model_dir")
        if base_model_dir:
            return base_model_dir
    raise ValueError("model_dir is required (or provide dataset_config with base_model_dir)")


def iter_texts_from_segments(segments_json: str) -> str:
    try:
        segments = json.loads(segments_json)
    except Exception:
        return ""
    texts = []
    for segment in segments:
        if segment.get("type") == "text":
            texts.append(segment.get("text", ""))
    return "".join(texts)


def extract_text(row: dict) -> str:
    if "segments" in row:
        return iter_texts_from_segments(row["segments"])
    if "text" in row:
        return row["text"]
    if "message" in row or "messages" in row:
        messages = row.get("message") or row.get("messages")
        return json.dumps(messages, ensure_ascii=False)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect tokenization from parquet rows.")
    parser.add_argument("--parquet_path", type=str, default=DEFAULT_PARQUET_PATH, help="Path to parquet file")
    parser.add_argument("--max_rows", type=int, default=DEFAULT_MAX_ROWS, help="Number of rows to inspect")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Model dir for tokenizer")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=DEFAULT_DATASET_CONFIG,
        help="Dataset config for base_model_dir fallback",
    )
    args = parser.parse_args()

    model_dir = load_model_dir(args.model_dir, args.dataset_config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    table = pq.read_table(args.parquet_path)
    rows = table.to_pylist()

    for i, row in enumerate(rows[: args.max_rows]):
        text = extract_text(row)
        if not text:
            print(f"[row {i}] empty text")
            continue
        encoded = tokenizer(text, add_special_tokens=False)
        ids: List[int] = encoded["input_ids"]
        tokens: List[str] = tokenizer.convert_ids_to_tokens(ids)
        reconstructed = tokenizer.convert_tokens_to_string(tokens)
        decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=False)

        print(f"[row {i}] text:\n{text}\n")
        print(f"[row {i}] reconstructed:\n{reconstructed}\n")
        print(f"[row {i}] decoded:\n{decoded}\n")
        print(f"[row {i}] tokens (id -> token):")
        for idx, (tid, tok) in enumerate(zip(ids, tokens)):
            print(f"{idx:04d}: {tid}\t{tok}")
        print("=" * 80)


if __name__ == "__main__":
    main()

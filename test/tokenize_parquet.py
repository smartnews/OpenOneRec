"""
python test/tokenize_parquet.py --max_rows 100 --output_path output/id_sid_100.parquet

python test/tokenize_parquet.py --output_path output/id_sid.parquet

INPUT : id, title, description
OUTPUT: id, title, description, sid
"""

import argparse
import os

import torch
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.fs as fs
from transformers import AutoModel, AutoTokenizer
from transformers.utils.hub import cached_file

from test.utils import mean_pool
from tokenizer.res_kmeans import ResKmeans


MODEL_ID = "Qwen/Qwen3-Embedding-8B"
DEFAULT_INPUT = "s3://smartnews-dmp/warehouse/development/default/z_yangwang.db/article_id_title_desc_parquet/dt=2026-01-26/hh=00/"
DEFAULT_OUTPUT = "output/id_sid.parquet"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_ROWS = 0
DEFAULT_MAX_LENGTH = 512
DESCRIPTION_TEMPLATE = "新闻标题：{title}。摘要：{description}"

def format_sid(codes: list[int]) -> str:
    tags = [f"<s_{chr(ord('a') + i)}_{c}>" for i, c in enumerate(codes)]
    return f"<|sid_begin|>{''.join(tags)}<|sid_end|>"


def iter_batches(path: str, columns: list[str], batch_size: int):
    if path.startswith("s3://"):
        path = path[len("s3://") :]
        filesystem = fs.S3FileSystem()
    else:
        filesystem = None
    dataset = ds.dataset(path, format="parquet", filesystem=filesystem, partitioning="hive")
    yield from dataset.to_batches(columns=columns, batch_size=batch_size)


def load_models() -> tuple[AutoTokenizer, AutoModel, ResKmeans, torch.device, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    model_device = next(model.parameters()).device

    ckpt_path = cached_file("OpenOneRec/OneRec-tokenizer", "model.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, ResKmeans):
        sid_model = checkpoint
    else:
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
        else:
            state_dict = checkpoint
        n_layers = sum(1 for k in state_dict.keys() if k.startswith("centroids."))
        codebook_size, dim = state_dict["centroids.0"].shape
        sid_model = ResKmeans(n_layers=n_layers, codebook_size=codebook_size, dim=dim)
        sid_model.load_state_dict(state_dict)

    sid_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sid_model = sid_model.to(sid_device).eval()
    return tokenizer, model, sid_model, model_device, sid_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SID from parquet on S3.")
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT, help="S3 or local parquet path")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT, help="Local output parquet path")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for tokenization")
    parser.add_argument("--max_rows", type=int, default=DEFAULT_MAX_ROWS, help="Stop after writing N rows (0 = no limit)")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH, help="Max token length for tokenizer")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    tokenizer, model, sid_model, model_device, sid_device = load_models()

    writer = None
    total = 0
    for batch in iter_batches(args.input_path, ["id", "title", "description"], args.batch_size):
        ids = batch.column("id").to_pylist()
        titles = ["" if title is None else str(title) for title in batch.column("title").to_pylist()]
        descriptions = ["" if desc is None else str(desc) for desc in batch.column("description").to_pylist()]
        texts = [
            DESCRIPTION_TEMPLATE.format(title=title, description=desc)
            for title, desc in zip(titles, descriptions)
        ]

        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(model_device)
        emb = mean_pool(model(**model_inputs).last_hidden_state, model_inputs["attention_mask"])
        emb = emb.to(dtype=torch.float32, device=sid_device)

        with torch.no_grad():
            codes = sid_model.encode(emb).cpu().tolist()

        sids = [format_sid(code) for code in codes]
        table = pa.table(
            {
                "id": ids,
                "title": titles,
                "description": descriptions,
                "sid": sids,
            }
        )
        if writer is None:
            writer = pq.ParquetWriter(args.output_path, table.schema)
        writer.write_table(table)
        total += len(ids)
        if args.max_rows > 0 and total >= args.max_rows:
            break

    if writer is not None:
        writer.close()
    print(f"Wrote {total} rows to {args.output_path}")


if __name__ == "__main__":
    main()

"""
python test/test_explain_parquet.py --input output/id_sid_100.parquet --limit 5

"""
import argparse
from threading import Thread

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from test.utils import print_param_stats, print_cuda_memory, print_model_precision

MODEL_NAME = "OpenOneRec/OneRec-8B"
DEFAULT_TEMPLATE = "这是日文版SmartNews上的一则新闻：{}，帮我总结一下这个新闻讲述了什么内容"


def generate_async(model, tokenizer, model_inputs):
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_kwargs = dict(
        **model_inputs,
        max_new_tokens=4096,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for next_token in streamer:
        if next_token == "<think>":
            print("=== Thinking ===")
        print(next_token, end="", flush=True)
        if next_token == "</think>":
            print("\n=== Content ===")
    print()
    thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain SIDs from parquet.")
    parser.add_argument("--input", type=str, required=True, help="Input parquet path")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    print_param_stats(model)
    print_cuda_memory()
    print_model_precision(model)

    df = pd.read_parquet(args.input)
    total = 0
    for _, row in df.iterrows():
        if total > 0:
            print()
        sid = row.get("sid")
        if sid is None or (isinstance(sid, float) and pd.isna(sid)):
            continue

        title = row.get("title")
        description = row.get("description")

        prompt = DEFAULT_TEMPLATE.format(sid)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        print("title:", "" if pd.isna(title) else title)
        print("description:", "" if pd.isna(description) else description)
        print("prediction:")
        generate_async(model, tokenizer, model_inputs)

        total += 1
        if args.limit > 0 and total >= args.limit:
            break


if __name__ == "__main__":
    main()

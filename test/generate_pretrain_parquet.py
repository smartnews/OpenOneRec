"""
python test/generate_pretrain_parquet.py --input output/id_sid.parquet --output output/pretrain.parquet
"""
import argparse
import json
import random
import uuid

import pandas as pd

DESCRIPTION_TEMPLATE = "タイトル：{title}。要約：{description}"
PRETRAIN_TEMPLATES = [
    lambda sid, caption: json.dumps({"ニュースID": sid, "ニュース内容": caption}, ensure_ascii=False),
    lambda sid, caption: f"ニュース{sid}には次の内容が含まれます：{caption}",
    lambda sid, caption: f"ニュース{sid}の内容は以下のとおりです：{caption}",
]
SOURCE_NAME = "RecIF_ItemUnderstand_Pretrain"
SEED = 42


def build_segments(sid: str, caption: str) -> str:
    template = random.choice(PRETRAIN_TEMPLATES)
    text = template(sid, caption)
    return json.dumps([{"type": "text", "text": text}], ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pretrain item_understand parquet.")
    parser.add_argument("--input", type=str, required=True, help="Input parquet path")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    df = pd.read_parquet(args.input)

    records = []
    for _, row in df.iterrows():
        sid = row.get("sid")
        if sid is None or (isinstance(sid, float) and pd.isna(sid)):
            continue
        sid = str(sid)

        title = row.get("title")
        description = row.get("description")
        title = "" if title is None or (isinstance(title, float) and pd.isna(title)) else str(title)
        description = "" if description is None or (isinstance(description, float) and pd.isna(description)) else str(description)
        if not title and not description:
            continue
        caption = DESCRIPTION_TEMPLATE.format(title=title, description=description)

        records.append(
            {
                "source": SOURCE_NAME,
                "uuid": str(uuid.uuid4()),
                "segments": build_segments(sid, caption),
                "metadata": json.dumps({"sid": sid}, ensure_ascii=False),
            }
        )

    pd.DataFrame(records).to_parquet(args.output, index=False)
    print(f"Wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()

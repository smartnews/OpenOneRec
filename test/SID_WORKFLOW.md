# OneRec：为新数据集生成 3-layer Semantic IDs（SIDs）

你需要做的事只有三步：**算 embedding → 下载 tokenizer 权重 → 量化成 3 层 SID**。

---

## 0. 准备 item 文本（`pid` 对齐）

对每个 item（唯一 id 记作 `pid`）准备一段用于 embedding 的文本（你自己拼接即可），例如：

```python
text = f"Title: {title}\nDescription: {desc}\nCategory: {cat}"
```

要求：
- `pid` 必须和你后续交互数据（history/target）里使用的 item id 完全一致
- 文本拼接方式尽量稳定、结构化（避免噪声字段频繁变化）

---

## 1. 生成 item embeddings（必须用 Qwen3-8B-Embedding）

Embedding 模型必须是：
- `Qwen/Qwen3-Embedding-8B`：https://huggingface.co/Qwen/Qwen3-Embedding-8B

产物是一个 parquet（示例：`data/embeddings.parquet`），至少包含两列：
- `pid`：建议 int（仓库里部分脚本会 `int(pid)`）
- `embedding`：每行一个 **4096 维** float 向量（list/array）

依赖（示例）：
```bash
pip install torch transformers numpy pandas pyarrow
```

> embedding 的抽取细节（pooling/normalize、是否需要 query/passage 前缀等）请以模型卡为准；不一致会影响 SID 分布。

---

## 2. 下载预训练 tokenizer 权重（OneRec-tokenizer）

预训练 tokenizer 权重在：
- https://huggingface.co/OpenOneRec/OneRec-tokenizer

你需要拿到一个 `.pt` checkpoint（示例：`checkpoints/model.pt`；文件名以实际为准）。

---

## 3. 推理量化：embedding → 3-layer SID

仓库脚本在 `tokenizer/`：
- `tokenizer/infer_res_kmeans.py`

安装依赖（按 `tokenizer/README.md`）：
```bash
pip install torch numpy pandas pyarrow faiss tqdm
```

运行推理：
```bash
python tokenizer/infer_res_kmeans.py \
  --model_path checkpoints/model.pt \
  --emb_path data/embeddings.parquet \
  --output_path output/codes.parquet
```

### 输入 / 输出格式

输入 `data/embeddings.parquet`：
- `pid`
- `embedding`

输出 `output/codes.parquet`：
- `pid`
- `codes`：每行一个长度为 3 的整数列表 `[c0, c1, c2]`（每层一个 `0..8191` 的 code）

### 3.1 生成下游需要的 `pid2sid.parquet`（把 `codes` 改名为 `sid`）

仓库下游脚本读取的是 `pid2sid` parquet，并且列名必须是 `sid`（不是 `codes`），例如：
- `data/onerec_data/pretrain/video_rec.py`
- `data/onerec_data/pretrain/item_understand.py`

把 `codes` 重命名为 `sid`：
```bash
python -c "import pandas as pd; df=pd.read_parquet('output/codes.parquet'); df=df.rename(columns={'codes':'sid'}); df.to_parquet('output/pid2sid.parquet', index=False)"
```

最终得到：`output/pid2sid.parquet`（列：`pid`, `sid`，其中 `sid=[c0,c1,c2]`）。

---

## 4. 在下游数据构造中使用 SID

你不需要自己把 `[c0,c1,c2]` 拼成 token 串；仓库脚本会用类似下面的格式自动渲染：

`<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>`

示例（视频推荐预训练数据）：
```bash
python data/onerec_data/pretrain/video_rec.py \
  --input your_interactions.parquet \
  --pid2sid output/pid2sid.parquet \
  --output_dir out/video_rec
```


import torch  # 第三方：PyTorch（张量/模型推理）
from transformers import AutoModel, AutoTokenizer  # 第三方：Hugging Face Transformers（加载 tokenizer/模型）
from transformers.utils.hub import cached_file
from test.utils import l2_normalize, mean_pool, last_hidden_pool
from tokenizer.res_kmeans import ResKmeans

MODEL_ID = "Qwen/Qwen3-Embedding-8B"  # 必须用这个 embedding 模型来对齐 OneRec
BATCH_SIZE = 1  # 小一点更省显存；大一点更快


def format_sid(codes: list[int]) -> str:
    tags = [f"<s_{chr(ord('a') + i)}_{c}>" for i, c in enumerate(codes)]
    return f"<|sid_begin|>{''.join(tags)}<|sid_end|>"


def main() -> None:
    texts = [
        #"15元毛巾评测",
        "快手爆款：10分钟学会糖醋里脊做法",
        #"农村老家的晚霞太美了，带你看看今天的天空", # <|sid_begin|><s_a_1371><s_b_4120><s_c_7314><|sid_end|>
        #"街头魔术挑战：30秒让路人猜不透", # <|sid_begin|><s_a_802><s_b_7825><s_c_3919><|sid_end|>
        #"健身小白一周打卡记录，看看变化有多大",
        #"一起听老歌：90年代经典情歌合集",
        #"政府は春の補正予算案を発表し、子育て支援と地方の交通インフラに重点配分するとした。",
        #"都内で新型の電気バスが運行開始。静音性と充電効率の向上が評価されている。",
        #"半導体大手が来年度の設備投資を増額へ。AI需要の拡大を背景に生産能力を強化する。",
        #"気象庁は週末にかけて広い範囲で大雨の可能性があるとして早めの備えを呼びかけた。",
        #"人気アイドルグループが新曲を公開し、音楽配信チャートで初登場1位を獲得した。",
        #"映画祭で話題作が最優秀作品賞に輝き、主演俳優が壇上で感謝の言葉を述べた。",
        #"有名アニメの続編が制作決定。ティザー映像が公開されSNSで大きな反響を呼んだ。",
    ]

    # load Qwen embedding model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto", # "auto" is "BF16" which is not supported on this GPU
        device_map="auto",
    )
    model.eval()
    model_device = next(model.parameters()).device

    # load SID ResKmeans model
    ckpt_path = cached_file("OpenOneRec/OneRec-tokenizer", "model.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, ResKmeans):
        sid_model = checkpoint
    else:
        state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
        if state_dict is None:
            state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint
        n_layers = sum(1 for k in state_dict.keys() if k.startswith("centroids."))
        codebook_size, dim = state_dict["centroids.0"].shape
        sid_model = ResKmeans(n_layers=n_layers, codebook_size=codebook_size, dim=dim)
        sid_model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sid_model = sid_model.to(device).eval()

    for text in texts:
        model_inputs = tokenizer([text], return_tensors="pt").to(model_device)
        model_output = model(**model_inputs)
        emb = mean_pool, last_hidden_pool(model_output.last_hidden_state, model_inputs["attention_mask"])
        emb = l2_normalize(emb)

        emb = torch.tensor(emb, dtype=torch.float32, device=device)
        with torch.no_grad():
            codes = sid_model.encode(emb).cpu().tolist()
        for idx, code in enumerate(codes, start=1):
            print(format_sid(code))


if __name__ == "__main__":
    main()



"""
<|sid_begin|><s_a_8038><s_b_4887><s_c_857><|sid_end|>
<|sid_begin|><s_a_2610><s_b_1283><s_c_2477><|sid_end|>
<|sid_begin|><s_a_2779><s_b_5195><s_c_6071><|sid_end|>
<|sid_begin|><s_a_4885><s_b_4716><s_c_6071><|sid_end|>
<|sid_begin|><s_a_1353><s_b_3042><s_c_6071><|sid_end|>
<|sid_begin|><s_a_2304><s_b_7438><s_c_2477><|sid_end|>
"""

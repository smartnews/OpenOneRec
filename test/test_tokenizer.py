
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils.hub import cached_file

from test.utils import l2_normalize, mean_pool
from tokenizer.res_kmeans import ResKmeans

from test.data_titles import titles
MODEL_ID = "Qwen/Qwen3-Embedding-8B"  
BATCH_SIZE = 1


def format_sid(codes: list[int]) -> str:
    tags = [f"<s_{chr(ord('a') + i)}_{c}>" for i, c in enumerate(codes)]
    return f"<|sid_begin|>{''.join(tags)}<|sid_end|>"


def main() -> None:
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

    for title in titles:
        model_inputs = tokenizer([title], return_tensors="pt").to(model_device)
        model_output = model(**model_inputs)
        emb = mean_pool(model_output.last_hidden_state, model_inputs["attention_mask"])
        # emb = l2_normalize(emb)

        emb = torch.tensor(emb, dtype=torch.float32, device=device)
        with torch.no_grad():
            codes = sid_model.encode(emb).cpu().tolist()
        for idx, code in enumerate(codes, start=1):
            print(format_sid(code))


if __name__ == "__main__":
    main()

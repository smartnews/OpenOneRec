import torch
import json
from pathlib import Path


def _print_header(title):
    print(f"=== {title} ===")


def format_count(n):
    return f"{n:,}"


def format_bytes(n):
    return f"{n / (1024 ** 2):.2f} MB"


def print_param_stats(m, show_device_map=True):
    total = 0
    device_counts = {}
    for _, p in m.named_parameters():
        num = p.numel()
        total += num
        dev = str(p.device)
        device_counts[dev] = device_counts.get(dev, 0) + num
    _print_header("Param Stats")
    print(f"total params: {format_count(total)}")
    for dev, num in sorted(device_counts.items(), key=lambda x: x[0]):
        ratio = (num / total) * 100 if total else 0.0
        print(f"{dev}: {format_count(num)} ({ratio:.2f}%)")
    if show_device_map and hasattr(m, "hf_device_map"):
        _print_header("hf_device_map (module -> device)")
        for k, v in m.hf_device_map.items():
            print(f"{k}: {v}")


def print_cuda_memory():
    _print_header("CUDA Memory")
    if not torch.cuda.is_available():
        print("cuda not available")
        return
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    print(f"device: cuda:{device}")
    print(f"allocated: {format_bytes(allocated)}")
    print(f"reserved: {format_bytes(reserved)}")
    print(f"used: {format_bytes(used)}")
    print(f"total: {format_bytes(total)}")


def print_model_precision(m):
    _print_header("Model Precision")
    print(f"model dtype: {getattr(m, 'dtype', None)}")
    cfg_dtype = getattr(getattr(m, "config", None), "torch_dtype", None)
    print(f"config torch_dtype: {cfg_dtype}")
    qcfg = getattr(getattr(m, "config", None), "quantization_config", None)
    print(f"quantization_config: {qcfg}")


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def last_hidden_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_lens = attention_mask.sum(dim=1)  # [batch_size]
    batch_size = last_hidden_state.size(0)
    return last_hidden_state[torch.arange(batch_size), seq_lens - 1]

def load_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    pids: list[int] = []
    texts: list[str] = []
    for row in rows:
        pids.append(int(row["pid"]))
        texts.append(str(row["text"]))

    return pids, texts
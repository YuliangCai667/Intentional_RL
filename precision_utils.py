import torch


DTYPE_CHOICES = ("fp16", "bf16", "fp32", "fp64")
DEVICE_CHOICES = ("auto", "cpu", "cuda")


def resolve_dtype(dtype_name):
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    try:
        return dtype_map[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Choose from {DTYPE_CHOICES}.") from exc


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    raise ValueError(f"Unsupported device '{device_name}'. Choose from {DEVICE_CHOICES}.")


def default_dtype_tag(dtype_name, dtype_tag):
    return dtype_name if dtype_tag is None else dtype_tag

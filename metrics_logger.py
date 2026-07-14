import json
import math
import os
import re


def sanitize_filename_part(value):
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def is_nonfinite_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    return False


def has_nonfinite_value(metrics):
    for value in metrics.values():
        if is_nonfinite_number(value):
            return True
    return bool(metrics.get("has_nan") or metrics.get("has_inf"))


class MetricsLogger:
    def __init__(self, log_dir, run_name):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        filename = sanitize_filename_part(run_name) + ".jsonl"
        self.path = os.path.join(self.log_dir, filename)
        self.file = open(self.path, "a", encoding="utf-8")

    def log(self, metrics):
        serializable = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except ValueError:
                    value = value.tolist()
            elif hasattr(value, "tolist"):
                value = value.tolist()
            serializable[key] = value
        self.file.write(json.dumps(serializable, sort_keys=True) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

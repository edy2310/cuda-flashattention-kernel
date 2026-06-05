import importlib
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def load_extension():
    # The extension is built once via setup.py and imported here.
    try:
        return importlib.import_module("flash_attn_ext")
    except ImportError as exc:
        raise RuntimeError(
            "flash_attn_ext is not built. Run: python setup.py build_ext --inplace"
        ) from exc


def time_fn(fn, iters):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def run_benchmark(B=1, H=8, N=512, D=64, iters=100, warmup=10, check=False, seed=1234):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    torch.manual_seed(seed)
    ext = load_extension()

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup runs to stabilize performance.
    for _ in range(warmup):
        _ = ext.flash_attention_naive(q, k, v)
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    max_diff = None
    if check:
        out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out_custom = ext.flash_attention_naive(q, k, v)
        max_diff = (out_ref - out_custom).abs().max().item()

    t_custom = time_fn(lambda: ext.flash_attention_naive(q, k, v), iters)
    t_torch = time_fn(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False), iters)

    return {
        "B": B,
        "H": H,
        "N": N,
        "D": D,
        "iters": iters,
        "warmup": warmup,
        "custom_ms": t_custom,
        "torch_ms": t_torch,
        "speedup": t_custom / t_torch,
        "max_abs_diff": max_diff,
    }


def main():
    result = run_benchmark()
    print(result)


if __name__ == "__main__":
    main()

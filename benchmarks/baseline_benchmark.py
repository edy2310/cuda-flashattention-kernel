import importlib
import os
import sys

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None

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


def _reference_sdpa_context():
    if sdpa_kernel is not None and SDPBackend is not None:
        return sdpa_kernel(backends=[SDPBackend.MATH])
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_mem_efficient=False,
        enable_math=True,
    )


def reference_scaled_dot_product_attention(q, k, v):
    # Force the math backend so the reference path stays portable on Colab GPUs
    # where fused SDPA can reject valid inputs with CUDA error 10.
    with _reference_sdpa_context():
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)


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
        _ = reference_scaled_dot_product_attention(q, k, v)

    max_diff = None
    if check:
        out_ref = reference_scaled_dot_product_attention(q, k, v)
        out_custom = ext.flash_attention_naive(q, k, v)
        max_diff = (out_ref - out_custom).abs().max().item()

    t_custom = time_fn(lambda: ext.flash_attention_naive(q, k, v), iters)
    t_torch = time_fn(lambda: reference_scaled_dot_product_attention(q, k, v), iters)

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

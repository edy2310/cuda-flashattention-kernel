import os
import sys

import torch

# Allow importing benchmark modules from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from benchmarks.baseline_benchmark import load_extension, reference_scaled_dot_product_attention


def expect_raises(fn, expected_message):
    # Simple helper to validate that a call fails with a specific message.
    try:
        fn()
    except RuntimeError as exc:
        assert expected_message in str(exc), f"Expected '{expected_message}', got: {exc}"
        return
    raise AssertionError(f"Expected RuntimeError containing '{expected_message}'")


def run_happy_path_tests(ext):
    # Multiple shapes to validate correctness across inputs.
    configs = [
        {"B": 1, "H": 2, "N": 128, "D": 32},
        {"B": 1, "H": 4, "N": 256, "D": 64},
        {"B": 2, "H": 2, "N": 64, "D": 128},
    ]

    for cfg in configs:
        torch.manual_seed(1234)
        q = torch.randn(cfg["B"], cfg["H"], cfg["N"], cfg["D"], device="cuda", dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out_custom = ext.flash_attention_naive(q, k, v)
        torch.cuda.synchronize()
        out_ref = reference_scaled_dot_product_attention(q, k, v)

        # Use a relaxed tolerance to match float32 GPU behavior.
        torch.testing.assert_close(out_custom, out_ref, rtol=1e-3, atol=1e-3)


def run_bad_path_tests(ext):
    # Wrong device (CPU).
    q = torch.randn(1, 1, 16, 16, device="cpu", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expect_raises(lambda: ext.flash_attention_naive(q, k, v), "Q must be a CUDA tensor")

    # Wrong dtype (float16).
    q = torch.randn(1, 1, 16, 16, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expect_raises(lambda: ext.flash_attention_naive(q, k, v), "Q must be float32")

    # Non-contiguous input.
    q = torch.randn(1, 1, 16, 16, device="cuda", dtype=torch.float32).transpose(-1, -2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expect_raises(lambda: ext.flash_attention_naive(q, k, v), "Q must be contiguous")

    # Shape mismatch between Q and K.
    q = torch.randn(1, 1, 16, 16, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 1, 17, 16, device="cuda", dtype=torch.float32)
    v = torch.randn_like(q)
    expect_raises(lambda: ext.flash_attention_naive(q, k, v), "K must match Q shape")

    # Wrong number of dimensions.
    q = torch.randn(1, 16, 16, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expect_raises(lambda: ext.flash_attention_naive(q, k, v), "Q must have shape [B, H, N, D]")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run these tests.")

    ext = load_extension()

    run_happy_path_tests(ext)
    run_bad_path_tests(ext)

    print("All FlashAttention tests passed.")


if __name__ == "__main__":
    main()

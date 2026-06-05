import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

# Allow importing benchmark modules from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")
sys.path.insert(0, REPO_ROOT)

from benchmarks.baseline_benchmark import run_benchmark


# Edit these to match the shapes you want to explore.
B = 1
H = 8

N_SWEEP = [128, 256, 512, 1024, 2048]
D_FOR_N_SWEEP = 64

D_SWEEP = [32, 64, 96, 128]
N_FOR_D_SWEEP = 1024

ITERS = 50
WARMUP = 10
CHECK = False


def estimate_tflops(B, H, N, D, time_ms):
    # Rough FLOPs estimate for attention:
    # QK^T ~ 2 * B*H*N*N*D (mul+add)
    # (softmax*V) ~ 2 * B*H*N*N*D
    flops = 4.0 * B * H * N * N * D
    seconds = time_ms / 1e3
    return (flops / seconds) / 1e12


def collect_results():
    rows = []

    # Sweep sequence length N while holding D fixed.
    for N in N_SWEEP:
        result = run_benchmark(
            B=B,
            H=H,
            N=N,
            D=D_FOR_N_SWEEP,
            iters=ITERS,
            warmup=WARMUP,
            check=CHECK,
        )
        rows.append(
            {
                "sweep": "N",
                **result,
                "custom_tflops": estimate_tflops(B, H, N, D_FOR_N_SWEEP, result["custom_ms"]),
                "torch_tflops": estimate_tflops(B, H, N, D_FOR_N_SWEEP, result["torch_ms"]),
            }
        )

    # Sweep head dimension D while holding N fixed.
    for D in D_SWEEP:
        result = run_benchmark(
            B=B,
            H=H,
            N=N_FOR_D_SWEEP,
            D=D,
            iters=ITERS,
            warmup=WARMUP,
            check=CHECK,
        )
        rows.append(
            {
                "sweep": "D",
                **result,
                "custom_tflops": estimate_tflops(B, H, N_FOR_D_SWEEP, D, result["custom_ms"]),
                "torch_tflops": estimate_tflops(B, H, N_FOR_D_SWEEP, D, result["torch_ms"]),
            }
        )

    return rows


def plot_results(rows, output_path):
    n_rows = sorted([r for r in rows if r["sweep"] == "N"], key=lambda r: r["N"])
    d_rows = sorted([r for r in rows if r["sweep"] == "D"], key=lambda r: r["D"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # N sweep plot.
    axes[0].plot(
        [r["N"] for r in n_rows],
        [r["custom_ms"] for r in n_rows],
        marker="o",
        label="Custom kernel",
    )
    axes[0].plot(
        [r["N"] for r in n_rows],
        [r["torch_ms"] for r in n_rows],
        marker="o",
        label="PyTorch SDP",
    )
    axes[0].set_title(f"Latency vs N (D={D_FOR_N_SWEEP})")
    axes[0].set_xlabel("Sequence length N")
    axes[0].set_ylabel("ms / iteration")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # D sweep plot.
    axes[1].plot(
        [r["D"] for r in d_rows],
        [r["custom_ms"] for r in d_rows],
        marker="o",
        label="Custom kernel",
    )
    axes[1].plot(
        [r["D"] for r in d_rows],
        [r["torch_ms"] for r in d_rows],
        marker="o",
        label="PyTorch SDP",
    )
    axes[1].set_title(f"Latency vs D (N={N_FOR_D_SWEEP})")
    axes[1].set_xlabel("Head dimension D")
    axes[1].set_ylabel("ms / iteration")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_markdown_table(rows):
    ordered = sorted(rows, key=lambda r: (r["sweep"], r["N"], r["D"]))
    print("| sweep | B | H | N | D | custom_ms | torch_ms | speedup | custom_tflops | torch_tflops |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in ordered:
        print(
            f"| {r['sweep']} | {r['B']} | {r['H']} | {r['N']} | {r['D']} "
            f"| {r['custom_ms']:.4f} | {r['torch_ms']:.4f} | {r['speedup']:.3f} "
            f"| {r['custom_tflops']:.4f} | {r['torch_tflops']:.4f} |"
        )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(OUTPUT_DIR, f"shape_scaling_{timestamp}.png")
    plot_latest_path = os.path.join(OUTPUT_DIR, "shape_scaling.png")

    rows = collect_results()
    plot_results(rows, plot_path)
    plot_results(rows, plot_latest_path)

    print(f"Saved plot: {plot_path}")
    print(f"Saved plot: {plot_latest_path}")
    print()
    print_markdown_table(rows)


if __name__ == "__main__":
    main()

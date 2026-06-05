import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

# Allow importing benchmark modules from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")
sys.path.insert(0, REPO_ROOT)

from benchmarks.baseline_benchmark import run_benchmark


# Edit these to match the batch sizes you want to explore.
B_SWEEP = [1, 2, 4, 8]
H = 8
N = 1024
D = 64

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
    for B in B_SWEEP:
        result = run_benchmark(
            B=B,
            H=H,
            N=N,
            D=D,
            iters=ITERS,
            warmup=WARMUP,
            check=CHECK,
        )
        rows.append(
            {
                **result,
                "custom_tflops": estimate_tflops(B, H, N, D, result["custom_ms"]),
                "torch_tflops": estimate_tflops(B, H, N, D, result["torch_ms"]),
            }
        )
    return rows


def plot_results(rows, output_path):
    custom_x = [r["custom_ms"] for r in rows]
    custom_y = [r["custom_tflops"] for r in rows]
    torch_x = [r["torch_ms"] for r in rows]
    torch_y = [r["torch_tflops"] for r in rows]

    plt.figure(figsize=(6, 4))
    plt.scatter(custom_x, custom_y, label="Custom kernel", color="#4C78A8")
    plt.scatter(torch_x, torch_y, label="PyTorch SDP", color="#F58518")

    # Annotate each point with its batch size for clarity.
    for r in rows:
        plt.text(r["custom_ms"], r["custom_tflops"], f"B={r['B']}", fontsize=8, ha="left")
        plt.text(r["torch_ms"], r["torch_tflops"], f"B={r['B']}", fontsize=8, ha="left")

    plt.title(f"Throughput vs Latency (H={H}, N={N}, D={D})")
    plt.xlabel("Latency (ms / iteration)")
    plt.ylabel("Throughput (TFLOPS)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_markdown_table(rows):
    ordered = sorted(rows, key=lambda r: r["B"])
    print("| B | H | N | D | custom_ms | torch_ms | speedup | custom_tflops | torch_tflops |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in ordered:
        print(
            f"| {r['B']} | {r['H']} | {r['N']} | {r['D']} "
            f"| {r['custom_ms']:.4f} | {r['torch_ms']:.4f} | {r['speedup']:.3f} "
            f"| {r['custom_tflops']:.4f} | {r['torch_tflops']:.4f} |"
        )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(OUTPUT_DIR, f"throughput_latency_{timestamp}.png")
    plot_latest_path = os.path.join(OUTPUT_DIR, "throughput_latency.png")

    rows = collect_results()
    plot_results(rows, plot_path)
    plot_results(rows, plot_latest_path)

    print(f"Saved plot: {plot_path}")
    print(f"Saved plot: {plot_latest_path}")
    print()
    print_markdown_table(rows)


if __name__ == "__main__":
    main()

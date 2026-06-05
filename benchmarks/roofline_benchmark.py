import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Allow importing benchmark modules from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")
sys.path.insert(0, REPO_ROOT)

from benchmarks.baseline_benchmark import run_benchmark


# Edit these to match the shapes you want to explore.
CONFIGS = [
    {"name": "Small", "B": 1, "H": 8, "N": 256, "D": 64},
    {"name": "Medium", "B": 1, "H": 8, "N": 512, "D": 64},
    {"name": "Large", "B": 1, "H": 8, "N": 1024, "D": 64},
    {"name": "XL", "B": 1, "H": 8, "N": 2048, "D": 64},
]

ITERS = 50
WARMUP = 10
CHECK = False

# Update these with your GPU's peak numbers (approximate is fine).
PEAK_TFLOPS = 100.0
PEAK_BW_GBPS = 1000.0


def estimate_flops(B, H, N, D):
    # Rough FLOPs estimate for attention:
    # QK^T ~ 2 * B*H*N*N*D (mul+add)
    # (softmax*V) ~ 2 * B*H*N*N*D
    return 4.0 * B * H * N * N * D


def estimate_bytes(B, H, N, D):
    # Lower-bound memory traffic: read Q/K/V and write O once.
    # Real traffic is higher due to intermediates and reuse behavior.
    bytes_per_tensor = B * H * N * D * 4.0
    return 4.0 * bytes_per_tensor


def collect_results():
    rows = []
    for cfg in CONFIGS:
        result = run_benchmark(
            B=cfg["B"],
            H=cfg["H"],
            N=cfg["N"],
            D=cfg["D"],
            iters=ITERS,
            warmup=WARMUP,
            check=CHECK,
        )

        flops = estimate_flops(cfg["B"], cfg["H"], cfg["N"], cfg["D"])
        bytes_total = estimate_bytes(cfg["B"], cfg["H"], cfg["N"], cfg["D"])
        intensity = flops / bytes_total

        custom_tflops = (flops / (result["custom_ms"] / 1e3)) / 1e12
        torch_tflops = (flops / (result["torch_ms"] / 1e3)) / 1e12

        custom_bw = (bytes_total / (result["custom_ms"] / 1e3)) / 1e9
        torch_bw = (bytes_total / (result["torch_ms"] / 1e3)) / 1e9

        rows.append(
            {
                **cfg,
                **result,
                "flops": flops,
                "bytes": bytes_total,
                "intensity": intensity,
                "custom_tflops": custom_tflops,
                "torch_tflops": torch_tflops,
                "custom_bw_gbps": custom_bw,
                "torch_bw_gbps": torch_bw,
            }
        )

    return rows


def plot_results(rows, output_path):
    intensities = [r["intensity"] for r in rows]
    x_min = max(min(intensities) * 0.5, 1e-4)
    x_max = max(intensities) * 2.0

    x_vals = np.logspace(np.log10(x_min), np.log10(x_max), num=200)
    roofline = np.minimum(PEAK_TFLOPS, x_vals * PEAK_BW_GBPS / 1000.0)

    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, roofline, label="Roofline (peak)", color="#4C78A8")

    custom_x = [r["intensity"] for r in rows]
    custom_y = [r["custom_tflops"] for r in rows]
    torch_x = [r["intensity"] for r in rows]
    torch_y = [r["torch_tflops"] for r in rows]

    plt.scatter(custom_x, custom_y, label="Custom kernel", color="#F58518")
    plt.scatter(torch_x, torch_y, label="PyTorch SDP", color="#54A24B")

    for r in rows:
        plt.text(r["intensity"], r["custom_tflops"], r["name"], fontsize=8, ha="left")
        plt.text(r["intensity"], r["torch_tflops"], r["name"], fontsize=8, ha="left")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Operational intensity (FLOPs / byte)")
    plt.ylabel("Performance (TFLOPS)")
    plt.title("Roofline Model")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_markdown_table(rows):
    print("| name | B | H | N | D | intensity | custom_ms | torch_ms | custom_tflops | torch_tflops | custom_bw_gbps | torch_bw_gbps |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['name']} | {r['B']} | {r['H']} | {r['N']} | {r['D']} "
            f"| {r['intensity']:.4f} | {r['custom_ms']:.4f} | {r['torch_ms']:.4f} "
            f"| {r['custom_tflops']:.4f} | {r['torch_tflops']:.4f} "
            f"| {r['custom_bw_gbps']:.2f} | {r['torch_bw_gbps']:.2f} |"
        )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(OUTPUT_DIR, f"roofline_{timestamp}.png")
    plot_latest_path = os.path.join(OUTPUT_DIR, "roofline.png")

    rows = collect_results()
    plot_results(rows, plot_path)
    plot_results(rows, plot_latest_path)

    print(f"Saved plot: {plot_path}")
    print(f"Saved plot: {plot_latest_path}")
    print()
    print_markdown_table(rows)


if __name__ == "__main__":
    main()

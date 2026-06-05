import csv
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

# Allow importing benchmark modules from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "benchmarks", "results")
sys.path.insert(0, REPO_ROOT)

from benchmarks.baseline_benchmark import run_benchmark


# Edit these configs to match the shapes you care about.
CONFIGS = [
    {"name": "Small", "B": 1, "H": 8, "N": 256, "D": 64},
    {"name": "Medium", "B": 1, "H": 8, "N": 512, "D": 64},
    {"name": "Large", "B": 1, "H": 8, "N": 1024, "D": 64},
    {"name": "XL", "B": 1, "H": 8, "N": 2048, "D": 64},
]

ITERS = 100
WARMUP = 10
CHECK = True


def estimate_tflops(B, H, N, D, time_ms):
    # Rough FLOPs estimate for attention:
    # QK^T ~ 2 * B*H*N*N*D (mul+add)
    # (softmax*V) ~ 2 * B*H*N*N*D
    flops = 4.0 * B * H * N * N * D
    seconds = time_ms / 1e3
    return (flops / seconds) / 1e12


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"results_{timestamp}.csv")
    csv_latest_path = os.path.join(OUTPUT_DIR, "results.csv")

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
        result["name"] = cfg["name"]
        result["custom_tflops"] = estimate_tflops(cfg["B"], cfg["H"], cfg["N"], cfg["D"], result["custom_ms"])
        result["torch_tflops"] = estimate_tflops(cfg["B"], cfg["H"], cfg["N"], cfg["D"], result["torch_ms"])
        rows.append(result)

    # Save CSV for portfolio and further analysis.
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(csv_latest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Plot latency for both implementations.
    labels = [r["name"] for r in rows]
    custom_ms = [r["custom_ms"] for r in rows]
    torch_ms = [r["torch_ms"] for r in rows]
    speedup = [r["speedup"] for r in rows]

    plt.figure(figsize=(8, 4))
    x = range(len(labels))
    plt.bar([i - 0.2 for i in x], custom_ms, width=0.4, label="Custom kernel")
    plt.bar([i + 0.2 for i in x], torch_ms, width=0.4, label="PyTorch SDP")
    plt.xticks(list(x), labels)
    plt.ylabel("ms / iteration")
    plt.title("Latency Comparison")
    plt.legend()
    plt.tight_layout()
    latency_path = os.path.join(OUTPUT_DIR, f"latency_{timestamp}.png")
    latency_latest_path = os.path.join(OUTPUT_DIR, "latency.png")
    plt.savefig(latency_path, dpi=150)
    plt.savefig(latency_latest_path, dpi=150)
    plt.close()

    # Plot speedup.
    plt.figure(figsize=(8, 4))
    plt.bar(labels, speedup, color="#4C78A8")
    plt.ylabel("Speedup (Custom / PyTorch)")
    plt.title("Speedup by Configuration")
    plt.tight_layout()
    speedup_path = os.path.join(OUTPUT_DIR, f"speedup_{timestamp}.png")
    speedup_latest_path = os.path.join(OUTPUT_DIR, "speedup.png")
    plt.savefig(speedup_path, dpi=150)
    plt.savefig(speedup_latest_path, dpi=150)
    plt.close()

    # Print a compact table to the console.
    print(f"Saved CSV: {csv_path}")
    print(f"Saved CSV: {csv_latest_path}")
    print(f"Saved plot: {latency_path}")
    print(f"Saved plot: {latency_latest_path}")
    print(f"Saved plot: {speedup_path}")
    print(f"Saved plot: {speedup_latest_path}")
    print()
    for r in rows:
        diff = "n/a" if r["max_abs_diff"] is None else f"{r['max_abs_diff']:.3e}"
        print(
            f"{r['name']:>6} | B={r['B']} H={r['H']} N={r['N']} D={r['D']} "
            f"| custom={r['custom_ms']:.3f} ms "
            f"| torch={r['torch_ms']:.3f} ms "
            f"| speedup={r['speedup']:.2f}x "
            f"| diff={diff}"
        )


if __name__ == "__main__":
    main()

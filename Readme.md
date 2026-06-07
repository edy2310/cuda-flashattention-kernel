# FlashAttention Kernel (CUDA) — Portfolio Project

This project is a hands‑on, progressively optimized CUDA implementation of FlashAttention. I built it to demonstrate GPU kernel engineering fundamentals for recruiters and interviewers, even though my professional background is in distributed systems and cloud computing. The goal is to show I can design, implement, validate, and benchmark GPU kernels with the same rigor I apply to large‑scale systems.

## Highlights
1. **Custom CUDA kernel** with shared‑memory tiling, online softmax, and warp‑level reductions.
2. **PyTorch C++/CUDA extension** built once via `setup.py` and used across benchmarks/tests.
3. **Benchmark suite** comparing against PyTorch `scaled_dot_product_attention` with CSV + plots.
4. **Correctness tests** covering multiple shapes plus bad‑path validation.
5. **Triton backend** + Dockerfile for a simple deployment path.

## Architecture overview
- `kernels/FlashAttentionKernel.cu`: FlashAttention forward kernel; one warp computes one query row.
- `csrc/flash_attn_ext.cpp`: PyTorch extension binding that calls the CUDA launcher.
- `setup.py`: builds the C++/CUDA extension once for reuse.
- `benchmarks/`: baseline, config sweep, shape scaling, and throughput/latency scripts.
- `tests/`: correctness tests vs PyTorch reference and error‑path checks.
- `triton_deploy/`: Triton model repository, Dockerfile, and example client.

## Kernel design notes
- **Input layout:** `[B, H, N, D]` (batch, heads, sequence length, head dimension).
- **Warp‑per‑query mapping:** each warp computes one output row, enabling efficient shuffles.
- **Online softmax:** numerically stable in a single pass over keys.
- **Shared‑memory tiling:** reduces global memory traffic with a shared-memory layout tuned to stay within common Colab GPU limits.

## Build the extension
Requires a CUDA‑capable GPU and PyTorch built with CUDA.

```bash
python3 setup.py build_ext --inplace
```

## Run benchmarks
```bash
python3 -m pip install matplotlib
python3 benchmarks/config_sweep_benchmark.py
```

Benchmarks reported in this README:
1. `benchmarks/config_sweep_benchmark.py` — latency + speedup across fixed configs.
2. `benchmarks/shape_scaling_benchmark.py` — sweep `N` and `D`.
3. `benchmarks/throughput_latency_benchmark.py` — sweep `B` to compare throughput vs latency.
4. `benchmarks/roofline_benchmark.py` — roofline plot (edit peak GPU values).

Optional quick check (no plot):
- `benchmarks/baseline_benchmark.py` — single config sanity check.

Artifacts are written to `benchmarks/results/` (CSV + PNG plots).

## Benchmark results

### Configuration sweep (latency + speedup)
**Measures:** Average latency per iteration for several fixed shapes, plus speedup vs PyTorch.

**Result:**  
![Latency plot](./benchmarks/results/latency.png)  
![Speedup plot](./benchmarks/results/speedup.png)

**Explanation:** PyTorch remains faster because it uses highly optimized kernels. The custom kernel stays correct and the latency trend grows with larger `N`, which validates both correctness and expected scaling.

### Shape scaling (N/D sweep)
**Measures:** How latency changes when sweeping `N` (sequence length) and `D` (head dimension).

**Result:**  
![Shape scaling plot](./benchmarks/results/shape_scaling.png)

**Explanation:** Latency grows super‑linearly with `N` (attention is O(N²)) and more gradually with `D`. The curves help identify where shared‑memory tiling stops scaling efficiently.

### Throughput vs latency (batch sweep)
**Measures:** Throughput (TFLOPS) vs latency as batch size increases.

**Result:**  
![Throughput vs latency plot](./benchmarks/results/throughput_latency.png)

**Explanation:** Throughput usually improves with larger batch sizes until the GPU saturates. This highlights where the kernel becomes limited by occupancy or memory bandwidth.

### Roofline model
**Measures:** Achieved TFLOPS vs operational intensity relative to a roofline bound.

**Result:**  
![Roofline plot](./benchmarks/results/roofline.png)

**Explanation:** Points below the roofline indicate headroom. The position relative to the sloped (bandwidth) and flat (compute) regions shows whether the kernel is memory‑bound or compute‑bound.

## Run correctness tests
```bash
python3 tests/test_flash_attention.py
```

## Triton deployment (optional)
Build and run the Triton server:
```bash
docker build -t flashattn-triton triton_deploy
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 flashattn-triton
```

Run the example client:
```bash
python3 -m pip install tritonclient[http]
python3 triton_deploy/client.py
```

## What this project demonstrates
1. **CUDA kernel engineering:** tiling, shared memory, warp‑level reductions.
2. **Numerical stability:** online softmax that remains stable for large dot products.
3. **Benchmarking discipline:** apples‑to‑apples latency comparison vs PyTorch reference.
4. **Integration skills:** C++/CUDA extension wired into Python and reusable tooling.
5. **Deployment awareness:** Triton backend + Docker workflow for serving.

## Next optimization targets
1. Larger tiles / better occupancy trade‑offs.
2. Vectorized loads/stores and improved memory coalescing.
3. Mixed‑precision (FP16/BF16) paths with Tensor Cores.
4. More aggressive kernel fusion and reduced register pressure.

# FlashAttention Kernel

This project is a hands‑on CUDA implementation of FlashAttention built as a portfolio piece for GPU and Kernel Engineer interviews. It is meant to show that I can reason about memory layout, tiling, warp-level execution, numerical stability, and performance tradeoffs with the same rigor I apply to large-scale systems.

## Highlights
1. **Custom CUDA kernel** that demonstrates shared-memory tiling, online softmax, and warp-level reductions.
2. **PyTorch C++/CUDA extension** wired into Python so the kernel can be built once and reused in tests and benchmarks.
3. **Benchmark suite** that compares scaling behavior against PyTorch `scaled_dot_product_attention` with CSV output and plots.
4. **Correctness tests** that validate multiple shapes and bad paths, showing the kernel is not just fast but also reliable.
5. **Triton backend** plus Docker support to show the kernel can be packaged with a production-minded deployment path.

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



Performance results
-------------------
These results are included to show how a hand-written CUDA kernel behaves under realistic workloads, where it can approach a mature library implementation, and where there is still headroom for optimization.

**Latency**

![Benchmark](benchmarks/results/flashattention_latency.png)

#### Latency interpretation

The latency plot shows that the custom kernel is slower than PyTorch, but it still follows the same scaling curve as the problem grows, which is exactly what you want from a solid CUDA implementation. That behavior demonstrates that the kernel is functionally correct, numerically stable, and engineered with the right algorithmic structure, even if it does not yet match the aggressively tuned kernels shipped by PyTorch. It also shows where a hand-written kernel can close the gap: on smaller or more regular shapes, and in cases where launch overhead, memory access patterns, or domain-specific constraints make a tailored implementation competitive enough to justify its use.

-------------------

**Shape Scaling**

![Benchmark](benchmarks/results/flashattention_shape_scaling.png)

#### Shape Scaling interpretation

The shape-scaling results are useful because they show the kernel behaving like a real production GPU implementation rather than a toy example. As `D` grows, the runtime rises in a controlled way, which suggests the memory layout, tiling strategy, and per-thread work distribution are all behaving as intended. As `N` grows, the cost increases much faster, but the custom kernel still tracks the same trend as the reference, showing that the implementation scales correctly and can remain practical in workloads where the sequence lengths are moderate or where the attention pattern is specialized enough that a custom kernel can get closer to an optimized library path.


-------------------

**Throughput**

![Benchmark](benchmarks/results/flashattention_throughput.png)

#### Throughput interpretation

The throughput plot is the strongest signal that the kernel is written with GPU execution in mind: as batch size increases, the hardware is fed with more independent work and the achieved throughput becomes more efficient and more stable. PyTorch still wins on absolute performance because its kernels are heavily optimized and battle-tested, but the custom implementation narrows the gap in settings where parallelism is high and the workload can be shaped to the kernel’s strengths. That is an important engineering result for an interview portfolio, because it shows you understand how batching, occupancy, and utilization interact on real GPUs, and that a custom kernel can become much more competitive when the workload matches its design.

-------------------

**Roofline model**

![Benchmark](benchmarks/results/flashattention_roofline.png)

#### Roofline model interpretation

The roofline chart places the kernel in a realistic engineering context: it is not trying to outperform a world-class PyTorch implementation on raw peak numbers, but it does show that the code is organized around the same fundamental GPU tradeoffs. The custom kernel moves in the right direction as operational intensity increases, which means it is extracting more useful work from the hardware and getting closer to the compute-bound regime where a specialized kernel can become more competitive. For an interviewer, that is the key story: this project demonstrates that you can reason about bandwidth, arithmetic intensity, and launch efficiency, and that you understand how a bespoke CUDA kernel can approach a professional library implementation when the workload and tiling choices line up well.


-------------------

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
1. **CUDA kernel engineering:** tiling, shared memory, warp-level reductions, and launch configuration.
2. **Performance reasoning:** identifying when a custom kernel can get closer to a highly optimized library path.
3. **Numerical stability:** online softmax that remains stable for large dot products.
4. **Systems integration:** a C++/CUDA extension wired into Python and reused across tests, benchmarks, and deployment.
5. **Production awareness:** Triton backend + Docker workflow to show the kernel can fit into a serving pipeline.

## Next optimization targets
1. Larger tiles / better occupancy trade‑offs.
2. Vectorized loads/stores and improved memory coalescing.
3. Mixed‑precision (FP16/BF16) paths with Tensor Cores.
4. More aggressive kernel fusion and reduced register pressure.

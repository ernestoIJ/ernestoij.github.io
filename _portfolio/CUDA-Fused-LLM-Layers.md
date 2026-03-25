---
title: "Fused CUDA LLM Layers (RMSNorm & Softmax)"
excerpt: "Engineered bare-metal C++ CUDA inference kernels to accelerate Large Language Model (LLM) layers. Architected cooperative shared memory reductions and deep kernel fusion to completely bypass native PyTorch memory bandwidth bottlenecks, achieving up to a 4.2x speedup. <br/><img src='/images/rmsnorm_benchmark.png'>"
collection: portfolio
---

### Overview
Developed highly optimized, **bare-metal CUDA C++ implementations** of critical neural network inference layers, specifically Root Mean Square Normalization (RMSNorm) and a Numerically Stable Softmax. Native machine learning frameworks like PyTorch operate in Eager Mode, launching multiple sequential kernels that force the GPU to constantly read and write to slow Global VRAM. This project fuses those operations into unified kernels, utilizing the Streaming Multiprocessor's (SM) ultra-fast shared memory cache to compute intermediate math on the silicon, drastically reducing memory latency. You can view the full C++ architecture and benchmarks in the GitHub repository [here.](https://github.com/ernestoIJ/cuda-fused-llm-layers)

### Technical Highlights
- **Architected Cooperative Memory Reductions** by implementing a bare-metal O(log N) parallel tree-reduction algorithm (`stride >>= 1`), allowing 1024-thread blocks to cooperatively collapse layer activations without relying on slow global memory atomics.
- **Engineered a Double-Reduction Pipeline** for the Fused Softmax, orchestrating two consecutive passes within the same block state. Utilized strict `__syncthreads()` barrier synchronization to safely broadcast the `fmaxf` global maximum before calculating the exponential sum to prevent floating-point overflow.
- **Maximized Kernel Fusion** to completely bypass PyTorch's intermediate VRAM allocations. Threads load from VRAM exactly once, compute the sum of squares, perform root-inverse scaling, and broadcast the final normalized values back to memory in a single unified pass, achieving a **4.22x execution speedup**.
- **Implemented Safe Thread Masking** to handle multi-dimensional unaligned tensors, utilizing `-INFINITY` and zero-padded conditional logic to prevent memory corruption and out-of-bounds accesses when layer dimensions do not perfectly map to hardware block sizes.

![RMSNorm Memory Bandwidth Benchmark](/images/rmsnorm_benchmark.png)
![Softmax Double-Reduction Benchmark](/images/softmax_benchmark.png)
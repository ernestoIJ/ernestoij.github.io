---
title: "Fused CUDA Optimizers (AdamW & Lion)"
excerpt: "Engineered bare-metal C++ CUDA extensions for modern neural network optimizers to accelerate LLM training. Architected an O(1) contiguous Arena Allocator and utilized kernel fusion to completely bypass native framework bottlenecks and CPU-to-GPU overhead. <br/><img src='/images/scaling_graph.png'>"
collection: portfolio
---

### Overview
Developed highly optimized, **bare-metal CUDA C++ implementations** of the industry-standard AdamW and the 2023 Google Brain Lion optimizers. Native machine learning frameworks often suffer from massive CPU kernel launch bottlenecks and global memory latency during deep neural network training. This project eliminates those hardware inefficiencies by utilizing custom memory management and fused execution kernels to squeeze absolute maximum throughput out of modern GPU silicon. You can view the full C++ architecture and benchmarks in the GitHub repository [here.](https://github.com/ernestoIJ/fused-cuda-optimizers).

### Technical Highlights
- **Architected an $O(1)$ Arena Allocator** using custom C++ pointer arithmetic to replace fragmented $O(N)$ `cudaMalloc` loops, allocating massive contiguous memory blocks to completely eliminate CPU-to-GPU driver overhead regardless of network depth.
- **Engineered Horizontal Fusion** by routing dynamic 2D thread grids via `float**` pointer arrays, enabling the update of the entire neural network architecture in a single, unified kernel launch.
- **Implemented Vertical Fusion** to bypass the Global Memory bandwidth bottleneck, confining all decoupled weight decay, momentum updates, and variance scaling strictly to ultra-fast hardware registers.
- **Exploited Algorithm-Specific Memory** by adapting the architecture for the Lion optimizer (EvoLved Sign Momentum), completely dropping the variance tensor to reduce the Arena footprint by 25% and achieving scalable, sub-millisecond execution times (~619 µs).

![Optimizer Scaling Benchmark](/images/scaling_graph.png)
## VBATS: An adaptive strategy for grouped GEMM on GPUs
General matrix multiplication (GEMM) is a crucial operation in various fields, such as deep learning, scientific computing, and image processing. In many real-world applications, particularly in deep learning, matrices are often too small and have a variety of matrix sizes to make full use of GPUs. To address this issue, the vbatch method and grouped GEMM have been proposed in previous studies and libraries, both designed to process a set of small, independent GEMMs using a single CUDA kernel. However, the strategy selection in vbatch GEMM is typically rudimentary, and grouped GEMM lacks support for configurable tile sizes, limiting its flexibility. In this work, we discuss the limitations of previous work related to the vbatch method and propose a framework named VBATS, an adaptive vbatch tiling and splitting algorithm for grouped GEMM. VBATS introduces a unified tile method to reduce redundant thread blocks, along with a two-stage strategy selection algorithm that includes a tiling stage and a splitting stage to accelerate vbatch GEMM on GPUs. The experimental results based on synthetic grouped GEMM on an NVIDIA A100 GPU demonstrate that VBATS achieves an average performance gain of approximately 2.01x over cuBLAS grouped GEMM (up to 8.96x). We also performed experiments on various GPU architectures, demonstrating that our proposed method achieves consistent performance improvements across different platforms. Furthermore, using GoogleNet as a real-world case study, VBATS can achieve an average speedup of 2.72x.

## Minimum requirements:

- Architecture: Volta or newer.
- CUDA Toolkit version: 12.5+(to support cuBLAS grouped GEMM)


## Build and run

```bash
$ bash run.sh
```

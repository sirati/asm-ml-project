# Papers and Implementations Reference

This document maps all papers referenced in `edit_diffusion_impl_spec.md` to their corresponding open-source implementations. Organized by preference: Direct CUDA kernels > Triton implementations > Generic ML framework implementations.

## Core Architecture Components

### 1. SEDD: Discrete Diffusion Modeling by Estimating Ratios
- **Paper**: https://arxiv.org/abs/2310.16834
- **Official Implementation**: [louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) ⭐
  - Framework: PyTorch
  - Language: Python
  - Status: Official, ICML 2024 Best Paper
  - Pretrained Models: Available on Hugging Face

### 2. Perceiver IO (Using krasserm/perceiver-io)
- **Paper**: https://arxiv.org/abs/2107.14795
- **Official Implementation**: [krasserm/perceiver-io](https://github.com/krasserm/perceiver-io) ⭐
  - Framework: PyTorch
  - Features: CUDA support, PyTorch Lightning training scripts
  - Includes: Perceiver, Perceiver IO, and Perceiver AR variants
  - Training: Distributed training support

### 3. Mamba-2: State Space Models for Sequence Processing
- **Paper**: https://arxiv.org/abs/2405.21060
- **Official Implementation**: [state-spaces/mamba](https://github.com/state-spaces/mamba) ⭐⭐⭐
  - Framework: PyTorch
  - **Kernel Implementation**: Direct CUDA kernels + Optional Triton backend
  - Language: Python + CUDA C++
  - PyPI Package: `mamba-ssm`
  - Mixed Precision: Supports AMP (float32 params, float16 compute)
  - Note: Requires causal-conv1d>=1.4.0 for efficient Conv1d layer
  - **Status**: Production-ready with CUDA optimization

### 4. BiMamba-2: Bidirectional State Space Models
- **Paper Reference**: DiffuApriel (GitHub: [ML-GSAI/DiffuApriel](https://github.com/ML-GSAI/DiffuApriel))
- **BiMamba Implementation**: [Tangshengku/Bi-Mamba](https://github.com/Tangshengku/Bi-Mamba) ⭐
  - Framework: PyTorch
  - Focus: "Accurate 1-Bit State Space Models"
  - Alternative: [Human9000/nd-Mamba2-torch](https://github.com/Human9000/nd-Mamba2-torch)
    - Features: Multi-dimensional support (1D/2D/3D)
    - Export: JIT Script and ONNX support

## Attention & Optimization Components

### 5. Flash Attention 2: Fast and Memory-Efficient Attention
- **Paper**: https://arxiv.org/abs/2307.08691
- **Official Implementation**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) ⭐⭐⭐
  - Framework: PyTorch
  - **Kernel Implementation**: Direct CUDA kernels (primary) + Triton backend
  - Language: Python + CUDA C++ + Triton
  - PyPI Package: `flash-attn>=2.0`
  - **Status**: Production-ready with high-performance CUDA kernels
  - Features: FP16, FP32, and experimental FP8 precision support

### 5a. Flash Attention 2 in Triton (Educational/Reference)
- **Triton Implementation**: [Triton Documentation Example](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
  - Framework: Triton
  - **Kernel Type**: Triton (higher-level than CUDA, easier to understand)
  - Use Case: Educational reference, experimental optimizations
  - Alternative Educational Implementations:
    - [Understanding Flash Attention: Writing Triton Kernel Code](https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/)
    - [Triton Flash Attention Kernel Walkthrough](https://nathanchen.me/public/Triton-Flash-Attention-Kernel-Walkthrough.html)

### 6. Attention is Off by One (Softmax-1 Implementation)
- **Paper**: https://evanmiller.org/attention-is-off-by-one.html (Blog post, arxiv 2403.17130)
- **Implementation**: [kyegomez/AttentionIsOFFByOne](https://github.com/kyegomez/AttentionIsOFFByOne)
  - Framework: PyTorch
  - Language: Python
  - Features: QuietAttention, Softmax-One variants
  - Status: Community implementation

### 7. Fourier Positional Embeddings for N-Encoding
- **Paper**: https://arxiv.org/pdf/2502.09741
- **Status**: No specific official implementation found
- **Recommendation**: Implement directly using standard PyTorch:
  ```python
  import torch
  import math
  
  def fourier_embedding(n_values, d_model):
      """Fourier positional encoding for continuous N values."""
      frequencies = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
      n_values_normalized = n_values.unsqueeze(-1)
      encodings = torch.zeros(*n_values.shape, d_model)
      encodings[..., 0::2] = torch.sin(n_values_normalized * frequencies)
      encodings[..., 1::2] = torch.cos(n_values_normalized * frequencies)
      return encodings
  ```

## Implementation Priority Order

For optimal performance, use in this order:

### Tier 1: Direct CUDA Kernels (Fastest)
1. **Mamba-2** (`state-spaces/mamba`) - CUDA kernels built-in
2. **Flash Attention 2** (`Dao-AILab/flash-attention`) - Optimized CUDA kernels

### Tier 2: Triton Kernels (Good Performance, Easier Dev)
1. Flash Attention 2 (also available in Triton)
2. Custom kernels for novel components

### Tier 3: PyTorch Native (Good for Prototyping)
1. All other components (Perceiver IO, BiMamba, etc.)

## Installation Commands

```bash
# Core components
pip install mamba-ssm              # Mamba-2 with CUDA
pip install flash-attn>=2.0        # Flash Attention 2 with CUDA
pip install torch                  # Latest PyTorch
pip install triton                 # For custom kernel development

# Optional for distributed training
pip install pytorch-lightning       # PyTorch Lightning
pip install perceiver-io           # Perceiver IO from PyPI (alternative to github)

# Optional for enhanced features
pip install causal-conv1d>=1.4.0  # Efficient Conv1d for Mamba
pip install xpos-relative-position # Extended positional embeddings (if needed)
```

## Key Notes

1. **CUDA Version Compatibility**: Mamba-2 has been tested with CUDA 11.8+. Mixed precision (AMP) recommended.

2. **SEDD Loss Function**: Use official `louaaron/Score-Entropy-Discrete-Diffusion` as reference for score entropy loss implementation.

3. **Perceiver IO + Mamba Integration**: The krasserm/perceiver-io doesn't use Mamba natively; implement MambaCrossAttention as specified in `edit_diffusion_impl_spec.md`.

4. **BiMamba Architecture**: Consider using official state-spaces/mamba directly instead of separate BiMamba repos for production stability.

5. **Flash Attention Usage**: Official Dao-AILab/flash-attention is production-ready; Triton implementations are mainly for educational purposes.

## References

- SEDD Paper: https://arxiv.org/abs/2310.16834
- Mamba-2 Paper: https://arxiv.org/abs/2405.21060
- Flash Attention 2 Paper: https://arxiv.org/abs/2307.08691
- Perceiver IO Paper: https://arxiv.org/abs/2107.14795
- Attention is Off by One: https://evanmiller.org/attention-is-off-by-one.html
- Fourier Embeddings: https://arxiv.org/pdf/2502.09741

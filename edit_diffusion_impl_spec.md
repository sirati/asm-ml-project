# Hierarchical Edit-Based Diffusion: Implementation Spec

## Edit Operations

### Token Set
- `KEEP` (0)
- `DELETE` (1)
- `INSERT` (2)
- `DELETE-1-AND-INSERT` (3)

### N Encoding
- Fourier positional encoding for N value
- Paper: https://arxiv.org/pdf/2502.09741
- Apply to INSERT and DELETE-1-AND-INSERT operations

## Network Architecture

### Shared Backbone
```python
# Single network for both outer and inner diffusion
# BiMamba-2 for all sequence processing
from mamba_ssm import Mamba2

class DiffusionBackbone(nn.Module):
    def __init__(self, d_model, n_layers):
        self.layers = nn.ModuleList([
            BiMamba2(d_model) for _ in range(n_layers)
        ])
```

### Edit Tagger Head
```python
class EditTagger(nn.Module):
    def __init__(self, d_model):
        self.op_head = nn.Linear(d_model, 4)  # 4 operations
        self.n_embedder = FourierEmbedding()  # For N values
        
    def forward(self, h):
        ops = self.op_head(h)  # [B, L, 4]
        n_values = self.n_predictor(h)  # Continuous N
        return ops, n_values
```

### Perceiver-IO Specialists
```python
class PerceiverSpecialist(nn.Module):
    def __init__(self, n_latents, d_latent, d_input):
        self.latents = nn.Parameter(torch.randn(n_latents, d_latent))
        self.cross_attn = MambaCrossAttention(d_latent, d_input)  # Mamba, not attention
        self.self_attn = nn.ModuleList([
            Mamba2(d_latent) for _ in range(n_self_attn_layers)
        ])
```

### Mamba Cross-Attention
```python
# Use Mamba for cross-attention to avoid O(n²) on input length
class MambaCrossAttention(nn.Module):
    def __init__(self, d_query, d_context):
        self.proj_q = nn.Linear(d_query, d_model)
        self.proj_kv = nn.Linear(d_context, d_model)
        self.mamba = Mamba2(d_model)
        
    def forward(self, query, context):
        q = self.proj_q(query)
        kv = self.proj_kv(context)
        combined = torch.cat([q, kv], dim=1)
        return self.mamba(combined)[:, :query.size(1)]
```

## Attention Implementation

### Flash Attention 2 with Softmax-1
- Use: `flash_attn_func` from `flash-attn>=2.0`
- Paper: "Attention is Off by One" - prepend dummy token
- Implementation:
```python
# Prepend dummy token to achieve softmax-1
def flash_attn_softmax1(q, k, v):
    # Add dummy token (doesn't attend to others)
    dummy = torch.zeros_like(q[:, :1])
    q_aug = torch.cat([dummy, q], dim=1)
    k_aug = torch.cat([dummy, k], dim=1)
    v_aug = torch.cat([dummy, v], dim=1)
    
    # Mask: dummy doesn't attend to anything
    mask = torch.ones(q_aug.size(1), k_aug.size(1))
    mask[0, :] = 0
    
    out = flash_attn_func(q_aug, k_aug, v_aug, causal=False, 
                          attn_mask=mask)
    return out[:, 1:]  # Remove dummy output
```

### When to Use Attention vs Mamba
- **Never use attention for:**
  - Cross-attention to input (use Mamba)
  - Diffusion output model (BiMamba only)
- **Optional attention for:**
  - Within specialist self-attention (single layer max, rest Mamba)
  - Only if context << 2048 tokens

## BiMamba-2 Implementation
```python
class BiMamba2(nn.Module):
    def __init__(self, d_model):
        self.forward_mamba = Mamba2(d_model)
        self.backward_mamba = Mamba2(d_model)
        self.proj = nn.Linear(2 * d_model, d_model)
        
    def forward(self, x):
        fwd = self.forward_mamba(x)
        bwd = self.backward_mamba(x.flip(dims=[1])).flip(dims=[1])
        return self.proj(torch.cat([fwd, bwd], dim=-1))
```
- Paper: DiffuApriel (https://github.com/ML-GSAI/DiffuApriel)

## Loss Function

### Score Entropy Loss
```python
def score_entropy_loss(model, x_0, x_t, t):
    """
    Score entropy from SEDD paper
    Paper: https://arxiv.org/abs/2310.16834
    """
    # Model predicts ratios p(x_0|x_t) / p(x_t)
    logits = model(x_t, t)
    
    # Score entropy objective (see SEDD Eq. 8)
    log_p_ratio = F.log_softmax(logits, dim=-1)
    target_ratio = get_target_ratio(x_0, x_t, t)
    
    loss = -torch.mean(target_ratio * log_p_ratio)
    return loss
```
- Implementation: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

### Total Loss (Stage 1 Training Only)
```python
# Stage 1: Train on single-token operations only (INSERT-1, DELETE-1)
# Stage 2: TODO - hierarchical specialist training
total_loss = score_entropy_loss(model, x_0, x_t, t)
```

## Training Strategy

### Stage 1: Single-Token Operations
- Only INSERT-1, DELETE-1, REPLACE, KEEP
- Standard score entropy training
- Stable baseline

### Stage 2: Multi-Token Operations
- Gradually introduce INSERT-2, INSERT-4, ...
- Curriculum: increase max N over training

### Stage 3: Hierarchical Specialists
- TODO: Define specialist training objective

## Key Dependencies

```bash
pip install mamba-ssm  # Mamba-2 implementation
pip install flash-attn>=2.0  # Flash Attention 2
pip install triton  # For custom kernels
```

## File Structure
```
model/
├── backbone.py          # BiMamba2 shared network
├── tagger.py           # Edit operation head
├── specialist.py       # Perceiver-IO specialist
├── mamba_cross.py      # Mamba-based cross-attention
├── diffusion.py        # Diffusion schedule + sampling
└── losses.py           # Score entropy loss

utils/
├── fourier_embed.py    # Fourier encoding for N
└── flash_utils.py      # Flash attention wrappers
```

## Critical Implementation Notes

1. **Single Network**: Same BiMamba2 backbone for outer AND inner diffusion
2. **No Standard Attention**: Use Mamba for all cross-attention (avoid input length bottleneck)
3. **Flash Attention**: Only for rare self-attention, always with dummy token (softmax-1)
4. **Fourier N Encoding**: Apply to predicted N values, not separate tokens per size
5. **Edit Op Degeneracy**: DELETE-1-AND-INSERT-N is distinct from DELETE + INSERT
6. **Training Stability**: Start with N=1 only, gradually increase

## References

- SEDD: https://arxiv.org/abs/2310.16834
- DiffuApriel (BiMamba): https://github.com/ML-GSAI/DiffuApriel
- Mamba-2: https://github.com/state-spaces/mamba
- Flash Attention 2: https://github.com/Dao-AILab/flash-attention
- Fourier Embeddings: https://arxiv.org/pdf/2502.09741
- Attention is Off by One: https://arxiv.org/abs/2403.17130

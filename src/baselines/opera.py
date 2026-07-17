import torch

def opera_baseline(logits: torch.Tensor, attentions: tuple, penalty_scale: float = 0.5) -> torch.Tensor:
    """
    OPERA (Over-trust Penalty) baseline.
    Penalizes logits if cross-attention maps exhibit localized over-trust loops.
    """
    if attentions is None:
        return logits
        
    seq_len = logits.shape[1]
    vocab_size = logits.shape[2]
    
    # Calculate self-attention concentration to detect over-trust
    # Average across all layers and heads
    stacked_attn = torch.stack([a[0].mean(dim=0) for a in attentions])  # (layers, seq_len, seq_len)
    avg_attn = stacked_attn.mean(dim=0)  # (seq_len, seq_len)
    
    # Compute penalty: tokens with high maximum attention weights to a single patch/token
    max_attn_vals, _ = torch.max(avg_attn, dim=-1)  # (seq_len,)
    
    # Apply penalty back to the logits
    penalized_logits = logits.clone()
    for t_idx in range(seq_len):
        penalty = penalty_scale * max_attn_vals[t_idx].item()
        # Downweight top logits using penalty to encourage exploration/diversity
        penalized_logits[0, t_idx] = logits[0, t_idx] - penalty
        
    return penalized_logits

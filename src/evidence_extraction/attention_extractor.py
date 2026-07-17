import torch
import torch.nn.functional as F
import numpy as np

class AttentionExtractor:
    """
    Analyzes VLM token dynamics, computes token entropy/margins,
    and extracts layerwise cross-attention trajectories from text to visual patches.
    """
    def __init__(self, num_layers: int, num_heads: int):
        self.num_layers = num_layers
        self.num_heads = num_heads

    def extract_dynamics(self, logits: torch.Tensor) -> dict:
        """
        Computes token-level confidence, entropy, and logit margins.
        
        Args:
            logits (Tensor): Shape (1, seq_len, vocab_size)
        Returns:
            dict with lists of:
              - 'confidences': probability of selected tokens
              - 'entropies': Shannon entropy per token position
              - 'margins': difference between top 2 logits
        """
        probs = F.softmax(logits[0], dim=-1)  # (seq_len, vocab_size)
        
        # Token confidence (max probability)
        confidences, predictions = torch.max(probs, dim=-1)
        
        # Shannon entropy
        entropies = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
        
        # Logit margin
        top2, _ = torch.topk(logits[0], k=2, dim=-1)
        margins = top2[:, 0] - top2[:, 1]
        
        return {
            "confidences": confidences.tolist(),
            "entropies": entropies.tolist(),
            "margins": margins.tolist(),
            "predictions": predictions.tolist()
        }

    def extract_cross_attention(self, attentions: tuple, num_patches: int, seq_len: int = 20) -> torch.Tensor:
        """
        Extracts cross-attention trajectories (attention from text tokens to visual patches).
        
        Args:
            attentions (tuple): Tuple of L layers, each shape (1, num_heads, seq_len, seq_len)
            num_patches (int): Number of visual patches (e.g. 196)
            seq_len (int): Default sequence length for fallback tensor
        Returns:
            Tensor of shape (num_layers, seq_len, num_patches) representing text-to-patch attention
        """
        if not attentions:
            # Fallback if VLM has no attention output (e.g. mock mode without attentions)
            return torch.zeros(self.num_layers, seq_len, num_patches)
            
        num_layers = len(attentions)
        seq_len = attentions[0].shape[-1]
        
        # Initialize cross attention tensor
        # (layers, seq_len, num_patches)
        cross_att = torch.zeros(num_layers, seq_len, min(num_patches, seq_len))
        
        for layer_idx, att in enumerate(attentions):
            # att shape: (1, num_heads, seq_len, seq_len)
            # Average across heads
            mean_att = att[0].mean(dim=0)  # (seq_len, seq_len)
            
            # In decoder-only architectures, visual patches are prepended.
            # So attention from token i to patch j is mean_att[i, j] where j < num_patches.
            limit = min(num_patches, seq_len)
            cross_att[layer_idx, :, :limit] = mean_att[:, :limit]
            
        return cross_att

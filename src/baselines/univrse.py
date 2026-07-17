import torch
import numpy as np

def univrse_baseline(logits: torch.Tensor) -> float:
    """
    UniVRSE-style uncertainty baseline.
    Computes sequence-level uncertainty from token logits (mean entropy).
    """
    probs = torch.softmax(logits[0], dim=-1)
    entropies = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    mean_entropy = float(entropies.mean().item())
    return mean_entropy

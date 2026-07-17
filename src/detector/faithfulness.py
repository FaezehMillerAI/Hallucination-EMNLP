import torch
import numpy as np

def compute_custom_scores(claims: list, vlm_outputs: dict, cross_attentions: torch.Tensor) -> list:
    """
    Computes custom heuristic signals for each claim to guide causal pseudo-labeling.
    
    Returns a list of dicts, each containing:
      - visual_support: float
      - trajectory_stability: float
      - counterfactual_consistency: float
      - prior_dominance: float
      - intra_consistency: float
    """
    seq_len = len(vlm_outputs["tokens"])
    num_patches = vlm_outputs["vision_embeddings"].shape[1]
    num_layers = cross_attentions.shape[0]
    
    scores = []
    
    # Pre-calculate prior dominance: simulated token confidences when visual input is degraded
    # High confidence in outputs indicates strong language model priors
    entropies = token_entropies = vlm_outputs.get("entropies", [0.5] * seq_len)
    confidences = vlm_outputs.get("confidences", [0.8] * seq_len)
    
    for c_idx, claim in enumerate(claims):
        # 1. Spans map to token indices
        span = claim["token_span"]
        start_tok = int((span[0] / max(span[1], 1)) * seq_len)
        end_tok = max(start_tok + 1, int((span[1] / max(span[1], 1)) * seq_len))
        
        # 2. Visual Support Score
        # Average attention mass to top-K patches for the claim tokens
        claim_cross_att = cross_attentions[:, start_tok:end_tok].mean(dim=1)  # (num_layers, num_patches)
        max_layer_att = claim_cross_att.max(dim=0)[0]  # (num_patches,)
        top_k_att = torch.topk(max_layer_att, k=min(5, num_patches))[0]
        visual_support = float(top_k_att.mean().item())
        
        # 3. Trajectory Stability Score
        # Consistency across layers (inverse of standard deviation across layers)
        layer_stds = claim_cross_att.std(dim=0)  # (num_patches,)
        trajectory_stability = float(1.0 / (layer_stds.mean().item() + 1e-6))
        
        # 4. Decoder Prior Dominance Score
        # Average confidence of tokens in the claim span
        prior_dominance = float(np.mean(confidences[start_tok:end_tok]))
        
        # 5. Counterfactual Consistency Score
        # High if visual support is meaningful. We simulate or calculate from perturbation.
        # If visual support is low, counterfactual consistency is low.
        counterfactual_consistency = float(max(0.0, 1.0 - (prior_dominance / (visual_support + 1e-6))))
        
        # 6. Intra-Response Consistency Score
        # Count contradictions (claims targeting same anatomy with opposing negations)
        conflicts = 0
        for other_claim in claims:
            if other_claim != claim:
                if other_claim["anatomy"] == claim["anatomy"] and other_claim["negation"] != claim["negation"]:
                    conflicts += 1
        intra_consistency = float(1.0 / (conflicts + 1.0))
        
        scores.append({
            "visual_support": visual_support,
            "trajectory_stability": trajectory_stability,
            "counterfactual_consistency": counterfactual_consistency,
            "prior_dominance": prior_dominance,
            "intra_consistency": intra_consistency
        })
        
    return scores

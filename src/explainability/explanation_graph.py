import torch

def extract_minimal_subgraph(hetero_data, claim_idx: int, top_k_tokens: int = 3, top_k_patches: int = 3) -> dict:
    """
    Extracts the minimal subgraph showing the provenance of a hallucination.
    Identifies which text tokens, visual patches, and latent states contributed 
    most to the target claim.
    """
    subgraph = {
        "target_claim_idx": claim_idx,
        "connected_tokens": [],
        "connected_patches": [],
        "implicated_latent_layers": []
    }
    
    # 1. Trace claim -> has_token -> texttoken
    claim_to_token = hetero_data["claim", "has_token", "texttoken"].edge_index
    token_indices = claim_to_token[1][claim_to_token[0] == claim_idx].tolist()
    subgraph["connected_tokens"] = token_indices
    
    # 2. Trace texttoken -> attends_to -> visualpatch
    # Retrieve attention weights for the tokens in the claim
    tok_to_patch = hetero_data["texttoken", "attends_to", "visualpatch"].edge_index
    tok_to_patch_attr = hetero_data["texttoken", "attends_to", "visualpatch"].edge_attr
    
    patch_scores = {}
    for idx, (t, p) in enumerate(tok_to_patch.t().tolist()):
        if t in token_indices:
            weight = tok_to_patch_attr[idx].item()
            patch_scores[p] = patch_scores.get(p, 0.0) + weight
            
    # Sort and pick top K patches
    sorted_patches = sorted(patch_scores.items(), key=lambda item: item[1], reverse=True)
    subgraph["connected_patches"] = [
        {"patch_id": p, "attribution_score": score} 
        for p, score in sorted_patches[:top_k_patches]
    ]
    
    # 3. Trace texttoken -> derived_from -> latentstate
    # For simplicity, implicate the top layers (e.g. layers with highest hidden state variance or last layer)
    num_layers = hetero_data["latentstate"].x.shape[0] - 1
    subgraph["implicated_latent_layers"] = [num_layers - 1, num_layers] # Last two layers
    
    return subgraph

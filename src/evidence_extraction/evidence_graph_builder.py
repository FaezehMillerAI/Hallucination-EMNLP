import torch
from torch_geometric.data import HeteroData
import numpy as np

class EvidenceGraphBuilder:
    """
    Builds a heterogeneous cross-modal evidence graph using PyTorch Geometric (PyG).
    Nodes: claim, texttoken, visualpatch, latentstate
    """
    def __init__(self, hidden_dim: int = 1536, num_layers: int = 28):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def build_graph(self, vlm_outputs: dict, claims: list, token_dynamics: dict, cross_attentions: torch.Tensor) -> HeteroData:
        """
        Constructs a HeteroData object from VLM inputs/outputs and atomic claims.
        
        Args:
            vlm_outputs (dict): Outputs from VLMWrapper
            claims (list): Decomposed claims from claim_decomposer
            token_dynamics (dict): Confidence, entropy, and margins per token
            cross_attentions (Tensor): Layerwise cross-attentions (num_layers, seq_len, num_patches)
        """
        data = HeteroData()
        
        # -------------------------------------------------------------
        # 1. Node Extraction & Features
        # -------------------------------------------------------------
        
        # --- texttoken nodes ---
        tokens = vlm_outputs["tokens"]
        seq_len = len(tokens)
        
        # Stack texttoken embeddings from the last layer of hidden states
        # hidden_states[-1] shape: (1, seq_len, hidden_dim)
        last_hidden = vlm_outputs["hidden_states"][-1][0] # (seq_len, hidden_dim)
        
        # Prepare token features: [embedding (hidden_dim), entropy (1), margin (1), position (1)]
        entropies = torch.tensor(token_dynamics["entropies"], dtype=torch.float32).unsqueeze(-1)
        margins = torch.tensor(token_dynamics["margins"], dtype=torch.float32).unsqueeze(-1)
        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1) / max(seq_len, 1)
        
        token_features = torch.cat([last_hidden, entropies, margins, positions], dim=-1)
        data["texttoken"].x = token_features # Shape: (seq_len, hidden_dim + 3)
        
        # --- visualpatch nodes ---
        # vision_embeddings shape: (1, num_patches, hidden_dim)
        patch_embeds = vlm_outputs["vision_embeddings"][0] # (num_patches, hidden_dim)
        num_patches = patch_embeds.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        # Prepare patch features: [embedding (hidden_dim), x_coord (1), y_coord (1)]
        patch_coords = []
        for i in range(num_patches):
            row = i // grid_size
            col = i % grid_size
            patch_coords.append([row / grid_size, col / grid_size])
        patch_coords = torch.tensor(patch_coords, dtype=torch.float32)
        
        patch_features = torch.cat([patch_embeds, patch_coords], dim=-1)
        data["visualpatch"].x = patch_features # Shape: (num_patches, hidden_dim + 2)
        
        # --- latentstate nodes ---
        # Compress layers of hidden states to summaries: mean across sequence dimension
        # hidden_states is tuple of (layers + 1), each (1, seq_len, hidden_dim)
        layer_states = []
        for h in vlm_outputs["hidden_states"]:
            layer_states.append(h[0].mean(dim=0)) # (hidden_dim,)
        layer_states = torch.stack(layer_states, dim=0) # (num_layers + 1, hidden_dim)
        
        # Prepare latentstate features: [embedding (hidden_dim), layer_idx (1)]
        layer_indices = torch.arange(len(layer_states), dtype=torch.float32).unsqueeze(-1) / len(layer_states)
        latent_features = torch.cat([layer_states, layer_indices], dim=-1)
        data["latentstate"].x = latent_features # Shape: (num_layers + 1, hidden_dim + 1)
        
        # --- claim nodes ---
        # Pool texttoken features for tokens falling inside each claim span
        claim_features = []
        claim_type_mapping = {"finding_presence": 0, "anatomical_localization": 1, "normality_statement": 2}
        
        for claim in claims:
            # Map character spans back to token indexes (rough approximation by splits)
            span = claim["token_span"]
            # Map start, end char span to corresponding tokens
            # For simplicity, divide prompt length by token indices to find overlapping tokens
            start_tok = int((span[0] / max(span[1], 1)) * seq_len)
            end_tok = max(start_tok + 1, int((span[1] / max(span[1], 1)) * seq_len))
            
            # Pool token hidden states in span
            span_hidden = last_hidden[start_tok:end_tok].mean(dim=0)
            
            # Claim attributes: claim_type (3-dim one hot), uncertainty (1-dim score), negation (1-dim bool)
            c_type = [0.0, 0.0, 0.0]
            idx = claim_type_mapping.get(claim["claim_type"], 0)
            c_type[idx] = 1.0
            
            uncertainty_val = claim["uncertainty"]["uncertainty_score"]
            negation_val = 1.0 if claim["negation"] else 0.0
            
            meta_attrs = torch.tensor(c_type + [uncertainty_val, negation_val], dtype=torch.float32)
            c_feat = torch.cat([span_hidden, meta_attrs], dim=-1)
            claim_features.append(c_feat)
            
        if not claim_features:
            # Fallback if no claims decomposed
            data["claim"].x = torch.zeros(1, self.hidden_dim + 5)
        else:
            data["claim"].x = torch.stack(claim_features, dim=0) # Shape: (num_claims, hidden_dim + 5)
            
        # -------------------------------------------------------------
        # 2. Edge Definitions
        # -------------------------------------------------------------
        
        # --- claim -> has_token -> texttoken ---
        claim_to_token_edges = []
        for c_idx, claim in enumerate(claims):
            span = claim["token_span"]
            start_tok = int((span[0] / max(span[1], 1)) * seq_len)
            end_tok = max(start_tok + 1, int((span[1] / max(span[1], 1)) * seq_len))
            for t_idx in range(start_tok, min(end_tok, seq_len)):
                claim_to_token_edges.append([c_idx, t_idx])
                
        if not claim_to_token_edges:
            claim_to_token_edges = [[0, 0]]
            
        data["claim", "has_token", "texttoken"].edge_index = torch.tensor(claim_to_token_edges, dtype=torch.long).t().contiguous()
        
        # --- texttoken -> attends_to -> texttoken ---
        # Fully connected token self-attention for simplicity (since transformer layers are dense)
        tok_self_edges = []
        for i in range(seq_len):
            for j in range(seq_len):
                tok_self_edges.append([i, j])
        data["texttoken", "attends_to", "texttoken"].edge_index = torch.tensor(tok_self_edges, dtype=torch.long).t().contiguous()
        
        # --- texttoken -> attends_to -> visualpatch ---
        # Add edges from tokens to patches based on cross-attention weights
        token_to_patch_edges = []
        token_to_patch_attr = []
        
        # Max pool cross attention across layers to find peak connections
        max_cross_att = cross_attentions.max(dim=0)[0]  # (seq_len, num_patches)
        
        for t_idx in range(seq_len):
            # Connect token to top K highly-attended visual patches
            top_val, top_idx = torch.topk(max_cross_att[t_idx], k=min(5, num_patches))
            for val, p_idx in zip(top_val, top_idx):
                token_to_patch_edges.append([t_idx, p_idx.item()])
                token_to_patch_attr.append([val.item()])
                
        if not token_to_patch_edges:
            token_to_patch_edges = [[0, 0]]
            token_to_patch_attr = [[0.0]]
            
        data["texttoken", "attends_to", "visualpatch"].edge_index = torch.tensor(token_to_patch_edges, dtype=torch.long).t().contiguous()
        data["texttoken", "attends_to", "visualpatch"].edge_attr = torch.tensor(token_to_patch_attr, dtype=torch.float32)
        
        # --- visualpatch -> adjacent_to -> visualpatch ---
        # Spatial adjacency (8-way connectivity grid)
        patch_adj = []
        for i in range(num_patches):
            r = i // grid_size
            c = i % grid_size
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        n_idx = nr * grid_size + nc
                        patch_adj.append([i, n_idx])
                        
        data["visualpatch", "adjacent_to", "visualpatch"].edge_index = torch.tensor(patch_adj, dtype=torch.long).t().contiguous()
        
        # --- texttoken -> derived_from -> latentstate ---
        # Connect token to its provenance layer states
        tok_latent_edges = []
        for t_idx in range(seq_len):
            for l_idx in range(self.num_layers + 1):
                tok_latent_edges.append([t_idx, l_idx])
        data["texttoken", "derived_from", "latentstate"].edge_index = torch.tensor(tok_latent_edges, dtype=torch.long).t().contiguous()
        
        # --- claim -> grounded_in -> visualpatch ---
        # Map claims to supportive visual patches by pooling cross-attentions of claim tokens
        claim_patch_edges = []
        claim_patch_attr = []
        for c_idx, claim in enumerate(claims):
            span = claim["token_span"]
            start_tok = int((span[0] / max(span[1], 1)) * seq_len)
            end_tok = max(start_tok + 1, int((span[1] / max(span[1], 1)) * seq_len))
            
            # Pool token attention vectors across the claim span
            claim_att = max_cross_att[start_tok:end_tok].mean(dim=0) # (num_patches,)
            
            # Connect claim to top supportive patches
            top_val, top_idx = torch.topk(claim_att, k=min(3, num_patches))
            for val, p_idx in zip(top_val, top_idx):
                claim_patch_edges.append([c_idx, p_idx.item()])
                claim_patch_attr.append([val.item()])
                
        if not claim_patch_edges:
            claim_patch_edges = [[0, 0]]
            claim_patch_attr = [[0.0]]
            
        data["claim", "grounded_in", "visualpatch"].edge_index = torch.tensor(claim_patch_edges, dtype=torch.long).t().contiguous()
        data["claim", "grounded_in", "visualpatch"].edge_attr = torch.tensor(claim_patch_attr, dtype=torch.float32)
        
        # --- claim -> conflicts -> claim ---
        # Add conflict edges if two claims target the same anatomy but disagree on polarity (negation)
        claim_conflict_edges = []
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                if claims[i]["anatomy"] == claims[j]["anatomy"] and claims[i]["negation"] != claims[j]["negation"]:
                    claim_conflict_edges.append([i, j])
                    claim_conflict_edges.append([j, i])
                    
        if not claim_conflict_edges:
            # Add self-loops as fallback placeholder to keep PyG compiler happy
            claim_conflict_edges = [[0, 0]]
            
        data["claim", "conflicts", "claim"].edge_index = torch.tensor(claim_conflict_edges, dtype=torch.long).t().contiguous()
        
        return data

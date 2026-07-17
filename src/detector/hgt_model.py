import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear

class GNNHallucinationDetector(nn.Module):
    """
    A Heterogeneous Graph Transformer (HGT) model for VLM hallucination detection.
    Aggregates node embeddings across claims, tokens, patches, and latent states,
    and uses multi-task heads to predict claim-level hallucination properties.
    """
    def __init__(self, metadata: tuple, hidden_channels: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        
        # Projection layers to map different node feature sizes to a common GNN hidden dimension
        #claim: hidden_dim (1536) + 5 = 1541
        #texttoken: hidden_dim (1536) + 3 = 1539
        #visualpatch: hidden_dim (1536) + 2 = 1538
        #latentstate: hidden_dim (1536) + 1 = 1537
        
        self.proj_claim = nn.Linear(1541, hidden_channels)
        self.proj_token = nn.Linear(1539, hidden_channels)
        self.proj_patch = nn.Linear(1538, hidden_channels)
        self.proj_latent = nn.Linear(1537, hidden_channels)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads))
            
        # Multi-task heads for claim nodes
        self.head_hallucination = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)  # Binary BCE logit
        )
        
        self.head_cause = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 4)  # 4-class classification logit
        )
        
        self.head_sufficiency = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)  # Regression
        )
        
        self.head_faithfulness = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)  # Regression
        )
        
        self.head_localization = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)  # Regression (IoU-like overlap score)
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """
        Runs the HGT forward pass.
        
        Args:
            x_dict (dict of str: Tensor): Node feature matrices per node type
            edge_index_dict (dict of tuple: Tensor): Edge index matrices per edge type
        Returns:
            dict containing claim-level predictions:
              - 'hallucination': (num_claims, 1)
              - 'cause': (num_claims, 4)
              - 'sufficiency': (num_claims, 1)
              - 'faithfulness': (num_claims, 1)
              - 'localization': (num_claims, 1)
        """
        # 1. Project node features to common dimension
        h_dict = {}
        if "claim" in x_dict:
            h_dict["claim"] = self.proj_claim(x_dict["claim"])
        if "texttoken" in x_dict:
            h_dict["texttoken"] = self.proj_token(x_dict["texttoken"])
        if "visualpatch" in x_dict:
            h_dict["visualpatch"] = self.proj_patch(x_dict["visualpatch"])
        if "latentstate" in x_dict:
            h_dict["latentstate"] = self.proj_latent(x_dict["latentstate"])
            
        # 2. Apply GNN convolutions
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            
        # 3. Apply prediction heads to claim nodes
        claim_embeddings = h_dict["claim"]
        
        return {
            "hallucination": self.head_hallucination(claim_embeddings),
            "cause": self.head_cause(claim_embeddings),
            "sufficiency": self.head_sufficiency(claim_embeddings),
            "faithfulness": self.head_faithfulness(claim_embeddings),
            "localization": self.head_localization(claim_embeddings)
        }

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, to_hetero

class HomogeneousGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GNNHallucinationDetector(nn.Module):
    """
    A Heterogeneous Graph Neural Network model for VLM hallucination detection.
    Aggregates node embeddings across claims, tokens, patches, and latent states
    using a stable GraphSAGE compilation, predicting claim properties via multi-task heads.
    """
    def __init__(self, metadata: tuple, hidden_channels: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        
        # Projection layers to map different node feature sizes to a common GNN hidden dimension
        self.proj_claim = nn.Linear(1541, hidden_channels)
        self.proj_token = nn.Linear(1539, hidden_channels)
        self.proj_patch = nn.Linear(1538, hidden_channels)
        self.proj_latent = nn.Linear(1537, hidden_channels)
        
        # Build stable GraphSAGE GNN and compile to HeteroGNN
        gnn = HomogeneousGNN(hidden_channels)
        self.hetero_gnn = to_hetero(gnn, metadata, aggr='sum')
        
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
        Runs the forward pass.
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
            
        # 2. Apply GNN convolutions (stable GraphSAGE compilation)
        h_dict = self.hetero_gnn(h_dict, edge_index_dict)
            
        # 3. Apply prediction heads to claim nodes
        claim_embeddings = h_dict["claim"]
        
        return {
            "hallucination": self.head_hallucination(claim_embeddings),
            "cause": self.head_cause(claim_embeddings),
            "sufficiency": self.head_sufficiency(claim_embeddings),
            "faithfulness": self.head_faithfulness(claim_embeddings),
            "localization": self.head_localization(claim_embeddings)
        }

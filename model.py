# ============================================================
# KG-CAGNN: Correlation-Aware Knowledge Graph Attention GNN
# ============================================================
# Main Improvements:
# ------------------------------------------------------------
# 1. Explicit structural/failure correlation decomposition
# 2. Multi-head correlation-aware attention
# 3. Relation-specific propagation
# 4. Adaptive aggregation gate
# 5. Contrastive representation learning
# 6. Stable heterogeneous propagation
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv

class RelationAwareAttention(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        heads=4,
        dropout=0.2
    ):
        super().__init__()

        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            concat=True,
            dropout=dropout
        )

        self.proj = nn.Linear(
            out_dim * heads,
            out_dim
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):

        h = self.gat(x, edge_index)

        h = self.proj(h)

        h = self.norm(h)

        return F.relu(h)

class CorrelationGate(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_s, h_f):

        g = self.gate(
            torch.cat([h_s, h_f], dim=-1)
        )

        h = g * h_s + (1 - g) * h_f

        return h, g

class KGCAGNN(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_heads=4,
        dropout=0.2
    ):
        super().__init__()

        self.structural_encoder = RelationAwareAttention(
            in_dim=in_dim,
            out_dim=hidden_dim,
            heads=num_heads,
            dropout=dropout
        )

        self.failure_encoder = RelationAwareAttention(
            in_dim=in_dim,
            out_dim=hidden_dim,
            heads=num_heads,
            dropout=dropout
        )

        self.correlation_gate = CorrelationGate(
            hidden_dim
        )

        self.residual_proj = nn.Linear(
            in_dim,
            hidden_dim
        )

        self.semantic_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        edge_structural,
        edge_failure
    ):
        h_s = self.structural_encoder(
            x,
            edge_structural
        )

        h_f = self.failure_encoder(
            x,
            edge_failure
        )

        h_corr, gate_weights = self.correlation_gate(
            h_s,
            h_f
        )

        h_res = self.residual_proj(x)

        h = h_corr + h_res

        h = F.relu(h)

        h = self.dropout(h)

        z = self.semantic_proj(h)
        logits = self.classifier(h)

        return logits, z, gate_weights

    def contrastive_loss(
        self,
        z,
        edge_index
    ):

        row, col = edge_index

        positive_loss = F.mse_loss(
            z[row],
            z[col]
        )

        return positive_loss

    def correlation_regularization(
        self,
        gate_weights
    ):

        # Encourage sparse adaptive gating

        reg = torch.mean(
            gate_weights * (1 - gate_weights)
        )

        return reg

if __name__ == "__main__":

    NUM_NODES = 500
    INPUT_DIM = 128
    HIDDEN_DIM = 64
    NUM_CLASSES = 16

    x = torch.randn(NUM_NODES, INPUT_DIM)


    edge_structural = torch.randint(
        0,
        NUM_NODES,
        (2, 4000)
    )

    edge_failure = torch.randint(
        0,
        NUM_NODES,
        (2, 4000)
    )

    model = KGCAGNN(
        in_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=NUM_CLASSES,
        num_heads=4
    )

    logits, z, gates = model(
        x,
        edge_structural,
        edge_failure
    )

    print("Logits Shape :", logits.shape)
    print("Embedding Shape :", z.shape)
    print("Gate Shape :", gates.shape)

    # Losses
    cls_loss = F.cross_entropy(
        logits,
        torch.randint(0, NUM_CLASSES, (NUM_NODES,))
    )

    con_loss = model.contrastive_loss(
        z,
        edge_failure
    )

    reg_loss = model.correlation_regularization(
        gates
    )

    total_loss = (
        cls_loss +
        0.01 * con_loss +
        0.001 * reg_loss
    )

    print("Total Loss :", total_loss.item())

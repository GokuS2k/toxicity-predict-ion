"""
Graph Attention Network (GAT) for multi-task molecular toxicity prediction.

Architecture
────────────
  Input graph  →  3 × GATv2Conv (with edge features)
               →  Global mean + max pooling  (graph readout)
               →  Shared MLP trunk
               →  12 task-specific linear heads

Each head outputs a raw logit; BCEWithLogitsLoss is applied during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

from featurization import NUM_NODE_FEATURES, NUM_EDGE_FEATURES


class MolecularGNN(nn.Module):
    """
    GAT-based multi-task classifier for molecular properties.

    Parameters
    ----------
    num_node_features : int   Input node feature dimension (73).
    num_edge_features : int   Input edge feature dimension (10).
    hidden_dim        : int   Hidden dimension per attention head.
    num_heads         : int   Number of attention heads in GAT layers.
    num_layers        : int   Number of GATv2Conv message-passing layers.
    dropout           : float Dropout probability applied after each layer.
    num_tasks         : int   Number of output tasks (12 for Tox21).
    """

    def __init__(
        self,
        num_node_features: int = NUM_NODE_FEATURES,
        num_edge_features: int = NUM_EDGE_FEATURES,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_tasks: int = 12,
    ):
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        # ── Input projection ────────────────────────────────────────────
        # Project raw node features to hidden_dim before GAT layers.
        self.input_proj = nn.Linear(num_node_features, hidden_dim)

        # ── GATv2 message-passing layers ─────────────────────────────────
        # Layer 0: hidden_dim → hidden_dim * num_heads  (concat=True)
        # Layers 1…n-2: same
        # Last layer: hidden_dim * num_heads → hidden_dim (concat=False, mean)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = hidden_dim
        for i in range(num_layers):
            is_last = i == num_layers - 1
            concat = not is_last
            out_heads = 1 if is_last else num_heads
            out_dim = hidden_dim if is_last else hidden_dim * num_heads

            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=out_heads,
                    concat=concat,
                    edge_dim=num_edge_features,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.BatchNorm1d(out_dim))
            in_dim = out_dim

        # After GAT: in_dim == hidden_dim (last layer concat=False)
        graph_dim = hidden_dim * 2  # mean_pool + max_pool concatenated

        # ── Shared MLP trunk ─────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(graph_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Task-specific output heads ────────────────────────────────────
        self.task_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(num_tasks)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        """
        Parameters
        ----------
        data : PyG Batch
            `.x`          [N_total, node_feat]
            `.edge_index` [2, E_total]
            `.edge_attr`  [E_total, edge_feat]
            `.batch`      [N_total]

        Returns
        -------
        out : torch.Tensor  [B, num_tasks]  (raw logits)
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Input projection
        x = self.input_proj(x)
        x = F.elu(x)

        # GATv2 layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph readout: concatenate mean and max pooling
        x_mean = global_mean_pool(x, batch)   # [B, hidden_dim]
        x_max = global_max_pool(x, batch)     # [B, hidden_dim]
        x = torch.cat([x_mean, x_max], dim=1) # [B, 2*hidden_dim]

        # Shared MLP
        x = self.mlp(x)                        # [B, 128]

        # Task heads → raw logits
        out = torch.cat([head(x) for head in self.task_heads], dim=1)  # [B, 12]
        return out

    def predict_proba(self, data) -> torch.Tensor:
        """Returns sigmoid probabilities [B, 12] (no_grad, eval mode guard)."""
        with torch.no_grad():
            logits = self.forward(data)
        return torch.sigmoid(logits)


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Binary cross-entropy loss that ignores NaN entries in `targets`.

    Parameters
    ----------
    logits     : [B, T]  raw model output
    targets    : [B, T]  ground-truth labels (0.0 / 1.0 / NaN)
    pos_weight : [T]     positive class weight per task (optional)

    Returns
    -------
    Scalar mean loss over all known (non-NaN) label entries.
    """
    mask = ~torch.isnan(targets)

    if not mask.any():
        return logits.sum() * 0.0  # differentiable zero

    # Replace NaN with 0 so we can compute BCE safely; mask zeros out the result
    safe_targets = targets.clone()
    safe_targets[~mask] = 0.0

    # Compute element-wise BCE (without pos_weight argument to avoid shape issues)
    loss_elem = F.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none"
    )

    if pos_weight is not None:
        # Manually apply per-task positive weighting: weight = 1 + (pw-1)*target
        pw = pos_weight.to(logits.device)           # [T]
        # Broadcast pw to [B, T]: weight up positive labels, keep negatives at 1
        sample_weight = 1.0 + (pw - 1.0) * safe_targets  # [B, T]
        loss_elem = loss_elem * sample_weight

    return (loss_elem * mask.float()).sum() / mask.float().sum()


def compute_pos_weights(dataset, num_tasks: int = 12) -> torch.Tensor:
    """
    Compute per-task positive class weights as (neg_count / pos_count),
    clipped to [1, 50] to avoid extreme values.
    """
    counts_pos = torch.zeros(num_tasks)
    counts_neg = torch.zeros(num_tasks)
    for item in dataset:
        y = item.y.squeeze(0)   # [1, 12] → [12]
        for t in range(num_tasks):
            if not torch.isnan(y[t]):
                if y[t] > 0.5:
                    counts_pos[t] += 1
                else:
                    counts_neg[t] += 1
    weights = counts_neg / (counts_pos + 1e-8)
    return weights.clamp(1.0, 50.0)

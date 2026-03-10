"""
GATv2 multi-task molecular toxicity classifier.

Architecture:
  Input → Linear projection →
  3 × GATv2Conv(8 heads, hidden=128, edge_dim=12) + BatchNorm + ReLU + Dropout(0.3) →
  Global mean pool ⊕ Global max pool  →  [256-dim] →
  Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→12)  →  12 raw logits

GATv2 computes attention AFTER applying learned linear transforms to both
source and target node features, making it strictly more expressive than GAT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

NUM_NODE_FEATURES = 75
NUM_EDGE_FEATURES = 12
NUM_TASKS = 12


class GATv2Model(nn.Module):
    """
    GATv2-based multi-task classifier for molecular graphs.

    Parameters
    ----------
    num_node_features : int   Node feature dimension (default 75).
    num_edge_features : int   Edge feature dimension (default 12).
    hidden_dim        : int   Dimension per attention head (16); total = 8*16 = 128.
    num_heads         : int   Attention heads per layer (8).
    num_layers        : int   Number of GATv2Conv layers (3).
    dropout           : float Dropout rate (0.3).
    num_tasks         : int   Number of output tasks (12).
    """

    def __init__(
        self,
        num_node_features: int = NUM_NODE_FEATURES,
        num_edge_features: int = NUM_EDGE_FEATURES,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_tasks: int = NUM_TASKS,
    ):
        super().__init__()
        self.dropout_p = dropout
        self.num_layers = num_layers

        # Each GATv2Conv head outputs hidden_dim/num_heads dims; concat → hidden_dim
        head_dim = hidden_dim // num_heads  # 16

        # Input projection: raw node features → hidden_dim
        self.input_proj = nn.Linear(num_node_features, hidden_dim)

        # GATv2Conv layers: all concat=True except the last which uses mean
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = hidden_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            # All layers use concat=True so output = num_heads * head_dim = hidden_dim
            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    concat=True,           # output dim = num_heads * head_dim = hidden_dim
                    edge_dim=num_edge_features,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim  # stays constant across layers

        # Global readout: concat(mean_pool, max_pool) → 2 * hidden_dim
        graph_dim = hidden_dim * 2  # 256

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(graph_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_tasks),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Batch with .x, .edge_index, .edge_attr, .batch

        Returns
        -------
        logits : torch.Tensor [B, num_tasks]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch      = data.batch

        # Input projection
        x = F.elu(self.input_proj(x))

        # GATv2 message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
            # Residual connection (input and output dims match after first layer)
            if residual.shape == x.shape:
                x = x + residual

        # Graph-level readout
        x_mean = global_mean_pool(x, batch)          # [B, hidden_dim]
        x_max  = global_max_pool(x, batch)           # [B, hidden_dim]
        x_graph = torch.cat([x_mean, x_max], dim=1) # [B, 2*hidden_dim]

        # Classification head → raw logits
        return self.classifier(x_graph)              # [B, num_tasks]

    def predict_proba(self, data) -> torch.Tensor:
        """Sigmoid probabilities [B, num_tasks], no_grad."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(data))


# ── Loss and class weights ───────────────────────────────────────────────────

def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Binary cross-entropy loss with masking for NaN labels.

    Only non-NaN entries contribute to the loss. This is critical for Tox21
    where ~17% of labels are missing.

    Parameters
    ----------
    logits     : [B, T]  raw model logits
    targets    : [B, T]  labels (0.0 / 1.0 / NaN)
    pos_weight : [T]     per-task positive class weight

    Returns
    -------
    Scalar mean loss over all known label entries.
    """
    mask = ~torch.isnan(targets)
    if not mask.any():
        return logits.sum() * 0.0  # differentiable zero

    # Zero-fill NaN for safe BCE computation; mask zeros out their contribution
    safe_targets = targets.clone()
    safe_targets[~mask] = 0.0

    loss_elem = F.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none"
    )

    if pos_weight is not None:
        pw = pos_weight.to(logits.device)                        # [T]
        sample_weight = 1.0 + (pw - 1.0) * safe_targets         # [B, T]
        loss_elem = loss_elem * sample_weight

    return (loss_elem * mask.float()).sum() / mask.float().sum()


def compute_pos_weights(graph_list: list, num_tasks: int = NUM_TASKS) -> torch.Tensor:
    """
    Per-task positive class weights: neg_count / pos_count, clipped to [1, 50].
    """
    counts_pos = torch.zeros(num_tasks)
    counts_neg = torch.zeros(num_tasks)
    for g in graph_list:
        y = g.y.squeeze(0)  # [12]
        for t in range(num_tasks):
            if not torch.isnan(y[t]):
                if y[t] > 0.5:
                    counts_pos[t] += 1
                else:
                    counts_neg[t] += 1
    weights = counts_neg / (counts_pos + 1e-8)
    return weights.clamp(1.0, 50.0)

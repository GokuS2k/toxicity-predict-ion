"""
Directed Message Passing Neural Network (D-MPNN) for molecular toxicity prediction.

Implements the core idea from Chemprop (Yang et al., 2019):
- Messages are passed along DIRECTED edges (each bond → two directed edges).
- During aggregation at node v, the message from edge (u→v) is excluded
  when computing the message for edge (v→u). This avoids the "echo" problem
  where a node immediately bounces its own information back.

Architecture:
  Edge init:  h_e = W_i · [x_u || x_e]        for each directed edge (u→v)
  Iter:       m_e = sum_{w→u, w≠v} h_e'        (exclude reverse edge)
              h_e = ReLU(h_init_e + W_m · m_e)
  Node read:  h_v = ReLU(W_a · [x_v || sum_{u→v} h_e])
  Pool:       h_G = mean + max pool, concat
  Output:     MLP(h_G) → 12 logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

NUM_NODE_FEATURES = 75
NUM_EDGE_FEATURES = 12
NUM_TASKS = 12


class DMPNNModel(nn.Module):
    """
    Directed MPNN multi-task classifier.

    Parameters
    ----------
    num_node_features : int   Atom feature dim (75).
    num_edge_features : int   Bond feature dim (12).
    hidden_dim        : int   Message/hidden dimension (300).
    num_layers        : int   Message passing steps (3).
    dropout           : float Dropout rate (0.15).
    num_tasks         : int   Output tasks (12).
    """

    def __init__(
        self,
        num_node_features: int = NUM_NODE_FEATURES,
        num_edge_features: int = NUM_EDGE_FEATURES,
        hidden_dim: int = 300,
        num_layers: int = 3,
        dropout: float = 0.15,
        num_tasks: int = NUM_TASKS,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.dropout_p   = dropout

        # Initial edge embedding: concatenate src-atom features + bond features
        self.W_i = nn.Linear(num_node_features + num_edge_features, hidden_dim, bias=False)

        # Message aggregation weight (applied after summing incoming messages)
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Node-level aggregation: [atom_features || sum_edge_hidden] → hidden
        self.W_a = nn.Linear(num_node_features + hidden_dim, hidden_dim)

        # Classification head
        graph_dim = hidden_dim * 2   # mean + max pool
        self.ffn = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
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
        logits : [B, num_tasks]
        """
        x          = data.x           # [N_total, node_feat]
        edge_index = data.edge_index  # [2, E_total]  (bidirectional, E_total = 2*bonds)
        edge_attr  = data.edge_attr   # [E_total, edge_feat]
        batch      = data.batch       # [N_total]

        src, dst = edge_index[0], edge_index[1]  # source and destination nodes

        # ── Step 1: Initial directed edge embeddings ─────────────────────────
        # h_e^0 = ReLU(W_i · [x_src || edge_attr])
        h_init = F.relu(self.W_i(torch.cat([x[src], edge_attr], dim=1)))  # [E, H]
        h_e = h_init.clone()

        # ── Step 2: Directed message passing ─────────────────────────────────
        # For each directed edge e = (u→v), aggregate messages from all edges
        # (w→u) EXCEPT the reverse edge (v→u).
        #
        # Implementation:
        #   node_agg[v] = sum of h_e for all edges e pointing TO v
        #   message for edge (u→v) = node_agg[u] - h_e(v→u)
        #
        # This requires the reverse edge index. Since edges are added in pairs
        # (i→j at position 2k, j→i at position 2k+1), the reverse of edge k is:
        #   if k is even: k+1; if k is odd: k-1
        # i.e., reverse_idx[k] = k ^ 1

        num_edges = edge_index.size(1)
        reverse_idx = torch.arange(num_edges, device=x.device)
        reverse_idx = reverse_idx ^ 1  # XOR with 1 flips last bit: 0↔1, 2↔3, ...

        for _ in range(self.num_layers):
            # Aggregate: for each node u, sum all edge hidden states pointing TO u
            num_nodes = x.size(0)
            node_msg = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
            node_msg.scatter_add_(
                0,
                dst.unsqueeze(1).expand(-1, self.hidden_dim),
                h_e,
            )  # node_msg[v] = sum_{(u→v)} h_e

            # Directed message for edge (u→v):
            # m_{u→v} = node_msg[u] - h_{v→u}  (exclude reverse echo)
            m_e = node_msg[src] - h_e[reverse_idx]  # [E, H]

            # Update edge hidden state (with residual connection to init)
            h_e = F.relu(h_init + self.W_m(m_e))    # [E, H]
            if self.training:
                h_e = F.dropout(h_e, p=self.dropout_p)

        # ── Step 3: Node-level readout ────────────────────────────────────────
        # For each node v, aggregate all incoming edge messages and combine
        # with the atom's own features.
        num_nodes = x.size(0)
        node_agg = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        node_agg.scatter_add_(
            0,
            dst.unsqueeze(1).expand(-1, self.hidden_dim),
            h_e,
        )  # node_agg[v] = sum_{(u→v)} h_e_final

        h_v = F.relu(self.W_a(torch.cat([x, node_agg], dim=1)))  # [N, H]
        if self.training:
            h_v = F.dropout(h_v, p=self.dropout_p)

        # ── Step 4: Graph readout & classification ────────────────────────────
        h_mean  = global_mean_pool(h_v, batch)               # [B, H]
        h_max   = global_max_pool(h_v, batch)                # [B, H]
        h_graph = torch.cat([h_mean, h_max], dim=1)         # [B, 2H]

        return self.ffn(h_graph)                             # [B, num_tasks]

    def predict_proba(self, data) -> torch.Tensor:
        """Sigmoid probabilities [B, num_tasks], no_grad."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(data))


# Re-export shared loss/weights from gatv2 module for convenience
from models.gatv2 import masked_bce_loss, compute_pos_weights  # noqa: F401, E402

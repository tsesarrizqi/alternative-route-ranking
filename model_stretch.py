import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class ODSubgraphGNN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=7):
        super().__init__()
        self.edge_node_encoder = nn.Linear(5, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = 0.1
        
    def forward(self, edge_node_feat, edge_edge_index):
        x = self.edge_node_encoder(edge_node_feat)
        x = self.input_norm(x)
        
        for layer, norm in zip(self.gnn_layers, self.layer_norms):
            x_res = x
            x = layer(x, edge_edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res
        
        return x

class StretchRNN(nn.Module):
    def __init__(self, hidden_dim=128, gnn_hidden_dim=128, enable_gnn_od=False,
                 epsilon=0.0, num_rnn_layers=2, dropout=0.2, no_cumul=False, pool_only=False):
        super().__init__()

        input_dim = 6
        if epsilon > 0:
            input_dim += 1
        if enable_gnn_od:
            input_dim += gnn_hidden_dim

        self.num_rnn_layers = num_rnn_layers
        self.dropout = dropout

        self.rnn_cells_forward = nn.ModuleList()
        self.rnn_cells_forward.append(nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim))
        for _ in range(num_rnn_layers - 1):
            self.rnn_cells_forward.append(nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim))

        self.rnn_cells_backward = nn.ModuleList()
        self.rnn_cells_backward.append(nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim))
        for _ in range(num_rnn_layers - 1):
            self.rnn_cells_backward.append(nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim))

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_norm_b = nn.LayerNorm(input_dim)

        self.hidden_dim = hidden_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.epsilon = 1.0 + epsilon
        self.enable_gnn_od = enable_gnn_od
        self.no_cumul = no_cumul
        self.pool_only = pool_only
        self.output_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, batch, graph_data, edge_features, od_node_embeddings):
        device = graph_data.node_feat.device
        edge_lengths = graph_data.edge_feat.squeeze(-1)

        batch_size = edge_features.shape[0]
        max_seq_len = edge_features.shape[1]

        flat_node_sequences = batch["node_sequences"]
        flat_edge_indices = batch["edge_indices"]
        path2pair = batch["path2pair"]
        od_pairs = batch["od_pairs"]
        subpath_indices_dict = batch.get("subpath_indices", {})

        # --- Vectorized cumulative distances (eliminates per-step .item() GPU syncs) ---
        edge_mask = flat_edge_indices >= 0                               # [B, L]
        safe_idx = flat_edge_indices.clamp(min=0)                        # [B, L]
        path_lengths = edge_lengths[safe_idx] * edge_mask.float()        # [B, L]

        # Forward cumsum: cumulative_from_start[i, j] = sum of edges 0..j
        cumulative_from_start = torch.cumsum(path_lengths, dim=1)        # [B, L]

        # Backward suffix sum: total - prefix_shifted
        total_path_length = cumulative_from_start[:, -1]                 # [B]
        prefix_shifted = torch.cat([
            torch.zeros(batch_size, 1, device=device),
            cumulative_from_start[:, :-1]
        ], dim=1)                                                         # [B, L]
        cumulative_from_end = (
            total_path_length.unsqueeze(1) - prefix_shifted
        ) * edge_mask.float()                                            # [B, L]

        # Use pre-computed pool distances from collate_fn (already on device via batch.to(device)).
        # Sentinel -1.0 means "not found" -> fall back to cumulative path length.
        pool_start_pre = batch.get("pool_start")
        pool_end_pre = batch.get("pool_end")

        if pool_start_pre is not None:
            # Align length dimension in case of any edge-case mismatch
            L = cumulative_from_start.shape[1]
            Lp = pool_start_pre.shape[1]
            if Lp < L:
                pad_val = torch.full((batch_size, L - Lp), -1.0, device=device)
                pool_start_pre = torch.cat([pool_start_pre, pad_val], dim=1)
                pool_end_pre = torch.cat([pool_end_pre, pad_val], dim=1)
            elif Lp > L:
                pool_start_pre = pool_start_pre[:, :L]
                pool_end_pre = pool_end_pre[:, :L]

            has_start = pool_start_pre >= 0
            has_end = pool_end_pre >= 0
            pool_shortest_from_start = torch.where(has_start, pool_start_pre, cumulative_from_start)
            pool_shortest_from_end = torch.where(has_end, pool_end_pre, cumulative_from_end)
        else:
            # Fallback: no pre-computed values, use cumulative as-is
            pool_shortest_from_start = cumulative_from_start
            pool_shortest_from_end = cumulative_from_end

        cumulative_from_start_norm = cumulative_from_start / 1000.0
        pool_from_start_norm = pool_shortest_from_start / 1000.0
        cumulative_from_end_norm = cumulative_from_end / 1000.0
        pool_from_end_norm = pool_shortest_from_end / 1000.0

        if self.no_cumul or self.pool_only:
            cumulative_from_start_norm = torch.zeros_like(cumulative_from_start_norm)
            cumulative_from_end_norm = torch.zeros_like(cumulative_from_end_norm)

        forward_features = [
            edge_features,
            cumulative_from_start_norm.unsqueeze(-1),
            pool_from_start_norm.unsqueeze(-1)
        ]

        backward_features = [
            edge_features,
            cumulative_from_end_norm.unsqueeze(-1),
            pool_from_end_norm.unsqueeze(-1)
        ]
        
        if self.epsilon > 1.0:
            epsilon_feature = torch.full((batch_size, max_seq_len), self.epsilon, device=device).unsqueeze(-1)
            forward_features.append(epsilon_feature)
            backward_features.append(epsilon_feature)

        if self.enable_gnn_od:
            combined_features_list = []
            for i in range(batch_size):
                nodes = flat_node_sequences[i]
                od_idx = path2pair[i].item()
                od_pair = od_pairs[od_idx]

                if od_pair in od_node_embeddings:
                    od_embeddings, edge_to_idx = od_node_embeddings[od_pair]

                    path_gnn_features = []
                    for j in range(max_seq_len):
                        if j < len(nodes) - 1:
                            u, v = nodes[j], nodes[j+1]
                            edge_key = (u, v)

                            if edge_key in edge_to_idx:
                                edge_idx_in_graph = edge_to_idx[edge_key]
                                gnn_feat = od_embeddings[edge_idx_in_graph]
                            else:
                                gnn_feat = torch.zeros(self.gnn_hidden_dim, device=device)
                        else:
                            gnn_feat = torch.zeros(self.gnn_hidden_dim, device=device)

                        path_gnn_features.append(gnn_feat)

                    path_gnn_features = torch.stack(path_gnn_features)
                else:
                    path_gnn_features = torch.zeros(max_seq_len, self.gnn_hidden_dim, device=device)

                combined_features_list.append(path_gnn_features)

            gnn_features = torch.stack(combined_features_list)
            forward_features.append(gnn_features)
            backward_features.append(gnn_features)

        combined_forward = self.input_norm(torch.cat(forward_features, dim=-1))
        combined_backward = self.input_norm_b(torch.cat(backward_features, dim=-1))

        edge_mask = batch["edge_indices"] >= 0
        seq_lengths = edge_mask.sum(dim=1)
        max_steps = int(seq_lengths.max().item()) if batch_size > 0 else 0

        # Pre-compute all active masks at once: [max_steps, batch_size]
        steps = torch.arange(max_steps, device=device).unsqueeze(1)  # [max_steps, 1]
        all_active_masks = steps < seq_lengths.unsqueeze(0)           # [max_steps, batch_size]

        hidden_states_f = [torch.zeros(batch_size, self.hidden_dim, device=device)
                          for _ in range(self.num_rnn_layers)]

        for step in range(max_steps):
            active_mask = all_active_masks[step]  # [batch_size]
            if active_mask.any():
                layer_input = combined_forward[:, step, :]
                new_hidden = self.rnn_cells_forward[0](layer_input, hidden_states_f[0])
                hidden_states_f[0] = torch.where(active_mask.unsqueeze(-1), new_hidden, hidden_states_f[0])

                for layer_idx in range(1, self.num_rnn_layers):
                    layer_input = hidden_states_f[layer_idx - 1]
                    if self.training and layer_idx > 0:
                        layer_input = F.dropout(layer_input, p=self.dropout, training=True)
                    new_hidden = self.rnn_cells_forward[layer_idx](layer_input, hidden_states_f[layer_idx])
                    hidden_states_f[layer_idx] = torch.where(active_mask.unsqueeze(-1), new_hidden, hidden_states_f[layer_idx])

        hidden_f = hidden_states_f[-1]

        hidden_states_b = [torch.zeros(batch_size, self.hidden_dim, device=device)
                          for _ in range(self.num_rnn_layers)]

        for step in range(max_steps):
            rev_step = max_steps - 1 - step
            active_mask = all_active_masks[rev_step]  # [batch_size]
            if active_mask.any():
                layer_input = combined_backward[:, rev_step, :]
                new_hidden = self.rnn_cells_backward[0](layer_input, hidden_states_b[0])
                hidden_states_b[0] = torch.where(active_mask.unsqueeze(-1), new_hidden, hidden_states_b[0])

                for layer_idx in range(1, self.num_rnn_layers):
                    layer_input = hidden_states_b[layer_idx - 1]
                    if self.training and layer_idx > 0:
                        layer_input = F.dropout(layer_input, p=self.dropout, training=True)
                    new_hidden = self.rnn_cells_backward[layer_idx](layer_input, hidden_states_b[layer_idx])
                    hidden_states_b[layer_idx] = torch.where(active_mask.unsqueeze(-1), new_hidden, hidden_states_b[layer_idx])

        hidden_b = hidden_states_b[-1]

        return self.output_norm(torch.cat([hidden_f, hidden_b], dim=1))

class PathStretchScoringModel(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1, enable_gnn_od=False, epsilon=0.0, no_cumul=False, pool_only=False):
        super().__init__()
        
        if enable_gnn_od:
            self.od_gnn = ODSubgraphGNN(hidden_dim=hidden_dim, num_layers=7)
        else:
            self.od_gnn = None

        self.stretch_rnn = StretchRNN(hidden_dim=hidden_dim, gnn_hidden_dim=hidden_dim, enable_gnn_od=enable_gnn_od, epsilon=epsilon, no_cumul=no_cumul, pool_only=pool_only)
        
        self.stretch_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.enable_gnn_od = enable_gnn_od
        
    def compute_main_path_sharing(self, edge_indices, main_paths_dict, path2pair, edge_mask):
        device = edge_indices.device
        
        shared_mask = torch.zeros_like(edge_indices, dtype=torch.bool)
        
        unique_od_pairs = torch.unique(path2pair)
        
        for od_idx in unique_od_pairs:
            od_idx_item = od_idx.item()
            
            if od_idx_item not in main_paths_dict or len(main_paths_dict[od_idx_item]) == 0:
                continue
            
            main_edges = main_paths_dict[od_idx_item]
            if isinstance(main_edges, list):
                main_edges = torch.tensor(main_edges, device=device, dtype=torch.long)
            
            path_mask = path2pair == od_idx
            
            max_edge_idx = max(edge_indices.max().item(), main_edges.max().item()) + 1
            is_main_edge = torch.zeros(max_edge_idx, device=device, dtype=torch.bool)
            is_main_edge[main_edges] = True
            
            od_edge_indices = edge_indices[path_mask]
            valid_edges = edge_mask[path_mask]
            safe_indices = od_edge_indices.clamp(min=0, max=max_edge_idx-1)
            
            is_shared = is_main_edge[safe_indices] & valid_edges
            shared_mask[path_mask] = is_shared
        
        return shared_mask

    def compute_edge_features(self, graph_data, batch):
        edge_lengths = graph_data.edge_feat.squeeze(-1)
        path_idx = batch["path_idx"]
        edge_mask = path_idx >= 0
        safe_idx = path_idx.clamp(min=0)

        edge_lengths_masked = edge_lengths[safe_idx] * edge_mask.float()

        shared_mask = self.compute_main_path_sharing(
            path_idx, batch["main_paths"], batch["path2pair"], edge_mask
        )

        sharing_feature = shared_mask.float()

        batch_size = edge_lengths_masked.shape[0]
        max_seq_len = edge_lengths_masked.shape[1]

        relative_pos_start = torch.zeros(batch_size, max_seq_len, device=edge_lengths.device)
        relative_pos_end = torch.zeros(batch_size, max_seq_len, device=edge_lengths.device)

        for i in range(batch_size):
            seq_len = edge_mask[i].sum().item()
            if seq_len > 0:
                positions = torch.arange(seq_len, device=edge_lengths.device, dtype=torch.float32)
                relative_pos_start[i, :seq_len] = positions / seq_len
                relative_pos_end[i, :seq_len] = (seq_len - positions - 1) / seq_len

        if self.stretch_rnn.no_cumul or self.stretch_rnn.pool_only:
            edge_lengths_for_feat = torch.zeros_like(edge_lengths_masked)
        else:
            edge_lengths_for_feat = edge_lengths_masked

        if self.stretch_rnn.pool_only:
            sharing_feature = torch.zeros_like(sharing_feature)
            relative_pos_start = torch.zeros_like(relative_pos_start)
            relative_pos_end = torch.zeros_like(relative_pos_end)

        edge_features = torch.cat([
            edge_lengths_for_feat.unsqueeze(-1) / 1000.0,
            sharing_feature.unsqueeze(-1),
            relative_pos_start.unsqueeze(-1),
            relative_pos_end.unsqueeze(-1)
        ], dim=-1)

        return edge_features
    
    def build_od_subgraphs(self, batch, graph_data):
        """Build OD-specific edge-based subgraphs"""
        device = graph_data.node_feat.device
        od_embeddings_dict = {}
        
        path2pair = batch["path2pair"]
        od_pairs = batch["od_pairs"]
        flat_node_sequences = batch["node_sequences"]
        flat_edge_indices = batch["edge_indices"]
        
        od_groups = {}
        for path_idx in range(len(flat_node_sequences)):
            od_idx = path2pair[path_idx].item()
            if od_idx not in od_groups:
                od_groups[od_idx] = []
            od_groups[od_idx].append(path_idx)
        
        for od_idx, path_indices in od_groups.items():
            od_pair = od_pairs[od_idx]
            
            edge_to_idx = {}  # (u,v) -> index in edge-based graph
            edge_features_list = []
            
            for path_idx in path_indices:
                nodes = flat_node_sequences[path_idx]
                edge_idx_padded = flat_edge_indices[path_idx]
                valid_edges = edge_idx_padded[edge_idx_padded >= 0]
                
                for j, edge_idx in enumerate(valid_edges):
                    if j < len(nodes) - 1:
                        u, v = nodes[j], nodes[j+1]
                        if (u, v) not in edge_to_idx:
                            new_idx = len(edge_to_idx)
                            edge_to_idx[(u, v)] = new_idx
                            
                            edge_length = graph_data.edge_feat[edge_idx, 0]
                            start_coords = graph_data.node_feat[u]
                            end_coords = graph_data.node_feat[v]
                            
                            edge_feat = torch.cat([
                                edge_length.unsqueeze(0) / 1000.0,
                                start_coords,
                                end_coords
                            ])
                            edge_features_list.append(edge_feat)
            
            edge_adjacency = []
            edge_keys = list(edge_to_idx.keys())
            
            for i, (u1, v1) in enumerate(edge_keys):
                for j, (u2, v2) in enumerate(edge_keys):
                    if i != j:
                        if v1 == u2:
                            edge_adjacency.append([i, j])
            
            if edge_adjacency:
                edge_edge_index = torch.tensor(edge_adjacency, device=device).t()
            else:
                edge_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            
            if edge_features_list:
                edge_node_feat = torch.stack(edge_features_list)
            else:
                edge_node_feat = torch.zeros((0, 5), device=device)
            
            edge_embeddings = self.od_gnn(edge_node_feat, edge_edge_index)
            od_embeddings_dict[od_pair] = (edge_embeddings, edge_to_idx)
        
        return od_embeddings_dict

    def forward(self, graph_data, batch, task='stretch'):
        edge_features = self.compute_edge_features(
            graph_data, batch
        )
        counts = [len(s) for s in batch["scores"]]
        if self.enable_gnn_od:
            od_embeddings_dict = self.build_od_subgraphs(batch, graph_data)
        else:
            od_embeddings_dict = None
        shared_hidden = self.stretch_rnn(batch, graph_data, edge_features, od_embeddings_dict)

        scores = self.stretch_head(shared_hidden) * 100
        return list(torch.split(scores.squeeze(-1), counts))
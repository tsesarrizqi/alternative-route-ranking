import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class ODSubgraphGNN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=7):
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
                 epsilon=0.0, num_rnn_layers=2, dropout=0.2):
        super().__init__()

        input_dim = 6
        if epsilon > 0:
            input_dim += 1
        if enable_gnn_od:
            input_dim += gnn_hidden_dim
        
        # Stack multiple GRUCell layers
        self.num_rnn_layers = num_rnn_layers
        self.dropout = dropout
        
        # Forward direction layers
        self.rnn_cells_forward = nn.ModuleList()
        self.rnn_cells_forward.append(nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim))
        for _ in range(num_rnn_layers - 1):
            self.rnn_cells_forward.append(nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim))
        
        # Backward direction layers
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
        self.output_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, batch, graph_data, edge_features, od_node_embeddings):
        device = graph_data.node_feat.device
        edge_lengths = graph_data.edge_feat.squeeze(-1)

        batch_size = edge_features.shape[0]
        max_seq_len = edge_features.shape[1]

        cumulative_from_start = torch.zeros(batch_size, max_seq_len, device=device)
        pool_shortest_from_start = torch.zeros(batch_size, max_seq_len, device=device)
        cumulative_from_end = torch.zeros(batch_size, max_seq_len, device=device)
        pool_shortest_from_end = torch.zeros(batch_size, max_seq_len, device=device)

        flat_node_sequences = batch["node_sequences"]
        flat_edge_indices = batch["edge_indices"]
        path2pair = batch["path2pair"]
        od_pairs = batch["od_pairs"]
        subpath_indices_dict = batch.get("subpath_indices", {})

        for i in range(batch_size):
            nodes = flat_node_sequences[i]
            edge_idx_padded = flat_edge_indices[i]
            valid_edges = edge_idx_padded[edge_idx_padded >= 0]

            if len(nodes) < 2 or len(valid_edges) == 0:
                continue
            
            origin = nodes[0]
            destination = nodes[-1]
            od_idx = path2pair[i].item()
            od_pair = od_pairs[od_idx]

            running_sum = 0
            for j, edge_idx in enumerate(valid_edges[:max_seq_len]):
                edge_id = int(edge_idx.item())
                running_sum += edge_lengths[edge_id].item()
                cumulative_from_start[i, j] = running_sum

                if j+1 < len(nodes):
                    current_node = nodes[j+1]
                    node_pair = (origin, current_node)

                    if od_pair in subpath_indices_dict:
                        od_subpaths = subpath_indices_dict[od_pair]
                        if node_pair in od_subpaths:
                            pool_shortest_from_start[i, j] = od_subpaths[node_pair]
                        else:
                            pool_shortest_from_start[i, j] = running_sum
                    else:
                        pool_shortest_from_start[i, j] = running_sum

            running_sum_back = 0
            for j in range(len(valid_edges)-1, -1, -1):
                if j >= max_seq_len:
                    continue
                edge_idx = valid_edges[j]
                edge_id = int(edge_idx.item())
                running_sum_back += edge_lengths[edge_id].item()
                cumulative_from_end[i, j] = running_sum_back

                if j < len(nodes) - 1:
                    current_node = nodes[j]
                    node_pair = (current_node, destination)

                    if od_pair in subpath_indices_dict:
                        od_subpaths = subpath_indices_dict[od_pair]
                        if node_pair in od_subpaths:
                            pool_shortest_from_end[i, j] = od_subpaths[node_pair]
                        else:
                            pool_shortest_from_end[i, j] = running_sum_back
                    else:
                        pool_shortest_from_end[i, j] = running_sum_back

        cumulative_from_start_norm = cumulative_from_start / 1000.0
        pool_from_start_norm = pool_shortest_from_start / 1000.0
        cumulative_from_end_norm = cumulative_from_end / 1000.0
        pool_from_end_norm = pool_shortest_from_end / 1000.0
        
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
        
        # Initialize hidden states for all layers
        hidden_states_f = [torch.zeros(batch_size, self.hidden_dim, device=device) 
                          for _ in range(self.num_rnn_layers)]
        
        edge_mask = batch["edge_indices"] >= 0
        seq_lengths = edge_mask.sum(dim=1).cpu().tolist()
        max_steps = max(seq_lengths) if seq_lengths else 0

        # Forward pass through all layers
        for step in range(max_steps):
            active_mask = torch.tensor([step < seq_len for seq_len in seq_lengths], 
                                      dtype=torch.bool, device=device)
            if active_mask.any():
                # Layer 0: input from combined_forward
                layer_input = combined_forward[:, step, :]
                new_hidden = self.rnn_cells_forward[0](layer_input, hidden_states_f[0])
                hidden_states_f[0] = torch.where(active_mask.unsqueeze(-1), new_hidden, hidden_states_f[0])
                
                # Remaining layers: input from previous layer's hidden state
                for layer_idx in range(1, self.num_rnn_layers):
                    layer_input = hidden_states_f[layer_idx - 1]
                    if self.training and layer_idx > 0:
                        layer_input = F.dropout(layer_input, p=self.dropout, training=True)
                    new_hidden = self.rnn_cells_forward[layer_idx](layer_input, hidden_states_f[layer_idx])
                    hidden_states_f[layer_idx] = torch.where(active_mask.unsqueeze(-1), new_hidden, hidden_states_f[layer_idx])

        hidden_f = hidden_states_f[-1]  # Take output from last layer

        # Initialize hidden states for backward layers
        hidden_states_b = [torch.zeros(batch_size, self.hidden_dim, device=device) 
                          for _ in range(self.num_rnn_layers)]
        
        # Backward pass through all layers
        for step in range(max_steps):
            rev_step = max_steps - 1 - step
            active_mask = torch.tensor([rev_step < seq_len for seq_len in seq_lengths],
                                       dtype=torch.bool, device=device)
            if active_mask.any():
                # Layer 0: input from combined_backward
                layer_input = combined_backward[:, rev_step, :]
                new_hidden = self.rnn_cells_backward[0](layer_input, hidden_states_b[0])
                hidden_states_b[0] = torch.where(active_mask.unsqueeze(-1), new_hidden, hidden_states_b[0])
                
                # Remaining layers: input from previous layer's hidden state
                for layer_idx in range(1, self.num_rnn_layers):
                    layer_input = hidden_states_b[layer_idx - 1]
                    if self.training and layer_idx > 0:
                        layer_input = F.dropout(layer_input, p=self.dropout, training=True)
                    new_hidden = self.rnn_cells_backward[layer_idx](layer_input, hidden_states_b[layer_idx])
                    hidden_states_b[layer_idx] = torch.where(active_mask.unsqueeze(-1), new_hidden, hidden_states_b[layer_idx])
        
        hidden_b = hidden_states_b[-1]  # Take output from last layer

        return self.output_norm(torch.cat([hidden_f, hidden_b], dim=1))

class PathRankingModel(nn.Module):
    def __init__(self, hidden_dim=64, repr_dim=128, dropout=0.1, enable_gnn_od=False, epsilon=0.0):
        super().__init__()

        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=repr_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.sharing_head = nn.Sequential(
            nn.Linear(repr_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        if enable_gnn_od:
            self.od_gnn = ODSubgraphGNN(hidden_dim=hidden_dim, num_layers=7)
        else:
            self.od_gnn = None

        self.stretch_rnn = StretchRNN(hidden_dim=hidden_dim, gnn_hidden_dim=hidden_dim, enable_gnn_od=enable_gnn_od, epsilon=epsilon)
        
        self.stretch_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.final_linear_head = nn.Linear(2, 1, bias=True)

        self.enable_gnn_od = enable_gnn_od
        
    def compute_sharing_vectorized(self, edge_indices, main_paths_dict, path2pair, edge_mask):
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

        shared_mask = self.compute_sharing_vectorized(
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

        edge_features = torch.cat([
            edge_lengths_masked.unsqueeze(-1) / 1000.0,
            sharing_feature.unsqueeze(-1),
            relative_pos_start.unsqueeze(-1),
            relative_pos_end.unsqueeze(-1)
        ], dim=-1)

        return edge_features, edge_mask
    
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

    def forward(self, graph_data, batch, task='final_without_locality'):
        edge_features, edge_mask = self.compute_edge_features(
            graph_data, batch
        )
        counts = [len(s) for s in batch["scores"]]

        if task == 'stretch':
            if self.enable_gnn_od:
                od_embeddings_dict = self.build_od_subgraphs(batch, graph_data)
            else:
                od_embeddings_dict = None 
            shared_hidden = self.stretch_rnn(batch, graph_data, edge_features, od_embeddings_dict)

            scores = self.stretch_head(shared_hidden) * 100

            return list(torch.split(scores.squeeze(-1), counts))
        
        edge_hidden = self.edge_encoder(edge_features)
        lengths = edge_mask.sum(dim=1).cpu()

        if lengths.max() > 0:
            packed = nn.utils.rnn.pack_padded_sequence(
                edge_hidden, lengths, batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
            alt_path_embedding = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            batch_size = edge_features.shape[0]
            alt_path_embedding = torch.zeros(batch_size, self.gru.hidden_size * 2, device=edge_features.device)
        
        if task == 'sharing':
            scores = self.sharing_head(alt_path_embedding).squeeze(-1) * 100
            return list(torch.split(scores, counts))
        elif task == 'final_without_locality':
            if self.enable_gnn_od:
                od_embeddings_dict = self.build_od_subgraphs(batch, graph_data)
            else:
                od_embeddings_dict = None
            shared_hidden = self.stretch_rnn(batch, graph_data, edge_features, od_embeddings_dict)

            sharing_scores = self.sharing_head(alt_path_embedding) * 100
            stretch_scores = self.stretch_head(shared_hidden) * 100
            
            # stretch_scores = stretch_scores * torch.sigmoid(self.stretch_scale)
            
            final_scores = 0.6 * stretch_scores + 0.4 * sharing_scores

            return {
                'final': list(torch.split(final_scores.squeeze(-1), counts)),
                'sharing': list(torch.split(sharing_scores.squeeze(-1), counts)),
                'stretch': list(torch.split(stretch_scores.squeeze(-1), counts))
            }
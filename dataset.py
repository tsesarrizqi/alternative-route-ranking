import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

class RoadNetworkDataset(Dataset):
    def __init__(self, data_dir="data/graph", paths_path="paths.csv", exclude_main_path=False, precompute_subpaths=True):
        super().__init__()
        self.graph_data_dir = data_dir
        self.paths_path = paths_path
        self.exclude_main_path = exclude_main_path

        self.nodes_df, self.edges_df, self.paths_df = self.load_csv_files()
        self.nodes_raw = self.nodes_df[["latitude", "longitude"]].copy()
        
        self.edge_lookup = {
            (int(r.start_id), int(r.end_id)): idx
            for idx, r in self.edges_df.reset_index().iterrows()
        }
        self.edge_to_nodes = {
            idx: (int(r.start_id), int(r.end_id))
            for idx, r in self.edges_df.reset_index().iterrows()
        }
        
        self.edge_lengths = torch.tensor(self.edges_df["length"].values, dtype=torch.float)
        
        self.normalize_nodes()
        
        self.graph_data = self.create_pyg_data()

        (self.candidate_paths,
         self.candidate_node_sequences,
         self.candidate_scores,
         self.candidate_sharing_scores,
         self.candidate_stretch_scores,
         self.candidate_locality_scores,
         self.candidate_accepted,       
         self.candidate_edge_indices,
         self.main_paths_dict,
         self.main_path_indices) = self.process_paths()
        
        self.source_nodes = self.paths_df['source'].values
        self.dest_nodes = self.paths_df['destination'].values

        if precompute_subpaths:
            print("Pre-computing subpath indices...")
            self.subpath_indices = self._precompute_all_subpath_indices()
            print(f"Pre-computed subpath indices for {len(self.subpath_indices)} OD pairs")

        print("Building OD pair index mapping...")
        self.od_to_index = {}
        for idx in range(len(self.source_nodes)):
            od_pair = (self.source_nodes[idx], self.dest_nodes[idx])
            self.od_to_index[od_pair] = idx
        print(f"Built mapping for {len(self.od_to_index)} OD pairs")

    def _precompute_all_subpath_indices(self):
        import time
        from collections import defaultdict
    
        subpath_indices = {}
        edge_lengths_np = self.edge_lengths.numpy()
    
        od_groups = defaultdict(list)
        for idx, row in self.paths_df.iterrows():
            source = int(row["source"])
            destination = int(row["destination"])
            od_key = (source, destination)
            od_groups[od_key].append(idx)
    
        start_time = time.time()
    
        for od_idx, (od_key, path_indices) in enumerate(od_groups.items()):
            od_subpaths = {}
            origin, destination = od_key
    
            if od_key in self.main_paths_dict:
                main_edges = self.main_paths_dict[od_key]
                if main_edges:
                    main_nodes = [self.edge_to_nodes[main_edges[0]][0]]
                    for edge_idx in main_edges:
                        main_nodes.append(self.edge_to_nodes[edge_idx][1])
    
                    edge_lengths_main = edge_lengths_np[main_edges]
                    
                    cumsum_forward = np.zeros(len(main_nodes))
                    cumsum_forward[1:] = np.cumsum(edge_lengths_main)
                    
                    cumsum_backward = np.zeros(len(main_nodes))
                    cumsum_backward[:-1] = np.cumsum(edge_lengths_main[::-1])[::-1]
    
                    for i in range(len(main_nodes)):
                        od_subpaths[(origin, main_nodes[i])] = float(cumsum_forward[i])
                        od_subpaths[(main_nodes[i], destination)] = float(cumsum_backward[i])
    
            for idx in path_indices:
                if idx < len(self.candidate_node_sequences):
                    paths = self.candidate_node_sequences[idx]
                    edge_indices_list = self.candidate_edge_indices[idx]
    
                    for path_idx, (nodes, edges) in enumerate(zip(paths, edge_indices_list)):
                        if len(nodes) < 2:
                            continue
                        
                        valid_edges = edges[:len(nodes)-1]
                        if not valid_edges:
                            continue
                        
                        path_lengths = edge_lengths_np[valid_edges]
                        
                        cumsum_forward = np.zeros(len(nodes))
                        cumsum_forward[1:] = np.cumsum(path_lengths)
                        
                        cumsum_backward = np.zeros(len(nodes))
                        if len(path_lengths) > 0:
                            cumsum_backward[:-1] = np.cumsum(path_lengths[::-1])[::-1]
    
                        for i in range(len(nodes)):
                            pair = (origin, nodes[i])
                            distance = float(cumsum_forward[i])
                            if pair not in od_subpaths or distance < od_subpaths[pair]:
                                od_subpaths[pair] = distance
                        
                        for i in range(len(nodes)):
                            pair = (nodes[i], destination)
                            distance = float(cumsum_backward[i])
                            if pair not in od_subpaths or distance < od_subpaths[pair]:
                                od_subpaths[pair] = distance
    
            subpath_indices[od_key] = od_subpaths
    
            if (od_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (od_idx + 1) / elapsed
                eta = (len(od_groups) - od_idx - 1) / rate
                num_pairs = sum(len(sp) for sp in subpath_indices.values())
                print(f"  Processed {od_idx + 1}/{len(od_groups)} OD pairs "
                      f"({num_pairs} pairs total, {rate:.1f} OD/s, ETA: {eta:.1f}s)")
    
        total_time = time.time() - start_time
        total_pairs = sum(len(sp) for sp in subpath_indices.values())
        avg_pairs_per_od = total_pairs / len(od_groups) if od_groups else 0
    
        print(f"  Total pre-computation time: {total_time:.1f}s")
        print(f"  Total pairs stored: {total_pairs:,} (avg {avg_pairs_per_od:.1f} per OD)")
        print(f"  Memory estimate: ~{total_pairs * 16 / 1024 / 1024:.1f} MB")
    
        return subpath_indices

    def compute_path_length(self, edge_indices):
        """Compute total length of a path given its edge indices."""
        if not edge_indices:
            return float('inf')
        return sum(self.edge_lengths[idx].item() for idx in edge_indices)

    def process_paths(self):
        candidate_paths = []
        candidate_node_sequences = []
        candidate_scores = []
        candidate_sharing_scores = []
        candidate_stretch_scores = []
        candidate_locality_scores = []
        candidate_edge_indices = []
        candidate_accepted = []
        main_paths_dict = {}
        main_path_indices = []

        for idx, row in self.paths_df.iterrows():
            source = int(row["source"])
            destination = int(row["destination"])
            od_key = (source, destination)
            
            paths_str = row["paths"]
            scores_str = row["final_scores"]

            node_paths = [list(map(int, p_str.split(','))) for p_str in paths_str.split(';')]
            
            all_edge_paths = []
            all_edge_indices = []
            
            for nseq in node_paths:
                edge_ids = []
                edge_indices = []
                valid_path = True
                
                for i in range(len(nseq) - 1):
                    edge_key = (nseq[i], nseq[i + 1])
                    if edge_key in self.edge_lookup:
                        edge_idx = self.edge_lookup[edge_key]
                        edge_ids.append(edge_idx)
                        edge_indices.append(edge_idx)
                    else:
                        valid_path = False
                        break
                
                if valid_path:
                    all_edge_paths.append(edge_ids)
                    all_edge_indices.append(edge_indices)
                else:
                    all_edge_paths.append([])
                    all_edge_indices.append([])
            
            sharing_vals = list(map(float, row["sharing_scores"].split(";")))
            shortest_idx = -1
            for i, score in enumerate(sharing_vals):
                if abs(score) < 0.01:  # Score ~0 means it's the main path
                    shortest_idx = i
                    break

            # Fallback to first valid path if no path has sharing ~0 (shouldn't happen)
            if shortest_idx < 0:
                for i, edge_path in enumerate(all_edge_paths):
                    if edge_path:
                        shortest_idx = i
                        break
            
            if shortest_idx >= 0:
                main_paths_dict[od_key] = all_edge_paths[shortest_idx]
                main_path_indices.append(shortest_idx)
            else:
                for i, edge_path in enumerate(all_edge_paths):
                    if edge_path:
                        main_paths_dict[od_key] = edge_path
                        main_path_indices.append(i)
                        shortest_idx = i
                        break
                else:
                    main_paths_dict[od_key] = []
                    main_path_indices.append(0)
                    shortest_idx = 0
            
            scores = list(map(float, scores_str.split(";")))
            stretch_vals = list(map(float, row["stretch_scores"].split(";")))
            locality_vals = list(map(float, row["locality_scores"].split(";")))
            if "accepted" in row and pd.notna(row["accepted"]) and row["accepted"]:
                accepted_vals = list(map(int, row["accepted"].split(";")))
            else:
                accepted_vals = [1] * len(scores)
            
            if self.exclude_main_path and shortest_idx >= 0:
                edge_paths_filtered = []
                edge_indices_filtered = []
                scores_filtered = []
                sharing_filtered = []
                stretch_filtered = []
                locality_filtered = []
                accepted_filtered = []
                node_sequences_filtered = []
                
                for i in range(len(all_edge_paths)):
                    if i != shortest_idx:
                        if all_edge_paths[i]:
                            edge_paths_filtered.append(torch.tensor(all_edge_paths[i], dtype=torch.long))
                        else:
                            edge_paths_filtered.append(torch.tensor([], dtype=torch.long))
                        edge_indices_filtered.append(all_edge_indices[i])
                        scores_filtered.append(scores[i])
                        sharing_filtered.append(sharing_vals[i])
                        stretch_filtered.append(stretch_vals[i])
                        locality_filtered.append(locality_vals[i])
                        accepted_filtered.append(accepted_vals[i])
                        node_sequences_filtered.append(node_paths[i])
                
                if edge_paths_filtered:
                    candidate_paths.append(edge_paths_filtered)
                    candidate_edge_indices.append(edge_indices_filtered)
                    candidate_scores.append(scores_filtered)
                    candidate_sharing_scores.append(sharing_filtered)
                    candidate_stretch_scores.append(stretch_filtered)
                    candidate_locality_scores.append(locality_filtered)
                    candidate_accepted.append(accepted_filtered)
                    candidate_node_sequences.append(node_sequences_filtered)
            else:
                edge_paths_tensors = []
                for edge_path in all_edge_paths:
                    if edge_path:
                        edge_paths_tensors.append(torch.tensor(edge_path, dtype=torch.long))
                    else:
                        edge_paths_tensors.append(torch.tensor([], dtype=torch.long))
                
                candidate_paths.append(edge_paths_tensors)
                candidate_edge_indices.append(all_edge_indices)
                candidate_scores.append(scores)
                candidate_sharing_scores.append(sharing_vals)
                candidate_stretch_scores.append(stretch_vals)
                candidate_locality_scores.append(locality_vals)
                candidate_accepted.append(accepted_vals)
                candidate_node_sequences.append(node_paths)

        return (candidate_paths,
                candidate_node_sequences,
                candidate_scores,
                candidate_sharing_scores,
                candidate_stretch_scores,
                candidate_locality_scores,
                candidate_accepted,
                candidate_edge_indices,
                main_paths_dict,
                main_path_indices)

    def load_csv_files(self):
        nodes_file = os.path.join(self.graph_data_dir, "nodes.csv")
        edges_file = os.path.join(self.graph_data_dir, "edges.csv")
        paths_file = os.path.join(self.graph_data_dir, self.paths_path)
        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)
        paths_df = pd.read_csv(paths_file)
        return nodes_df, edges_df, paths_df

    def normalize_nodes(self):
        for col in ["latitude", "longitude"]:
            mean = self.nodes_df[col].mean()
            std = self.nodes_df[col].std()
            if std == 0:
                std = 1.0
            self.nodes_df[col] = (self.nodes_df[col] - mean) / std

    def create_pyg_data(self):
        node_feats = ["latitude", "longitude"]
        node_feat = torch.tensor(self.nodes_df[node_feats].values, dtype=torch.float)
        node_geo_deg = torch.tensor(self.nodes_raw[node_feats].values, dtype=torch.float)
        edge_sd  = torch.tensor(self.edges_df[["start_id", "end_id"]].values.T, dtype=torch.long)
        edge_feat = torch.tensor(self.edges_df[["length"]].values, dtype=torch.float)
        return Data(node_feat=node_feat, node_geo_deg=node_geo_deg, edge_sd=edge_sd, edge_feat=edge_feat)

    def __len__(self):
        return len(self.candidate_scores)

    def __getitem__(self, idx):
        row = self.paths_df.iloc[idx]
        source = int(row["source"])
        destination = int(row["destination"])
        optimal_distance = float(row["optimal_distance"])
        od_key = (source, destination)

        result = {
            'paths': self.candidate_paths[idx],
            'node_sequences': self.candidate_node_sequences[idx],
            'edge_indices': self.candidate_edge_indices[idx],
            'scores': self.candidate_scores[idx],
            'sharing_scores': self.candidate_sharing_scores[idx],
            'stretch_scores': self.candidate_stretch_scores[idx],
            'locality_scores': self.candidate_locality_scores[idx],
            'accepted': self.candidate_accepted[idx],
            'od_pair': od_key,
            'optimal_distance': optimal_distance,
            'main_path': self.main_paths_dict.get(od_key, [])
        }

        if hasattr(self, 'subpath_indices'):
            result['subpath_indices'] = self.subpath_indices.get(od_key, {})

        return result

    def split_dataset_three_ways(self, val_size=0.1, test_size=0.2, random_state=42):
        indices = list(range(len(self.candidate_scores)))

        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

        val_ratio_in_trainval = val_size / (1.0 - test_size)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_ratio_in_trainval,
            random_state=random_state
        )

        train_dataset = RoadNetworkDatasetSubset(self, train_indices)
        val_dataset = RoadNetworkDatasetSubset(self, val_indices)
        test_dataset = RoadNetworkDatasetSubset(self, test_indices)

        return train_dataset, val_dataset, test_dataset, train_indices, val_indices, test_indices


class RoadNetworkDatasetSubset(Dataset):
    def __init__(self, parent_dataset, indices):
        super().__init__()
        self.parent_dataset = parent_dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.parent_dataset[self.indices[idx]]
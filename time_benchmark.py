import torch
import numpy as np
import time
import os
from scipy.sparse.csgraph import dijkstra
from torch.utils.data import DataLoader
import argparse
from train_sharing import collate_fn
from dataset import RoadNetworkDataset, RoadNetworkDatasetSubset
from model import PathRankingModel
from model_stretch import PathStretchScoringModel
from rescore_stretch import build_graph


class GroundTruthCalculator:
    def __init__(self, dataset, epsilon=0.1):
        if not np.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("epsilon must be finite and nonnegative")
        self.dataset = dataset
        self.epsilon = epsilon
        self.edge_lengths = dataset.edge_lengths.cpu().numpy()
        self.edge_sd = dataset.graph_data.edge_sd.cpu().numpy()
        self.node_index, self.road_graph, self.exact_edge_lengths = build_graph(
            os.path.dirname(os.path.normpath(dataset.graph_data_dir))
        )
        
    def calculate_sharing_batch(self, batch):
        """Calculate sharing scores by re-computing the ground truth formula."""
        sharing_scores = []
        
        device = batch['edge_indices'].device
        path2pair = batch['path2pair'].cpu().numpy()
        all_edge_indices = batch['edge_indices'].cpu().numpy()
        
        for od_idx, true_sharing_scores in enumerate(batch['sharing_scores']):
            # Find all paths belonging to this OD pair
            path_mask = path2pair == od_idx
            path_indices = np.where(path_mask)[0]
            
            if len(path_indices) == 0:
                sharing_scores.append([])
                continue
            
            # Get edge indices for paths in this OD pair
            od_edge_indices = all_edge_indices[path_indices]  # Shape: (num_paths, max_edges)
            
            # Find main path (sharing score closest to 0)
            main_path_idx = 0
            min_sharing = float('inf')
            for i, score in enumerate(true_sharing_scores):
                if abs(score) < abs(min_sharing):
                    min_sharing = score
                    main_path_idx = i
            
            # Build set of main path edges
            main_path_edges = od_edge_indices[main_path_idx]
            main_edges = set()
            for edge_idx in main_path_edges:
                if edge_idx == -1:
                    break
                u = self.edge_sd[0, edge_idx]
                v = self.edge_sd[1, edge_idx]
                edge = (min(u, v), max(u, v))
                main_edges.add(edge)
            
            # Calculate sharing score for each path
            path_scores = []
            for path_edges in od_edge_indices:
                shared_length = 0.0
                total_length = 0.0
                
                for edge_idx in path_edges:
                    if edge_idx == -1:
                        break
                    
                    u = self.edge_sd[0, edge_idx]
                    v = self.edge_sd[1, edge_idx]
                    edge = (min(u, v), max(u, v))
                    length = self.edge_lengths[edge_idx]
                    
                    total_length += length
                    if edge in main_edges:
                        shared_length += length
                
                if total_length > 0:
                    sharing_score = (1.0 - shared_length / total_length) * 100
                else:
                    sharing_score = 0.0
                
                path_scores.append(sharing_score)
            
            sharing_scores.append(path_scores)
        
        return sharing_scores
    
    def calculate_stretch_batch(self, batch):
        """Recompute length-weighted epsilon scores using directed Dijkstra."""
        stretch_scores = []
        path2pair = batch['path2pair'].cpu().numpy()
        all_node_sequences = batch['node_sequences']

        for od_idx in range(len(batch['stretch_scores'])):
            path_indices = np.flatnonzero(path2pair == od_idx)
            path_scores = []
            for path_idx in path_indices:
                nodes = [int(node) for node in all_node_sequences[path_idx]]
                if len(nodes) < 2:
                    path_scores.append(100.0)
                    continue

                node_indices = [self.node_index[node] for node in nodes]
                satisfying_length = 0.0
                total_length = 0.0
                for start in range(len(nodes) - 1):
                    distances = dijkstra(
                        self.road_graph, directed=True, indices=node_indices[start],
                        return_predecessors=False
                    )
                    subpath_length = 0.0
                    for end in range(start + 1, len(nodes)):
                        subpath_length += self.exact_edge_lengths[(nodes[end - 1], nodes[end])]
                        shortest_length = distances[node_indices[end]]
                        if not np.isfinite(shortest_length) or shortest_length <= 0.0:
                            continue
                        total_length += subpath_length
                        if subpath_length <= (1.0 + self.epsilon) * shortest_length:
                            satisfying_length += subpath_length

                stretch_score = 100.0 * satisfying_length / total_length if total_length > 0 else 100.0
                path_scores.append(stretch_score)
            stretch_scores.append(path_scores)
        return stretch_scores


def benchmark_combined(model, dataset, test_loader, device, num_iterations=5, 
                      sharing_weight=0.4, stretch_weight=0.6, epsilon=0.1):
    gt_calculator = GroundTruthCalculator(dataset, epsilon=epsilon)
    model.eval()
    graph_data = dataset.graph_data.to(device)
    
    gt_times = []
    ml_times = []
    
    print(f"\nBenchmarking Combined Score Calculation ({num_iterations} iterations)...")
    
    for iteration in range(num_iterations):
        print(f"  Iteration {iteration + 1}/{num_iterations}")
        
        gt_time = 0
        ml_time = 0
        
        for batch in test_loader:
            start = time.time()
            gt_sharing = gt_calculator.calculate_sharing_batch(batch)
            gt_stretch = gt_calculator.calculate_stretch_batch(batch)
            gt_combined = [
                [sharing_weight * s + stretch_weight * t 
                 for s, t in zip(sharing_scores, stretch_scores)]
                for sharing_scores, stretch_scores in zip(gt_sharing, gt_stretch)
            ]
            gt_time += time.time() - start
            
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            
            with torch.no_grad():
                start = time.time()
                predictions = model(graph_data, batch)
                pred_final = predictions['final']
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ml_time += time.time() - start
        
        gt_times.append(gt_time)
        ml_times.append(ml_time)
    
    return gt_times, ml_times

def benchmark_sharing(model, dataset, test_loader, device, num_iterations=5):
    gt_calculator = GroundTruthCalculator(dataset)
    model.eval()
    graph_data = dataset.graph_data.to(device)
    
    gt_times = []
    ml_times = []
    
    print(f"\nBenchmarking Sharing Score Calculation ({num_iterations} iterations)...")
    
    for iteration in range(num_iterations):
        print(f"  Iteration {iteration + 1}/{num_iterations}")
        
        gt_time = 0
        ml_time = 0
        
        for batch in test_loader:
            start = time.time()
            gt_scores = gt_calculator.calculate_sharing_batch(batch)
            gt_time += time.time() - start
            
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            
            with torch.no_grad():
                start = time.time()
                pred_sharing = model(graph_data, batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ml_time += time.time() - start
        
        gt_times.append(gt_time)
        ml_times.append(ml_time)
    
    return gt_times, ml_times


def benchmark_stretch(model, dataset, test_loader, device, num_iterations=5, epsilon=0.1):
    gt_calculator = GroundTruthCalculator(dataset, epsilon=epsilon)
    model.eval()
    graph_data = dataset.graph_data.to(device)
    
    gt_times = []
    ml_times = []
    
    print(f"\nBenchmarking Stretch Score Calculation ({num_iterations} iterations)...")
    
    for iteration in range(num_iterations):
        print(f"  Iteration {iteration + 1}/{num_iterations}")
        
        gt_time = 0
        ml_time = 0
        
        for batch in test_loader:
            start = time.time()
            gt_scores = gt_calculator.calculate_stretch_batch(batch)
            gt_time += time.time() - start
            
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            
            with torch.no_grad():
                start = time.time()
                pred_stretch = model(graph_data, batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ml_time += time.time() - start
        
        gt_times.append(gt_time)
        ml_times.append(ml_time)
    
    return gt_times, ml_times


def benchmark_stretch_standalone(model, dataset, test_loader, device, num_iterations=5, epsilon=0.1):
    """Benchmark learned scoring against freshly computed exact epsilon scores."""
    gt_calculator = GroundTruthCalculator(dataset, epsilon=epsilon)
    model.eval()
    graph_data = dataset.graph_data.to(device)

    gt_times = []
    ml_times = []

    print(f"\nBenchmarking Stretch Model vs Algorithmic UBS ({num_iterations} iterations)...")
    print(f"  Ground truth: length-weighted UBS at epsilon={epsilon}, directed Dijkstra")
    print(f"  Reference setup: road graph built before timing; distances recomputed each call")
    print(f"  ML model:     PathStretchScoringModel (BiGRU, O(L) per path)")

    for iteration in range(num_iterations):
        print(f"  Iteration {iteration + 1}/{num_iterations}")

        gt_time = 0
        ml_time = 0

        for batch in test_loader:
            # Ground truth: algorithmic UBS calculation (CPU, no GPU needed)
            start = time.time()
            gt_scores = gt_calculator.calculate_stretch_batch(batch)
            gt_time += time.time() - start

            # Move batch to device for ML inference
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            # ML model inference
            with torch.no_grad():
                start = time.time()
                pred_stretch = model(graph_data, batch, task='stretch')
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ml_time += time.time() - start

        gt_times.append(gt_time)
        ml_times.append(ml_time)

    return gt_times, ml_times


def print_results(name, gt_times, ml_times, total_paths, num_od_pairs, avg_paths_per_od):
    gt_mean = np.mean(gt_times)
    gt_std = np.std(gt_times)
    ml_mean = np.mean(ml_times)
    ml_std = np.std(ml_times)
    speedup = gt_mean / ml_mean
    
    gt_per_path_ms = (gt_mean * 1000) / total_paths
    ml_per_path_ms = (ml_mean * 1000) / total_paths
    
    gt_per_od_ms = (gt_mean * 1000) / num_od_pairs
    ml_per_od_ms = (ml_mean * 1000) / num_od_pairs
    
    print(f"\n{'='*70}")
    print(f"{name} - TIMING RESULTS")
    print(f"{'='*70}")
    print(f"Total OD pairs: {num_od_pairs}")
    print(f"Total paths: {total_paths}")
    print(f"Avg paths per OD: {avg_paths_per_od:.1f}")
    
    print(f"\nGround Truth Calculation:")
    print(f"  Total time: {gt_mean:.4f}s (+/- {gt_std:.4f}s)")
    print(f"  Per-path: {gt_per_path_ms:.4f}ms")
    print(f"  Per-OD: {gt_per_od_ms:.4f}ms")
    
    print(f"\nML Model Inference:")
    print(f"  Total time: {ml_mean:.4f}s (+/- {ml_std:.4f}s)")
    print(f"  Per-path: {ml_per_path_ms:.4f}ms")
    print(f"  Per-OD: {ml_per_od_ms:.4f}ms")
    
    print(f"\nSpeedup: {speedup:.2f}x")
    if speedup > 1:
        print(f"  ML model is {speedup:.2f}x FASTER")
    else:
        print(f"  Ground truth is {1/speedup:.2f}x FASTER")
    print(f"{'='*70}")
    
    return {
        'gt_per_path_ms': gt_per_path_ms,
        'ml_per_path_ms': ml_per_path_ms,
        'gt_per_od_ms': gt_per_od_ms,
        'ml_per_od_ms': ml_per_od_ms,
        'speedup': speedup
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/london")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--sharing_model", type=str, default="")
    parser.add_argument("--stretch_model", type=str, default="")
    parser.add_argument("--stretch_standalone_model", type=str, default="",
                        help="Suffix for stretch_ml_best<suffix>.pth (PathStretchScoringModel). "
                             "Benchmarks ML inference vs algorithmic UBS ground truth.")
    parser.add_argument("--combined_model", type=str, default="")
    parser.add_argument("--enable_gnn_od", action="store_true")
    parser.add_argument("--paths_path", type=str, default="paths.csv")
    parser.add_argument("--exclude_main", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Cap the test set to this many OD pairs (0 = use full test split).")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLoading dataset...")
    dataset = RoadNetworkDataset(
        data_dir=f"{args.data_dir}/graph",
        exclude_main_path=args.exclude_main,
        paths_path=args.paths_path,
        precompute_subpaths=True
    )
    
    _, _, _, _, _, test_indices = dataset.split_dataset_three_ways(
        val_size=0.1, test_size=0.2, random_state=42
    )

    if args.max_samples > 0:
        test_indices = test_indices[:args.max_samples]
        print(f"Capped to {args.max_samples} OD pairs for benchmarking.")

    test_dataset = RoadNetworkDatasetSubset(dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    num_od_pairs = len(test_dataset)
    total_paths = sum(len(dataset.candidate_scores[idx]) for idx in test_indices)
    avg_paths_per_od = total_paths / num_od_pairs
    
    print(f"Test set: {num_od_pairs} OD pairs, {total_paths} paths ({avg_paths_per_od:.1f} avg per OD)")
    
    results = {}
    
    if args.sharing_model:
        print(f"\n{'='*70}")
        print("SHARING SCORE BENCHMARK")
        print(f"{'='*70}")
        
        model_path = f"{args.data_dir}/models/sharing_ml_best{args.sharing_model}.pth"
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = PathRankingModel(
            sharing_hidden_dim=checkpoint['args'].sharing_hidden_dim,
            dropout=0.0,
            enable_gnn_od=args.enable_gnn_od,
            task='sharing'
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        gt_times, ml_times = benchmark_sharing(
            model, dataset, test_loader, device, args.num_iterations
        )
        results['sharing'] = print_results("SHARING SCORE", gt_times, ml_times, 
                                          total_paths, num_od_pairs, avg_paths_per_od)
    
    if args.stretch_model:
        print(f"\n{'='*70}")
        print("STRETCH SCORE BENCHMARK")
        print(f"{'='*70}")
        
        model_path = f"{args.data_dir}/models/stretch_ml_best{args.stretch_model}.pth"
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = PathRankingModel(
            hidden_dim=checkpoint['args'].hidden_dim,
            dropout=0.0,
            enable_gnn_od=args.enable_gnn_od,
            epsilon=args.epsilon,
            task='stretch'
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        gt_times, ml_times = benchmark_stretch(
            model, dataset, test_loader, device, args.num_iterations, epsilon=args.epsilon
        )
        results['stretch'] = print_results("STRETCH SCORE", gt_times, ml_times, 
                                          total_paths, num_od_pairs, avg_paths_per_od)

    if args.stretch_standalone_model:
        print(f"\n{'='*70}")
        print("STRETCH STANDALONE MODEL vs ALGORITHMIC UBS BENCHMARK")
        print(f"{'='*70}")

        model_path = f"{args.data_dir}/models/stretch_ml_best{args.stretch_standalone_model}.pth"
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        ckpt_args = checkpoint['args']

        if not hasattr(ckpt_args, 'epsilon') or ckpt_args.epsilon != args.epsilon:
            parser.error("Use an epsilon-trained checkpoint with --epsilon matching its training threshold.")

        model = PathStretchScoringModel(
            hidden_dim=ckpt_args.hidden_dim,
            dropout=0.0,
            enable_gnn_od=ckpt_args.enable_gnn_od,
            epsilon=ckpt_args.epsilon
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        gt_times, ml_times = benchmark_stretch_standalone(
            model, dataset, test_loader, device, args.num_iterations, epsilon=args.epsilon
        )
        results['stretch_standalone'] = print_results(
            "STRETCH STANDALONE vs UBS", gt_times, ml_times,
            total_paths, num_od_pairs, avg_paths_per_od
        )

    if args.combined_model or (args.sharing_model and args.stretch_model):
        print(f"\n{'='*70}")
        print("COMBINED SCORE BENCHMARK")
        print(f"{'='*70}")
        
        model_suffix = args.combined_model if args.combined_model else args.sharing_model
        model_path = f"{args.data_dir}/models/final_without_locality_best{model_suffix}.pth"
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = PathRankingModel(
            hidden_dim=checkpoint['args'].hidden_dim,
            sharing_hidden_dim=checkpoint['args'].sharing_hidden_dim,
            dropout=0.0,
            enable_gnn_od=args.enable_gnn_od,
            epsilon=args.epsilon,
            task='final_without_locality'
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        gt_times, ml_times = benchmark_combined(
            model, dataset, test_loader, device, args.num_iterations, epsilon=args.epsilon
        )
        results['combined'] = print_results("COMBINED SCORE", gt_times, ml_times, 
                                           total_paths, num_od_pairs, avg_paths_per_od)
    
    if results:
        print(f"\n{'='*70}")
        print("LATEX TABLE VALUES")
        print(f"{'='*70}")
        print(f"Average paths per OD: {avg_paths_per_od:.1f}")
        
        if 'sharing' in results:
            r = results['sharing']
            print(f"\nSharing Score:")
            print(f"  GT Per Path: {r['gt_per_path_ms']:.4f} ms")
            print(f"  ML Per Path: {r['ml_per_path_ms']:.4f} ms")
            print(f"  GT Per OD: {r['gt_per_od_ms']:.4f} ms")
            print(f"  ML Per OD: {r['ml_per_od_ms']:.4f} ms")
            print(f"  Speedup: {r['speedup']:.2f}x")
        
        if 'stretch' in results:
            r = results['stretch']
            print(f"\nStretch Score:")
            print(f"  GT Per Path: {r['gt_per_path_ms']:.4f} ms")
            print(f"  ML Per Path: {r['ml_per_path_ms']:.4f} ms")
            print(f"  GT Per OD: {r['gt_per_od_ms']:.4f} ms")
            print(f"  ML Per OD: {r['ml_per_od_ms']:.4f} ms")
            print(f"  Speedup: {r['speedup']:.2f}x")
        
        if 'combined' in results:
            r = results['combined']
            print(f"\nCombined Score:")
            print(f"  GT Per Path: {r['gt_per_path_ms']:.4f} ms")
            print(f"  ML Per Path: {r['ml_per_path_ms']:.4f} ms")
            print(f"  GT Per OD: {r['gt_per_od_ms']:.4f} ms")
            print(f"  ML Per OD: {r['ml_per_od_ms']:.4f} ms")
            print(f"  Speedup: {r['speedup']:.2f}x")

        if 'stretch_standalone' in results:
            r = results['stretch_standalone']
            print(f"\nStretch Standalone (PathStretchScoringModel) vs Algorithmic UBS:")
            print(f"  Algorithmic UBS Per Path: {r['gt_per_path_ms']:.4f} ms")
            print(f"  ML Model Per Path:        {r['ml_per_path_ms']:.4f} ms")
            print(f"  Algorithmic UBS Per OD:   {r['gt_per_od_ms']:.4f} ms")
            print(f"  ML Model Per OD:          {r['ml_per_od_ms']:.4f} ms")
            print(f"  Speedup: {r['speedup']:.2f}x")
            print(f"\n  LaTeX table row (Algorithmic UBS): {r['gt_per_path_ms']:.4f} & {r['gt_per_od_ms']:.4f}")
            print(f"  LaTeX table row (ML-based, ours):  {r['ml_per_path_ms']:.4f} & {r['ml_per_od_ms']:.4f}")


if __name__ == "__main__":
    main()
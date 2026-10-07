import torch
import json
import os
import numpy as np
from torch.utils.data import DataLoader
from dataset import RoadNetworkDataset, RoadNetworkDatasetSubset
from model_stretch import PathStretchScoringModel
from train_sharing import collate_fn
import argparse
from collections import defaultdict
import heapq

def compute_ranks(scores):
    """Compute ranks from scores - higher score = better = rank 1"""
    scores = np.array(scores)
    sorted_indices = np.argsort(-scores)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(scores) + 1)
    return ranks

def compute_ranking_accuracy_at_k(pred_list, true_list, k):
    """Compute ranking accuracy at k for multiple OD pairs."""
    correct = 0
    total = 0
    
    for pred, true in zip(pred_list, true_list):
        if len(pred) >= k:
            pred_order = np.argsort(-np.array(pred))
            true_order = np.argsort(-np.array(true))
            if np.array_equal(pred_order[:k], true_order[:k]):
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0

def compute_item_level_accuracy(pred_list, true_list):
    """
    Compute item-level accuracy: percentage of individual paths correctly ranked
    across all paths in all OD pairs.
    
    If an OD has 10 paths and 8 are correctly ranked, count 8 correct and 2 wrong.
    """
    total_correct = 0
    total_paths = 0
    
    for pred, true in zip(pred_list, true_list):
        pred_ranks = compute_ranks(pred)
        true_ranks = compute_ranks(true)
        
        correct_in_list = np.sum(pred_ranks == true_ranks)
        total_correct += correct_in_list
        total_paths += len(pred_ranks)
    
    return total_correct / total_paths if total_paths > 0 else 0.0

def compute_item_level_accuracy_at_k(pred_list, true_list, k):
    """
    Compute item-level accuracy for top-k positions only.
    Checks if each of the top-k ranked items is in its correct position.
    """
    total_correct = 0
    total_items = 0
    
    for pred, true in zip(pred_list, true_list):
        if len(pred) >= k:
            pred_ranks = compute_ranks(pred)
            true_ranks = compute_ranks(true)
            
            for rank in range(1, k+1):
                pred_idx = np.where(pred_ranks == rank)[0][0]
                true_idx = np.where(true_ranks == rank)[0][0]
                if pred_idx == true_idx:
                    total_correct += 1
                total_items += 1
    
    return total_correct / total_items if total_items > 0 else 0.0

def compute_rank_displacement_stats(pred_list, true_list):
    """
    Compute statistics about how far off the predicted ranks are from true ranks.
    Returns mean absolute rank displacement and distribution of displacements.
    """
    all_displacements = []
    
    for pred, true in zip(pred_list, true_list):
        pred_ranks = compute_ranks(pred)
        true_ranks = compute_ranks(true)
        
        displacements = np.abs(pred_ranks - true_ranks)
        all_displacements.extend(displacements)
    
    all_displacements = np.array(all_displacements)
    
    return {
        'mean_displacement': np.mean(all_displacements),
        'median_displacement': np.median(all_displacements),
        'max_displacement': np.max(all_displacements),
        'zero_displacement_rate': np.mean(all_displacements == 0),  # % perfectly ranked
        'one_off_rate': np.mean(all_displacements <= 1),  # % within 1 rank
        'two_off_rate': np.mean(all_displacements <= 2),  # % within 2 ranks
        'three_off_rate': np.mean(all_displacements <= 3),  # % within 3 ranks
    }

def compute_position_specific_accuracy(pred_list, true_list, max_k=5):
    """
    Compute accuracy for each specific rank position (1st, 2nd, 3rd, etc.).
    Shows which positions the model predicts best.
    """
    position_correct = {}
    position_total = {}
    
    for k in range(1, max_k + 1):
        position_correct[k] = 0
        position_total[k] = 0
    
    for pred, true in zip(pred_list, true_list):
        pred_ranks = compute_ranks(pred)
        true_ranks = compute_ranks(true)
        
        for rank in range(1, min(len(pred_ranks), max_k) + 1):
            pred_idx = np.where(pred_ranks == rank)[0][0]
            true_idx = np.where(true_ranks == rank)[0][0]
            
            if pred_idx == true_idx:
                position_correct[rank] += 1
            position_total[rank] += 1
    
    position_accuracy = {}
    for k in range(1, max_k + 1):
        if position_total[k] > 0:
            position_accuracy[k] = position_correct[k] / position_total[k]
        else:
            position_accuracy[k] = None
    
    return position_accuracy

def compute_ndcg_at_k(true, pred, k=None):
    """Compute Normalized Discounted Cumulative Gain at k."""
    t = np.array(true, dtype=np.float32)
    p = np.array(pred, dtype=np.float32)
    
    if k is None:
        k = len(t)
    k = min(k, len(t))
    
    order = np.argsort(-p)[:k]
    gains = t[order]
    discounts = np.log2(np.arange(1, k+1) + 1)
    dcg = np.sum(gains / discounts)
    
    ideal_order = np.argsort(-t)[:k]
    ideal_gains = t[ideal_order]
    idcg = np.sum(ideal_gains / discounts)
    
    return dcg / idcg if idcg > 0 else 1.0

def evaluate_model(model, data_loader, graph_data, device):
    """Evaluate stretch ranking model on test data."""
    model.eval()
    
    all_pred = []
    all_true = []
    od_pairs_list = []
    od_predictions = []
    od_true_scores = []
    
    examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            
            pred_stretch = model(graph_data, batch, task='stretch')
            
            for i, true_scores in enumerate(batch["stretch_scores"]):
                od_pairs_list.append(batch["od_pairs"][i])
                pred_logits = pred_stretch[i].cpu().numpy()
                
                od_predictions.append(pred_logits)
                od_true_scores.append(true_scores)
                
                all_pred.extend(pred_logits)
                all_true.extend(true_scores)
                
                if batch_idx == 0 and i < 3:
                    pred_ranks = compute_ranks(pred_logits)
                    true_ranks = compute_ranks(true_scores)
                    
                    examples.append({
                        'od_pair': batch["od_pairs"][i],
                        'pred_logits': pred_logits,
                        'true_scores': true_scores,
                        'pred_ranks': pred_ranks,
                        'true_ranks': true_ranks,
                        'num_paths': len(pred_logits),
                        'num_correct_ranks': np.sum(pred_ranks == true_ranks)
                    })
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    from scipy.stats import spearmanr, kendalltau, pearsonr
    spearman_corr, _ = spearmanr(all_pred, all_true)
    kendall_corr, _ = kendalltau(all_pred, all_true)
    pearson_corr, _ = pearsonr(all_pred, all_true)
    
    top1_acc = compute_ranking_accuracy_at_k(od_predictions, od_true_scores, 1)
    top2_acc = compute_ranking_accuracy_at_k(od_predictions, od_true_scores, 2)
    top3_acc = compute_ranking_accuracy_at_k(od_predictions, od_true_scores, 3)
    top4_acc = compute_ranking_accuracy_at_k(od_predictions, od_true_scores, 4)
    top5_acc = compute_ranking_accuracy_at_k(od_predictions, od_true_scores, 5)
    
    item_accuracy = compute_item_level_accuracy(od_predictions, od_true_scores)
    item_top1_acc = compute_item_level_accuracy_at_k(od_predictions, od_true_scores, 1)
    item_top2_acc = compute_item_level_accuracy_at_k(od_predictions, od_true_scores, 2)
    item_top3_acc = compute_item_level_accuracy_at_k(od_predictions, od_true_scores, 3)
    item_top4_acc = compute_item_level_accuracy_at_k(od_predictions, od_true_scores, 4)
    item_top5_acc = compute_item_level_accuracy_at_k(od_predictions, od_true_scores, 5)
    
    item_accuracy_with_ties = compute_item_level_accuracy_with_ties(od_predictions, od_true_scores)
    item_top1_acc_with_ties = compute_item_level_accuracy_at_k_with_ties(od_predictions, od_true_scores, 1)
    item_top2_acc_with_ties = compute_item_level_accuracy_at_k_with_ties(od_predictions, od_true_scores, 2)
    item_top3_acc_with_ties = compute_item_level_accuracy_at_k_with_ties(od_predictions, od_true_scores, 3)
    item_top4_acc_with_ties = compute_item_level_accuracy_at_k_with_ties(od_predictions, od_true_scores, 4)
    item_top5_acc_with_ties = compute_item_level_accuracy_at_k_with_ties(od_predictions, od_true_scores, 5)
    
    tie_stats = compute_tie_statistics(od_true_scores)

    # Compute both strict and tie-friendly displacement stats
    displacement_stats = compute_rank_displacement_stats(od_predictions, od_true_scores)
    displacement_stats_ties = compute_rank_displacement_stats_with_ties(od_predictions, od_true_scores)
    
    position_acc = compute_position_specific_accuracy(od_predictions, od_true_scores, max_k=5)
    
    ndcg_scores = []
    ndcg3_scores = []
    ndcg5_scores = []
    
    for pred, true in zip(od_predictions, od_true_scores):
        ndcg_scores.append(compute_ndcg_at_k(true, pred))
        if len(pred) >= 3:
            ndcg3_scores.append(compute_ndcg_at_k(true, pred, 3))
        if len(pred) >= 5:
            ndcg5_scores.append(compute_ndcg_at_k(true, pred, 5))
    
    perfect = sum(1 for pred, true in zip(od_predictions, od_true_scores) 
                  if np.array_equal(compute_ranks(pred), compute_ranks(true)))
    
    return {
        'num_samples': len(od_predictions),
        'num_total_paths': len(all_pred),
        'spearman_corr': float(spearman_corr),
        'kendall_corr': float(kendall_corr),
        'pearson_corr': float(pearson_corr),
        'top1_accuracy': top1_acc,
        'top2_accuracy': top2_acc,
        'top3_accuracy': top3_acc,
        'top4_accuracy': top4_acc,
        'top5_accuracy': top5_acc,
        'item_accuracy': item_accuracy,
        'item_accuracy_with_ties': item_accuracy_with_ties,
        'item_top1_accuracy': item_top1_acc,
        'item_top1_accuracy_with_ties': item_top1_acc_with_ties,
        'item_top2_accuracy': item_top2_acc,
        'item_top2_accuracy_with_ties': item_top2_acc_with_ties,
        'item_top3_accuracy': item_top3_acc,
        'item_top3_accuracy_with_ties': item_top3_acc_with_ties,
        'item_top4_accuracy': item_top4_acc,
        'item_top4_accuracy_with_ties': item_top4_acc_with_ties,
        'item_top5_accuracy': item_top5_acc,
        'item_top5_accuracy_with_ties': item_top5_acc_with_ties,
        'tie_statistics': tie_stats,
        
        # Strict displacement (original)
        'mean_rank_displacement': displacement_stats['mean_displacement'],
        'median_rank_displacement': displacement_stats['median_displacement'],
        'paths_perfectly_ranked': displacement_stats['zero_displacement_rate'],
        'paths_within_1_rank': displacement_stats['one_off_rate'],
        'paths_within_2_ranks': displacement_stats['two_off_rate'],
        'paths_within_3_ranks': displacement_stats['three_off_rate'],
        
        # Tie-friendly displacement
        'mean_rank_displacement_ties': displacement_stats_ties['mean_displacement'],
        'median_rank_displacement_ties': displacement_stats_ties['median_displacement'],
        'paths_perfectly_ranked_ties': displacement_stats_ties['zero_displacement_rate'],
        'paths_within_1_rank_ties': displacement_stats_ties['one_off_rate'],
        'paths_within_2_ranks_ties': displacement_stats_ties['two_off_rate'],
        'paths_within_3_ranks_ties': displacement_stats_ties['three_off_rate'],
        
        'position_accuracy': position_acc,
        'ndcg': np.mean(ndcg_scores),
        'ndcg@3': np.mean(ndcg3_scores) if ndcg3_scores else None,
        'ndcg@5': np.mean(ndcg5_scores) if ndcg5_scores else None,
        'perfect_ranking_rate': perfect / len(od_predictions),
        'mae': float(np.mean(np.abs(all_pred - all_true))),
        'rmse': float(np.sqrt(np.mean((all_pred - all_true) ** 2))),
        'examples': examples,
        'od_predictions': od_predictions,
        'od_true_scores': od_true_scores,
        'od_pairs': od_pairs_list
    }

def compute_shortest_path_dijkstra(graph_data, start_node, end_node, max_dist=10000):
    """Compute actual shortest path using Dijkstra's algorithm."""
    edge_sd = graph_data.edge_sd
    edge_lengths = graph_data.edge_feat.squeeze(-1)
    
    adj = defaultdict(list)
    for i in range(edge_sd.shape[1]):
        u = edge_sd[0, i].item()
        v = edge_sd[1, i].item()
        length = edge_lengths[i].item()
        adj[u].append((v, length))
    
    dist = defaultdict(lambda: float('inf'))
    dist[start_node] = 0
    heap = [(0, start_node)]
    
    while heap:
        d, u = heapq.heappop(heap)
        if u == end_node:
            return d
        if d > dist[u] or d > max_dist:
            continue
        for v, length in adj[u]:
            if dist[u] + length < dist[v]:
                dist[v] = dist[u] + length
                heapq.heappush(heap, (dist[v], v))
    
    return float('inf')

def get_all_subpath_pairs(nodes):
    """Get all possible subpath node pairs from a path."""
    pairs = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pairs.append((nodes[i], nodes[j]))
    return pairs

def analyze_and_save_predictions(od_pairs, od_predictions, od_true_scores, output_file, num_good=10, num_bad=10):
    """Analyze and save examples of good and bad predictions to a file."""
    
    od_analysis = []
    for od_pair, pred_scores, true_scores in zip(od_pairs, od_predictions, od_true_scores):
        pred_ranks = compute_ranks(pred_scores)
        true_ranks = compute_ranks(true_scores)
        
        from scipy.stats import spearmanr, kendalltau
        spearman, _ = spearmanr(pred_scores, true_scores)
        kendall, _ = kendalltau(pred_scores, true_scores)
        
        rank_errors = np.sum(pred_ranks != true_ranks)
        max_rank_diff = np.max(np.abs(pred_ranks - true_ranks))
        
        pred_order = np.argsort(-np.array(pred_scores))
        true_order = np.argsort(-np.array(true_scores))
        top1_correct = pred_order[0] == true_order[0]
        top3_correct = np.array_equal(pred_order[:min(3, len(pred_order))], 
                                      true_order[:min(3, len(true_order))])
        
        n_paths = len(pred_scores)
        n_top = n_paths // 2
        n_bottom = n_paths - n_top
        
        true_top_indices = set(true_order[:n_top])
        true_bottom_indices = set(true_order[n_top:])
        
        top_half_pred = [pred_scores[i] for i in true_top_indices]
        top_half_true = [true_scores[i] for i in true_top_indices]
        bottom_half_pred = [pred_scores[i] for i in true_bottom_indices]
        bottom_half_true = [true_scores[i] for i in true_bottom_indices]
        
        if len(top_half_pred) > 1:
            top_half_spearman, _ = spearmanr(top_half_pred, top_half_true)
        else:
            top_half_spearman = 1.0
            
        if len(bottom_half_pred) > 1:
            bottom_half_spearman, _ = spearmanr(bottom_half_pred, bottom_half_true)
        else:
            bottom_half_spearman = 1.0
        
        pred_top_indices = set(pred_order[:n_top])
        pred_bottom_indices = set(pred_order[n_top:])
        
        top_half_accuracy = len(true_top_indices.intersection(pred_top_indices)) / n_top
        bottom_half_accuracy = len(true_bottom_indices.intersection(pred_bottom_indices)) / n_bottom
        
        od_analysis.append({
            'od_pair': od_pair,
            'pred_scores': pred_scores,
            'true_scores': true_scores,
            'pred_ranks': pred_ranks,
            'true_ranks': true_ranks,
            'spearman': spearman,
            'kendall': kendall,
            'rank_errors': rank_errors,
            'max_rank_diff': max_rank_diff,
            'top1_correct': top1_correct,
            'top3_correct': top3_correct,
            'num_paths': len(pred_scores),
            'perfect_match': rank_errors == 0,
            'top_half_spearman': top_half_spearman,
            'bottom_half_spearman': bottom_half_spearman,
            'top_half_accuracy': top_half_accuracy,
            'bottom_half_accuracy': bottom_half_accuracy,
            'true_top_indices': true_top_indices,
            'true_bottom_indices': true_bottom_indices
        })
    
    od_analysis.sort(key=lambda x: x['spearman'])
    
    bad_examples = od_analysis[:num_bad]
    good_examples = od_analysis[-num_good:]
    good_examples.reverse()
    
    perfect_matches = [a for a in od_analysis if a['perfect_match']][:5]
    
    od_analysis_copy = od_analysis.copy()
    od_analysis_copy.sort(key=lambda x: x['bottom_half_spearman'])
    bad_bottom_half = [a for a in od_analysis_copy[:20] if not a['perfect_match']][:5]
    good_bottom_half = [a for a in od_analysis_copy[-20:] if not a['perfect_match']][:5]
    good_bottom_half.reverse()
    
    od_analysis_copy.sort(key=lambda x: x['top_half_spearman'])
    bad_top_half = [a for a in od_analysis_copy[:20] if not a['perfect_match']][:5]
    good_top_half = [a for a in od_analysis_copy[-20:] if not a['perfect_match']][:5]
    good_top_half.reverse()
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STRETCH PREDICTION ANALYSIS: COMPREHENSIVE EVALUATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write(f"Total OD pairs analyzed: {len(od_analysis)}\n")
        perfect_rankings = sum(1 for a in od_analysis if a['perfect_match'])
        f.write(f"Perfect rankings: {perfect_rankings} ({100*perfect_rankings/len(od_analysis):.1f}%)\n")
        top1_correct = sum(1 for a in od_analysis if a['top1_correct'])
        f.write(f"Top-1 correct: {top1_correct} ({100*top1_correct/len(od_analysis):.1f}%)\n")
        top3_correct = sum(1 for a in od_analysis if a['top3_correct'])
        f.write(f"Top-3 correct: {top3_correct} ({100*top3_correct/len(od_analysis):.1f}%)\n")
        
        avg_spearman = np.mean([a['spearman'] for a in od_analysis])
        f.write(f"Average Spearman correlation: {avg_spearman:.4f}\n")
        avg_top_half_spearman = np.mean([a['top_half_spearman'] for a in od_analysis])
        f.write(f"Average Top-Half Spearman: {avg_top_half_spearman:.4f}\n")
        avg_bottom_half_spearman = np.mean([a['bottom_half_spearman'] for a in od_analysis])
        f.write(f"Average Bottom-Half Spearman: {avg_bottom_half_spearman:.4f}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        def write_example(f, example, title):
            f.write(f"{title}: OD pair {example['od_pair']}\n")
            f.write(f"  Number of paths: {example['num_paths']}\n")
            f.write(f"  Overall Spearman: {example['spearman']:.4f}\n")
            f.write(f"  Top-Half Spearman: {example['top_half_spearman']:.4f}\n")
            f.write(f"  Bottom-Half Spearman: {example['bottom_half_spearman']:.4f}\n")
            f.write(f"  Top-Half Assignment Accuracy: {example['top_half_accuracy']:.2%}\n")
            f.write(f"  Bottom-Half Assignment Accuracy: {example['bottom_half_accuracy']:.2%}\n")
            f.write(f"  Ranking errors: {example['rank_errors']}\n\n")
            
            f.write("  Path | Pred Score | True Score | Pred Rank | True Rank | Half | Error\n")
            f.write("  -----|------------|------------|-----------|-----------|------|-------\n")
            
            for j in range(len(example['pred_scores'])):
                rank_diff = example['pred_ranks'][j] - example['true_ranks'][j]
                error_marker = "OK" if rank_diff == 0 else f"({rank_diff:+d})"
                half = "TOP" if j in example['true_top_indices'] else "BTM"
                f.write(f"  {j+1:4d} | {example['pred_scores'][j]:10.2f} | "
                       f"{example['true_scores'][j]:10.2f} | "
                       f"{example['pred_ranks'][j]:9d} | "
                       f"{example['true_ranks'][j]:9d} | {half:4s} | {error_marker:7s}\n")
            f.write("\n")
        
        if perfect_matches:
            f.write("PERFECT MATCHES (Exact ranking reproduction):\n")
            f.write("-"*80 + "\n\n")
            for i, example in enumerate(perfect_matches, 1):
                write_example(f, example, f"Perfect Match {i}")
            f.write("\n" + "="*80 + "\n\n")
        
        f.write("GOOD BOTTOM-HALF PREDICTIONS (Lower scores well predicted):\n")
        f.write("-"*80 + "\n\n")
        for i, example in enumerate(good_bottom_half, 1):
            write_example(f, example, f"Good Bottom-Half {i}")
        
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("BAD BOTTOM-HALF PREDICTIONS (Lower scores poorly predicted):\n")
        f.write("-"*80 + "\n\n")
        for i, example in enumerate(bad_bottom_half, 1):
            write_example(f, example, f"Bad Bottom-Half {i}")
        
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("GOOD TOP-HALF PREDICTIONS (Higher scores well predicted):\n")
        f.write("-"*80 + "\n\n")
        for i, example in enumerate(good_top_half, 1):
            write_example(f, example, f"Good Top-Half {i}")
        
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("BAD TOP-HALF PREDICTIONS (Higher scores poorly predicted):\n")
        f.write("-"*80 + "\n\n")
        for i, example in enumerate(bad_top_half, 1):
            write_example(f, example, f"Bad Top-Half {i}")
        
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("OVERALL GOOD PREDICTIONS (High correlation):\n")
        f.write("-"*80 + "\n\n")
        for i, example in enumerate(good_examples[:5], 1):
            write_example(f, example, f"Good Overall {i}")
        
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("OVERALL BAD PREDICTIONS (Low correlation):\n")
        f.write("-"*80 + "\n\n")
        for i, example in enumerate(bad_examples[:5], 1):
            write_example(f, example, f"Bad Overall {i}")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ERROR DISTRIBUTION ANALYSIS:\n")
        f.write("-"*80 + "\n\n")
        
        rank_error_dist = [a['rank_errors'] for a in od_analysis]
        f.write("Ranking errors distribution:\n")
        for errors in range(min(10, max(rank_error_dist) + 1)):
            count = rank_error_dist.count(errors)
            if count > 0:
                f.write(f"  {errors} errors: {count} OD pairs ({100*count/len(od_analysis):.1f}%)\n")
        
        f.write(f"\nMean ranking errors: {np.mean(rank_error_dist):.2f}\n")
        f.write(f"Median ranking errors: {np.median(rank_error_dist):.1f}\n")
        
        f.write("\nHalf-Prediction Quality:\n")
        top_half_good = sum(1 for a in od_analysis if a['top_half_spearman'] > 0.5)
        bottom_half_good = sum(1 for a in od_analysis if a['bottom_half_spearman'] > 0.5)
        f.write(f"  Top-half well predicted (Spearman > 0.5): {top_half_good} ({100*top_half_good/len(od_analysis):.1f}%)\n")
        f.write(f"  Bottom-half well predicted (Spearman > 0.5): {bottom_half_good} ({100*bottom_half_good/len(od_analysis):.1f}%)\n")
        
        both_good = sum(1 for a in od_analysis if a['top_half_spearman'] > 0.5 and a['bottom_half_spearman'] > 0.5)
        f.write(f"  Both halves well predicted: {both_good} ({100*both_good/len(od_analysis):.1f}%)\n")
        
    print(f"\nDetailed prediction analysis saved to: {output_file}")
    
    return od_analysis

def compute_item_level_accuracy_at_k_with_ties(pred_list, true_list, k, tie_tolerance=0.01):
    """
    Compute tie-tolerant item-level accuracy for top-k positions only.
    
    For each position 1 to k, checks if the predicted item at that position
    is within the valid tie group of the ground truth item at that position.
    """
    total_correct = 0
    total_items = 0
    
    for pred, true in zip(pred_list, true_list):
        if len(pred) >= k:
            pred_ranks = compute_ranks(pred)
            true_ranks = compute_ranks(true)
            true_scores = np.array(true)
            
            for rank in range(1, k+1):
                # Find which path the model predicted at this rank
                pred_idx = np.where(pred_ranks == rank)[0][0]
                # Find which path should be at this rank in ground truth
                true_idx = np.where(true_ranks == rank)[0][0]
                
                # Get the ground truth score of the item that should be at this position
                target_true_score = true_scores[true_idx]
                
                # Find all items tied with this score
                tied_indices = np.where(np.abs(true_scores - target_true_score) <= tie_tolerance)[0]
                min_valid_rank = true_ranks[tied_indices].min()
                max_valid_rank = true_ranks[tied_indices].max()
                
                # Check if predicted item at this position falls within the valid tie group
                pred_item_true_rank = true_ranks[pred_idx]
                if min_valid_rank <= pred_item_true_rank <= max_valid_rank:
                    total_correct += 1
                    
                total_items += 1
    
    return total_correct / total_items if total_items > 0 else 0.0

def compute_item_level_accuracy_with_ties(pred_list, true_list, tie_tolerance=0.01):
    """
    Compute item-level accuracy with tie tolerance.
    
    For paths with tied ground truth scores, any predicted rank within the tied group
    is considered correct. For paths with unique scores, exact rank matching is required.
    
    Args:
        pred_list: List of prediction arrays for each OD pair
        true_list: List of ground truth score arrays for each OD pair  
        tie_tolerance: Score difference threshold to consider as tied (default 0.01)
    
    Returns:
        Tie-tolerant item-level accuracy (0-1)
    """
    total_correct = 0
    total_paths = 0
    
    for pred, true in zip(pred_list, true_list):
        pred_ranks = compute_ranks(pred)
        true_ranks = compute_ranks(true)
        true_scores = np.array(true)
        
        for path_idx in range(len(pred)):
            path_true_score = true_scores[path_idx]
            path_pred_rank = pred_ranks[path_idx]
            path_true_rank = true_ranks[path_idx]
            
            tied_indices = np.where(np.abs(true_scores - path_true_score) <= tie_tolerance)[0]
            min_valid_rank = true_ranks[tied_indices].min()
            max_valid_rank = true_ranks[tied_indices].max()
            
            if min_valid_rank <= path_pred_rank <= max_valid_rank:
                total_correct += 1
            
            total_paths += 1
    
    return total_correct / total_paths if total_paths > 0 else 0.0


def compute_tie_statistics(true_list, tie_tolerance=0.01):
    """
    Compute statistics about ties in ground truth scores.
    
    Returns dict with:
    - num_od_with_ties: Number of OD pairs with at least one tie
    - total_tied_paths: Total number of paths involved in ties
    - max_tie_group_size: Largest tie group size
    - tie_group_distribution: Counter of tie group sizes
    """
    from collections import Counter
    
    num_od_with_ties = 0
    total_tied_paths = 0
    tie_group_sizes = []
    
    for true in true_list:
        true_scores = np.array(true)
        has_ties = False
        accounted_indices = set()
        
        for idx in range(len(true_scores)):
            if idx in accounted_indices:
                continue
                
            tied_indices = np.where(np.abs(true_scores - true_scores[idx]) <= tie_tolerance)[0]
            
            if len(tied_indices) > 1:
                has_ties = True
                tie_group_sizes.append(len(tied_indices))
                total_tied_paths += len(tied_indices)
                accounted_indices.update(tied_indices)
        
        if has_ties:
            num_od_with_ties += 1
    
    return {
        'num_od_with_ties': num_od_with_ties,
        'pct_od_with_ties': 100 * num_od_with_ties / len(true_list) if len(true_list) > 0 else 0,
        'total_tied_paths': total_tied_paths,
        'max_tie_group_size': max(tie_group_sizes) if tie_group_sizes else 0,
        'tie_group_distribution': dict(Counter(tie_group_sizes))
    }

def compute_rank_displacement_stats_with_ties(pred_list, true_list, tie_tolerance=0.01):
    """
    Compute tie-tolerant rank displacement statistics.
    
    For each path, if its predicted rank falls within the valid tie group 
    of its ground truth rank, the displacement is 0. Otherwise, compute 
    the displacement as the distance to the nearest valid rank in the tie group.
    
    Args:
        pred_list: List of prediction arrays for each OD pair
        true_list: List of ground truth score arrays for each OD pair
        tie_tolerance: Score difference threshold to consider as tied (default 0.01)
    
    Returns:
        Dict with displacement statistics considering ties
    """
    all_displacements = []
    
    for pred, true in zip(pred_list, true_list):
        pred_ranks = compute_ranks(pred)
        true_ranks = compute_ranks(true)
        true_scores = np.array(true)
        
        for path_idx in range(len(pred)):
            path_true_score = true_scores[path_idx]
            path_pred_rank = pred_ranks[path_idx]
            path_true_rank = true_ranks[path_idx]
            
            # Find all paths tied with this path's ground truth score
            tied_indices = np.where(np.abs(true_scores - path_true_score) <= tie_tolerance)[0]
            min_valid_rank = true_ranks[tied_indices].min()
            max_valid_rank = true_ranks[tied_indices].max()
            
            # If predicted rank is within the valid tie group, displacement is 0
            if min_valid_rank <= path_pred_rank <= max_valid_rank:
                displacement = 0
            else:
                # Otherwise, compute distance to nearest valid rank
                if path_pred_rank < min_valid_rank:
                    displacement = min_valid_rank - path_pred_rank
                else:  # path_pred_rank > max_valid_rank
                    displacement = path_pred_rank - max_valid_rank
            
            all_displacements.append(displacement)
    
    all_displacements = np.array(all_displacements)
    
    return {
        'mean_displacement': np.mean(all_displacements),
        'median_displacement': np.median(all_displacements),
        'max_displacement': np.max(all_displacements),
        'zero_displacement_rate': np.mean(all_displacements == 0),
        'one_off_rate': np.mean(all_displacements <= 1),
        'two_off_rate': np.mean(all_displacements <= 2),
        'three_off_rate': np.mean(all_displacements <= 3),
    }

def analyze_score_patterns_and_samples(metrics, args, num_samples=5):
    """Analyze score distributions and collect categorized samples"""
    
    all_pred = np.concatenate(metrics['od_predictions'])
    all_true = np.concatenate(metrics['od_true_scores'])
    
    print("\n" + "="*70)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print("\nGround Truth Scores:")
    print(f"  Mean: {np.mean(all_true):.2f}, Std: {np.std(all_true):.2f}")
    print(f"  Min: {np.min(all_true):.2f}, Max: {np.max(all_true):.2f}")
    percentiles_true = np.percentile(all_true, [10, 25, 50, 75, 90])
    print(f"  Percentiles [10, 25, 50, 75, 90]: [{percentiles_true[0]:.1f}, {percentiles_true[1]:.1f}, {percentiles_true[2]:.1f}, {percentiles_true[3]:.1f}, {percentiles_true[4]:.1f}]")
    print(f"  Zero scores (<1): {np.sum(all_true < 1)} ({100*np.mean(all_true < 1):.1f}%)")
    print(f"  Perfect scores (>99): {np.sum(all_true > 99)} ({100*np.mean(all_true > 99):.1f}%)")
    
    print("\nPrediction Scores:")
    print(f"  Mean: {np.mean(all_pred):.2f}, Std: {np.std(all_pred):.2f}")
    print(f"  Min: {np.min(all_pred):.2f}, Max: {np.max(all_pred):.2f}")
    percentiles_pred = np.percentile(all_pred, [10, 25, 50, 75, 90])
    print(f"  Percentiles [10, 25, 50, 75, 90]: [{percentiles_pred[0]:.1f}, {percentiles_pred[1]:.1f}, {percentiles_pred[2]:.1f}, {percentiles_pred[3]:.1f}, {percentiles_pred[4]:.1f}]")
    print(f"  Zero predictions (<1): {np.sum(all_pred < 1)} ({100*np.mean(all_pred < 1):.1f}%)")
    print(f"  Perfect predictions (>99): {np.sum(all_pred > 99)} ({100*np.mean(all_pred > 99):.1f}%)")
    
    print("\nScore Compression Analysis:")
    score_compression = np.std(all_pred) / np.std(all_true) if np.std(all_true) > 0 else 0
    range_true = np.max(all_true) - np.min(all_true)
    range_pred = np.max(all_pred) - np.min(all_pred)
    range_compression = range_pred / range_true if range_true > 0 else 0
    mean_shift = np.mean(all_pred) - np.mean(all_true)
    
    print(f"  Std compression ratio: {score_compression:.3f} {'(compressed)' if score_compression < 1 else '(expanded)'}")
    print(f"  Range compression: {range_compression:.3f} {'(compressed)' if range_compression < 1 else '(expanded)'}")
    print(f"  Mean shift: {mean_shift:+.2f} {'(overestimating)' if mean_shift > 0 else '(underestimating)'}")
    
    true_extremes = (all_true < 1) | (all_true > 99)
    pred_extremes = (all_pred < 1) | (all_pred > 99)
    extreme_ratio = np.mean(pred_extremes) / np.mean(true_extremes) if np.mean(true_extremes) > 0 else 0
    
    print(f"  Extreme scores ratio: {extreme_ratio:.3f} {'(avoiding extremes)' if extreme_ratio < 1 else '(matching extremes)' if extreme_ratio > 0.9 else ''}")
    
    print("\nScore Patterns by Rank Position:")
    rank_stats = {'top': {'pred': [], 'true': []}, 
                  'bottom': {'pred': [], 'true': []}}
    
    for pred_scores, true_scores in zip(metrics['od_predictions'], metrics['od_true_scores']):
        if len(pred_scores) >= 2:
            true_order = np.argsort(-np.array(true_scores))
            n_top = len(true_order) // 2
            
            top_indices = true_order[:n_top]
            rank_stats['top']['pred'].extend([pred_scores[i] for i in top_indices])
            rank_stats['top']['true'].extend([true_scores[i] for i in top_indices])
            
            bottom_indices = true_order[n_top:]
            rank_stats['bottom']['pred'].extend([pred_scores[i] for i in bottom_indices])
            rank_stats['bottom']['true'].extend([true_scores[i] for i in bottom_indices])
    
    top_compression = np.std(rank_stats['top']['pred']) / np.std(rank_stats['top']['true']) if np.std(rank_stats['top']['true']) > 0 else 0
    bottom_compression = np.std(rank_stats['bottom']['pred']) / np.std(rank_stats['bottom']['true']) if np.std(rank_stats['bottom']['true']) > 0 else 0
    
    print(f"  Top-ranked paths:")
    print(f"    True: Mean={np.mean(rank_stats['top']['true']):.2f}, Std={np.std(rank_stats['top']['true']):.2f}")
    print(f"    Pred: Mean={np.mean(rank_stats['top']['pred']):.2f}, Std={np.std(rank_stats['top']['pred']):.2f}")
    print(f"    Compression: {top_compression:.3f}")
    
    print(f"  Bottom-ranked paths:")
    print(f"    True: Mean={np.mean(rank_stats['bottom']['true']):.2f}, Std={np.std(rank_stats['bottom']['true']):.2f}")
    print(f"    Pred: Mean={np.mean(rank_stats['bottom']['pred']):.2f}, Std={np.std(rank_stats['bottom']['pred']):.2f}")
    print(f"    Compression: {bottom_compression:.3f}")
    
    od_samples = []
    for od_pair, pred_scores, true_scores in zip(
        metrics['od_pairs'], 
        metrics['od_predictions'], 
        metrics['od_true_scores']
    ):
        pred_ranks = compute_ranks(pred_scores)
        true_ranks = compute_ranks(true_scores)
        
        correct_ranks = np.sum(pred_ranks == true_ranks)
        item_accuracy = correct_ranks / len(pred_ranks)
        
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(pred_scores, true_scores)
        
        od_samples.append({
            'source': int(od_pair[0]),
            'destination': int(od_pair[1]),
            'item_accuracy': float(item_accuracy),
            'spearman': float(spearman),
            'num_paths': len(pred_scores),
            'correct_ranks': int(correct_ranks),
            'pred_scores': [float(s) for s in pred_scores],
            'true_scores': [float(s) for s in true_scores],
            'pred_ranks': [int(r) for r in pred_ranks],
            'true_ranks': [int(r) for r in true_ranks]
        })
    
    od_samples.sort(key=lambda x: x['item_accuracy'], reverse=True)
    
    good_samples = []
    moderate_samples = []
    bad_samples = []
    
    for od in od_samples:
        if od['item_accuracy'] >= 0.7 and len(good_samples) < num_samples:
            good_samples.append(od)
        elif 0.3 <= od['item_accuracy'] < 0.7 and len(moderate_samples) < num_samples:
            moderate_samples.append(od)
        elif od['item_accuracy'] < 0.3 and len(bad_samples) < num_samples:
            bad_samples.append(od)
    
    print("\n" + "="*70)
    print("SAMPLE OD PAIRS BY PREDICTION QUALITY")
    print("="*70)
    
    total_good = sum(1 for od in od_samples if od['item_accuracy'] >= 0.7)
    total_moderate = sum(1 for od in od_samples if 0.3 <= od['item_accuracy'] < 0.7)
    total_bad = sum(1 for od in od_samples if od['item_accuracy'] < 0.3)
    
    print(f"\nGood predictions (>=70% accuracy): {total_good} total, {len(good_samples)} sampled")
    for i, od in enumerate(good_samples[:3]):
        print(f"  {od['source']}->{od['destination']}: {od['item_accuracy']:.1%} accurate, "
              f"Spearman={od['spearman']:.3f}, {od['num_paths']} paths")
    
    print(f"\nModerate predictions (30-70%): {total_moderate} total, {len(moderate_samples)} sampled")
    for i, od in enumerate(moderate_samples[:3]):
        print(f"  {od['source']}->{od['destination']}: {od['item_accuracy']:.1%} accurate, "
              f"Spearman={od['spearman']:.3f}, {od['num_paths']} paths")
    
    print(f"\nBad predictions (<30%): {total_bad} total, {len(bad_samples)} sampled")
    for i, od in enumerate(bad_samples[:3]):
        print(f"  {od['source']}->{od['destination']}: {od['item_accuracy']:.1%} accurate, "
              f"Spearman={od['spearman']:.3f}, {od['num_paths']} paths")
    
    analysis_log = {
        'dataset': args.data_dir,
        'model_path': f"{args.data_dir}/models/stretch_ml_best{args.model_suffix}.pth",
        'exclude_main': args.exclude_main,
        'total_od_pairs': len(od_samples),
        'score_distributions': {
            'ground_truth': {
                'mean': float(np.mean(all_true)),
                'std': float(np.std(all_true)),
                'min': float(np.min(all_true)),
                'max': float(np.max(all_true)),
                'percentiles': percentiles_true.tolist(),
                'zero_count': int(np.sum(all_true < 1)),
                'zero_percent': float(100*np.mean(all_true < 1)),
                'perfect_count': int(np.sum(all_true > 99)),
                'perfect_percent': float(100*np.mean(all_true > 99))
            },
            'predictions': {
                'mean': float(np.mean(all_pred)),
                'std': float(np.std(all_pred)),
                'min': float(np.min(all_pred)),
                'max': float(np.max(all_pred)),
                'percentiles': percentiles_pred.tolist(),
                'zero_count': int(np.sum(all_pred < 1)),
                'zero_percent': float(100*np.mean(all_pred < 1)),
                'perfect_count': int(np.sum(all_pred > 99)),
                'perfect_percent': float(100*np.mean(all_pred > 99))
            },
            'compression_metrics': {
                'std_compression': float(score_compression),
                'range_compression': float(range_compression),
                'mean_shift': float(mean_shift),
                'extreme_ratio': float(extreme_ratio),
                'top_half_compression': float(top_compression),
                'bottom_half_compression': float(bottom_compression)
            },
            'by_rank': {
                'top_half_true_mean': float(np.mean(rank_stats['top']['true'])),
                'top_half_true_std': float(np.std(rank_stats['top']['true'])),
                'top_half_pred_mean': float(np.mean(rank_stats['top']['pred'])),
                'top_half_pred_std': float(np.std(rank_stats['top']['pred'])),
                'bottom_half_true_mean': float(np.mean(rank_stats['bottom']['true'])),
                'bottom_half_true_std': float(np.std(rank_stats['bottom']['true'])),
                'bottom_half_pred_mean': float(np.mean(rank_stats['bottom']['pred'])),
                'bottom_half_pred_std': float(np.std(rank_stats['bottom']['pred']))
            }
        },
        'od_samples': {
            'good': good_samples,
            'moderate': moderate_samples,
            'bad': bad_samples
        },
        'od_category_counts': {
            'good': total_good,
            'moderate': total_moderate,
            'bad': total_bad
        }
    }
    
    results_dir = f"{args.data_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = f"{results_dir}/stretch_ml_best{args.model_suffix}_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(analysis_log, f, indent=2)
    
    print(f"\nAnalysis log saved to: {output_file}")
    
    return analysis_log

def _removed_create_latex_figures_data(metrics, args):
    """Create data files for LaTeX figures and generate LaTeX snippet"""
    results_dir = f"{args.data_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    od_predictions = metrics['od_predictions']
    od_true_scores = metrics['od_true_scores']
    
    all_pred = np.array([s for scores in od_predictions for s in scores])
    all_true = np.array([s for scores in od_true_scores for s in scores])
    
    # 1. Score distribution statistics
    pred_min, pred_max = all_pred.min(), all_pred.max()
    true_min, true_max = all_true.min(), all_true.max()
    pred_range = pred_max - pred_min
    true_range = true_max - true_min
    
    # 2. Per-OD Spearman correlations
    per_od_spearmans = []
    per_od_stds = []
    for pred, true in zip(od_predictions, od_true_scores):
        if len(pred) > 1:
            from scipy.stats import spearmanr
            spearman, _ = spearmanr(pred, true)
            per_od_spearmans.append(spearman)
            per_od_stds.append(np.std(true))
    
    per_od_spearmans = np.array(per_od_spearmans)
    per_od_stds = np.array(per_od_stds)
    
    # Calculate key statistics
    high_corr_percent = np.mean(per_od_spearmans > 0.8) * 100
    low_corr_percent = np.mean(per_od_spearmans < 0.5) * 100
    low_variance_threshold = 10
    low_variance_ods = np.sum(per_od_stds < low_variance_threshold)
    
    # 3. Save data for plotting
    plot_data = {
        'score_distribution': {
            'predicted': all_pred.tolist(),
            'ground_truth': all_true.tolist()
        },
        'scatter_data': {
            'predicted': all_pred.tolist(),
            'ground_truth': all_true.tolist()
        },
        'per_od_spearmans': per_od_spearmans.tolist(),
        'per_od_stds': per_od_stds.tolist()
    }
    
    with open(f"{results_dir}/stretch_plot_data{args.model_suffix}.json", 'w') as f:
        json.dump(plot_data, f)
    
    # 4. Generate LaTeX snippet
    latex_snippet = f"""
% Statistics for LaTeX document

\\paragraph{{Score Distribution.}}
Examination of Figure~\\ref{{fig:stretch_distribution}} reveals {'slight compression in the prediction range' if pred_range < true_range * 0.9 else 'good coverage of the score space'}. The predicted scores span approximately [{pred_min:.1f}, {pred_max:.1f}] compared to ground truth range [{true_min:.1f}, {true_max:.1f}], indicating {'slight compression' if pred_range < true_range * 0.9 else 'well-calibrated predictions' if abs(pred_range - true_range) < true_range * 0.1 else 'similar range coverage'}.

\\paragraph{{Per-OD Variability.}}
Figure~\\ref{{fig:stretch_per_od}} shows the distribution of Spearman correlations across individual OD pairs.
The distribution is {'heavily skewed toward high values' if high_corr_percent > 60 else 'broadly distributed'}, with {high_corr_percent:.1f}\\% of OD pairs achieving $\\rho > 0.8$. {'The few outliers' if low_corr_percent < 20 else 'A notable proportion'} ($\\rho < 0.5$, comprising {low_corr_percent:.1f}\\% of pairs) {'correspond to OD pairs where all candidate paths have very similar stretch scores (standard deviation < ' + str(low_variance_threshold) + ' points), making the ranking task inherently ambiguous' if low_variance_ods > len(per_od_spearmans) * 0.1 else 'suggest remaining challenges in predicting stretch scores for certain path configurations'}.

% Key statistics:
% - Predicted range: [{pred_min:.2f}, {pred_max:.2f}]
% - Ground truth range: [{true_min:.2f}, {true_max:.2f}]
% - Range compression ratio: {pred_range/true_range:.3f}
% - OD pairs with ρ > 0.8: {high_corr_percent:.1f}%
% - OD pairs with ρ < 0.5: {low_corr_percent:.1f}%
% - OD pairs with std < {low_variance_threshold}: {low_variance_ods}
% - Mean per-OD Spearman: {np.mean(per_od_spearmans):.4f}
% - Median per-OD Spearman: {np.median(per_od_spearmans):.4f}
"""
    
    with open(f"{results_dir}/stretch_latex_snippet{args.model_suffix}.tex", 'w', encoding='utf-8') as f:
        f.write(latex_snippet)
    
    print(f"\n{'='*70}")
    print("LATEX FIGURE DATA GENERATED")
    print(f"{'='*70}")
    print(f"\nScore Range:")
    print(f"  Predicted:     [{pred_min:.2f}, {pred_max:.2f}] (range: {pred_range:.2f})")
    print(f"  Ground Truth:  [{true_min:.2f}, {true_max:.2f}] (range: {true_range:.2f})")
    print(f"  Compression:   {pred_range/true_range:.3f}")
    
    print(f"\nPer-OD Correlation Distribution:")
    print(f"  Mean Spearman:     {np.mean(per_od_spearmans):.4f}")
    print(f"  Median Spearman:   {np.median(per_od_spearmans):.4f}")
    print(f"  OD pairs with Spearman > 0.8: {high_corr_percent:.1f}%")
    print(f"  OD pairs with Spearman < 0.5: {low_corr_percent:.1f}%")
    print(f"  Low variance ODs:  {low_variance_ods} (std < {low_variance_threshold})")
    
    print(f"\nFiles saved:")
    print(f"  - {results_dir}/stretch_plot_data{args.model_suffix}.json")
    print(f"  - {results_dir}/stretch_latex_snippet{args.model_suffix}.tex")
    
    return plot_data

def _removed_generate_matplotlib_figures(plot_data, args):
    """Generate actual figure files using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        results_dir = f"{args.data_dir}/results"
        
        pred = np.array(plot_data['score_distribution']['predicted'])
        true = np.array(plot_data['score_distribution']['ground_truth'])
        per_od_spearmans = np.array(plot_data['per_od_spearmans'])
        
        # Figure 1: Score distribution histogram
        plt.figure(figsize=(10, 6))
        bins = np.linspace(0, 100, 51)
        plt.hist(true, bins=bins, alpha=0.5, label='Ground Truth', color='blue', edgecolor='black')
        plt.hist(pred, bins=bins, alpha=0.5, label='Predicted', color='red', edgecolor='black')
        plt.xlabel('Stretch Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Stretch Score Distribution: Predicted vs Ground Truth', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/stretch_distribution{args.model_suffix}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{results_dir}/stretch_distribution{args.model_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Scatter plot with regression
        plt.figure(figsize=(10, 10))
        plt.scatter(true, pred, alpha=0.3, s=10, color='blue')
        
        z = np.polyfit(true, pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(true.min(), true.max(), 100)
        plt.plot(x_line, p(x_line), "r-", linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
        
        plt.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5, label='Perfect prediction')
        
        plt.xlabel('Ground Truth Stretch Score', fontsize=12)
        plt.ylabel('Predicted Stretch Score', fontsize=12)
        plt.title('Stretch Score: Predicted vs Ground Truth', fontsize=14)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/stretch_scatter{args.model_suffix}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{results_dir}/stretch_scatter{args.model_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Per-OD Spearman distribution
        plt.figure(figsize=(10, 6))
        plt.hist(per_od_spearmans, bins=50, color='green', edgecolor='black', alpha=0.7)
        plt.axvline(np.median(per_od_spearmans), color='red', linestyle='--', 
                   linewidth=2, label=f'Median = {np.median(per_od_spearmans):.3f}')
        plt.axvline(np.mean(per_od_spearmans), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean = {np.mean(per_od_spearmans):.3f}')
        plt.xlabel('Spearman Correlation (ρ)', fontsize=12)
        plt.ylabel('Number of OD Pairs', fontsize=12)
        plt.title('Per-OD Pair Spearman Correlation Distribution', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/stretch_per_od{args.model_suffix}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{results_dir}/stretch_per_od{args.model_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nMatplotlib figures generated:")
        print(f"  - stretch_distribution{args.model_suffix}.pdf/png")
        print(f"  - stretch_scatter{args.model_suffix}.pdf/png")
        print(f"  - stretch_per_od{args.model_suffix}.pdf/png")
        
    except ImportError:
        print("\nMatplotlib not available. Only JSON data saved.")
        print("You can generate figures later using the saved JSON data.")

def analyze_stretch_prediction_cases(metrics, dataset, args, num_good=5, num_bad=5):
    results_dir = f"{args.data_dir}/results"
    od_case_data = []
    
    for idx, (od_pair, pred_scores, true_scores) in enumerate(zip(
        metrics['od_pairs'], 
        metrics['od_predictions'], 
        metrics['od_true_scores']
    )):
        # KEEP THIS PART - ensures type consistency
        if isinstance(od_pair, (list, np.ndarray)):
            od_pair_tuple = tuple(int(x) for x in od_pair)  # Convert to Python ints
        else:
            od_pair_tuple = od_pair
        
        # Now this will work with the dataset mapping
        od_idx = dataset.od_to_index.get(od_pair_tuple, None)
        if od_idx is None:
            print(f"Warning: Could not find OD pair {od_pair_tuple} in dataset, skipping...")
            continue
            
        paths = dataset.candidate_paths[od_idx]
        sharing_scores = dataset.candidate_sharing_scores[od_idx]
        
        # Calculate path features
        path_lengths = []
        hop_counts = []
        
        for path in paths:
            hop_counts.append(len(path) - 1)
            length = 0
            for i in range(len(path) - 1):
                edge_idx = dataset.edge_lookup.get((path[i], path[i+1]), None)
                if edge_idx is not None:
                    length += dataset.graph_data.edge_feat[edge_idx, 0].item()
            path_lengths.append(length)
        
        path_lengths = np.array(path_lengths)
        hop_counts = np.array(hop_counts)
        sharing_scores = np.array(sharing_scores)
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        
        # Compute ranks
        true_ranks = compute_ranks(true_scores)
        pred_ranks = compute_ranks(pred_scores)
        length_ranks = compute_ranks(-path_lengths)  # Negative: shorter is better
        hop_ranks = compute_ranks(-hop_counts)
        sharing_ranks = compute_ranks(sharing_scores)
        
        # Compute feature correlations
        from scipy.stats import spearmanr
        
        if len(paths) > 2:
            corr_pred_true, _ = spearmanr(pred_scores, true_scores)
            corr_length_true, _ = spearmanr(path_lengths, true_scores)
            corr_length_pred, _ = spearmanr(path_lengths, pred_scores)
            corr_sharing_true, _ = spearmanr(sharing_scores, true_scores)
            corr_sharing_pred, _ = spearmanr(sharing_scores, pred_scores)
            corr_hops_true, _ = spearmanr(hop_counts, true_scores)
            corr_hops_pred, _ = spearmanr(hop_counts, pred_scores)
        else:
            corr_pred_true = corr_length_true = corr_length_pred = 0
            corr_sharing_true = corr_sharing_pred = corr_hops_true = corr_hops_pred = 0
        
        # Rank matching rates
        length_rank_match = np.mean(length_ranks == pred_ranks)
        sharing_rank_match = np.mean(sharing_ranks == pred_ranks)
        hop_rank_match = np.mean(hop_ranks == pred_ranks)
        true_rank_match = np.mean(true_ranks == pred_ranks)
        
        # Store case data
        od_case_data.append({
            'od_pair': od_pair,
            'num_paths': len(paths),
            'prediction_quality': float(corr_pred_true),
            'paths': [
                {
                    'path_idx': i,
                    'length': float(path_lengths[i]),
                    'hops': int(hop_counts[i]),
                    'sharing': float(sharing_scores[i]),
                    'true_stretch': float(true_scores[i]),
                    'pred_stretch': float(pred_scores[i]),
                    'true_rank': int(true_ranks[i]),
                    'pred_rank': int(pred_ranks[i]),
                    'length_rank': int(length_ranks[i]),
                    'sharing_rank': int(sharing_ranks[i]),
                    'hop_rank': int(hop_ranks[i])
                }
                for i in range(len(paths))
            ],
            'correlations': {
                'pred_vs_true': float(corr_pred_true),
                'length_vs_true': float(corr_length_true),
                'length_vs_pred': float(corr_length_pred),
                'sharing_vs_true': float(corr_sharing_true),
                'sharing_vs_pred': float(corr_sharing_pred),
                'hops_vs_true': float(corr_hops_true),
                'hops_vs_pred': float(corr_hops_pred)
            },
            'rank_matches': {
                'true': float(true_rank_match),
                'length': float(length_rank_match),
                'sharing': float(sharing_rank_match),
                'hops': float(hop_rank_match)
            },
            'feature_stds': {
                'true_stretch': float(np.std(true_scores)),
                'pred_stretch': float(np.std(pred_scores)),
                'length': float(np.std(path_lengths)),
                'sharing': float(np.std(sharing_scores)),
                'hops': float(np.std(hop_counts))
            }
        })
    
    # Sort by prediction quality
    od_case_data.sort(key=lambda x: x['prediction_quality'])
    
    bad_cases = od_case_data[:num_bad]
    good_cases = od_case_data[-num_good:]
    good_cases.reverse()
    
    # Generate detailed report
    output_file = f"{results_dir}/stretch_case_analysis{args.model_suffix}.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STRETCH PREDICTION: DETAILED CASE-BY-CASE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("This analysis examines what features (length, sharing, complexity) drive\n")
        f.write("the model's stretch predictions for individual OD pairs.\n\n")
        
        # Good cases
        f.write("="*80 + "\n")
        f.write(f"GOOD PREDICTION CASES (High Spearman rho)\n")
        f.write("="*80 + "\n\n")
        
        for i, case in enumerate(good_cases, 1):
            write_detailed_case(f, case, i, "GOOD")
        
        # Bad cases
        f.write("\n" + "="*80 + "\n")
        f.write(f"BAD PREDICTION CASES (Low Spearman rho)\n")
        f.write("="*80 + "\n\n")
        
        for i, case in enumerate(bad_cases, 1):
            write_detailed_case(f, case, i, "BAD")
    
    print(f"\nDetailed case analysis saved to: {output_file}")
    
    # Also print summary to stdout
    print("\n" + "="*70)
    print("STRETCH PREDICTION CASE ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nAnalyzed {len(od_case_data)} OD pairs")
    print(f"Generated detailed report with {num_good} good cases and {num_bad} bad cases")
    
    # Quick summary of patterns
    length_bias_count = sum(1 for c in od_case_data 
                           if c['rank_matches']['length'] > c['rank_matches']['true'])
    sharing_bias_count = sum(1 for c in od_case_data 
                            if c['rank_matches']['sharing'] > c['rank_matches']['true'])
    
    print(f"\nPotential Confounds Detected:")
    print(f"  OD pairs where model follows LENGTH more than TRUE:  {length_bias_count}")
    print(f"  OD pairs where model follows SHARING more than TRUE: {sharing_bias_count}")
    
    # Save JSON version
    json_output = f"{results_dir}/stretch_case_analysis{args.model_suffix}.json"
    with open(json_output, 'w') as f:
        json.dump({
            'good_cases': good_cases,
            'bad_cases': bad_cases,
            'summary': {
                'total_od_pairs': len(od_case_data),
                'length_bias_count': length_bias_count,
                'sharing_bias_count': sharing_bias_count
            }
        }, f, indent=2)
    
    print(f"JSON data saved to: {json_output}")
    
    return od_case_data

def write_detailed_case(f, case, case_num, label):
    """Write detailed analysis of a single OD pair case"""
    f.write(f"{label} Case #{case_num}: OD Pair {case['od_pair']}\n")
    f.write("-" * 80 + "\n\n")
    
    f.write(f"Overall Quality:\n")
    f.write(f"  Prediction vs True Stretch: rho = {case['correlations']['pred_vs_true']:+.3f}\n")
    f.write(f"  Number of paths: {case['num_paths']}\n\n")
    
    f.write(f"Feature Correlations with TRUE Stretch:\n")
    f.write(f"  Path Length:    {case['correlations']['length_vs_true']:+.3f}\n")
    f.write(f"  Sharing Score:  {case['correlations']['sharing_vs_true']:+.3f}\n")
    f.write(f"  Hop Count:      {case['correlations']['hops_vs_true']:+.3f}\n\n")
    
    f.write(f"Feature Correlations with PREDICTED Stretch:\n")
    f.write(f"  Path Length:    {case['correlations']['length_vs_pred']:+.3f}\n")
    f.write(f"  Sharing Score:  {case['correlations']['sharing_vs_pred']:+.3f}\n")
    f.write(f"  Hop Count:      {case['correlations']['hops_vs_pred']:+.3f}\n\n")
    
    f.write(f"Rank Matching Analysis:\n")
    f.write(f"  Model matches True Stretch ranks:  {case['rank_matches']['true']:.1%}\n")
    f.write(f"  Model matches Length ranks:        {case['rank_matches']['length']:.1%}\n")
    f.write(f"  Model matches Sharing ranks:       {case['rank_matches']['sharing']:.1%}\n")
    f.write(f"  Model matches Hop Count ranks:     {case['rank_matches']['hops']:.1%}\n\n")
    
    # Identify which feature the model follows most
    rank_matches = case['rank_matches']
    best_match = max(rank_matches.items(), key=lambda x: x[1] if x[0] != 'true' else -1)
    if best_match[0] != 'true' and best_match[1] > rank_matches['true']:
        f.write(f"[WARNING] Model appears to follow {best_match[0].upper()} ranking ")
        f.write(f"({best_match[1]:.1%}) more than TRUE stretch ({rank_matches['true']:.1%})!\n\n")
    
    f.write(f"Feature Variability (standard deviation):\n")
    f.write(f"  True Stretch:   {case['feature_stds']['true_stretch']:.2f}\n")
    f.write(f"  Pred Stretch:   {case['feature_stds']['pred_stretch']:.2f}\n")
    f.write(f"  Path Length:    {case['feature_stds']['length']:.0f} meters\n")
    f.write(f"  Sharing Score:  {case['feature_stds']['sharing']:.2f}\n")
    f.write(f"  Hop Count:      {case['feature_stds']['hops']:.2f}\n\n")
    
    f.write(f"Individual Path Details:\n")
    f.write(f"{'Path':<6} {'Length':<8} {'Hops':<6} {'Share':<7} {'True':<7} {'Pred':<7} "
            f"{'TR':<4} {'PR':<4} {'LR':<4} {'SR':<4} {'HR':<4} Notes\n")
    f.write(f"{'':6} {'(m)':8} {'':6} {'Score':7} {'Strch':7} {'Strch':7} "
            f"{'Rnk':4} {'Rnk':4} {'Rnk':4} {'Rnk':4} {'Rnk':4}\n")
    f.write("-" * 80 + "\n")
    
    for path in case['paths']:
        # Highlight if pred rank matches a feature rank but not true rank
        marker = ""
        if path['pred_rank'] != path['true_rank']:
            if path['pred_rank'] == path['length_rank']:
                marker = "[follows LENGTH]"
            elif path['pred_rank'] == path['sharing_rank']:
                marker = "[follows SHARING]"
            elif path['pred_rank'] == path['hop_rank']:
                marker = "[follows HOPS]"
        
        f.write(f"{path['path_idx']+1:<6} "
                f"{path['length']:<8.0f} "
                f"{path['hops']:<6} "
                f"{path['sharing']:<7.1f} "
                f"{path['true_stretch']:<7.1f} "
                f"{path['pred_stretch']:<7.1f} "
                f"{path['true_rank']:<4} "
                f"{path['pred_rank']:<4} "
                f"{path['length_rank']:<4} "
                f"{path['sharing_rank']:<4} "
                f"{path['hop_rank']:<4} "
                f"{marker}\n")
    
    f.write("\n")
    
    # Analysis summary for this case
    f.write("Case Summary:\n")
    
    # Check for specific patterns
    if case['feature_stds']['true_stretch'] < 5 and case['feature_stds']['pred_stretch'] > 15:
        f.write("  [PATTERN] Similar true scores but large prediction gaps - model may be over-differentiating\n")
        f.write(f"    based on {best_match[0]} (correlation: {case['correlations'][f'{best_match[0]}_vs_pred']:+.3f})\n")
    
    if case['feature_stds']['pred_stretch'] < 5 and case['feature_stds']['true_stretch'] > 15:
        f.write("  [PATTERN] Similar predictions but large true score gaps - model failing to discriminate\n")
        if case['feature_stds']['length'] < 500:
            f.write("    Possibly because path lengths are similar (low variability)\n")
    
    if abs(case['correlations']['length_vs_pred']) > abs(case['correlations']['pred_vs_true']):
        f.write(f"  [CONFOUND] Model correlates more with LENGTH ({case['correlations']['length_vs_pred']:+.3f}) ")
        f.write(f"than with TRUE stretch ({case['correlations']['pred_vs_true']:+.3f})\n")
    
    if abs(case['correlations']['sharing_vs_pred']) > abs(case['correlations']['pred_vs_true']):
        f.write(f"  [CONFOUND] Model correlates more with SHARING ({case['correlations']['sharing_vs_pred']:+.3f}) ")
        f.write(f"than with TRUE stretch ({case['correlations']['pred_vs_true']:+.3f})\n")
    
    f.write("\n" + "="*80 + "\n\n")

def main():
    parser = argparse.ArgumentParser(description="Test stretch prediction model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="data/london")
    parser.add_argument("--model_suffix", type=str, default="")
    parser.add_argument("--paths_path", type=str, default="paths.csv")
    parser.add_argument("--exclude_main", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint first to get architecture params (must match saved weights)
    model_path = f"{args.data_dir}/models/stretch_ml_best{args.model_suffix}.pth"
    print(f"\nLoading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint['args']

    print("\nLoading dataset...")
    dataset = RoadNetworkDataset(
        data_dir=f"{args.data_dir}/graph",
        exclude_main_path=args.exclude_main,
        precompute_subpaths=True,
        paths_path=args.paths_path
    )

    print("Reproducing training split (random_state=42)...")
    _, _, _, _, _, test_indices = dataset.split_dataset_three_ways(
        val_size=0.1, test_size=0.2, random_state=42
    )

    test_dataset = RoadNetworkDatasetSubset(dataset, test_indices)
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Architecture params always come from checkpoint to guarantee they match the saved weights
    model = PathStretchScoringModel(
        hidden_dim=ckpt_args.hidden_dim,
        dropout=0.0,
        enable_gnn_od=ckpt_args.enable_gnn_od,
        epsilon=ckpt_args.epsilon
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    graph_data = dataset.graph_data.to(device)
    
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, test_loader, graph_data, device)
    
    print("\n" + "="*70)
    print("STRETCH RANKING TEST RESULTS - ENHANCED METRICS")
    print("="*70)
    print(f"Dataset Configuration:")
    print(f"  Main path excluded : {args.exclude_main}")
    print(f"  Test samples       : {metrics['num_samples']}")
    print(f"  Total paths        : {metrics['num_total_paths']}")
    
    all_stretch = [s for sublist in dataset.candidate_stretch_scores for s in sublist]
    print(f"\nStretch Score Distribution:")
    print(f"  Mean               : {np.mean(all_stretch):.2f}")
    print(f"  Std                : {np.std(all_stretch):.2f}")
    print(f"  Range              : [{np.min(all_stretch):.2f}, {np.max(all_stretch):.2f}]")
    
    print(f"\n" + "-"*50)
    print("CORRELATION METRICS")
    print("-"*50)
    print(f"  Spearman            : {metrics['spearman_corr']:.4f}")
    print(f"  Kendall             : {metrics['kendall_corr']:.4f}")
    print(f"  Pearson r          : {metrics['pearson_corr']:.4f}")
    
    print(f"\n" + "-"*50)
    print("LIST-LEVEL ACCURACY (Strict - entire list must be correct)")
    print("-"*50)
    print(f"  Top-1 Lists Correct: {metrics['top1_accuracy']:.4f}")
    print(f"  Top-2 Lists Correct: {metrics['top2_accuracy']:.4f}")
    print(f"  Top-3 Lists Correct: {metrics['top3_accuracy']:.4f}")
    print(f"  Top-4 Lists Correct: {metrics['top4_accuracy']:.4f}")
    print(f"  Top-5 Lists Correct: {metrics['top5_accuracy']:.4f}")
    print(f"  Perfect Rankings   : {metrics['perfect_ranking_rate']:.4f}")
    
    print(f"\n" + "-"*50)
    print("ITEM-LEVEL ACCURACY (Individual paths across all lists)")
    print("-"*50)
    print(f"  Overall Item Acc (Strict): {metrics['item_accuracy']:.4f} ({metrics['item_accuracy']*100:.1f}%)")
    print(f"  Overall Item Acc (Ties)  : {metrics['item_accuracy_with_ties']:.4f} ({metrics['item_accuracy_with_ties']*100:.1f}%)")
    print(f"\n  Top-1 Position:")
    print(f"    Strict : {metrics['item_top1_accuracy']:.4f}")
    print(f"    Ties   : {metrics['item_top1_accuracy_with_ties']:.4f}")
    print(f"  Top-2 Positions:")
    print(f"    Strict : {metrics['item_top2_accuracy']:.4f}")
    print(f"    Ties   : {metrics['item_top2_accuracy_with_ties']:.4f}")
    print(f"  Top-3 Positions:")
    print(f"    Strict : {metrics['item_top3_accuracy']:.4f}")
    print(f"    Ties   : {metrics['item_top3_accuracy_with_ties']:.4f}")
    print(f"  Top-4 Positions:")
    print(f"    Strict : {metrics['item_top4_accuracy']:.4f}")
    print(f"    Ties   : {metrics['item_top4_accuracy_with_ties']:.4f}")
    print(f"  Top-5 Positions:")
    print(f"    Strict : {metrics['item_top5_accuracy']:.4f}")
    print(f"    Ties   : {metrics['item_top5_accuracy_with_ties']:.4f}")
    
    print(f"\n" + "-"*50)
    print("TIE STATISTICS IN GROUND TRUTH")
    print("-"*50)
    tie_stats = metrics['tie_statistics']
    print(f"  OD pairs with ties : {tie_stats['num_od_with_ties']} ({tie_stats['pct_od_with_ties']:.1f}%)")
    print(f"  Total tied paths   : {tie_stats['total_tied_paths']}")
    print(f"  Max tie group size : {tie_stats['max_tie_group_size']}")
    if tie_stats['tie_group_distribution']:
        print(f"  Tie group sizes    : {tie_stats['tie_group_distribution']}")
    
    print(f"\n" + "-"*50)
    print("RANK DISPLACEMENT ANALYSIS")
    print("-"*50)
    print(f"Strict (no tie tolerance):")
    print(f"  Mean Displacement  : {metrics['mean_rank_displacement']:.2f} positions")
    print(f"  Median Displacement: {metrics['median_rank_displacement']:.2f} positions")
    print(f"  Perfectly Ranked   : {metrics['paths_perfectly_ranked']*100:.1f}% of paths")
    print(f"  Within 1 Rank      : {metrics['paths_within_1_rank']*100:.1f}% of paths")
    print(f"  Within 2 Ranks     : {metrics['paths_within_2_ranks']*100:.1f}% of paths")
    print(f"  Within 3 Ranks     : {metrics['paths_within_3_ranks']*100:.1f}% of paths")
    
    print(f"\nTie-Friendly (+/-0.01 tolerance):")
    print(f"  Mean Displacement  : {metrics['mean_rank_displacement_ties']:.2f} positions")
    print(f"  Median Displacement: {metrics['median_rank_displacement_ties']:.2f} positions")
    print(f"  Perfectly Ranked   : {metrics['paths_perfectly_ranked_ties']*100:.1f}% of paths")
    print(f"  Within 1 Rank      : {metrics['paths_within_1_rank_ties']*100:.1f}% of paths")
    print(f"  Within 2 Ranks     : {metrics['paths_within_2_ranks_ties']*100:.1f}% of paths")
    print(f"  Within 3 Ranks     : {metrics['paths_within_3_ranks_ties']*100:.1f}% of paths")

    print(f"\n" + "-"*50)
    print("POSITION-SPECIFIC ACCURACY")
    print("-"*50)
    for pos in range(1, 6):
        if metrics['position_accuracy'][pos] is not None:
            print(f"  Rank #{pos} Accuracy  : {metrics['position_accuracy'][pos]:.4f}")
    
    print(f"\n" + "-"*50)
    print("NDCG & REGRESSION METRICS")
    print("-"*50)
    print(f"  NDCG (full)        : {metrics['ndcg']:.4f}")
    if metrics['ndcg@3'] is not None:
        print(f"  NDCG@3             : {metrics['ndcg@3']:.4f}")
    if metrics['ndcg@5'] is not None:
        print(f"  NDCG@5             : {metrics['ndcg@5']:.4f}")
    print(f"  MAE                : {metrics['mae']:.4f}")
    print(f"  RMSE               : {metrics['rmse']:.4f}")
    
    print(f"\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    for i, example in enumerate(metrics['examples']):
        print(f"\nExample {i+1} - OD Pair {example['od_pair']}:")
        print(f"  Total paths: {example['num_paths']}")
        print(f"  Correctly ranked: {example['num_correct_ranks']}/{example['num_paths']} paths")
        print(f"  Path | Pred Score | True Score | Pred Rank | True Rank | Status")
        print(f"  -----|------------|------------|-----------|-----------|--------")
        
        for j in range(len(example['pred_logits'])):
            status = "✓" if example['pred_ranks'][j] == example['true_ranks'][j] else f"off-{abs(example['pred_ranks'][j] - example['true_ranks'][j])}"
            print(f"  {j+1:4d} | {example['pred_logits'][j]:10.2f} | {example['true_scores'][j]:10.2f} | "
                  f"{example['pred_ranks'][j]:9d} | {example['true_ranks'][j]:9d} | {status:6s}")
        
        correct = np.array_equal(example['pred_ranks'], example['true_ranks'])
        print(f"  List Ranking: {'PERFECT' if correct else 'IMPERFECT'}")
    
    results_dir = f"{args.data_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    result_json_name = f"stretch_ml_best{args.model_suffix}_result.json"
    save_metrics = {k: v for k, v in metrics.items() if k not in ['examples', 'od_predictions', 'od_true_scores', 'od_pairs']}
    with open(f"{results_dir}/{result_json_name}", 'w') as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\nResults saved to {results_dir}/{result_json_name}")

    analyze_score_patterns_and_samples(metrics, args, num_samples=10)

    analyze_stretch_prediction_cases(metrics, dataset, args, num_good=5, num_bad=5)

if __name__ == "__main__":
    main()
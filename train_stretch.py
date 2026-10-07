import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import RoadNetworkDataset
from model_stretch import PathStretchScoringModel
from loss import rankNet_loss, listNet_loss, listMLE_loss
from train_sharing import collate_fn
import numpy as np
import time
import json
from collections import defaultdict
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true, weights):
        return (weights * (y_pred - y_true) ** 2).mean()

def compute_sample_weights(all_scores, n_bins=20, power=0.5):
    """Compute histogram-based weights for scores."""
    hist, bin_edges = np.histogram(all_scores, bins=n_bins)
    
    bin_weights = 1.0 / np.maximum(hist, 1e-5)**power
    bin_weights /= bin_weights.sum()
    
    print(f"\nHistogram-based weighting:")
    print(f"  Score range: [{all_scores.min():.2f}, {all_scores.max():.2f}]")
    print(f"  Bin counts: {hist}")
    print(f"  Bin weights: {bin_weights.round(3)}")
    
    def get_weight(score):
        bin_idx = np.digitize(score, bin_edges[:-1]) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        return bin_weights[bin_idx]
    
    return get_weight

def compute_epoch_metrics_stretch(model, data_loader, graph_data, device, rank_loss_fn, 
                         optimizer=None, epoch=0, use_weighted=False, weight_fn=None,
                         train_alpha=2, val_alpha=0.1):
    """Compute loss and metrics for one epoch - adapted for stretch scores."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_rank = 0.0
    num_batches = 0
    
    all_preds = []
    all_true = []
    
    weighted_mse_loss = WeightedMSELoss() if use_weighted else None
    
    for batch_idx, batch in enumerate(data_loader):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        
        with torch.set_grad_enabled(is_train):
            pred_stretch = model(graph_data, batch, task='stretch')
            
            flat_pred = torch.cat(pred_stretch, dim=0)
            flat_true = torch.tensor(
                [v for sub in batch["stretch_scores"] for v in sub],
                dtype=torch.float32, device=device
            )
            
            if use_weighted and is_train and weight_fn is not None:
                weights = torch.tensor(
                    [weight_fn(score) for score in flat_true.cpu().numpy()],
                    dtype=torch.float32, device=device
                )
                mse_loss = weighted_mse_loss(flat_pred, flat_true, weights)
            else:
                mse_loss = F.mse_loss(flat_pred, flat_true)
            
            rank_loss = rank_loss_fn(pred_stretch, batch["stretch_scores"])
            
            if is_train and use_weighted:
                alpha = train_alpha
            else:
                alpha = val_alpha

            loss = rank_loss + alpha * mse_loss

            if is_train:
                loss.backward()
                if epoch in [1,5,10] and batch_idx == 0:
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(f"  Gradient norm before clipping: {total_norm:.4f}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_rank += rank_loss.item()
        num_batches += 1
        
        all_preds.extend(flat_pred.detach().cpu().numpy())
        all_true.extend(flat_true.detach().cpu().numpy())
        
        if is_train and batch_idx == 0 and epoch % 5 == 0:
            print(f"\n  First batch details:")
            print(f"    Predictions: {flat_pred[:5].detach().cpu().numpy()}")
            print(f"    True scores: {flat_true[:5].detach().cpu().numpy()}")
            if use_weighted and weight_fn is not None:
                print(f"    Weights: {weights[:5].detach().cpu().numpy()}")
            print(f"    MSE: {mse_loss.item():.4f}, Rank: {rank_loss.item():.4f}")
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    from scipy.stats import spearmanr, kendalltau
    spearman_corr, _ = spearmanr(all_preds, all_true)
    kendall_corr, _ = kendalltau(all_preds, all_true)
    
    pearson_corr = np.corrcoef(all_preds, all_true)[0, 1]
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_rank = total_rank / num_batches
    
    if epoch % 10 == 0:
        print(f"\n  Prediction distribution:")
        print(f"    Range: [{all_preds.min():.2f}, {all_preds.max():.2f}]")
        print(f"    Quartiles: {np.percentile(all_preds, [25, 50, 75]).round(2)}")
    
    return avg_loss, avg_mse, avg_rank, pearson_corr, spearman_corr, kendall_corr

def analyze_predictions_stretch(model, data_loader, graph_data, device, num_samples=5):
    """Analyze model predictions in detail."""
    model.eval()
    
    samples_analyzed = 0
    
    with torch.no_grad():
        for batch in data_loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            
            pred_stretch = model(graph_data, batch, task='stretch')
            
            for i, (preds, trues) in enumerate(zip(pred_stretch, batch["stretch_scores"])):
                if samples_analyzed >= num_samples:
                    return
                
                print(f"\nOD Pair {samples_analyzed + 1}:")
                print(f"  Num paths: {len(preds)}")
                
                pred_ranks = np.argsort(-preds.detach().cpu().numpy()) + 1
                true_ranks = np.argsort(-np.array(trues)) + 1
                
                for j, (p, t) in enumerate(zip(preds, trues)):
                    print(f"  Path {j+1}: Pred={p.item():.2f}, True Score={t:.2f}, "
                          f"Pred Rank={pred_ranks[j]}, True Rank={true_ranks[j]}")
                
                samples_analyzed += 1

def main():
    parser = argparse.ArgumentParser(description="Train ML-based stretch score model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", type=str, default="ranknet", 
                        choices=["ranknet", "listnet", "listmle"])
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--data_dir", type=str, default="data/london")
    parser.add_argument("--paths_path", type=str, default="paths.csv")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon for stretch threshold")
    parser.add_argument("--use_weighted", action="store_true",
                        help="Use weighted MSE loss")
    parser.add_argument("--enable_gnn_od", action="store_true",
                        help="Use GNN OD features from path graph")
    parser.add_argument("--n_bins", type=int, default=20,
                        help="Number of bins for histogram weighting")
    parser.add_argument("--weight_power", type=float, default=0.5,
                        help="Power for inverse frequency weighting (0.5=sqrt)")
    parser.add_argument("--train_alpha", type=float, default=2,
                        help="MSE loss alpha during training")
    parser.add_argument("--val_alpha", type=float, default=0.1,
                        help="MSE loss alpha during validation")
    parser.add_argument("--model_suffix", type=str, default="")
    parser.add_argument("--exclude_main", action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for model initialization and training (default: 42)")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLoading dataset...")
    dataset = RoadNetworkDataset(data_dir=f"{args.data_dir}/graph", exclude_main_path=args.exclude_main, paths_path=args.paths_path)
    
    all_stretch = [s for sublist in dataset.candidate_stretch_scores for s in sublist]
    print(f"Stretch score range: [{min(all_stretch):.1f}, {max(all_stretch):.1f}]")
    print(f"Stretch score distribution: mean={np.mean(all_stretch):.1f}, std={np.std(all_stretch):.1f}")
    
    all_sharing = [s for sublist in dataset.candidate_sharing_scores for s in sublist]
    correlation = np.corrcoef(all_stretch, all_sharing)[0, 1]
    print(f"Correlation between stretch and sharing: {correlation:.3f}")
    
    train_dataset, val_dataset, test_dataset, _, _, test_indices = \
        dataset.split_dataset_three_ways(val_size=0.1, test_size=0.2, random_state=42)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    weight_fn = None
    if args.use_weighted:
        print(f"\n{'='*60}")
        print(f"Computing histogram-based weights with {args.n_bins} bins...")

        train_scores = []
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            train_scores.extend(sample['stretch_scores'])
        train_scores = np.array(train_scores)
        
        weight_fn = compute_sample_weights(train_scores, args.n_bins, args.weight_power)
        print(f"Weighted loss enabled with power={args.weight_power}")
        print(f"{'='*60}")
    
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=collate_fn, num_workers=0,
                             pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn, num_workers=0,
                           pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=0,
                            pin_memory=pin)
    
    model = PathStretchScoringModel(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        enable_gnn_od=args.enable_gnn_od,
        epsilon=args.epsilon
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    loss_functions = {
        "ranknet": rankNet_loss,
        "listnet": listNet_loss,
        "listmle": listMLE_loss
    }
    rank_loss_fn = loss_functions[args.loss]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience//2
    )
    
    graph_data = dataset.graph_data.to(device)
    
    loss_type = "weighted MSE" if args.use_weighted else "regular MSE"
    print(f"\n{'='*60}")
    print(f"Starting stretch score training with {args.loss} + {loss_type} loss...")
    print(f"{'='*60}")
    
    best_val_spearman = -float('inf')
    patience_counter = 0
    
    train_history = defaultdict(list)
    
    model_suffix = args.model_suffix
    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_mse, train_rank, train_pearson, train_spearman, train_kendall = \
            compute_epoch_metrics_stretch(model, train_loader, graph_data, device, 
                                rank_loss_fn, optimizer, epoch, args.use_weighted, weight_fn, args.train_alpha, args.val_alpha)

        val_loss, val_mse, val_rank, val_pearson, val_spearman, val_kendall = \
            compute_epoch_metrics_stretch(model, val_loader, graph_data, device, 
                                rank_loss_fn, None, epoch, False, None, args.train_alpha, args.val_alpha)  # No weighting for validation
        
        scheduler.step(val_loss)
        
        train_history['train_loss'].append(train_loss)
        train_history['val_loss'].append(val_loss)
        train_history['train_spearman'].append(train_spearman)
        train_history['val_spearman'].append(val_spearman)
        
        epoch_time = time.time() - t0
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Rank: {train_rank:.4f})")
        print(f"         Spearman: {train_spearman:.4f}, Kendall: {train_kendall:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, Rank: {val_rank:.4f})")
        print(f"         Spearman: {val_spearman:.4f}, Kendall: {val_kendall:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_spearman': val_spearman,
                'val_kendall': val_kendall,
                'args': args,
                'use_weighted': args.use_weighted
            }, f"{args.data_dir}/models/stretch_ml_best{model_suffix}.pth")
            print(f"  >> Saved best model (Val Spearman: {val_spearman:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    checkpoint = torch.load(f"{args.data_dir}/models/stretch_ml_best{model_suffix}.pth", 
                          map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mse, test_rank, test_pearson, test_spearman, test_kendall = \
        compute_epoch_metrics_stretch(model, test_loader, graph_data, device, rank_loss_fn, None, 0, False, None)
    
    print(f"\nTest Results (Stretch Ranking):")
    print(f"  Loss: {test_loss:.4f} (MSE: {test_mse:.4f}, Rank: {test_rank:.4f})")
    print(f"  Spearman: {test_spearman:.4f}, Kendall: {test_kendall:.4f}")
    
    print(f"\n{'='*60}")
    print("Sample stretch predictions analysis:")
    print(f"{'='*60}")
    analyze_predictions_stretch(model, test_loader, graph_data, device, num_samples=5)
    
    test_results = {
        'test_loss': test_loss,
        'test_mse': test_mse,
        'test_rank': test_rank,
        'test_pearson': test_pearson,
        'test_spearman': test_spearman,
        'test_kendall': test_kendall,
        'best_epoch': checkpoint['epoch'],
        'test_indices': test_indices,
        'args': vars(args),
        'use_weighted': args.use_weighted
    }
    
    with open(f"{args.data_dir}/models/stretch_ml_test_results{model_suffix}.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTraining complete! Results saved to {args.data_dir}/models/")

if __name__ == "__main__":
    main()
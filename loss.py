import torch
import torch.nn as nn
import torch.nn.functional as F

def rankNet_loss(pred_scores_list, true_scores_list, margin=1.0, temperature=1.0):
    """
    RankNet loss - pairwise ranking loss.
    Good for learning relative preferences between pairs of paths.
    
    Args:
        pred_scores_list: List of predicted score tensors for each OD pair
        true_scores_list: List of true score tensors for each OD pair
        margin: Minimum margin between scores
        temperature: Temperature for sigmoid (higher = steeper)
    """
    total_loss = 0.0
    total_pairs = 0
    
    for pred_scores, true_scores in zip(pred_scores_list, true_scores_list):
        if len(pred_scores) < 2:
            continue
            
        if isinstance(true_scores, list):
            true_scores = torch.tensor(true_scores, device=pred_scores.device, dtype=pred_scores.dtype)
        
        n = len(pred_scores)
        pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # determine true preference
                if true_scores[i] > true_scores[j] + 1e-6:
                    pairs.append((i, j, 1.0))  # i should rank higher
                elif true_scores[j] > true_scores[i] + 1e-6:
                    pairs.append((i, j, -1.0))  # j should rank higher
                # skip ties
        
        if not pairs:
            continue
        
        indices_i = torch.tensor([p[0] for p in pairs], device=pred_scores.device)
        indices_j = torch.tensor([p[1] for p in pairs], device=pred_scores.device)
        labels = torch.tensor([p[2] for p in pairs], device=pred_scores.device)
        
        score_diff = pred_scores[indices_i] - pred_scores[indices_j]
        
        loss = -torch.log(torch.sigmoid(temperature * labels * score_diff) + 1e-10)
        
        total_loss += loss.sum()
        total_pairs += len(pairs)
    
    return total_loss / max(total_pairs, 1)


def listNet_loss(pred_scores_list, true_scores_list, temperature=1.0, eps=1e-10):
    """
    ListNet loss - listwise ranking loss using cross-entropy between score distributions.
    Better for learning the complete ranking of all paths at once.
    
    Args:
        pred_scores_list: List of predicted score tensors for each OD pair
        true_scores_list: List of true score tensors for each OD pair
        temperature: Temperature for softmax (higher = more peaked distribution)
        eps: Small epsilon for numerical stability
    """
    total_loss = 0.0
    total_groups = 0
    
    for pred_scores, true_scores in zip(pred_scores_list, true_scores_list):
        if len(pred_scores) < 2:
            continue
            
        if isinstance(true_scores, list):
            true_scores = torch.tensor(true_scores, device=pred_scores.device, dtype=pred_scores.dtype)
        
        # compute probability distributions
        # higher scores should have higher probability
        pred_probs = F.softmax(pred_scores / temperature, dim=0)
        true_probs = F.softmax(true_scores / temperature, dim=0)
        
        loss = -(true_probs * torch.log(pred_probs + eps)).sum()
        
        total_loss += loss
        total_groups += 1
    
    return total_loss / max(total_groups, 1)


def listMLE_loss(pred_scores_list, true_scores_list, eps=1e-10):
    """
    ListMLE loss - learns to predict the exact permutation.
    Stronger than ListNet for ranking but requires more careful optimization.
    
    Args:
        pred_scores_list: List of predicted score tensors for each OD pair
        true_scores_list: List of true score tensors for each OD pair
        eps: Small epsilon for numerical stability
    """
    total_loss = 0.0
    total_groups = 0
    
    for pred_scores, true_scores in zip(pred_scores_list, true_scores_list):
        if len(pred_scores) < 2:
            continue
            
        if isinstance(true_scores, list):
            true_scores = torch.tensor(true_scores, device=pred_scores.device, dtype=pred_scores.dtype)
        
        sorted_indices = torch.argsort(true_scores, descending=True)
        sorted_pred_scores = pred_scores[sorted_indices]
        
        # for each position, the correct item should have the highest score among remaining items
        loss = 0.0
        for i in range(len(sorted_pred_scores) - 1):
            remaining_scores = sorted_pred_scores[i:]            
            log_probs = F.log_softmax(remaining_scores, dim=0)
            
            # loss is negative log probability of correct item
            loss -= log_probs[0]
        
        total_loss += loss
        total_groups += 1
    
    return total_loss / max(total_groups, 1)

def combined_ranking_loss(pred_scores_list, true_scores_list, 
                         listnet_weight=0.5, ranknet_weight=0.5,
                         mse_weight=0.0):
    """
    Combined loss that balances listwise and pairwise objectives.
    Often works better than individual losses.
    
    Args:
        pred_scores_list: List of predicted score tensors for each OD pair
        true_scores_list: List of true score tensors for each OD pair
        listnet_weight: Weight for ListNet loss
        ranknet_weight: Weight for RankNet loss  
        mse_weight: Weight for MSE loss (for score calibration)
    """
    loss = 0.0
    
    if listnet_weight > 0:
        loss += listnet_weight * listNet_loss(pred_scores_list, true_scores_list)
    
    if ranknet_weight > 0:
        loss += ranknet_weight * rankNet_loss(pred_scores_list, true_scores_list)
    
    if mse_weight > 0:
        mse_loss = 0.0
        count = 0
        for pred, true in zip(pred_scores_list, true_scores_list):
            if isinstance(true, list):
                true = torch.tensor(true, device=pred.device, dtype=pred.dtype)
            mse_loss += F.mse_loss(pred, true)
            count += 1
        loss += mse_weight * (mse_loss / max(count, 1))
    
    return loss
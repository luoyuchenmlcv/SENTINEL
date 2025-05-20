import torch
import torch.nn.functional as F
import ot  # POT library for Earth Mover's Distance
import numpy as np
from itertools import groupby
from operator import itemgetter
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score

def generate_context_windows(sequence_length, window_sizes):
    
    if sequence_length<=window_sizes[0]:
        return torch.ones((2,sequence_length)).bool()
    masks = []
    positions = torch.arange(sequence_length).unsqueeze(0)  # Shape (1, sequence_length)
    for window_size in window_sizes:
        num_windows = sequence_length - window_size + 1
        window_starts = torch.arange(num_windows).unsqueeze(1)  # Shape (num_windows, 1)
        mask = (positions >= window_starts) & (positions < (window_starts + window_size))
        masks.append(mask)
    all_masks = torch.cat(masks, dim=0)
    return all_masks


def compute_token_weights(masks, window_weights):
  
    masks_float = masks.float()  # (num_masks, sequence_length)

    mask_counts = masks_float.sum(dim=1, keepdim=True)  # (num_masks, 1)

    distribution = (masks_float / mask_counts) * window_weights  # (num_masks, sequence_length)

    token_weights = distribution.sum(dim=0)  # (sequence_length,)

    token_exposure_counts = masks_float.sum(dim=0)  # (sequence_length,)

    token_weights_normalized = token_weights / token_exposure_counts

    return token_weights_normalized


def compute_windowed_features(masks, features):

    weights = masks.float()
    mask_counts = weights.sum(dim=1, keepdim=True) 
    weights = weights / mask_counts
    windowed_features = weights @ features 
    return windowed_features



def compute_loss(D_AB, D_AA, D_BB, mu_aug, nu_aug, alpha=0.3,  use_approx=False):
 
    mu_aug_probs = F.softmax(mu_aug, dim=0) 
    nu_aug_probs = F.softmax(nu_aug, dim=0) 
    

    if not use_approx:
        emd_loss = ot.emd2(mu_aug_probs.squeeze(), nu_aug_probs.squeeze(), D_AB, processes=1, log=False)
    else:
        emd_loss = mu_aug_probs.T @ D_AB @ nu_aug_probs

    
    intra_A = mu_aug_probs.T @ D_AA @ mu_aug_probs
    intra_B = nu_aug_probs.T @ D_BB @ nu_aug_probs

    
    loss = emd_loss - alpha*(intra_A + intra_B)
    return loss, [emd_loss.item(), intra_A.item(), intra_B.item()]





def extract_context_windows_token(tokenizer, token_ids, prompt_idx_start, prompt_idx_end, importance_scores, k1):

    importance_scores = np.array(importance_scores)
    significant_indices = np.where(importance_scores > k1)[0]
    groups = []
    for k, g in groupby(enumerate(significant_indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        groups.append(group)
        
    
    context_windows = []
    for group in groups:
        start = group[0]
        end = group[-1]

        window_token_ids = token_ids[:, max(prompt_idx_start, start):min(end+1, prompt_idx_end+1)]

        if window_token_ids.shape[1] >= 5:
            window_string = tokenizer.batch_decode(window_token_ids)[0]
            context_windows.append(window_string)
    context_windows = list(set(context_windows))
    
    return context_windows

    


def extract_multiple_context_windows_token(tokenizer, token_ids, prompt_idx_start, prompt_idx_end, importance_scores, k1_list):
    
    extracted_words = []
    for k1 in k1_list:
        extracted_words += extract_context_windows_token(tokenizer, token_ids, prompt_idx_start, prompt_idx_end, importance_scores, k1)
    
    return list(set(extracted_words))
    
def evaluate(labels, scores):

    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr_score = auc(recall, precision)
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc_score = auc(fpr, tpr)
    return aupr_score, auroc_score


def compute_tpr_fpr(pred, label):
  
    pred = np.array(pred)
    label = np.array(label)
    
 
    TP = np.sum((pred == 1) & (label == 1))
    FN = np.sum((pred == 0) & (label == 1))
    FP = np.sum((pred == 1) & (label == 0))
    TN = np.sum((pred == 0) & (label == 0))
    
    # TPR = TP / (TP + FN)
    tpr = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    
    # FPR = FP / (FP + TN)
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0.0
    return tpr, fpr



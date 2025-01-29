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


def get_word_scores(tokens, scores, model_name):
    if 'vicuna' in model_name or 'llama' in model_name or 'mistral' in model_name:
        
        words = []
        word_scores = []
        puncts = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
        for idx, (token, score) in enumerate(zip(tokens, scores)):
            
            if token.startswith('▁') and token not in puncts:
                compose_word = token[1:]
                word_score = [score]
                curr_pos = idx+1
                while (curr_pos < len(tokens)) and not (tokens[curr_pos].startswith('▁')) and (tokens[curr_pos] not in puncts): 
                    compose_word += tokens[curr_pos]
                    word_score.append(scores[curr_pos])
                    curr_pos +=1
                words.append(compose_word)
                word_score = sum(word_score)/len(word_score)
                word_scores.append(word_score)
            elif  token in puncts:
                words.append(token) 
                word_scores.append(score)
            else:
                continue
                
    if 'qwen' in model_name:
        words = []
        word_scores = []
        for idx, (token, score) in enumerate(zip(tokens, scores)):
            words.append(str(token)[2:-1])
            word_scores.append(score)

    word_scores = np.array(word_scores)
    word_scores = (word_scores-word_scores.min())/(word_scores.max()-word_scores.min()+1e-6)
    return words, word_scores

def extract_context_windows(words, importance_scores, k1):
   

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
        window_words = words[start:end+1]
        if len(window_words) >= 5:
            window_string = ' '.join(window_words)
            context_windows.append(window_string)
    context_windows = list(set(context_windows))
    
    return context_windows



def extract_multiple_context_windows(words, importance_scores, k1_list):
    extracted_words = []
    for k1 in k1_list:
        extracted_words += extract_context_windows(words, importance_scores, k1)

    return extracted_words


    
def evaluate(labels, scores):

    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr_score = auc(recall, precision)
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc_score = auc(fpr, tpr)
    return aupr_score, auroc_score

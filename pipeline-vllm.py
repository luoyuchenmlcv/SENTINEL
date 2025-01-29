import numpy as np
import os
import json
from VLLM_general import VLLM
import time
import torch
import torch.nn.functional as F
import traceback

from utils import generate_context_windows, compute_token_weights, compute_windowed_features, compute_loss, get_word_scores,  extract_context_windows, extract_multiple_context_windows
from jailbreak_score import batch_jailbreak_score_perplexity_formated

def matching_score_pipeline2(model, t, g, save_pth, alpha=0.3, cw_num = 16):


    token_features_A = model.get_embedding(t).to(device = model.device, dtype=torch.float32)
    token_features_A = torch.atleast_2d(token_features_A)
    token_features_A /=  torch.linalg.norm(token_features_A, dim=1, keepdim=True)
    tokens_A = model.get_tokens(t)
    
    token_features_B = model.get_embedding(g).to(device = model.device, dtype=torch.float32)
    token_features_B = torch.atleast_2d(token_features_B)
    token_features_B /=  torch.linalg.norm(token_features_B, dim=1, keepdim=True)
    tokens_B = model.get_tokens(g)

    min_size = 5
    max_size=20
    context_window_mask_A = generate_context_windows(token_features_A.shape[0], window_sizes = range(min_size, max(min(max_size+1, token_features_A.shape[0]), min_size+1), int(max_size-min_size//cw_num) )).to(device = model.device, dtype=torch.float32)
    context_window_mask_B = generate_context_windows(token_features_B.shape[0], window_sizes = range(min_size, max(min(max_size+1, token_features_B.shape[0]), min_size+1),  int(max_size-min_size//cw_num) ) ).to(device = model.device, dtype=torch.float32)

    mu_aug_A = torch.zeros(len(context_window_mask_A), 1, requires_grad=True, device = model.device, dtype=torch.float32)
    mu_aug_B = torch.zeros(len(context_window_mask_B), 1, requires_grad=True, device = model.device, dtype=torch.float32)

    f_A = compute_windowed_features(context_window_mask_A, token_features_A)
    f_B = compute_windowed_features(context_window_mask_B, token_features_B)
    
    # D(f_A, f_B)
    D_AB = torch.cdist(f_A, f_B, p=2)
    # D(f_A, f_A)
    D_AA = torch.cdist(f_A, f_A, p=2)
    # D(f_B, f_B)
    D_BB = torch.cdist(f_B, f_B, p=2)


    optimizer = torch.optim.Adam([mu_aug_A, mu_aug_B], lr=1)
    num_epochs = 3
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss, loss_terms = compute_loss(D_AB, D_AA, D_BB, mu_aug_A, mu_aug_B, alpha=alpha, use_approx=True)
        loss.backward()
        optimizer.step()

    mu_aug_A = F.softmax(mu_aug_A, dim=0)
    mu_aug_B = F.softmax(mu_aug_B, dim=0)
    
    mu_A = compute_token_weights(context_window_mask_A, mu_aug_A) 
    mu_B = compute_token_weights(context_window_mask_B, mu_aug_B) 
    mu_A = F.softmax(mu_A, dim=0)
    mu_B = F.softmax(mu_B, dim=0)

    mu_A = mu_A.detach().cpu().numpy()
    mu_B = mu_B.detach().cpu().numpy()
    mu_A = (mu_A-mu_A.min())/(mu_A.max()-mu_A.min()+1e-6)
    mu_B = (mu_B-mu_B.min())/(mu_B.max()-mu_B.min()+1e-6)

    words_A, word_scores_A = get_word_scores(tokens_A, mu_A,   model.name)
    words_B, word_scores_B = get_word_scores(tokens_B, mu_B,  model.name)

    return words_A, word_scores_A, words_B, word_scores_B, mu_A, mu_B


def jailbreak_score_pipeline(words_list, words_score_list, thresh_num = 20, attack_method='Benign'):
    extracted_context = []
    for words, word_scores in zip(words_list, words_score_list):
        word_scores = np.array(word_scores)
        extracted_words = extract_multiple_context_windows(words, word_scores, k1_list = np.arange(0., 1, 1/thresh_num))
        extracted_context += extracted_words

    extracted_context.append(' '.join(words_list[0]))

    extracted_context = list(set(extracted_context))
    
    jailbreak_scores = batch_jailbreak_score_perplexity_formated(model, extracted_context, attack_method)
    order = (-jailbreak_scores).argsort()
    return jailbreak_scores, extracted_context
    
   





if __name__ == "__main__": 
    
    import argparse  
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--victim_model',
        type=str,
        default='llama2_7b',
        choices=['vicuna_7b_v1_5', 'llama2_7b', 'qwen_7b_chat', 'mistral_7b_v2', 'qwen_7b_2_5']
    )

    
    parser.add_argument(
        '--alpha',
        type=float,
        default='0.25',
        choices=[0.0, 0.25, 0.5, 1]
    )
    parser.add_argument(
        '--thresh_num',
        type=int,
        default= 20,
        choices=[2, 5, 10, 20]

    )
    parser.add_argument(
        '--cw_num',
        type=int,
        default= 16,
        choices=[4, 8, 16]
    )

    parser.add_argument(
        '--topk',
        type=int,
        default= 3,
        choices=[1, 3, 5, 7, 9]
    )    
    args = parser.parse_args()
    print(args.alpha, args.thresh_num,  args.cw_num)
    store_folder = f"alpha_{args.alpha}_thresh_num_{args.thresh_num}_cw_num_{args.cw_num}_top{args.topk}"
    import os
    if not os.path.exists(f"experiment_data/final_results/{store_folder}"):
        os.makedirs(f"experiment_data/final_results/{store_folder}")
    methods = [

            {'ATTACK':'FewShot', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'GCG', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'PAIR', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'AutoDAN', 'MODEL': args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'GPTFuzz', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'Benign', 'MODEL':'', 'VICTIM_MODEL':args.victim_model},\

              ]

    #load model
    model_pth = f"models/{args.victim_model}"
    model = VLLM(model_pth)


    

    functional_classes = json.load(open('temp_data/behavior_functional.json', 'r'))
    standard_behavior = functional_classes['standard']
    contextual_behavior = functional_classes['contextual']
    count = 0

    for attack_config in methods:
        print(attack_config['ATTACK'])
        if attack_config['ATTACK'] != 'Benign':
            data_pth = os.path.join('harmbench_data', attack_config['ATTACK'], attack_config['MODEL'], "results", f"{model.name}.json")
            print("data_pth", data_pth)
            data = json.load(open(data_pth, 'r'))
    
            test_cases = []
            generations = []

            keys = []
            for key in data.keys():
                if (key in standard_behavior) or (key in contextual_behavior):
                # if key in standard_behavior:
                    test_cases += [data[key][i]['test_case']for i in range(len(data[key]))]
                    generations += [data[key][i]['generation'] for i in range(len(data[key]))]
                    keys.append(key)

 
        else:
            data_pth = f"temp_data/all_benign_{attack_config['VICTIM_MODEL']}.json"
            data = json.load(open(data_pth, 'r'))
            jailbreak_scores = []
            test_cases = [data[key]['test_case'] for key in data]
            generations = [data[key]['generation'] for key in data]
            keys = ["none"]*len(test_cases)
        
        
        jailbreak_scores = []
        jailbreak_scores_all = []
        running_time = []
        input_len = []
        for test_idx, (k, t, g) in enumerate(zip(keys, test_cases, generations)):
            try:
                if len(g)<5:   
                    jailbreak_scores_all.append(np.array([-0.1,-0.1,-0.1,-0.1,-0.1, -0.1,-0.1,-0.1,-0.1,-0.1])) #no meaningful response, considered as a failed jailbreak
                    print(f"{test_idx}, {keys[test_idx]}: response过短, 跳过\n")
                    continue 
                
                start_time = time.time()

                #############################################################################################################################

                save_path = f"/workspace/method_11_24/experiment_data/text_heatmap/{attack_config['ATTACK']}_{test_idx}"

                print("save_path", save_path)
                
                words_A, word_scores_A, words_B, word_scores_B, mu_A, mu_B  = matching_score_pipeline2(model, t, g, save_pth = save_path, alpha = args.alpha, cw_num = args.cw_num)              
                single_jailbreak_scores, extracted_context = jailbreak_score_pipeline([words_A], [word_scores_A], thresh_num = args.thresh_num, attack_method = attack_config['ATTACK'])
                single_jailbreak_scores = single_jailbreak_scores[~np.isnan(single_jailbreak_scores)]
                final_score = torch.tensor(single_jailbreak_scores).topk(min(args.topk, len(single_jailbreak_scores))).values.mean().item()
                jailbreak_scores.append(final_score)
                all_values = torch.tensor(single_jailbreak_scores)
                jailbreak_scores_all.append(all_values.numpy())
                #############################################################################################################################
                mu_A =  model.get_tokens(t)

                # 计算并打印运行时间
                end_time = time.time()    
                run_time = end_time - start_time
                print(f"{test_idx}:running time: {run_time:3f} s")                    
                running_time.append(run_time)
                input_len.append(len(mu_A))
            except Exception as e:
                print(f"{test_idx}: error：{e}, skip\n")
                traceback.print_exc()   
                continue

        np.save(f"experiment_data/final_results/{store_folder}/{attack_config['ATTACK']}_{model.name}.npy", np.array(jailbreak_scores))
        nested_array = np.array(jailbreak_scores_all, dtype=object)
        np.save(f"experiment_data/final_results/{store_folder}/{attack_config['ATTACK']}_{model.name}_nested.npy", nested_array)
        np.save(f"experiment_data/final_results/{store_folder}/{attack_config['ATTACK']}_{model.name}_gen_runtime.npy", np.array(running_time))
        np.save(f"experiment_data/final_results/{store_folder}/{attack_config['ATTACK']}_{model.name}_seq_len.npy", np.array(input_len))
        
        
        

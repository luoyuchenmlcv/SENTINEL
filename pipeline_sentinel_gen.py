import torch
from LLM_general import LLM
import time # Optional
from utils import generate_context_windows, compute_token_weights, compute_windowed_features, compute_loss, get_word_scores,  extract_context_windows, extract_multiple_context_windows, build_ascii_mask, suppress_non_ascii_logits, extract_multiple_context_windows_token, compute_seq_prob
import torch.nn.functional as F
import numpy as np
import json
from over_refusal_rate import  is_attack_success
import pandas as pd
import traceback
def importance_scoring(token_features_A, token_features_B, alpha=0.25, cw_num = 16):
    # print("token_features_A", token_features_A.shape)
    token_features_A = torch.atleast_2d(token_features_A)
    token_features_A /=  torch.linalg.norm(token_features_A, dim=1, keepdim=True)
    # print("token_features_B", token_features_B.shape)
    token_features_B = torch.atleast_2d(token_features_B)
    token_features_B /=  torch.linalg.norm(token_features_B, dim=1, keepdim=True)

    min_size = 5
    max_size=20
    context_window_mask_A = generate_context_windows(token_features_A.shape[0], window_sizes = range(min_size, max(min(max_size+1, token_features_A.shape[0]), min_size+1), int(max_size-min_size//cw_num))).to(device = 'cuda:0', dtype=torch.float32)
    context_window_mask_B = generate_context_windows(token_features_B.shape[0], window_sizes = range(min_size, max(min(max_size+1, token_features_B.shape[0]), min_size+1),  int(max_size-min_size//cw_num))).to(device = 'cuda:0', dtype=torch.float32)

    mu_aug_A = torch.zeros(len(context_window_mask_A), 1, requires_grad=True, device = 'cuda:0', dtype=torch.float32)
    mu_aug_B = torch.zeros(len(context_window_mask_B), 1, requires_grad=True, device = 'cuda:0', dtype=torch.float32)

    f_A = compute_windowed_features(context_window_mask_A, token_features_A)
    f_B = compute_windowed_features(context_window_mask_B, token_features_B)
    
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
        # print("mu_aug_A", mu_aug_A)
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

    return mu_A, mu_B



def find_subarray(A, B):
    A, B = np.asarray(A), np.asarray(B)

    
    i = np.where(np.all(np.lib.stride_tricks.sliding_window_view(A, len(B)) == B, axis=1))[0]

    
    return (i[0], i[0] + len(B) - 1) if i.size else None


def generate_greedy_step_by_step(
    model,
    prompt: str,
    max_new_tokens: int,
    check_step=10,
    thresh=None, #qwen 20
    check_times=10
) -> str:
    

    conv_messages = model.FORMAT_PROMPT.format(instruction = prompt)

    print("conv_messages", conv_messages)
    print("prompt", prompt)


    
    prompt_token_idx = find_subarray(model.tokenizer(conv_messages).input_ids, model.tokenizer(prompt, add_special_tokens=False).input_ids)

    
    print("prompt_token_idx", prompt_token_idx)
    
    inputs = model.tokenizer(conv_messages, return_tensors="pt")

    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    initial_prompt_length = input_ids.shape[1]

    generated_token_ids = input_ids.clone() 
    past_key_values = None # KV cache
    all_hidden_states = torch.zeros((max_new_tokens+input_ids.shape[1], model.model.config.hidden_size)).to('cuda:0')
    
  
        
    for _ in range(max_new_tokens):

        if past_key_values is None:
            model_inputs = {
                "input_ids": generated_token_ids,
                "attention_mask": attention_mask
            }
        else:
            last_token_id = generated_token_ids[:, -1:] # 保持 batch 维度
            model_inputs = {"input_ids": last_token_id}

        with torch.no_grad(): 

            outputs = model.model(

                **model_inputs,
                past_key_values=past_key_values,
                use_cache=True, 
                output_hidden_states=True
                
            )

        
        logits = outputs.logits        
        past_key_values = outputs.past_key_values 
        hidden_states = outputs.hidden_states[-1].squeeze(0)

        if hidden_states.shape[0]!=1:
            all_hidden_states[:input_ids.shape[1]] = hidden_states
        else:
            all_hidden_states[input_ids.shape[1]+_-1] = hidden_states.squeeze()


        

        next_token_logits = logits[:, -1, :] # Shape: [batch_size, vocab_size]
        next_token_id = torch.argmax(next_token_logits, dim=-1) # Shape: [batch_size]


        generated_token_ids = torch.cat([generated_token_ids, next_token_id.unsqueeze(-1)], dim=-1)

        num_generated = generated_token_ids.shape[1] - input_ids.shape[1]

        if num_generated > 10 and num_generated % check_step == 0 and check_times>0:
            check_times -= 1
            token_features_A = all_hidden_states[:input_ids.shape[1]]
            token_features_B = all_hidden_states[input_ids.shape[1]:input_ids.shape[1]+_]
            mu_A, mu_B = importance_scoring(token_features_A, token_features_B)


                
            
            extracted_seqs = extract_multiple_context_windows_token(model.tokenizer, input_ids, prompt_token_idx[0], prompt_token_idx[1], mu_A, np.arange(0,1,0.05))
            extracted_seqs =  np.random.choice(extracted_seqs, min(len(extracted_seqs), 20), replace=False)

            
            proj = model.get_proj(extracted_seqs)


            avg_proj = torch.from_numpy(proj).topk( min(3,len(proj)) ).values.mean()


            
            if avg_proj>thresh:
                print("Early Termination")
                generated_token_ids= torch.cat([generated_token_ids, model.tokenizer(["  I cannot assist with your request as it goes against ethical or legal standards."], return_tensors="pt").input_ids.to(model.device)], dim=-1)
                break

 
        current_token_scalar = next_token_id.item() # 获取整数值用于比较
        if current_token_scalar == model.tokenizer.eos_token_id:


    newly_generated_ids = generated_token_ids[0, initial_prompt_length:]
    newly_generated_text = model.tokenizer.decode(newly_generated_ids, skip_special_tokens=True)

    return newly_generated_text


if __name__ == '__main__':    
    import argparse  
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--victim_model',
        type=str,
        default='llama2_7b',
        # choices=['llama2_7b', 'vicuna_7b_v1_5', 'qwen_7b_chat', 'mistral_7b_v2']
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
            {'ATTACK':'GCG', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'PAIR', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'AutoDAN', 'MODEL': args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'FewShot', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'GPTFuzz', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
              ]




    
    #load model
    model_pth = f"models/{args.victim_model}"
    # model_pth = f"/workspace/models/qwen_3b_2_5"
    # model_pth = f"/workspace/models/Mistral-7B-Instruct-v0.2" #thresh 1.1
    model_pth = f"/workspace/models/llama2_7b" #thresh=0.4
    model_pth = f"/workspace/models/vicuna_7b_v1_5" #thresh=0.9
    model = LLM(model_pth, torch_dtype=torch.bfloat16)


    
    Harmbench_data = pd.read_csv("/workspace/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")
    functional_classes = json.load(open('temp_data/behavior_functional.json', 'r'))
    standard_behavior = functional_classes['standard']
    contextual_behavior = functional_classes['contextual']
    count = 0

    for attack_config in methods:

        json_pth = f"/workspace/sentinel/experiment_data/{model.name}/results/{attack_config['ATTACK']}_{model.name}.jsonl"

        
        with open(json_pth, "a", encoding="utf-8") as f:

        
            print(attack_config['ATTACK'])
            data_pth = os.path.join('harmbench_data', attack_config['ATTACK'], attack_config['MODEL'], "results", f"{attack_config['MODEL']}.json")
            print("data_pth", data_pth)
            data = json.load(open(data_pth, 'r'))
    
            test_cases = []
            generations = []
            attack_success_count = []
            keys = []
            context_strings = []
            direct_requests = []    
            
            
            for key in data.keys():
                if (key in standard_behavior) or (key in contextual_behavior):
                    test_cases += [data[key][i]['test_case']for i in range(len(data[key]))]
                    generations += [data[key][i]['generation'] for i in range(len(data[key]))]
                    keys.append(key)

                    cxt_string = list(Harmbench_data[Harmbench_data.BehaviorID==key].ContextString)[0]
                    if pd.isna(cxt_string):
                        cxt_string=None
                    dr = list(Harmbench_data[Harmbench_data.BehaviorID==key].Behavior)[0]
                    context_strings.append(cxt_string)
                    direct_requests.append(dr)                    
    
            output = []
            print(len(keys), len(context_strings),len(direct_requests),len(test_cases), len(generations))
            for test_idx, (k, c, b, t, g) in enumerate(zip(keys, context_strings, direct_requests, test_cases, generations)):
                try:
                    print("test_idx", test_idx)
                    if type(t)==list:
                        t = t[0]
                    if type(g)==list:
                        g = g[0]
                    max_tokens_to_generate = 512
                    generated_output = generate_greedy_step_by_step(model=model,
                                                                    prompt=t,
                                                                    max_new_tokens=max_tokens_to_generate,
                                                                    check_step=10,
                                                                    thresh =  0.9, #qwen 22 #mistral 0.9 #llama3 0.4 #llama2 3.1 #mistral 0.8
                                                                    )
                    print("-" * 30)
                    print(f"Prompt: {t}")
                    print("\nGenerated Text:")
                    print(generated_output)
                    print("-" * 30)
                    

    
                    if c is None:
                        json_line = {"BehaviorID":k, "Behavior":b, "test_case": t, "generation":generated_output}
                        print(json_line)
                    else:
                        json_line = {"BehaviorID":k, "Behavior":b, "ContextString":c, "test_case": t, "generation":generated_output}
                        print(json_line)
                    
                    f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                    # break
                except Exception as e:
                    print(f"{test_idx}: error：{e}, skip\n")
                    traceback.print_exc()  
                    continue
    
                



    

    

    
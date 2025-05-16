from VLLM_general import VLLM
import torch
import json
import os
from datasets import load_dataset
import numpy as np
import pandas as pd


# model_pth = '/workspace/models/qwen_3b_2_5'
# model_pth = '/workspace/models/qwen_3b_2_5'
# model_pth = '/workspace/models/Mistral-7B-Instruct-RR'
# model_pth = '/workspace/models/Llama-3-8B-Instruct-RR'
# model_pth = "/workspace/models/LAT--robust-llama3-8b-instruct"
# model_pth = '/workspace/models/Llama3-8B-Instruct'
# model_pth = '/workspace/models/Mistral-7B-Instruct-v0.2'
model_pth = '/workspace/models/vicuna_7b_v1_5'



model = VLLM(model_pth)



if __name__ == '__main__':    
    import argparse  
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--victim_model',
        type=str,
        default='vicuna_7b_v1_5',
        choices=['llama2_7b', 'vicuna_7b_v1_5', 'qwen_7b_chat', 'mistral_7b_v2']
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

    methods = [
            {'ATTACK':'GCG', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'PAIR', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'AutoDAN', 'MODEL': args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'FewShot', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
            {'ATTACK':'GPTFuzz', 'MODEL':args.victim_model, 'VICTIM_MODEL':args.victim_model},\
              ]


    Harmbench_data = pd.read_csv("/workspace/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")
    functional_classes = json.load(open('temp_data/behavior_functional.json', 'r'))
    standard_behavior = functional_classes['standard']
    contextual_behavior = functional_classes['contextual']

    for attack_config in methods:

        json_pth = f"experiment_data/standard_generation/{model.name}/results/{attack_config['ATTACK']}.jsonl"
        if not os.path.exists(f"experiment_data/standard_generation/{model.name}/results"):
            os.makedirs(f"experiment_data/standard_generation/{model.name}/results")       
            
        with open(json_pth, "a", encoding="utf-8") as f:

        
            print(attack_config['ATTACK'])
            data_pth = os.path.join('harmbench_data', attack_config['ATTACK'], attack_config['MODEL'], "results", f"{attack_config['MODEL']}.json")
            print("data_pth", data_pth)
            data = json.load(open(data_pth, 'r'))
    
            test_cases = []
            generations = []
            keys = []
            context_strings = []
            direct_requests = []
            
            print("length", len(data))
            for key in list(data.keys()):
                if (key in standard_behavior) or (key in contextual_behavior):
                    test_cases += [data[key][i]['test_case'] for i in range(len(data[key]))]
                    generations += [data[key][i]['generation'] for i in range(len(data[key]))]
                    keys.append(key)
                    # print(test_cases)
                    cxt_string = list(Harmbench_data[Harmbench_data.BehaviorID==key].ContextString)[0]
                    if pd.isna(cxt_string):
                        cxt_string=None
                    dr = list(Harmbench_data[Harmbench_data.BehaviorID==key].Behavior)[0]
                    context_strings.append(cxt_string)
                    direct_requests.append(dr)

                    
            test_cases = [g[0] if isinstance(g, list) else g for g in test_cases]
            generations = [g[0] if isinstance(g, list) else g for g in generations]

            # print("test_cases", len(test_cases))
            # print("test_cases", test_cases[0])

            # quit()
            
            output = model.generate(test_cases)

  
            for k, c, b, t, g in list(zip(keys, context_strings, direct_requests, test_cases, output)):
                if c is None:
                    json_line = {"BehaviorID":k, "Behavior":b, "test_case": t, "generation":g}
                    print(json_line)
                else:
                    json_line = {"BehaviorID":k, "Behavior":b, "ContextString":c, "test_case": t, "generation":g}
                    print(json_line)


                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        
    
    

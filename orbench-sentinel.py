from VLLM_general import VLLM
from LLM_general import LLM
import torch
import json
import os
from datasets import load_dataset
import numpy as np
from pipeline_sentinel_gen import generate_greedy_step_by_step
from over_refusal_rate import is_overrefusal


model_pth = '/workspace/models/Llama3-8B-Instruct'
model_pth = "/workspace/models/Mistral-7B-Instruct-v0.2"
model_pth = "/workspace/models/llama2_7b"
model_pth = "/workspace/models/vicuna_7b_v1_5"

model = LLM(model_pth)

json_pth = os.path.join('orbench_data', 'DirectRequest', model.name, "results", f"{model.name}_sentinel.json")


from datasets import load_dataset

ds = load_dataset("orbench-llm/or-bench",  "or-bench-80k")

cls_list, num_list = np.unique([i['category'] for i in ds['train']], return_counts=True)

np.random.seed(42)
full_set = []
for cls, num in zip(cls_list, num_list):
    sample_idx = np.random.choice(num, 200, replace=False).tolist()
    subset = []
    for idx, test_case in enumerate(ds['train']):
        if test_case['category']==cls:
            subset.append(test_case['prompt'])
    
    sample_set = np.array(subset)[sample_idx]
    full_set.append(sample_set)


inputs = np.hstack(full_set)   


json_new_pth = os.path.join('orbench_data', 'sentinel', model.name, "results", f"{model.name}.jsonl")
overrefusal_count = []
with open(json_new_pth, "a", encoding="utf-8") as f:

    for idx, input in enumerate(inputs):
        output = generate_greedy_step_by_step(
        model,
        input,
        max_new_tokens=128,
        check_step=10,
        thresh = 0.9
        )

        overrefusal = is_overrefusal(input, output)
        overrefusal_count.append(overrefusal)
        print("idx", idx, "overrefusal:", np.mean(overrefusal_count))
        f.write(json.dumps({"test_case":input, "generation":output}, ensure_ascii=False) + "\n")

    
    
        



data = {}
for idx, (i, o) in enumerate(zip(inputs, outputs)):
    print(o)
    data[idx]={"test_case":i, "generation":o}



with open(json_pth, "w") as f:
    json.dump(data, f)
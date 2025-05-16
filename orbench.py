from VLLM_general import VLLM
import torch
import json
import os
from datasets import load_dataset
import numpy as np


model_pth = '/workspace/models/Llama-3-8B-Instruct'
model_pth = "/workspace/models/llama2_7b"
model_pth = "/workspace/models/vicuna_7b_v1_5"
model_pth = "/workspace/models/Mistral-7B-Instruct-v0.2"

model = VLLM(model_pth)

json_pth = os.path.join('orbench_data', 'DirectRequest', model.name, "results", f"{model.name}.json")
print("save pth", json_pth)

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


outputs = model.generate(inputs)



data = {}
for idx, (i, o) in enumerate(zip(inputs, outputs)):
    print(o)
    data[idx]={"test_case":i, "generation":o}



with open(json_pth, "w") as f:
    json.dump(data, f)
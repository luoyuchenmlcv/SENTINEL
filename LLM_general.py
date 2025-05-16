from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM
import torch
import json
import os
import torch.nn.functional as F
import numpy as np
from jaxtyping import Int, Float
from typing import List
from torch import Tensor
import contextlib
from typing import List, Tuple, Callable
import functools

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_activations_pre_hook(layer, cache: Float[Tensor, "n_samples pos layer d_model"], n_samples, positions: List[int]):
    def hook_fn(module, input):
        # print("input", input)
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:] = activation[:, positions, :].squeeze() #batch, last_few_positions, d_model
    return hook_fn





class LLM:
    def __init__(self, model_pth, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, load_dir=True):
        self.name = model_pth.split('/')[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = AutoModelForCausalLM.from_pretrained(model_pth, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2")
        self.model = AutoModelForCausalLM.from_pretrained(model_pth, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=trust_remote_code)        
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_pth, padding_side='left', trust_remote_code=True)     
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer_kwargs = {"return_tensors":"pt", "padding": True, "truncation": None, "max_length": None}

        if self.name == "Mistral-Small-24B-Instruct-2501":
            self.FORMAT_PROMPT = "[INST]{instruction}[/INST]"
        if self.name == "Mistral-7B-Instruct-v0.2":
            self.FORMAT_PROMPT = "[INST] {instruction} [/INST]"

            
        elif self.name == "llama2_7b" or self.name == "llama2_13b":
            self.FORMAT_PROMPT = "[INST] {instruction} [/INST] "
            
        elif self.name == "Llama3.1-8B-Instruct" or self.name == "Llama3-8B-Instruct":
            self.FORMAT_PROMPT = "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>"
            # self.FORMAT_PROMPT = '<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.name == "qwen_7b_2_5" or self.name == "qwen_14b_2_5" or self.name == "qwen_3b_2_5":
            # self.FORMAT_PROMPT = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"  
            self.FORMAT_PROMPT = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            
            self.tokenizer.pad_token = "<|endoftext|>"
        elif self.name == "zephyr_7b":
            self.FORMAT_PROMPT = "<|user|>\n{instruction}</s>\n<|assistant|>\n"
        elif self.name == "vicuna_7b_v1_5" or self.name == 'vicuna_13b_v1_5':
            # self.FORMAT_PROMPT = "Human: {instruction}\n Assistant:"
            self.FORMAT_PROMPT =  "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {instruction} ASSISTANT:"
            

        
        else:
            # raise
            pass
        
    
        self.generation_kwargs = {
                'num_return_sequences':1,
                'do_sample': False,
                'return_dict_in_generate': True,
                'num_beams':1,  
                'use_cache':True  
             }
        if load_dir:
            direction_dir = f'/workspace/refusal_direction/pipeline/runs/{self.name}/'
            direction_pth = os.path.join(direction_dir, 'direction.pt')
            direction = torch.load(direction_pth)
            self.direction = direction/torch.norm(direction)
            direction_meta = json.load(open(os.path.join(direction_dir, 'direction_metadata.json'), 'r'))
            self.sel_token, self.sel_layer = direction_meta['pos'], direction_meta['layer']
    
    def generate(self, prompt, return_token=False, return_logits=False, max_new_tokens=512, do_sample=False, prob=True):
        outputs = dict()
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs['output_logits'] = return_logits

        inputs = [self.FORMAT_PROMPT.format(instruction=p) for p in prompt]
        print("inputs", inputs)


        
        inputs = self.tokenizer(inputs, **self.tokenizer_kwargs)    
        inputs_len = inputs['input_ids'].shape[1]
        generation_kwargs['max_new_tokens'] = max_new_tokens
        generation_kwargs['do_sample'] = do_sample
        generation_outputs = self.model.generate(**inputs.to(self.device), **generation_kwargs)
        generation = self.tokenizer.batch_decode(generation_outputs.sequences[:, inputs_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        outputs['generation'] = generation
        
        if return_token:
            outputs_tokens = [self.tokenizer.convert_ids_to_tokens(seq_id) for seq_id in generation_outputs.sequences[:, inputs_len:]]
            outputs['outputs_tokens'] = outputs_tokens

            tokens = [self.tokenizer.convert_ids_to_tokens(seq_id) for seq_id in generation_outputs.sequences]
            outputs['tokens'] = tokens        
        
        if return_logits:
            if prob:
                probs = F.softmax(generation_outputs['logits'][0], dim=-1)
                outputs['prob'] = probs
            else:
                outputs['logits'] = generation_outputs['logits']
        return outputs



    def get_embedding(self, prompt):
        tokens = self.tokenizer(prompt, **self.tokenizer_kwargs, add_special_tokens=False)
        with torch.no_grad():
            out = self.model(**tokens.to(self.device), output_hidden_states=True)
        last_hidden_state = out.hidden_states[-1].squeeze()
        return torch.atleast_2d(last_hidden_state).to(dtype=torch.float32)



 




    def get_proj_batch(self, instructions, batch_size=1):
        import math
        torch.cuda.empty_cache()
    
        d_model = self.model.config.hidden_size
        proj_all = []
    
        def chunked(iterable, size):
            for i in range(0, len(iterable), size):
                yield iterable[i:i + size]
    
        for batch in chunked(instructions, batch_size):
            conv_messages = [self.FORMAT_PROMPT.format(instruction=p) for p in batch]
            inputs = self.tokenizer(conv_messages, **self.tokenizer_kwargs).to(self.device)
    
            n_samples = len(batch)
            activations = torch.zeros((n_samples, d_model), dtype=torch.float32, device=self.device)
    
            fwd_pre_hooks = [
                (
                    self.model.model.layers[self.sel_layer],
                    get_activations_pre_hook(
                        layer=self.sel_layer,
                        cache=activations,
                        n_samples=n_samples,
                        positions=[self.sel_token]
                    )
                )
            ]
    
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                with torch.no_grad():
                    _ = self.model.model(**inputs)
    
            proj = torch.sum(activations * self.direction, dim=-1).cpu().numpy()
            proj_all.append(proj)
    
        return np.concatenate(proj_all, axis=0)



    def get_tokens(self, prompt):
        input = self.tokenizer(prompt, **self.tokenizer_kwargs, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(input.input_ids.tolist()[0])
        return tokens

    def print_info(self):
        print(self.model)
        print(self.tokenizer)

    def toks2string(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)

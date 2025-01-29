from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import torch.nn.functional as F
import numpy as np
class LLM:
    def __init__(self, model_pth, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True):
        self.name = model_pth.split('/')[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_pth, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=trust_remote_code)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_pth, padding_side='left', trust_remote_code=True)     
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer_kwargs = {"return_tensors":"pt", "padding": True, "truncation": None, "max_length": None}

        if self.name == "mistral_7b_v2":
            self.FORMAT_PROMPT = "[INST] {instruction} [/INST]"
        elif self.name == "llama2_7b" or self.name == "llama2_13b":
            self.FORMAT_PROMPT = "[INST] {instruction} [/INST] "
        elif self.name == "zephyr_7b":
            self.FORMAT_PROMPT = "<|user|>\n{instruction}</s>\n<|assistant|>\n"
        elif self.name == "vicuna_7b_v1_5" or self.name == 'vicuna_13b_v1_5':
            self.FORMAT_PROMPT = "Human: {instruction}\n Assistant:"
        elif self.name == "qwen_7b_chat": #does not support batch parallelism
            self.FORMAT_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            self.tokenizer.pad_token = "<|extra_0|>"
            self.tokenizer.eos_token = "<|im_end|>"
            self.tokenizer.bos_token = "<|im_start|>"
        elif self.name == "baichuan2_7b":
            self.FORMAT_PROMPT = "<reserved_106>{instruction}<reserved_107>"
        elif self.name == "gemma2_9b" or self.name == "gemma2_9b_SimPO":
            self.FORMAT_PROMPT = "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
            self.tokenizer.bos_token = "<bos>"
            self.tokenizer.eos_token = "<eos>"
            self.tokenizer.pad_token = "<pad>"
        
    
        self.generation_kwargs = {
                'num_return_sequences':1,
                'do_sample': False,
                'return_dict_in_generate': True,
                'num_beams':1,  
                'use_cache':True  
             }

    def generate(self, prompt, return_token=False, return_logits=False, max_new_tokens=512, do_sample=False, prob=True):
        outputs = dict()
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs['output_logits'] = return_logits

        inputs = [self.FORMAT_PROMPT.format(instruction=p) for p in prompt]
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
        tokens = self.tokenizer(prompt, **self.tokenizer_kwargs)
        with torch.no_grad():
            out = self.model(**tokens.to(self.device), output_hidden_states=True)
        last_hidden_state = out.hidden_states[-1].squeeze()
        return last_hidden_state



    def compute_perplexity_batch(self, prompt, target, batch_size=5):

        prompt = np.array(prompt)
        target = np.array(target)
        

        import itertools  
        concat_inputs = np.array(list(itertools.product(prompt, target)))
        concat_indices = list(itertools.product(range(len(prompt)), range(len(target))))
        perplexities = []
        lls = []
        with torch.no_grad():
        
            batches = np.arange(len(concat_inputs)//batch_size*batch_size).reshape(-1, batch_size).tolist()
            batches.append([len(concat_inputs)//batch_size*batch_size+i for i in range(len(concat_inputs)%batch_size)])
            batches = [b for b  in batches if b]
            for batch_idx in batches:
                
                batch_concat_inputs = [i+j for i,j in concat_inputs[batch_idx]]
                batch_inputs = self.tokenizer(batch_concat_inputs, **self.tokenizer_kwargs)
                batch_target = [j for i,j in concat_inputs[batch_idx]]
    
                labels = torch.zeros(batch_inputs.input_ids.shape, device=self.device, dtype=torch.long)-100
                batch_target_len = [len(i)-1 for i in self.tokenizer(batch_target).input_ids]
        
                for i, (t, t_len) in enumerate(zip(batch_target, batch_target_len)):
                    labels[i, -t_len:] = batch_inputs.input_ids[i, -t_len:]
                batch_inputs['labels'] = labels
        
                
                generation_outputs = self.model(**batch_inputs.to(self.device))
                logits = generation_outputs.logits
                
                from torch.nn import CrossEntropyLoss 
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()


                
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, logits.shape[-1])
               
                
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                losses = loss_fct(shift_logits, shift_labels)
                losses = losses.view(batch_inputs.input_ids.shape[0], -1)
                batch_nlls = losses.sum(dim=1)  
                lengths = (losses != 0).sum(dim=1) 
                average_losses = batch_nlls / lengths
                batch_perplexities = torch.exp(average_losses)

                perplexities.append(batch_perplexities.cpu().numpy())
                lls.append(-batch_nlls.cpu().numpy())
                
            lls = np.hstack(lls)
            perplexities = np.hstack(perplexities)

            lls_matrix = np.zeros((len(prompt), len(target)))
            lls_matrix[list(zip(*concat_indices))[0], list(zip(*concat_indices))[1]] = lls.copy()
            
            perplexities_matrix = np.zeros((len(prompt), len(target)))
            perplexities_matrix[list(zip(*concat_indices))[0], list(zip(*concat_indices))[1]] = perplexities.copy()
                    
        return lls_matrix 

    def get_tokens(self, prompt):
        input = self.tokenizer(prompt,   **self.tokenizer_kwargs)
        tokens = self.tokenizer.convert_ids_to_tokens(input.input_ids.tolist()[0])
        return tokens

    def print_info(self):
        print(self.model)
        print(self.tokenizer)

    def toks2string(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)

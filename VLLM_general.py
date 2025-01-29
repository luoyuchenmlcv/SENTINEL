from vllm import LLM, SamplingParams
import torch
import json
import os
import torch.nn.functional as F
import numpy as np
import itertools

class VLLM:
    def __init__(self, model_pth):
        self.name = model_pth.split('/')[-1]
        self.model = LLM(model=model_pth, trust_remote_code=True)
        self.device = 'cuda:0'
        ###################################chat-template###################################
        if self.name == "mistral_7b_v2":
            self.FORMAT_PROMPT = "[INST] {instruction} [/INST]"
        elif self.name == "llama2_7b":
            self.FORMAT_PROMPT = "[INST] {instruction} [/INST] "
        elif self.name == "llama3_1_8B":
            self.FORMAT_PROMPT = "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>"
            self.model.llm_engine.tokenizer.tokenizer.pad_token = self.model.llm_engine.tokenizer.tokenizer.eos_token
        elif self.name == "qwen_7b_2_5":
            self.FORMAT_PROMPT = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            self.model.llm_engine.tokenizer.tokenizer.pad_token = "<|endoftext|>"
        elif self.name == "zephyr_7b":
            self.FORMAT_PROMPT = "<|user|>\n{instruction}</s>\n<|assistant|>\n"
        elif self.name == "vicuna_7b_v1_5" or self.name == 'vicuna_13b_v1_5':
            self.FORMAT_PROMPT = "Human: {instruction}\n Assistant:"
        elif self.name == "qwen_7b_chat": 
            self.FORMAT_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            self.model.llm_engine.tokenizer.tokenizer.pad_token = "<|extra_0|>"
            self.model.llm_engine.tokenizer.tokenizer.eos_token = "<|im_end|>"
            self.model.llm_engine.tokenizer.tokenizer.bos_token = "<|im_start|>"
        elif self.name == "baichuan2_7b":
            self.FORMAT_PROMPT = "<reserved_106>{instruction}<reserved_107>"
        elif self.name == "gemma2_9b":
            self.FORMAT_PROMPT = "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
            self.model.llm_engine.tokenizer.tokenizer.bos_token = "<bos>"
            self.model.llm_engine.tokenizer.tokenizer.eos_token = "<eos>"
            self.model.llm_engine.tokenizer.tokenizer.pad_token = "<pad>"
        ###################################chat-template###################################


    def generate(self, prompt,  max_new_tokens=512, do_sample=False):
        prompt =  [self.FORMAT_PROMPT.format(instruction=p) for p in prompt]
        if do_sample == False:
            sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        generation_kwargs = dict(sampling_params=sp, use_tqdm=True)
        responses = self.model.generate(prompt, **generation_kwargs)
        return [r.outputs[0].text for r in responses]




    def get_embedding(self, prompt):
        input_ids = self.model.get_tokenizer()(prompt).input_ids
        input_ids = torch.tensor(input_ids).to(self.device)
        embedding = self.model.llm_engine.model_executor.driver_worker.model_runner.model.model.get_input_embeddings(input_ids)
        return embedding






    def compute_perplexity_batch(self, inputs, suffixes, batch_size):
        sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=True)
        generation_kwargs = dict(sampling_params=sp, use_tqdm=False)

        # Tokenize inputs and suffixes
        inputs_ids = self.model.get_tokenizer()(inputs).input_ids

        inputs_tokens = [self.model.get_tokenizer().tokenize(s) for s in inputs]

        suffixes_ids = self.model.get_tokenizer()(suffixes).input_ids
        suffixes_tokens = [self.model.get_tokenizer().tokenize(s) for s in suffixes]

        # Create all combinations of inputs and suffixes
        concat_indices = list(itertools.product(range(len(inputs)), range(len(suffixes))))
        concat_inputs = [inputs[i] + suffixes[j] for i, j in concat_indices]
    
        # Initialize the result log probability matrix
        logprob_matrix = np.zeros((len(inputs), len(suffixes)))
    
        # Calculate the total number of batches
        total_batches = (len(concat_inputs) + batch_size - 1) // batch_size
    
        # Use tqdm for the outer loop
        from tqdm import tqdm
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for batch_start in range(0, len(concat_inputs), batch_size):
                batch_end = min(batch_start + batch_size, len(concat_inputs))
                batch_inputs = concat_inputs[batch_start:batch_end]
    
                # Generate responses for the current batch
                responses = self.model.generate(batch_inputs, **generation_kwargs)
    
                # Process responses and fill logprob_matrix
                for k, response in enumerate(responses):
                    index = batch_start + k
                    i, j = concat_indices[index]  # Retrieve the original input-suffix indices

                    logprob_ij_dict = response.prompt_logprobs[-len(suffixes_tokens[j]):]
                    
                    logprob_ij = 0

                    suffix_id = suffixes_ids[j][1:]
                    suffix_token = suffixes_tokens[j]

                    for token_id, token_logprob_ij_dict in zip(suffix_id, logprob_ij_dict):
                        token_logprob = token_logprob_ij_dict[token_id].logprob
                        logprob_ij += token_logprob
                    logprob_matrix[i, j] = logprob_ij

                pbar.update(1)
        return logprob_matrix
    
        
    def get_tokens(self, prompt):
        return self.model.get_tokenizer().tokenize(prompt)


    
    def toks2string(self, prompt):
        return self.model.get_tokenizer().convert_tokens_to_string(prompt)


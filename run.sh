#using VLLM framework for infrence efficiency, qwen-7b-chat too old to be compatible with current version of vllm, need to shift to transformers library
python pipeline-vllm.py --victim_model vicuna_7b_v1_5
python pipeline-vllm.py --victim_model llama2_7b
python pipeline-vllm.py --victim_model mistral_7b_v2
python pipeline.py --victim_model qwen_7b_chat

python calculate_asr-multiple.py --folder_path experiment_data/final_results/alpha_0.25_thresh_num_20_cw_num_16_top3 
#llama2_7b
python pipeline_sentinel_gen.py --victim_model llama2_7b --harm_thresh 3.25
python attack_success_rate.py --victim_model llama2_7b
python orbench-sentinel.py --victim_model llama2_7b --harm_thresh 3.25


# #mistral_7b_v2 
python pipeline_sentinel_gen.py --victim_model mistral_7b_v2 --harm_thresh 1.1
python attack_success_rate.py --victim_model mistral_7b_v2
python orbench-sentinel.py --victim_model mistral_7b_v2 --harm_thresh 1.1

# Llama3-8B-Instruct
python pipeline_sentinel_gen.py --victim_model Llama3-8B-Instruct --harm_thresh 0.85
python attack_success_rate.py --victim_model Llama3-8B-Instruct
python orbench-sentinel.py --victim_model Llama3-8B-Instruct --harm_thresh 0.85


# vicuna_7b_v1_5
python pipeline_sentinel_gen.py --victim_model vicuna_7b_v1_5 --harm_thresh 1.2
python attack_success_rate.py --victim_model vicuna_7b_v1_5
python orbench-sentinel.py --victim_model vicuna_7b_v1_5 --harm_thresh 1.2



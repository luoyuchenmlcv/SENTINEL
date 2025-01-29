apt-get update
apt-get install zip unzip
#python packages
pip install fschat huggingface_hub transformers matplotlib openai accelerate einops scikit-learn sentence-transformers nltk pandas protobuf google sentencepiece transformers_stream_generator vllm

#optional
echo 'export TRANSFORMERS_CACHE=/workspace/cache/transformer_cache' >> ~/.bashrc
echo 'export HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface_hub_cache' >> ~/.bashrc
source ~/.bashrc
#download models
python hf_download.py --model lmsys/vicuna-7b-v1.5 --save_dir ./models
python hf_download.py --model mistralai/Mistral-7B-Instruct-v0.2 --save_dir ./models
python hf_download.py --model Qwen/Qwen-7B-Chat --save_dir ./models
python hf_download.py --model meta-llama/Llama-2-7b-chat-hf --save_dir ./models 
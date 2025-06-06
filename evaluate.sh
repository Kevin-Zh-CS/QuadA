#!/bin/bash

# Which GPUs you would like to use for this script:
# For 4 GPUs, set it to 0,1,2,3; for 2 GPUs, set it to 0,1, etc.
cuda_devices=0,1,2,3

# Which layer of the transformer block will the activation perturbation be added to:
# Options are "LN" and "FFN"
noise_source="LN"

# How much perturbation/noise will be added (based on standard deviation):
noise_std=0.075

# Uncomment ONLY one of the models below that you would like to test:

# LLMs w/o QuadA:
# model_name="meta-llama/Llama-2-7b-chat-hf"
model_name="meta-llama/Llama-3.1-8B-Instruct"
# model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_name="mistralai/Mistral-7B-Instruct-v0.3"
# model_name="HuggingFaceH4/zephyr-7b-beta"
# model_name="Qwen/Qwen2.5-1.5B-Instruct"
# model_name="Qwen/Qwen2.5-7B-Instruct"
# model_name="microsoft/Phi-3.5-mini-instruct"
# model_name="microsoft/Phi-3-mini-4k-instruct"

# LLMs w/ QuadA:
# model_name="./output/alignment/Llama-2-7b-chat-hf-dpo"
# model_name="./output/alignment/Llama-3.1-8B-Instruct-dpo"
# model_name="./output/alignment/Mistral-7B-Instruct-v0.3-dpo"
# model_name="./output/alignment/Mixtral-8x7B-Instruct-v0.1-dpo"
# model_name="./output/alignment/zephyr-7b-beta-dpo"
# model_name="./output/alignment/Qwen2.5-1.5B-Instruct-dpo"
# model_name="./output/alignment/Qwen2.5-7B-Instruct-dpo"
# model_name="./output/alignment/Phi-3.5-mini-instruct-dpo"
# model_name="./output/alignment/Phi-3-mini-4k-instruct-dpo"

# Uncomment the absolute path to the modeling file of the model you are testing below:
modeling_file_path="./transformers/src/transformers/models/llama/modeling_llama.py"
# modeling_file_path="./transformers/src/transformers/models/mistral/modeling_mistral.py"
# modeling_file_path="./transformers/src/transformers/models/qwen2/modeling_qwen2.py"
# modeling_file_path="~/.cache/huggingface/modules/transformers_modules/microsoft/Phi-3.5-mini-instruct/<path-to-your>/modeling_phi3.py"
# modeling_file_path="~/.cache/huggingface/modules/transformers_modules/microsoft/Phi-3-mini-4k-instruct/<path-to-your>/modeling_phi3.py"

# ------------------------------------------------------------------------------
# DO NOT CHANGE ANYTHING BELOW
# ------------------------------------------------------------------------------

# Reset noise to zero
python evaluation/std.py "$noise_source" 0.00 0.00 "$modeling_file_path"

# Increase noise std to the desired value
python evaluation/std.py "$noise_source" "$noise_std" "$noise_std" "$modeling_file_path"

# Evaluate model
CUDA_VISIBLE_DEVICES=$cuda_devices python evaluation/core.py "$noise_source" "$noise_std" "$model_name"

# Reset noise to zero
python evaluation/std.py "$noise_source" 0.00 0.00 "$modeling_file_path"

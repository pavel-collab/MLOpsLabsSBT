#!/bin/bash

# Description: This script automates the installation and inference process for a Hugging Face model using TensorRT-LLM. Ensure that Git and Git LFS ('apt-get install git-lfs') are installed before running this script. Before running this script, run the following scripts sequentially: 1. install_git_and_lfs.sh 2. install_tensorrt_llm.sh
HF_MODEL_NAME="Trendyol-LLM-7b-chat-v1.0"
HF_MODEL_PATH="Trendyol/Trendyol-LLM-7b-chat-v1.0"

# Clone the Hugging Face model repository
mkdir -p /tensorrt/models && cd /tensorrt/models && git clone https://huggingface.co/$HF_MODEL_PATH
# Convert the model checkpoint to TensorRT format
python /tensorrt/v0.8.0/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py \
    --model_dir /tensorrt/models/$HF_MODEL_NAME \
    --output_dir /tensorrt/tensorrt-models/$HF_MODEL_NAME/v0.8.0/trt-checkpoints/fp16/1-gpu/ \
    --dtype float16

# Build TensorRT engine
trtllm-build --checkpoint_dir /tensorrt/tensorrt-models/$HF_MODEL_NAME/v0.8.0/trt-checkpoints/fp16/1-gpu/ \
    --output_dir /tensorrt/tensorrt-models/$HF_MODEL_NAME/v0.8.0/trt-engines/fp16/1-gpu/ \
    --remove_input_padding enable \
    --context_fmha enable \
    --gemm_plugin float16 \
    --max_input_len 32768 \
    --strongly_typed
    
# Run inference with the TensorRT engine
python3 /tensorrt/v0.8.0/tensorrtllm_backend/tensorrt_llm/examples/run.py \
    --max_output_len=250 \
    --tokenizer_dir /tensorrt/models/$HF_MODEL_NAME \
    --engine_dir=/tensorrt/tensorrt-models/$HF_MODEL_NAME/v0.8.0/trt-engines/fp16/1-gpu/ \
    --max_attention_window_size=4096 \
    --temperature=0.3 \
    --top_k=50 \
    --top_p=0.9 \
    --repetition_penalty=1.2 \
    --input_text="[INST] Sen yardımsever bir asistansın ve sana verilen talimatlar doğrultusunda en iyi cevabı üretmeye çalışacaksın.\n\nTürkiye'nin doğusunda ne var? [/INST]"
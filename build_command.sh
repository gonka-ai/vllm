DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai-validation --build-arg max_jobs=64 --build-arg nvcc_threads=2 --build-arg LLM_MAX_SIZE_MB=600 --build-arg PYTHON_VERSION=3.10 --build-arg CUDA_VERSION="12.6.0"

# docker run -it --runtime nvidia --gpus all \
#     -v /mnt/ramdisk/cache/huggingface:/root/.cache/huggingface \
#     --entrypoint /bin/sh \
#     --env "HUGGING_FACE_HUB_TOKEN=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" \
#     -p 8000:8000 \
#     --ipc=host \
#     -v $(pwd)/vllm:/usr/local/lib/python3.12/dist-packages/vllm/ \
#     vllm_v073

# docker run --runtime nvidia --gpus all \
#     -v /mnt/ramdisk/cache/:/root/.cache/huggingface \
#     -v $(pwd)/vllm/model_executor:/usr/local/lib/python3.12/dist-packages/vllm/model_executor \
#     -v $(pwd)/vllm/entrypoints/:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/ \
#     -v $(pwd)/vllm/sampling_params.py:/usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py \
#     --env "HUGGING_FACE_HUB_TOKEN=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" \
#     -p 8000:8000 \
#     --ipc=host \
#     vllm_v073 \
#     --model unsloth/llama-3-8b-Instruct


# curl -X POST "0.0.0.0:8000/v1/chat/completions" -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
#     "model": "unsloth/llama-3-8b-Instruct",
#     "messages": [
#       {"role": "user", "content": "Who won the world series in 2020? Сгенерируй интересный смешной и оригинальный текст"}
#     ],
#     "max_tokens": 80,
#     "temperature": 0.5,
#     "seed": 42,
#     "stream": false,
#     "logprobs": 1,
#     "top_logprobs": 3
#   }' | jq


# curl -X POST "0.0.0.0:8000/v1/chat/completions" -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
#     "model": "unsloth/llama-3-8b-Instruct",
#     "messages": [
#       {"role": "user", "content": "Who won the world series in 2020? Сгенерируй интересный смешной и оригинальный текст"}
#     ],
#     "max_tokens": 80,
#     "temperature": 0.5,
#     "seed": 42,
#     "stream": false,
#     "logprobs": 1,
#     "top_logprobs": 3,
#     "enforced_str": "The thrilling tale of the 2020 World Series! It was a season like no other, with teams battling it out on the field while also navigating the challenges of a global pandemic. But in the end, there could only be one champion.\n\nAnd that champion was... the Los Angeles Dodgers! ThatHUI right, the Boys in Blue brought home their first World Series title since 1988, and"
#   }' | jq




# Models Proposal

This is a proposal to add the [Qwen3-30B-A3B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8), [Qwen3-235B-A22B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8), [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528), and [Gemma-3-27B](https://huggingface.co/RedHatAI/gemma-3-27b-it-FP8-dynamic) models to the Gonka inference network. 

Validation thresholds for all the models were computed using the standard procedure described in [models/README.md](../README.md).

For each model, there are respective notebooks with the details of experiments and gdrive folders with raw inference-validation data:

Although Qwen3-235B-A22B-Instruct-2507-FP8 model is already onchain, there is a parameters adjustment required due to too high False Positive Rate. Current recomputation of the the threshold is intended to fix this.

| Parameter | Qwen3-30B | Qwen3-235B | DeepSeek-R1-0528 | Gemma-3-27B |
|-----------|-----------|-------------|-------------------|-------------|
| Notebook | [qwen30B_thresholds.ipynb](./qwen30B_thresholds.ipynb) | [qwen235B_thresholds.ipynb](./qwen235B_thresholds.ipynb) | [deepseek_thresholds.ipynb](./deepseek_thresholds.ipynb) | [gemma27B_thresholds.ipynb](./gemma27B_thresholds.ipynb) |
| Validation Data | [Qwen30B Validation Data](https://drive.google.com/drive/folders/1JqZ4wFsOr-RRZStk5bhnWppiV5g_-SpY?usp=sharing) | [Qwen3-235B Validation Data](https://drive.google.com/drive/folders/1yPCZg_hh3Ab4upvF7TcXoeMLJUIpBif2?usp=sharing) | [DeepSeek-R1-0528 Validation Data](https://drive.google.com/drive/folders/15ooQa3zjm1MCrN7NHt7z-f1pVK-A-pBt?usp=sharing) | [Gemma-3-27B Validation Data](https://drive.google.com/drive/folders/1RLlzLdTUj1vroQuD878Lr7wuEb3erCQZ?usp=sharing) |
| Model Len | 100000 | 240000 | 64000 | 64000 |
| Validation Thresholds | 0.023 | 0.042 | 0.053* | 0.05* |
| Fraud Accuracy | 52% | 24% | 83% | 99% |
| Tested Against | [Qwen3-30B-A3B-Instruct-2507-AWQ-4bit](https://huggingface.co/cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit) | [Qwen3-235B-A22B-Instruct-2507-INT4-W4A16](https://huggingface.co/chriswritescode/Qwen3-235B-A22B-Instruct-2507-INT4-W4A16) | [DeepSeek-R1-0528-quantized.w4a16](https://huggingface.co/RedHatAI/DeepSeek-R1-0528-quantized.w4a16) | [gemma-3-27b-it-GPTQ-4b-128g](https://huggingface.co/ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g) |
| VRAM (example setup) | $\geq$ 48GB (2xRT3090 or 1xH100) | ~320GB (4xH100 or 4xH200) | ~1128GB (8xH200) | $\geq$ 81GB (4x3090 or 1xH100) |

*Value of validation threshold for DeepSeek-R1-0528 and Gemma-3-27B were increased by +0.02 and +0.10 respectively, because upper vaidation threshold were computed on a relatively small sample (400 and 200 samples) AND both models are quite good in terms of fraud detection. It won't hurt the safety from fraud, but will significantly decrease False Postive chance. 

For the reproduction of raw data, the inference script producing the raw data is here: [link](https://github.com/gonka-ai/gonka/blob/tg/qwen_models/mlnode/packages/benchmarks/scripts/inference.py). You'll also need to set up configs in this script, you'll find them in GDrive with the raw data.

All experiments were conducted using MLNode v3.0.8.

**Qwen3-30B** is suggested to be deployed with with the following parameters:
```python
additional_args=[
    '--max-model-len', '100000', #Fits the minimum 48GB
    '--enable-auto-tool-choice',  # Optional: enables automatic tool choice
    '--tool-call-parser', 'hermes',  # Optional: specifies the Hermes tool call parser
]
```

**Qwen3-235B** is suggested to be deployed with the following parameters:
```python
additional_args=[
    '--max-model-len', '240000',
    '--enable-auto-tool-choice',  # Optional: enables automatic tool choice
    '--tool-call-parser', 'hermes',  # Specifies the Hermes tool call parser
]
```


**DeepSeek-R1-0528** is suggested to be deployed with the following parameters:
```python
additional_args=[
    '--quantization', 'fp8',
    '--enable-expert-parallel',
    '--max_model_len', '64000',
    '--enable-auto-tool-choice',  # Optional: enables automatic tool choice
    '--tool-call-parser', 'deepseek_v3',  # Specifies the DeepSeek V3 tool call parser
]
```

**Gemma-3-27B** is suggested to be deployed with the following parameters:
```python
additional_args=[
    '--max-model-len', '64000',
    '--gpu-memory-utilization', '0.95',
    '--enable-chunked-prefill',
    '--enable-auto-tool-choice',  # Optional: enables automatic tool choice
    '--tool-call-parser', 'pythonic',  # Specifies the Pythonic tool call parser
]
```





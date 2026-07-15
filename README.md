## mini-llm

Implementation of a Qwen 3.5 level LLM from first principles.


## Pretraining:

- Using the [common crawl sample dataset](https://huggingface.co/datasets/agentlans/common-crawl-sample)


## Resources:

- https://sebastianraschka.com/llms-from-scratch/
- https://qwen.ai/blog?id=qwen3.5


## Flow:

1. Download Pretraining Data as 
```
uv run python -m llm.pretraining.download_data
```
import tiktoken
import torch
from mini_llm.training.mini_llm_gpt import GPTModel
from mini_llm.config import GPT_CONFIG_124M


def main():
    # 1. tokenization stage
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    # 2. initialize instance and feed it the tokenized batch
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print(f"Output shape: {logits.shape}")
    print(logits)


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetLoader(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # tokenize the entire text
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            # uses a sliding window to chunk the book into overlapping sequences of max_length
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # returns the totla number of rows in the dataset
        return len(self.input_ids)

    def __getitem__(self, index):
        # returns a single row from the dataset
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # init the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetLoader(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # drop last True drops the last batch if it is shorter than the specified batch size to prevent loss spikes during training
        drop_last=drop_last,
        # number of CPU processes to use for preprocessing
        num_workers=num_workers,
    )
    return dataloader

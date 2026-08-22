import torch 
from torch.utils.data import Dataset, DataLoader
from tokenizer_run import tokenizer

class GPTDatasetLoader(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        # tokenize the entire text 
        token_ids = tokenizer.encode(txt)
        
        for i in range(0,len(token_ids) - max_length , stride):
            # uses a sliding window to chunk the book into overlapping sequences of max_length
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
             
    def __len__(self):
        # returns the totla number of rows in the dataset 
        return len(self.input_ids)

    def __getitem__(self, index):
        # returns a single row from the dataset 
        return self.input_ids[index],self.target_ids[index]
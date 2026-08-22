from pathlib import Path
#from tokenizer_run import tokenizer
from dataset_loader import create_dataloader_v1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = PROJECT_ROOT / "data" / "the_verdict.txt"

raw_text = FILE_PATH.read_text(encoding="utf-8")

print(f"Total number of characters: {len(raw_text)}")
print(raw_text[:99])

# enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

# # remove the first 50 tokens from the dataset for demonstration purposes,as it results in a slightly more interesting text passage in the next steps

# enc_sample = enc_text[50:]

# # create two variables x and y, where x contains the input tokens and y contains the targets, which are the inputs shifted by 1 

# context_size = 4
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]
# print(f"x: {x}")
# print(f"y: {y}")

dataloader = create_dataloader_v1(
    raw_text,batch_size=1,max_length=4,stride=1,shuffle=False
)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# more batch sizes 

dataloader2 = create_dataloader_v1(
    raw_text,batch_size=8,max_length=4,stride=4,
    shuffle=False
)

data_iter = iter(dataloader2)
inputs, targets = next(data_iter)
print("inputs: \n",inputs)
print("\ntargets: \n",targets)
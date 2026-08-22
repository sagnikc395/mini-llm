from pathlib import Path 
import re 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = PROJECT_ROOT / "data" / "the_verdict.txt"
raw_text = FILE_PATH.read_text(encoding="utf-8")

# preprocess the data
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# remove whitespace as well, handles other types of punctuation, like question marks, quotation marks.
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

# building a dictionary 
all_words = sorted(set(preprocessed))
vocab_size= len(all_words)
print(vocab_size)

# print the vocab 

vocab = {token: integer for integer, token in enumerate(all_words)}
#print the first 50 items of the vocabulary 
for k,v in enumerate(vocab.items()):
    print(v)
    if k >= 50:
        break 
      
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
all_tokens = sorted(set(preprocessed))
# extend with the eot and unk
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab_size= len(all_tokens)
print(vocab_size)

# print the vocab 

vocab = {token: integer for integer, token in enumerate(all_tokens)}
# #print the first 50 items of the vocabulary 
# for k,v in enumerate(vocab.items()):
#     print(v)
#     if k >= 50:
#         break 
      
class SimpleTokenizer:
    def __init__(self,vocab):
        # store the vocabulary as a class attribute for accessing in the encode and decode methods
        self.str_to_int = vocab 
        # create an inverse vocabulary that will map token IDS back to the original text tokens 
        self.int_to_str = {i:s for s,i in vocab.items()}
        
    def encode(self,text):
        # process input text into token IDS
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)',text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids 
    
    def decode(self,ids):
        # convert token IDS back to text 
        text = " ".join([self.int_to_str[i] for i in ids])
        # remove spaces before the specified punctuation 
        text = re.sub(r'\s+([,.?!"()\'])',r'\1',text) 
        return text 
    
tokenizer = SimpleTokenizer(vocab)
sample_text = raw_text[:500]
# turn the embeddings into ids
ids = tokenizer.encode(sample_text)
print(ids)

# and them turn them back 
print(tokenizer.decode(ids))

        
import torch

from dataset_loader import create_dataloader_v1 
from read_data import raw_text

# input_ids = torch.tensor([2,3,5,1])

# # using actual level size we might wanna use  
# #vocab_size = 6 
# vocab_size = 50257
# #output_dim = 3 
# output_dim = 256 

# torch.manual_seed(123)
# token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
# print(token_embedding_layer.weight)

# # applying a token ID to obtain the embedding vector 
# print(token_embedding_layer(torch.tensor([3])))

# # applying to all 4 input ids 
# print(token_embedding_layer(input_ids))

vocab_size = 50257
output_dim = 256 
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)

max_length = 4
dataloader = create_dataloader_v1(
raw_text, batch_size=8, max_length=max_length,
stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# using the embedding layer to embed the token IDS into 256 dimensioanl vectors 
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# for GPT model absolute embedding , we just need to create another embedding layer that has the same embedding dimensions as the token_embedding_layer 

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

print(pos_embeddings.shape)

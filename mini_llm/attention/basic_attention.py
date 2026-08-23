import torch 

inputs = torch.tensor(
    [[0.43,0.15,0.89],
     [0.55,0.87,0.66],
     [0.57,0.85,0.64],
     [0.22,0.58,0.33],
     [0.77,0.25,0.10],
     [0.05,0.80,0.55]]
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i,query)
print(attn_scores_2)

res = 0 
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print(res)
print(torch.dot(inputs[0],query))

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print(f"Attention weights: {attn_weights_2_tmp}")
print(f"Sum: {attn_weights_2_tmp.sum()}")

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print(f"attention weights: {attn_weights_2_naive}")
print(f"sum: {attn_weights_2_naive.sum()}")

# using pytorch implementation of softmax 
attn_weights_2 = torch.softmax(attn_scores_2,dim=0)
print(f"attention weights: {attn_weights_2}")
print(f"sum: {attn_weights_2.sum()}")
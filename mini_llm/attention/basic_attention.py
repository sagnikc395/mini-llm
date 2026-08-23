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

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i 
print(context_vec_2)

## calculating the attention score 
# attn_scores = torch.empty(6,6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i,j] = torch.dot(x_i,x_j)

# print(attn_scores)

attn_scores = inputs @ inputs.T 
print(attn_scores)

# then normalize each row so that the values in each row sum to 1 
attn_weights = torch.softmax(attn_scores,dim=-1)
print(attn_weights)

# can verify the rows indeed all sum to 1 
row_2_sums = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print(f"row 2 sum: {row_2_sums}")
print(f"all row sums: {attn_weights.sum(dim=-1)}")

all_context_vecs = attn_weights @ inputs 
print(all_context_vecs)
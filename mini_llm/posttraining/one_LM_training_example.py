import torch

logits = model(
    input_ids
).logits  # (B,L,V) -> score for every vocab token, at every position
logits = logits[
    :, :-1, :
]  # drop the last position -> its prediction has no nxt token to score
labels = input_ids[
    :, 1:
]  # the target at each position is just the actual next token (the shift)

completion_mask = completion_mask[
    :, 1:
]  # shift the mask the same way so that is lines up with labels (1 = completion)

log_probs = logits.log_softmax(
    dim=-1
)  # normalize the logits into log probabilities over the vocab -> (B,L-1,V)

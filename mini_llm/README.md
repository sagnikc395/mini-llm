## most easiest and dumbest inference 

1. Assume after post training the model checkpoints are like these 
```
checkpoints 
-> model weights 
-> model config 
    -> hidden_size 
    -> num_layers 
    -> num_heads 
    -> vocab_size 
    -> context_length
-> tokenizer
```

2. First inference engine would look like
```
prompt
  ↓
tokenize
  ↓
[input token IDs]
  ↓
load model weights
  ↓
forward pass
  ↓
logits for last token
  ↓
sampling
  ↓
next token
  ↓
append token
  ↓
forward again
  ↓
...
```

3. Load the checkpoints , feed token IDs into the model, produce one next token correctly and then repeat.

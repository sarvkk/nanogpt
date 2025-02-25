# NanoGPT

This train.py is based on the principles from the "Attention is All You Need" paper by Vaswani et al., which introduced the Transformer architecture. The Transformer architecture is the foundation for many subsequent models, including GPT-2 and GPT-3.

## Introduction

This project implements a simplified version of a GPT-like model using the Transformer architecture. It includes self-attention mechanisms, multi-head attention, feed-forward networks, layer normalization, and positional encoding.

## Key Points

Here are the key points that align with the "Attention is All You Need" paper:

1. Self-Attention Mechanism: The Head and MultiHeadAttention classes implement the self-attention mechanism, which is a core component of the Transformer architecture.

2. Multi-Head Attention: The MultiHeadAttention class combines multiple attention heads, allowing the model to focus on different parts of the input sequence simultaneously.

3. Feed-Forward Networks: The FeedForward class implements the position-wise feed-forward networks used in the Transformer architecture.

4. Layer Normalization: The Block class includes layer normalization, which is used in the Transformer architecture to stabilize and accelerate training.

5. Positional Encoding: The BigramLanguageModel class includes positional embeddings, which are used to provide the model with information about the position of tokens in the sequence.

While the codebase is inspired by the "Attention is All You Need" paper, it also incorporates elements specific to language modeling, such as token and position embeddings, which are used in models like GPT-2 and GPT-3. However, the code does not implement the full GPT-2 or GPT-3 architecture, which includes additional components and optimizations.

In summary, train.py is primarily based on the "Attention is All You Need" paper, with some adaptations for language modeling tasks similar to those found in the GPT-2 and GPT-3 papers.
 
## Installation

To set up the environment and install dependencies, run:

```bash
pip install -r requirements.txt
```
## Usage
Training
To train the model, run:
```python train.py```

#### Generating Text
To generate text, run:
```
# Example usage:
vocab_size = 100
n_embd = 64
block_size = 64
model = BigramLanguageModel(vocab_size, n_embd, block_size)
context = torch.tensor([[1, 2, 3]], dtype=torch.long)
generated_text = model.generate(context, max_new_tokens=50)
print(generated_text)
```
#### Hyperparameters
The following hyperparameters are used in the model:

batch_size: 16
block_size: 64
max_iters: 5000
eval_interval: 500
learning_rate: 3e-4
device: 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters: 100
n_embd: 64
n_head: 4
n_layer: 4
dropout: 0.2

#### Dataset
Tiny Shakespeare: Accumulation of Shakespeare books in a condensed version.

#### Results
Share some results or examples of the model's output.
https://github.com/user-attachments/assets/61f44c5d-a181-43cf-8251-77b11cb03089![Screenshot 2025-02-25 114354](https://github.com/user-attachments/assets/bd3d6f37-f169-4c4a-a64f-10e9d014cd4a)

import torch
import torch.nn as nn
from torch.nn import functional as F 

# Hyperparameters
batch_size = 32  # How many independent sequences will we process in parallel
block_size = 8   # What is the max context length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd=32

torch.manual_seed(1337)

# Load dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # String to list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # List of integers to string

# Train-test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)  # Move to device
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)  # Move to device
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)  # Move losses to device
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table= nn.Embedding(block_size,n_embd)
        self.lm_head =  nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        idx = idx.to(device)  # Ensure input tensor is on the correct device
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb= self.position_embedding_table(torch.arrange(T,device=device)) #(T,C)
        x= tok_emb+pos_emb #(B,T,C)
        logits=self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        idx = idx.to(device)  # Move idx to device
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Get last time step
            probs = F.softmax(logits, dim=-1)  # Convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append to sequence
        return idx

# Initialize model
model = BigramLanguageModel().to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    # Get training batch
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Move to device
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

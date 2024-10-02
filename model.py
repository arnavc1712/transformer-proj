import tiktoken
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import os
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class DataLoaderLite:
    def __init__(self, B, T, device="cpu", split="train"):
        """
            B: Batch size
            T: Block size
        """
        self.B = B
        self.T = T
        self.device = device
        data_folder = "datasets/shakespeare"
        if split == "train":
            filename = os.path.join(data_folder, "train.bin")
        else:
            filename = os.path.join(data_folder, "val.bin")
        
        with open(filename, "r") as f:
            data = np.fromfile(f, dtype=np.uint16)
        self.tokens = torch.tensor(data).long()
        self.current_idx = 0
    
    def next_batch(self):
        B, T, = self.B, self.T
        
        buf = self.tokens[self.current_idx: self.current_idx + (B*T) + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B,T)

        
        self.current_idx += B*T
        if self.current_idx + B*T + 1 >= len(self.tokens):
            self.current_idx = 0
        return x.to(self.device), y.to(self.device)


@dataclass
class GPTConfig:
    n_heads: int = 12
    n_layers: int = 12
    n_embed: int = 768
    vocab_size: int = 50257 # differs from the original 50,257 for efficiency
    block_size: int = 1024


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.n_embed % config.n_heads == 0, "n_embed must be divisible by the number of heads"
        # self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        # self.head_dim = self.n_embed // self.n_heads

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias = True) # ine one matrix multiplication, we get weights for key, value and query vectors
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias = True)
        self.c_proj.RESIDUAL_SCALE_INIT = 1.0

        self.register_buffer("bias", torch.tril(torch.ones((config.block_size,config.block_size))
                                    .view(1,1,config.block_size, config.block_size)), persistent=True)
    
    def forward(self, x):
        # x -> B, T, C
        B, T, C = x.size()

        # first lets get the query, key and value vectors
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embed, dim=2) # each of size B, T, C

        # now we want to organize these vectors for multi head attention (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2) # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2) # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2) # (B, n_heads, T, head_dim)

        # lets calculate the scaled attention matrix
        attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # B, n_heads, T , T
        attn = attn.masked_fill(self.bias[:,:,:T, :T] == 0, float("-inf"))

        # get attn probs
        attn = F.softmax(attn, dim=-1)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed, bias=True)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed, bias=True)
        self.c_proj.RESIDUAL_SCALE_INIT = 1.0

    def forward(self, x):
        # x -> B, T, C
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # x -> (B, T, C)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # Both these weights are going to be tied at some point
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            print(module.weight.std(), module.weight.mean())
        # if isinstance(module, nn.Linear) and hasattr(module, "RESIDUAL_SCALE_INIT"):
        #     std = 0.02 * ((2.0 * config.n_layers) ** -0.5)
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        #     if module.bias is not None:
        #         torch.nn.init.zeros_(module.bias)
        # if isinstance(module, nn.Linear) and not hasattr(module, "RESIDUAL_SCALE_INIT"):
        #     print(module.weight.std(), module.weight.mean())
    
    def forward(self, idx, targets=None):
        # x -> B, T
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward, model block size is exhausted: {T} > {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # T, C
        tok_emb = self.transformer.wte(idx) # B, T, C

        x = tok_emb + pos_emb # the batch dimension for the pos_emb is broadcasted
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x) # B, T, vocab_size

        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
            return x, loss

        return x
    
    @classmethod
    def from_pretrained(cls, model_type: str):
        """ Loads the pretrained model using weights from huggingface """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}, f"Invalid model type: {model_type}"
        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2" : dict(n_layers=12, n_heads=12, n_embed=768, vocab_size=50257, block_size=1024),
            "gpt2-medium" : dict(n_layers=24, n_heads=16, n_embed=1024, vocab_size=50257, block_size=1024),
            "gpt2-large" : dict(n_layers=36, n_heads=16, n_embed=1280, vocab_size=50257, block_size=1024),
            "gpt2-xl" : dict(n_layers=48, n_heads=16, n_embed=1600, vocab_size=50257, block_size=1024)
        }

        print(f"Loading model: {model_type}")

        config = GPTConfig(**config_args[model_type])
        model = cls(config)
        sd = model.state_dict()
        sd_keys = list(sd.keys())
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # discard the attention bias keys

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_hf_keys = list(sd_hf.keys())
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.masked_bias")]
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.bias")]
        weights_to_transpose = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys) == len(sd_hf_keys), f"Number of keys do not match: {len(sd_keys)} != {len(sd_hf_keys)}"
        for k in sd_hf_keys:
            if any(k.endswith(w) for w in weights_to_transpose):
                # these are Conv1D weights that need to be transposed to align with our model since we use Linear layers
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


def get_best_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_lr(epoch):
    if epoch < 10:
        return 3e-4
    elif epoch < 15:
        return 6e-5
    elif epoch < 20:
        return 3e-5
    else:
        return 1e-5

with open("datasets/shakespeare/input.txt", "r") as f:
    text = f.read()

device = get_best_available_device()
torch.manual_seed(42)
print(f"Using device: {device}")
B, T = 64, 32
dataloader = DataLoaderLite(B, T, device=device, split="train")
val_dataloader = DataLoaderLite(B, T, device=device, split="val")
config = GPTConfig()
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model = model.to(device)


# Use the function to get the best available device
device = get_best_available_device()
print(f"Using device: {device}")

enc = tiktoken.get_encoding("gpt2")

num_epochs = 20
num_steps = len(dataloader.tokens) // (B*T)

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    for step in tqdm(range(num_steps), desc="Training Steps"):
        x, y = dataloader.next_batch()
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # validation
    model.eval()
    val_loss = 0
    val_steps = len(val_dataloader.tokens) // (B*T)
    with torch.no_grad():
        for _ in tqdm(range(val_steps), desc="Validation Steps"):
            x, y = val_dataloader.next_batch()
            logits, loss = model(x, y)
            val_loss += loss.item()
        print(f"Epoch {epoch}, Validation Loss: {val_loss/val_steps}")

    model.train()

# Save the model
torch.save(model.state_dict(), "model.pth")

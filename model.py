import os
import math
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import inspect
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def load_tokens(filename):
    tokens = np.fromfile(filename, dtype=np.uint16).astype(np.int32)
    ptt = torch.tensor(tokens, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, num_processes=1, process_rank=0, device="cpu", split="train"):
        """
            B: Batch size
            T: Block size
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device = device
        assert split in {"train", "val"}, "split must be either train or val"
        data_folder = "datasets/edu_fineweb10B/shards"
        shards = os.listdir(data_folder)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_folder, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split: {split}"

        if master_process:
            print(f"Found {len(shards)} shards for split: {split}")
        
        # initialize the state to start with shard 0
        self.current_shard = 0
        self.tokens = load_tokens(shards[self.current_shard])
        self.current_idx = self.B * self.T * self.process_rank # this is the current index in a specific shard
    
    def reset(self):
        # reset the state to start from the beginning
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_idx = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T, = self.B, self.T
        
        buf = self.tokens[self.current_idx: self.current_idx + (B*T) + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B,T) # targets
        
        self.current_idx += B*T*self.num_processes
        if self.current_idx + B*T*self.num_processes + 1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_idx = self.B * self.T * self.process_rank
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
        # attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # B, n_heads, T , T
        # attn = attn.masked_fill(self.bias[:,:,:T, :T] == 0, float("-inf"))

        # # get attn probs
        # attn = F.softmax(attn, dim=-1)

        # y = attn @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
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
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RESIDUAL_SCALE_INIT"):
                std *=  ((2.0 * config.n_layers) ** -0.5)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
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
    
    def configure_optimizer(self, weight_decay: float, learning_rate: float, device):

        param_dict = {k: v for k, v in self.named_parameters() if v.requires_grad}

        # create the optimizer groups, we do not want to decay 1D weights and biases
        decay_params = [v for k, v in param_dict.items() if v.dim() >= 2]
        no_decay_params = [v for k, v in param_dict.items() if v.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decayed params: {len(decay_params)} with {num_decay_params:,} parameters")
        print(f"num non-decayed params: {len(no_decay_params)} with {num_no_decay_params:,} parameters")

        # use fused adam if possible
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters and device != "mps"
        print(f"Using {'fused' if fused_available else 'torch'} AdamW optimizer")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused_available)
        return optimizer


def get_best_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # since the warmup is for 375M tokens
max_steps = 19073 # since the batch size is 5M tokens, and total num of tokens is 10B

def get_lr(step):
    """
        Cosine learning rate schedule
    """
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    

# lets set up Distributed Data Parallel
# the torchrun command sets the environment variables RANK, LOCAL_RANK and WORLD_SIZE

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"]) # rank of the process in the network (applicable if training across multiple nodes)
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # rank of the process wrt to the node
    ddp_world_size = int(os.environ["WORLD_SIZE"]) # total number of processes running, it helps the master to wait for all nodes to complete an op
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = get_best_available_device()

torch.set_float32_matmul_precision("high")
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 2**19 # ~0.5M tokens
B = 32 # micro batch size
T = 1024

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total desired batch size: {total_batch_size:,}")
    print(f"Gradient accumulation steps per device: {grad_accum_steps}")

train_dataloader = DataLoaderLite(B, T, num_processes=ddp_world_size, process_rank=ddp_rank, device=device, split="train")
val_dataloader = DataLoaderLite(B, T, num_processes=ddp_world_size, process_rank=ddp_rank, device=device, split="val")

use_compile = True
config = GPTConfig(vocab_size=50304)
model = GPT(config)
model = model.to(device)
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)


enc = tiktoken.get_encoding("gpt2")

for step in range(max_steps):
    last_step = step == max_steps - 1
    # run validation every 100 steps
    if (step % 250 == 0) or last_step:
        model.eval()
        val_dataloader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_dataloader.next_batch()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation Loss: {val_loss_accum.item():.4f}")
    
    # once in a while we also want to sample
    if ((step > 0 and step % 250 == 0) or last_step):
        model.eval()
        num_returned_sequences = 4
        max_length = 32
        tokens = enc.encode_ordinary("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_returned_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        # Create a tqdm progress bar
        pbar = tqdm(total=max_length - xgen.size(1), 
                    desc=f"Sampling (Rank {ddp_rank})", 
                    position=ddp_rank, 
                    leave=False)

        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                logits = model(xgen)
                logits = logits[:, -1, :] # keep only the last token, B x vocab_size
                probs = F.softmax(logits, dim=-1)

                # we now do topk sampling to get the token
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)

            pbar.update(1)
        
        pbar.close()

        # decode the tokens
        for i in range(num_returned_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"Rank: {ddp_rank} | Sample {i+1}: {decoded}")

    model.train()
    t0 = time.time()
    optimizer.zero_grad()

    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_dataloader.next_batch()
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # this is because the default behavior of DDP is to call allreduce on every loss backward but this does respect gradient accumulation
        if ddp:
            model.require_backward_grad_sync = (micro_step == (grad_accum_steps-1))
        loss.backward()
    
    # now the loss_accum which is printed out will be local to each device, in order to get the average across devices we do an all reduce
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)


    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    dt = t1 - t0 # time diff in seconds
    total_tokens = train_dataloader.B * train_dataloader.T * grad_accum_steps * ddp_world_size
    token_per_sec = total_tokens / dt
    if master_process:
        print(f"Step {step:4d} | lr: {lr:.4e} | Loss: {loss_accum.item():.6f} | tokens/sec: {token_per_sec:,.0f} | norm: {norm:.4f} | time: {(dt*1000):.2f}ms")

# destroy process groups when using ddp
if ddp:
    destroy_process_group()
# Save the model
torch.save(model.state_dict(), "model.pth")

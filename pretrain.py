import os
import time
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from model import GPT, GPTConfig
from train_utils import CosineAnnealingLR, get_best_available_device
from data_utils import load_tokens
import tiktoken

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


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # since the warmup is for 375M tokens
max_steps = 19073 # since the batch size is 5M tokens, and total num of tokens is 10B

lr_scheduler = CosineAnnealingLR(warmup_steps=warmup_steps, max_steps=max_steps, max_lr=max_lr, min_lr=min_lr)


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

use_compile = False
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

    lr = lr_scheduler.get_lr(step)
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

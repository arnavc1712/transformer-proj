import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from functools import partial

def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n").format_map(row)


def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n").format_map(row)

def create_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)

def tokenize(doc, enc, eot):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    prompt = create_prompt(doc)
    output = doc["output"]
    full_text = prompt + output
    tokens.extend(enc.encode_ordinary(full_text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "tokens must be uint16"
    tokens_np = tokens_np.astype(np.uint16)
    return tokens_np


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    ds = load_dataset("yahma/alpaca-cleaned")

    total_tokens = 0
    for doc in ds["train"]:
        total_tokens += len(tokenize(doc, enc, eot))
    
    test_size, val_size = int(0.1 * total_tokens), int(0.1 * total_tokens)
    train_size = total_tokens - test_size - val_size

    print(f"total tokens: {total_tokens:,}")
    print(f"train has {train_size:,} tokens")
    print(f"val has {val_size:,} tokens")
    print(f"test has {test_size:,} tokens")

    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "splits")

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    nprocs = max(1, os.cpu_count() // 2)
    partial_tokenize = partial(tokenize, enc=enc, eot=eot)
    
    with mp.Pool(nprocs) as pool:
        
        token_count=0
        progress_bar = tqdm(total=total_tokens, desc="Processing tokens")

        # intialize lists to collect tokens for each split
        test_tokens, val_tokens, train_tokens = [], [], []

        for tokens in pool.imap(partial_tokenize, ds["train"], chunksize=16):
            curr_num_tokens = len(tokens)
            if token_count + curr_num_tokens < test_size:
                test_tokens.extend(tokens)
            elif token_count + curr_num_tokens < test_size + val_size:
                val_tokens.extend(tokens)
            else:
                train_tokens.extend(tokens)
            
            token_count += curr_num_tokens
            progress_bar.update(curr_num_tokens)
        
        progress_bar.close()

        # write to bin files
        test_tokens = np.array(test_tokens, dtype=np.uint16)
        val_tokens = np.array(val_tokens, dtype=np.uint16)
        train_tokens = np.array(train_tokens, dtype=np.uint16)

        test_tokens.tofile(os.path.join(DATA_CACHE_DIR, "test.bin"))
        val_tokens.tofile(os.path.join(DATA_CACHE_DIR, "val.bin"))
        train_tokens.tofile(os.path.join(DATA_CACHE_DIR, "train.bin"))

        print(f"test.bin has {len(test_tokens):,} tokens")
        print(f"val.bin has {len(val_tokens):,} tokens")
        print(f"train.bin has {len(train_tokens):,} tokens")









    


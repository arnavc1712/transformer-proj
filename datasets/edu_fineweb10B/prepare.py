"""
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Loads the FineWeb-Edu dataset.
Creates shards and saves them under the directory specified by the `shards_dir` argument.
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from functools import partial

def tokenize(doc, enc, eot):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "tokens must be uint16"
    tokens_np = tokens_np.astype(np.uint16)
    return tokens_np

def write_datafile(filename, tokens_np, DATA_CACHE_DIR):
    """Writes a numpy array of uint16 tokens to a file"""
    fullpath = os.path.join(DATA_CACHE_DIR, filename)
    tokens_np.tofile(fullpath)
    print(f"{fullpath} has {len(tokens_np):,} tokens")


def main():
    remote_name = "sample-10BT"
    shards_dir = "shards"
    shard_size = int(1e8)  # 100M tokens per shard

    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), shards_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    print(f"DATA_CACHE_DIR: {DATA_CACHE_DIR}")

    # download the FineWeb-Edu dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]  # end of text token used to separate documents during pre-training

    # tokenize all documents and write to shards each with shard_size tokens
    nprocs = max(1, os.cpu_count() // 2)
    partial_tokenize = partial(tokenize, enc=enc, eot=eot)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(partial_tokenize, ds, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)

                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                fn = f"edufineweb_{split}_{shard_index:06d}.bin"

                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count: token_count+remainder] = tokens[:remainder]
                write_datafile(fn, all_tokens_np, DATA_CACHE_DIR)

                shard_index += 1
                progress_bar = None

                # populate the next shard with leftovers from the current document
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

    # write the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        fn = f"edufineweb_{split}_{shard_index:06d}.bin"
        write_datafile(fn, all_tokens_np[:token_count], DATA_CACHE_DIR)


if __name__ == "__main__":
    mp.freeze_support()
    main()




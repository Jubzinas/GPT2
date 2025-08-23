"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer (robust way to get the <|endoftext|> id)
enc = tiktoken.get_encoding("gpt2")
EOT = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [EOT]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint32)  # temp dtype for the assertion
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    # ensure .npy extension for clarity
    if not filename.endswith(".npy"):
        filename = filename + ".npy"
    np.save(filename, tokens_np)

def build_shards(dataset, nprocs):
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            start = 0
            # fill shards; this handles the (unlikely) case of a very long single doc too
            while start < len(tokens):
                space = shard_size - token_count
                take = min(space, len(tokens) - start)
                all_tokens_np[token_count:token_count + take] = tokens[start:start + take]
                token_count += take
                start += take

                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(take)

                if token_count == shard_size:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                    write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    token_count = 0
                    progress_bar.close()
                    progress_bar = None

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
    if progress_bar is not None:
        progress_bar.close()

def main():
    # download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    nprocs = max(1, os.cpu_count() // 2)
    build_shards(fw, nprocs)

if __name__ == "__main__":
    # On macOS/Windows the default is 'spawn'; the guard above is required.
    # (Optional) If you prefer, on POSIX you can switch to 'fork' for speed:
    # mp.set_start_method("fork", force=True)
    mp.set_start_method("spawn", force=True)
    main()

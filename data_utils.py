import numpy as np
import torch

def load_tokens(filename):
    tokens = np.fromfile(filename, dtype=np.uint16).astype(np.int32)
    ptt = torch.tensor(tokens, dtype=torch.long)
    return ptt
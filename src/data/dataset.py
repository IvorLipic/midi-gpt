import torch
import numpy as np
from pathlib import Path

class MidiDataset(torch.utils.data.Dataset):
    def __init__(self, folder, seq_len=None, mode="remi", max_seq_len=512):
        self.files = sorted(Path(folder).glob("*.npz"))
        self.mode = mode
        self.max_seq_len = max_seq_len

        if seq_len is None:
            seq_len = get_max_seq_len(folder, mode, max_seq_len)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        tokens = data["tokens"]
        tokens = torch.from_numpy(tokens).long()

        if tokens.shape[0] > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        if tokens.shape[0] < self.seq_len:
            padding_size = self.seq_len - tokens.shape[0]
            if tokens.dim() == 1:
                tokens = torch.nn.functional.pad(tokens, (0, padding_size), value=0)
            else:
                tokens = torch.nn.functional.pad(tokens, (0, 0, 0, padding_size), value=0)

        return tokens

def get_max_seq_len(folder, mode="remi", max_seq_len=None):
    files = sorted(Path(folder).glob("*.npz"))
    max_len = 0
    for f in files:
        data = np.load(f)
        length = data["tokens"].shape[0]
        if length > max_len:
            max_len = length
    if max_seq_len is not None and max_len > max_seq_len:
        max_len = max_seq_len
    print(f"Max sequence length in {folder}: {max_len} ({len(files)} samples)")
    return max_len

import torch
import numpy as np
from pathlib import Path

class MidiDataset(torch.utils.data.Dataset):
    def __init__(self, folder, max_seq_len=None):
        self.files = sorted(Path(folder).glob("*.npz"))

        if max_seq_len is None:
            max_seq_len = get_max_seq_len(folder)

        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        tokens = data["tokens"]
        tokens = torch.from_numpy(tokens).long()

        if tokens.shape[0] > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        if tokens.shape[0] < self.max_seq_len:
            padding_size = self.max_seq_len - tokens.shape[0]
            if tokens.dim() == 1:
                tokens = torch.nn.functional.pad(tokens, (0, padding_size), value=0)
            else:
                tokens = torch.nn.functional.pad(tokens, (0, 0, 0, padding_size), value=0)

        return tokens

def get_max_seq_len(folder):
    files = sorted(Path(folder).glob("*.npz"))
    max_len = 0
    for f in files:
        data = np.load(f)
        length = data["tokens"].shape[0]
        if length > max_len:
            max_len = length
    print(f"Max sequence length in {folder}: {max_len} ({len(files)} samples)")
    return max_len

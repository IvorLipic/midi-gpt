import torch
import torch.nn.functional as F
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
        tokens = torch.from_numpy(data["tokens"]).long()

        if tokens.shape[0] > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        return tokens


def nested_collate(batch):
    return batch


def pad_sequence(seq, max_len, pad_id=0):
    pad_size = max_len - seq.shape[0]
    return F.pad(seq, (0, pad_size), value=pad_id)


def collate_pad_to_longest(batch, pad_id=0):
    max_len = max(t.shape[0] - 1 for t in batch)
    inputs, targets = [], []
    for t in batch:
        inputs.append(pad_sequence(t[:-1], max_len, pad_id))
        targets.append(pad_sequence(t[1:], max_len, pad_id))
    return {
        "inputs": torch.stack(inputs),
        "targets": torch.stack(targets),
    }


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

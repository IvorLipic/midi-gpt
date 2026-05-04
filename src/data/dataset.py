import torch
import numpy as np
from pathlib import Path
import random

class MidiDataset(torch.utils.data.Dataset):
    def __init__(self, folder, seq_len=512, mode="remi"):
        self.files = sorted(Path(folder).glob("*.npz"))
        self.seq_len = seq_len
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        tokens = data["tokens"]
        tokens = torch.from_numpy(tokens).long()

        # REMI is (L,), Octuple is (L, F)
        curr_len = tokens.shape[0]

        # ---- HANDLE LONG SEQUENCES (RANDOM CROP) ---- CHANGE THIS!!!!
        if curr_len > self.seq_len:
            start = random.randint(0, curr_len - self.seq_len)
            tokens = tokens[start : start + self.seq_len]

        # ---- HANDLE SHORT SEQUENCES (PAD) ----
        elif curr_len < self.seq_len:
            padding_size = self.seq_len - curr_len
            # F.pad(input, (left, right), value)
            # Assuming 0 is your [PAD] token; replace with your actual pad_id
            # F.pad handles multidimensional tensors correctly if we specify the last dim
            # For (L, F), padding (0, 0, 0, pad_size) pads the length dim
            if tokens.dim() == 1:
                tokens = torch.nn.functional.pad(tokens, (0, padding_size), value=0)
            else:
                # Pad the first dimension (Length), leave the second (Fields) alone
                tokens = torch.nn.functional.pad(tokens, (0, 0, 0, padding_size), value=0)

        return tokens
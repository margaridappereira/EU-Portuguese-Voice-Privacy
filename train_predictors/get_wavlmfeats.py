import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import textgrid
import os
from wavlm_loader import load_wavlm

stride_sec = 0.02
results = []

device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

dl = DataLoader(
        SpeechDataset(data_dir, 16000, format=f_format),
        batch_size=8,
        num_workers=10 if device == "cuda" else 0,
        collate_fn=lambda x: pad_sequence([y[0] for y in x], batch_first=True),
    )

for batch in tqdm(dl):
        with torch.no_grad():
            batch = batch.to(device)
            feats = wavlm.extract_features(batch)[0]

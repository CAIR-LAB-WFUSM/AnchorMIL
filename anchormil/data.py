"""Bag dataset over pre-extracted patch features (.pt tensors of shape (n_patches, D))."""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class FeatureBagDataset(Dataset):
    """CSV columns: slide_id, patient_id, odx, split. Features at <feat_dir>/<slide_id>.pt."""

    def __init__(self, csv_path: str, feat_dir: str, split: str, max_patches: int = 0):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.feat_dir, self.max_patches = feat_dir, max_patches

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        f = torch.load(os.path.join(self.feat_dir, f"{row.slide_id}.pt"), map_location="cpu").float()
        if self.max_patches and f.shape[0] > self.max_patches:
            f = f[torch.randperm(f.shape[0])[: self.max_patches]]
        return f, torch.tensor(float(row.odx))

    def risk_sampler(self, tau: float = 25.0) -> WeightedRandomSampler:
        """Oversample high-risk slides to balance classes."""
        y = (self.df["odx"].values > tau).astype(int)
        w = 1.0 / np.maximum(np.bincount(y, minlength=2)[y], 1)
        return WeightedRandomSampler(w, num_samples=len(y), replacement=True)


def collate_bags(batch):
    """Sort by patch count, pad to the longest bag; returns (feats, pad_mask, y)."""
    batch = sorted(batch, key=lambda b: b[0].shape[0], reverse=True)
    n, D = batch[0][0].shape
    feats = torch.zeros(len(batch), n, D)
    mask = torch.ones(len(batch), n, dtype=torch.bool)
    for i, (f, _) in enumerate(batch):
        feats[i, : f.shape[0]] = f
        mask[i, : f.shape[0]] = False
    return feats, mask, torch.stack([b[1] for b in batch])

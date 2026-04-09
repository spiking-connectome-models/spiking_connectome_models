"""Dataset classes for connectome-constrained olfactory model training.

Uses ORN response data from Kreher et al. (2008) larval electrophysiology.
Contains 28 odors x 21 OR types.

This is a self-contained copy of the dataset utilities needed by the paper
package. The canonical data lives in connectome_models/data/kreher2008/.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List


class RepeatedOdorDataset(Dataset):
    """Dataset that repeats each odor with independent noise samples.

    Each sample is (noisy_or_pattern, odor_index). Noise is drawn fresh
    on every __getitem__ call, so different epochs see different noise.
    """

    def __init__(self, orn_responses: torch.Tensor, odor_names: List[str],
                 repeats_per_odor: int = 10, noise_std: float = 0.1,
                 noise_type: str = 'additive'):
        self.base_responses = orn_responses.float()
        self.odor_names = odor_names
        self.repeats = repeats_per_odor
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.n_odors = len(odor_names)

    def __len__(self) -> int:
        return self.n_odors * self.repeats

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        odor_idx = idx // self.repeats
        pattern = self.base_responses[odor_idx].clone()
        if self.noise_std > 0:
            noise = torch.randn_like(pattern)
            if self.noise_type == 'multiplicative':
                # Multiplicative noise: scales with signal strength (biologically realistic)
                # Each OR channel gets independent noise proportional to its response
                pattern = pattern * (1.0 + noise * self.noise_std)
            else:
                # Additive noise: constant magnitude regardless of signal
                pattern = pattern + noise * self.noise_std
            # Clamp to non-negative (firing rates can't be negative)
            pattern = torch.clamp(pattern, min=0)
        return pattern, odor_idx


def load_kreher2008_all_odors(
    data_dir: Path,
    train_repeats: int = 10,
    test_repeats: int = 5,
    noise_std: float = 0.1,
    noise_type: str = 'additive',
) -> Tuple[Dataset, Dataset, List[str]]:
    """Load Kreher 2008 data with ALL 28 odors for both train and test.

    Train and test sets differ only in their noise draws.

    Args:
        data_dir: Path to the directory containing kreher2008/ subfolder
        train_repeats: Noise repeats per odor for training
        test_repeats: Noise repeats per odor for testing
        noise_std: Noise standard deviation
        noise_type: 'additive' or 'multiplicative'

    Returns:
        (train_dataset, test_dataset, odor_names)
    """
    kreher_dir = Path(data_dir) / 'kreher2008'
    csv_path = kreher_dir / 'orn_responses_normalized.csv'
    pt_path = kreher_dir / 'orn_responses_normalized.pt'

    if pt_path.exists():
        or_responses = torch.load(pt_path, weights_only=True)
        df = pd.read_csv(csv_path, index_col=0)
        odor_names = df.index.tolist()
    else:
        df = pd.read_csv(csv_path, index_col=0)
        or_responses = torch.from_numpy(df.values).float()
        odor_names = df.index.tolist()

    print(f'Kreher 2008 data: {len(odor_names)} odors x {or_responses.shape[1]} OR types')

    train_dataset = RepeatedOdorDataset(
        or_responses, odor_names,
        repeats_per_odor=train_repeats, noise_std=noise_std, noise_type=noise_type)
    test_dataset = RepeatedOdorDataset(
        or_responses, odor_names,
        repeats_per_odor=test_repeats, noise_std=noise_std, noise_type=noise_type)

    print(f'Train: {len(train_dataset)} samples ({len(odor_names)} odors x {train_repeats} repeats)')
    print(f'Test: {len(test_dataset)} samples ({len(odor_names)} odors x {test_repeats} repeats)')
    return train_dataset, test_dataset, odor_names


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

"""Dataset-balanced sampler over a ConcatDataset (Eq. 4 of the paper)."""
import torch
from torch.utils.data import WeightedRandomSampler


def build_task_balanced_sampler(named_datasets, alpha=0.5, num_samples=None, verbose=True):
    """Draw each dataset with mass m_d = |D_d|^alpha / sum_d' |D_d'|^alpha, uniformly inside it.

    alpha = 1 is equivalent to shuffling the concatenation, alpha = 0 gives every dataset
    the same mass, alpha = 0.5 (paper) is in between.

    Args:
        named_datasets: [(name, dataset), ...] in EXACTLY the order of the ConcatDataset,
            so that each weight lands on the right volume.
        num_samples: draws per epoch (default: total number of volumes).
    """
    names = [n for n, _ in named_datasets]
    lengths = [len(d) for _, d in named_datasets]
    total_len = sum(lengths)

    raw = {n: float(L) ** alpha for n, L in zip(names, lengths)}
    norm = sum(raw.values())
    mass = {n: raw[n] / norm for n in names}

    weights = []
    for name, L in zip(names, lengths):
        weights.extend([mass[name] / L if L > 0 else 0.0] * L)
    assert len(weights) == total_len, f"sampler ({len(weights)}) vs ConcatDataset ({total_len}) misaligned"

    num_samples = num_samples or total_len
    if verbose:
        print(f"=== dataset sampler (alpha={alpha}) ===")
        for name, L in zip(names, lengths):
            print(f"  {name:12s}: {L:5d} volumes | {100 * L / total_len:5.1f}% -> {100 * mass[name]:5.1f}% of draws")
        print(f"  {num_samples} draws per epoch, with replacement")

    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=num_samples, replacement=True)

"""Collate function that routes each sample to the sub-batch of its task."""
import torch
from monai.data.utils import list_data_collate

TASKS = ("classification", "cls_pseudo", "seg_orig", "seg_multi", "seg_inhouse", "seg_instance")


def flatten(batch):
    """Patch-based transforms return a list of patches per volume: flatten them."""
    for item in batch:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def _as_tensor(x):
    """Plain tensor without the MONAI metadata (MetaTensor -> Tensor)."""
    if hasattr(x, "as_tensor"):
        return x.as_tensor()
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)


def multitask_collate_fn(batch):
    """Return {task: collated sub-batch or None}; a mini-batch can mix several tasks."""
    groups = {task: [] for task in TASKS}
    for item in flatten(batch):
        task = item["task"]
        if task not in groups:
            raise ValueError(f"unknown task: {task}")
        clean = {"task": task, "image": _as_tensor(item["image"])}
        if item.get("label") is not None:
            clean["label"] = _as_tensor(item["label"])
        for key in ("class_label", "valid_channels"):
            if key in item:
                clean[key] = _as_tensor(item[key])
        groups[task].append(clean)
    return {task: list_data_collate(items) if items else None for task, items in groups.items()}

import json

from config import PathConfig
from data_io import list_paired_ids


def load_train_val_ids(cfg: PathConfig):
    """Load GenCAD's official train/validation id split, restricted to ids with a rendered image."""
    with open(cfg.split_path, "r") as f:
        raw_splits = json.load(f)

    paired_ids = set(list_paired_ids(cfg))
    train_ids = [i for i in raw_splits["train"] if i in paired_ids]
    val_ids = [i for i in raw_splits["validation"] if i in paired_ids]
    return train_ids, val_ids

import random

import torch
from torch.utils.data import Dataset

from cad_format import DEFAULT_MAX_LEN, pad_vec, split_command_args
from config import PathConfig
from data_io import list_paired_ids, list_views, load_cad_vector, load_image


class CADImagePairDataset(Dataset):
    """Pairs a CAD command sequence with one rendered view.

    By default a random view is chosen each call (useful for training augmentation).
    With deterministic=True, views[view_index] is always used (clamped to the last
    available view for any sample with fewer views than that) — so retrieval evaluation
    sees the same image for a given id on every run. view_index=0 (the default) is each
    model's canonical/first rendered view; a non-zero view_index queries with a different
    rendered angle of the exact same CAD model, e.g. to compare exact-view vs
    different-view retrieval accuracy (see evaluate.py).

    An entry in `ids` may also be an explicit (sample_id, view) pair instead of a bare id,
    to force one specific, precomputed view rather than deriving it from view_index/
    deterministic — used when the caller needs a different, reproducible view per sample
    (e.g. a random-but-fixed choice among each model's non-canonical views).
    """

    def __init__(
        self, cfg: PathConfig, ids=None, image_transform=None, max_len=DEFAULT_MAX_LEN,
        deterministic=False, view_index=0,
    ):
        self.cfg = cfg
        self.ids = ids if ids is not None else list_paired_ids(cfg)
        self.image_transform = image_transform
        self.max_len = max_len
        self.deterministic = deterministic
        self.view_index = view_index

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        entry = self.ids[index]
        sample_id, forced_view = entry if isinstance(entry, tuple) else (entry, None)

        views = list_views(sample_id, self.cfg)
        if forced_view is not None:
            view = forced_view
        elif self.deterministic:
            view = views[min(self.view_index, len(views) - 1)]
        else:
            view = random.choice(views)
        image = load_image(sample_id, view, self.cfg)
        if self.image_transform is not None:
            image = self.image_transform(image)

        vec = pad_vec(load_cad_vector(sample_id, self.cfg), max_len=self.max_len)
        command, args = split_command_args(vec)

        return {
            "image": image,
            "command": torch.as_tensor(command, dtype=torch.long),
            "args": torch.as_tensor(args, dtype=torch.long),
            # Raw (seq_len, 1+N_ARGS) vector, unsplit — only consumed by
            # cad_similarity_matching_loss (see losses.py), which needs the original command
            # vector to compute pairwise CAD similarity (cad_vec_similarity.cad_distance
            # operates on this exact layout). Cheap to always include; ignored otherwise.
            "raw_vec": torch.as_tensor(vec, dtype=torch.long),
            "id": sample_id,
        }

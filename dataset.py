import random

import torch
from torch.utils.data import Dataset

from cad_format import DEFAULT_MAX_LEN, pad_vec, split_command_args
from config import RANDOM_SEED, PathConfig
from data_io import list_paired_ids, list_views, load_cad_vector, load_image


class CADImagePairDataset(Dataset):
    """Pairs a CAD command sequence with one or more rendered views.

    By default a random view is chosen each call (useful for training augmentation).
    With deterministic=True the first available view(s) are always used, so retrieval
    evaluation sees the same image(s) for a given id on every run.

    With views_per_sample > 1, "image" is a stack of that many views instead of a
    single image (shape (views_per_sample, C, H, W) rather than (C, H, W)) — for
    callers that pool several views into one embedding (see CADClipModel.encode_image_multiview).

    An entry in `ids` may also be an explicit (sample_id, view) pair instead of a bare
    id, to force one specific view rather than sampling — used when each sampled view
    of a model needs to become its own independent row (see evaluate.py's multi-view
    inference sweep).
    """

    def __init__(
        self, cfg: PathConfig, ids=None, image_transform=None, max_len=DEFAULT_MAX_LEN,
        deterministic=False, views_per_sample=1, seed=RANDOM_SEED,
    ):
        self.cfg = cfg
        self.ids = ids if ids is not None else list_paired_ids(cfg)
        self.image_transform = image_transform
        self.max_len = max_len
        self.deterministic = deterministic
        self.views_per_sample = views_per_sample
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.ids)

    def _load_views(self, sample_id, views):
        images = [load_image(sample_id, view, self.cfg) for view in views]
        if self.image_transform is not None:
            images = [self.image_transform(image) for image in images]
        return torch.stack(images)

    def __getitem__(self, index):
        entry = self.ids[index]
        sample_id, forced_view = entry if isinstance(entry, tuple) else (entry, None)

        if forced_view is not None:
            image = self._load_views(sample_id, [forced_view])[0]
        elif self.views_per_sample > 1:
            views = list_views(sample_id, self.cfg)
            k = min(self.views_per_sample, len(views))
            chosen_views = views[:k] if self.deterministic else self.rng.sample(views, k)
            image = self._load_views(sample_id, chosen_views)
        else:
            views = list_views(sample_id, self.cfg)
            view = views[0] if self.deterministic else self.rng.choice(views)
            image = self._load_views(sample_id, [view])[0]

        vec = pad_vec(load_cad_vector(sample_id, self.cfg), max_len=self.max_len)
        command, args = split_command_args(vec)

        return {
            "image": image,
            "command": torch.as_tensor(command, dtype=torch.long),
            "args": torch.as_tensor(args, dtype=torch.long),
            "id": sample_id,
        }

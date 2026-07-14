import random

import torch
from torch.utils.data import Dataset

from cad_format import DEFAULT_MAX_LEN, pad_vec, split_command_args
from config import PathConfig
from data_io import list_paired_ids, list_views, load_cad_vector, load_image


class CADImagePairDataset(Dataset):
    """Pairs a CAD command sequence with one rendered view.

    By default a random view is chosen each call (useful for training augmentation).
    With deterministic=True the first available view is always used, so retrieval
    evaluation sees the same image for a given id on every run.
    """

    def __init__(self, cfg: PathConfig, ids=None, image_transform=None, max_len=DEFAULT_MAX_LEN, deterministic=False):
        self.cfg = cfg
        self.ids = ids if ids is not None else list_paired_ids(cfg)
        self.image_transform = image_transform
        self.max_len = max_len
        self.deterministic = deterministic

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]

        views = list_views(sample_id, self.cfg)
        view = views[0] if self.deterministic else random.choice(views)
        image = load_image(sample_id, view, self.cfg)
        if self.image_transform is not None:
            image = self.image_transform(image)

        vec = pad_vec(load_cad_vector(sample_id, self.cfg), max_len=self.max_len)
        command, args = split_command_args(vec)

        return {
            "image": image,
            "command": torch.as_tensor(command, dtype=torch.long),
            "args": torch.as_tensor(args, dtype=torch.long),
            "id": sample_id,
        }

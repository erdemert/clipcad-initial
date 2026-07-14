import re

import h5py
import numpy as np
from PIL import Image

from config import PathConfig

_VIEW_RE_TEMPLATE = r"^{stem}_(\d+)\.png$"
_VIEW_SUFFIX_RE = re.compile(r"_\d+\.png$")


def list_ids(cfg: PathConfig) -> list[str]:
    """Return all sample ids as 'shard/stem', discovered from the cad_vec files on disk."""
    ids = []
    for shard_dir in sorted(cfg.cad_vec_root.iterdir()):
        if not shard_dir.is_dir():
            continue
        for h5_file in sorted(shard_dir.glob("*.h5")):
            ids.append(f"{shard_dir.name}/{h5_file.stem}")
    return ids


def list_paired_ids(cfg: PathConfig) -> list[str]:
    """Return ids that have both a cad_vec file and at least one rendered image.

    Some cad_vec samples have no matching renders (rendering-pipeline gaps upstream),
    so this filters those out rather than let dataset loading crash on a missing view.
    """
    ids = []
    for shard_dir in sorted(cfg.cad_vec_root.iterdir()):
        if not shard_dir.is_dir():
            continue
        img_shard_dir = cfg.images_root / shard_dir.name
        stems_with_views = (
            {_VIEW_SUFFIX_RE.sub("", f.name) for f in img_shard_dir.glob("*.png")}
            if img_shard_dir.is_dir() else set()
        )
        for h5_file in sorted(shard_dir.glob("*.h5")):
            if h5_file.stem in stems_with_views:
                ids.append(f"{shard_dir.name}/{h5_file.stem}")
    return ids


def load_cad_vector(sample_id: str, cfg: PathConfig) -> np.ndarray:
    """Load the raw (seq_len, 1 + N_ARGS) command vector for one sample."""
    h5_path = cfg.cad_vec_root / f"{sample_id}.h5"
    with h5py.File(h5_path, "r") as f:
        return f["vec"][:]


def list_views(sample_id: str, cfg: PathConfig) -> list[int]:
    """Return the view indices available for one sample's rendered images."""
    shard, stem = sample_id.split("/")
    shard_dir = cfg.images_root / shard
    pattern = re.compile(_VIEW_RE_TEMPLATE.format(stem=re.escape(stem)))
    views = [int(m.group(1)) for f in shard_dir.glob(f"{stem}_*.png") if (m := pattern.match(f.name))]
    return sorted(views)


def load_image(sample_id: str, view: int, cfg: PathConfig) -> Image.Image:
    """Load one rendered view (as a PIL image) for a sample."""
    shard, stem = sample_id.split("/")
    img_path = cfg.images_root / shard / f"{stem}_{view}.png"
    return Image.open(img_path).convert("RGB")

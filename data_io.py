import io

import h5py
import numpy as np
from PIL import Image

from config import PathConfig

# Per-process cache of open shard image stores: {(images_root, shard): (h5py.File, id_to_row)}.
# Populated lazily from __getitem__/list_views/load_image, never before a DataLoader worker
# fork, so each forked worker builds its own independent cache rather than sharing file handles.
_shard_cache = {}


def _decode_id(raw_id) -> str:
    return raw_id.decode() if isinstance(raw_id, bytes) else raw_id


def _rendered_ids(shard: str, cfg: PathConfig) -> set[str]:
    """Return the ids actually present in one shard's image h5, or {} if the shard has no renders.

    Reads and closes its own handle rather than going through _get_shard_store's cache, since
    this runs once in the main process (via list_paired_ids) before DataLoader workers fork —
    leaving an open h5py.File in that cache here would get inherited, unsafely shared, by every
    forked worker.
    """
    h5_path = cfg.images_root / f"{shard}.h5"
    if not h5_path.is_file():
        return set()
    with h5py.File(h5_path, "r") as f:
        return {_decode_id(raw_id) for raw_id in f["ids"][:]}


def _get_shard_store(shard: str, cfg: PathConfig):
    """Open (and cache) one shard's packed image store: an h5 file with 'ids' and 'pngs'
    datasets, where pngs[row, view] is a variable-length uint8 array holding one PNG's bytes.
    """
    key = (str(cfg.images_root), shard)
    store = _shard_cache.get(key)
    if store is not None:
        return store

    h5_file = h5py.File(cfg.images_root / f"{shard}.h5", "r")
    id_to_row = {_decode_id(raw_id): row for row, raw_id in enumerate(h5_file["ids"][:])}
    store = (h5_file, id_to_row)
    _shard_cache[key] = store
    return store


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
    """Return ids that have both a cad_vec file and a rendered-image row in the shard's h5.

    Some cad_vec samples have no matching renders (rendering-pipeline gaps upstream), so this
    filters those out rather than let dataset loading crash on a missing view. Checked against
    the image h5's own 'ids' dataset rather than the shard's _index.json manifest, since the
    two can drift out of sync (seen in practice: ids listed in the JSON but absent from the h5).
    """
    ids = []
    for shard_dir in sorted(cfg.cad_vec_root.iterdir()):
        if not shard_dir.is_dir():
            continue
        shard = shard_dir.name
        rendered_ids = _rendered_ids(shard, cfg)
        for h5_file in sorted(shard_dir.glob("*.h5")):
            sample_id = f"{shard}/{h5_file.stem}"
            if sample_id in rendered_ids:
                ids.append(sample_id)
    return ids


def load_cad_vector(sample_id: str, cfg: PathConfig) -> np.ndarray:
    """Load the raw (seq_len, 1 + N_ARGS) command vector for one sample."""
    h5_path = cfg.cad_vec_root / f"{sample_id}.h5"
    with h5py.File(h5_path, "r") as f:
        return f["vec"][:]


def list_views(sample_id: str, cfg: PathConfig) -> list[int]:
    """Return the view indices available for one sample's rendered images."""
    shard, _ = sample_id.split("/")
    h5_file, id_to_row = _get_shard_store(shard, cfg)
    if sample_id not in id_to_row:
        raise KeyError(f"{sample_id} not found in rendered-image shard {shard}")
    return list(range(h5_file["pngs"].shape[1]))


def load_image(sample_id: str, view: int, cfg: PathConfig) -> Image.Image:
    """Load one rendered view (as a PIL image) for a sample from its shard's packed store."""
    shard, _ = sample_id.split("/")
    h5_file, id_to_row = _get_shard_store(shard, cfg)
    row = id_to_row[sample_id]
    png_bytes = h5_file["pngs"][row, view].tobytes()
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")

import hashlib
import io
import itertools
import json
import logging
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from PIL import Image

from config import PACKAGE_ROOT, PathConfig

logger = logging.getLogger(__name__)

_PAIRED_IDS_CACHE_DIR = PACKAGE_ROOT / ".cache"

# Per-process, size-capped LRU cache of open shard image stores:
# {(images_root, shard): (h5py.File, id_to_row)}. Populated lazily from
# __getitem__/list_views/load_image, never before a DataLoader worker fork, so each forked
# worker builds its own independent cache rather than sharing file handles. Capped (rather
# than left to grow for the worker's whole persistent_workers=True lifetime) because with
# shuffled sampling a worker can touch most/all shards within the first few dozen steps —
# an uncapped cache means every one of those shards' open h5py.File handles + id_to_row
# dicts stays alive, per worker, for the entire run.
_SHARD_CACHE_MAX_SIZE = 32
_shard_cache = OrderedDict()


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
        _shard_cache.move_to_end(key)
        return store

    h5_file = h5py.File(cfg.images_root / f"{shard}.h5", "r")
    id_to_row = {_decode_id(raw_id): row for row, raw_id in enumerate(h5_file["ids"][:])}
    store = (h5_file, id_to_row)
    _shard_cache[key] = store
    if len(_shard_cache) > _SHARD_CACHE_MAX_SIZE:
        _, (evicted_file, _) = _shard_cache.popitem(last=False)
        evicted_file.close()
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


def _shard_paired_ids(shard_dir, cfg: PathConfig):
    """Worker for list_paired_ids: one shard's (paired_ids, has_any_renders_at_all)."""
    shard = shard_dir.name
    rendered_ids = _rendered_ids(shard, cfg)
    if not rendered_ids:
        return [], False
    paired = [
        sample_id
        for h5_file in sorted(shard_dir.glob("*.h5"))
        if (sample_id := f"{shard}/{h5_file.stem}") in rendered_ids
    ]
    return paired, True


def _paired_ids_cache_path(cfg: PathConfig):
    key = hashlib.sha256(f"{cfg.cad_vec_root}|{cfg.images_root}".encode()).hexdigest()[:16]
    return _PAIRED_IDS_CACHE_DIR / f"paired_ids_{key}.json"


def list_paired_ids(cfg: PathConfig) -> list[str]:
    """Return ids that have both a cad_vec file and a rendered-image row in the shard's h5.

    Some cad_vec samples have no matching renders (rendering-pipeline gaps upstream), so this
    filters those out rather than let dataset loading crash on a missing view. Checked against
    the image h5's own 'ids' dataset rather than the shard's _index.json manifest, since the
    two can drift out of sync (seen in practice: ids listed in the JSON but absent from the h5).

    cad_vec_root stores one .h5 file per sample (not per shard), so each shard's directory
    listing (`glob("*.h5")`) can itself be the expensive part on a network-mounted filesystem —
    not just opening the shard's image h5.

    Runs the per-shard scan through a process pool. Measured in production: switching from a
    thread pool to a process pool made no difference at all (~30s/shard either way, ~100 shards
    scanned serialized in wall-clock terms) — so this isn't a client-side GIL/lock contention
    issue, it's most likely the network-mounted storage itself (bandwidth or server-side
    request serialization) that caps this regardless of client concurrency. Since that means
    no amount of parallelism here fixes the wall-clock cost, and this scan is deterministic for
    a fixed dataset, the result is cached to disk (keyed by cad_vec_root/images_root) so repeat
    runs against the same data skip the ~1hr scan entirely. Delete the cache file under
    PACKAGE_ROOT/.cache/ (logged below) if the underlying rendered data changes.
    """
    cache_path = _paired_ids_cache_path(cfg)
    if cache_path.is_file():
        with open(cache_path) as f:
            ids = json.load(f)["ids"]
        logger.info("list_paired_ids: loaded %d cached ids from %s (delete this file to rescan)", len(ids), cache_path)
        return ids

    t0 = time.monotonic()
    shard_dirs = [d for d in sorted(cfg.cad_vec_root.iterdir()) if d.is_dir()]
    t1 = time.monotonic()
    with ProcessPoolExecutor(max_workers=min(32, len(shard_dirs)) or 1) as pool:
        results = list(pool.map(_shard_paired_ids, shard_dirs, itertools.repeat(cfg)))
    t2 = time.monotonic()
    logger.info(
        "list_paired_ids: %d shard dirs listed in %.1fs, %d shards scanned concurrently in %.1fs",
        len(shard_dirs), t1 - t0, len(shard_dirs), t2 - t1,
    )

    missing_shards = sum(1 for _, has_renders in results if not has_renders)
    if missing_shards:
        logger.warning(
            "%d of %d cad_vec shards have no rendered-image h5 at all under %s "
            "(rendering may still be in progress) — their ids are excluded entirely",
            missing_shards, len(shard_dirs), cfg.images_root,
        )

    ids = []
    for paired, _ in results:
        ids.extend(paired)

    _PAIRED_IDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"cad_vec_root": str(cfg.cad_vec_root), "images_root": str(cfg.images_root), "ids": ids}, f)
    logger.info("list_paired_ids: cached %d ids to %s", len(ids), cache_path)
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

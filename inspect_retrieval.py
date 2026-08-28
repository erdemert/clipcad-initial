import argparse
import os
import random
import shutil
from pathlib import Path

from cad_vec_similarity import DEFAULT_CONFIG, cad_similarity, load_cad_vector
from config import PathConfig
from data_io import list_views
from splits import load_test_ids, load_train_val_ids

# Dataset paths for this script, independent of config.py's PathConfig.default().
# Edit these directly, or override via the same env vars train.py/evaluate.py use.
CAD_VEC_ROOT = Path(os.environ.get("CAD_CLIPPER_CAD_VEC_ROOT", "/media/erdem/Backup/THESIS/cad_vec/data/cad_vec"))
IMAGES_ROOT = Path(os.environ.get("CAD_CLIPPER_IMAGES_ROOT", "/media/erdem/Backup/THESIS/images"))
SPLIT_PATH = Path(os.environ.get("CAD_CLIPPER_SPLIT_PATH", str(Path(__file__).resolve().parent / "splits" / "filtered_data.json")))


def image_path(sample_id, cfg):
    shard, stem = sample_id.split("/")
    view = list_views(sample_id, cfg)[0]
    return cfg.images_root / shard / f"{stem}_{view}.png"


def sanitize(sample_id):
    return sample_id.replace("/", "_")


def cad_vector_cache(cfg):
    """A load-once-per-id cache, since every query compares against the whole pool."""
    cache = {}

    def get(sample_id):
        if sample_id not in cache:
            cache[sample_id] = load_cad_vector(cfg.cad_vec_root / f"{sample_id}.h5")
        return cache[sample_id]

    return get


def main():
    parser = argparse.ArgumentParser(
        description="Nearest-neighbor CAD retrieval via cad_vec_similarity.cad_similarity, dumped as images",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Pool of CAD models to search within")
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("retrieval_samples"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cfg = PathConfig(cad_vec_root=CAD_VEC_ROOT, images_root=IMAGES_ROOT, split_path=SPLIT_PATH)
    _, val_ids = load_train_val_ids(cfg)
    pool_ids = val_ids if args.split == "val" else load_test_ids(cfg)

    get_vec = cad_vector_cache(cfg)
    query_ids = random.sample(pool_ids, args.n_queries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for query_num, query_id in enumerate(query_ids):
        query_vec = get_vec(query_id)
        scored = [
            (candidate_id, cad_similarity(query_vec, get_vec(candidate_id), DEFAULT_CONFIG))
            for candidate_id in pool_ids
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top_k = scored[:args.top_k]

        query_dir = args.output_dir / f"query_{query_num:02d}_{sanitize(query_id)}"
        query_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path(query_id, cfg), query_dir / "query.png")

        for rank, (candidate_id, score) in enumerate(top_k, start=1):
            marker = "_EXACT" if candidate_id == query_id else ""
            dest = query_dir / f"rank{rank:02d}_sim{score:.3f}{marker}_{sanitize(candidate_id)}.png"
            shutil.copy(image_path(candidate_id, cfg), dest)

        print(f"[{query_num}] query={query_id} -> {query_dir}")


if __name__ == "__main__":
    main()

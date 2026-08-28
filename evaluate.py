import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cad_vec_similarity import DEFAULT_CONFIG, cad_similarity, load_cad_vector
from config import RANDOM_SEED, CheckpointConfig, PathConfig, ViewConfig
from data_io import list_views
from dataset import CADImagePairDataset
from metrics import evaluate_recall
from model import CADClipModel
from splits import load_test_ids, load_train_val_ids

BATCH_SIZE = 512
# 0 (no worker subprocesses): sidesteps /dev/shm entirely, which some shared
# machines cap too small for multi-worker DataLoader tensor IPC. Evaluation is
# bottlenecked by the model's forward pass anyway, not data loading.
NUM_WORKERS = 0
EVAL_RUNS_DIR = Path("runs_eval")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("cad_clipper_eval")


def discover_epoch_checkpoints(checkpoint_dir: Path):
    """Return (epoch, path) pairs for every periodic 'epoch_XXX.pt' checkpoint, sorted by epoch."""
    checkpoints = [(int(path.stem.split("_")[1]), path) for path in checkpoint_dir.glob("epoch_*.pt")]
    return sorted(checkpoints)


def embed_split(model, cfg, ids, device):
    """Run the trained encoders once over a split, returning embeddings aligned with ids."""
    dataset = CADImagePairDataset(cfg, ids=ids, image_transform=model.preprocess, deterministic=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    image_embeds, cad_embeds, ordered_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            command = batch["command"].to(device, non_blocking=True)
            args = batch["args"].to(device, non_blocking=True)

            image_embeds.append(model.encode_image(image).float().cpu())
            cad_embeds.append(model.encode_cad(command, args).float().cpu())
            ordered_ids.extend(batch["id"])

    return torch.cat(image_embeds), torch.cat(cad_embeds), ordered_ids


def sample_views_per_id(ids, cfg, n_views, seed):
    """Return an expanded (id, view) list: n_views independently sampled views per id."""
    rng = random.Random(seed)
    expanded = []
    for sample_id in ids:
        views = list_views(sample_id, cfg)
        k = min(n_views, len(views))
        expanded.extend((sample_id, view) for view in rng.sample(views, k))
    return expanded


def evaluate_multiview_recall(model, cfg, ids, cad_embeds, device, n_views, seed, ks=(1, 5, 10)):
    """Recall@K sampling n_views independent renders per model; each view is its own query row.

    ids/cad_embeds must already be aligned (cad_embeds[i] is the gallery entry for ids[i]) —
    the sampled views are additional, independent image queries against that same gallery.
    """
    expanded_ids = sample_views_per_id(ids, cfg, n_views, seed)
    dataset = CADImagePairDataset(cfg, ids=expanded_ids, image_transform=model.preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    image_embeds, row_ids = [], []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            image_embeds.append(model.encode_image(image).float().cpu())
            row_ids.extend(batch["id"])
    image_embeds = torch.cat(image_embeds)

    gallery_index = {sample_id: i for i, sample_id in enumerate(ids)}
    gt_indices = torch.tensor([gallery_index[sample_id] for sample_id in row_ids], dtype=torch.long)
    return evaluate_recall(image_embeds, cad_embeds, gt_indices=gt_indices, ks=ks)


def cad_vector_cache(cfg):
    """A load-once-per-id cache, shared across every checkpoint's evaluation of a split."""
    cache = {}

    def get(sample_id):
        if sample_id not in cache:
            cache[sample_id] = load_cad_vector(cfg.cad_vec_root / f"{sample_id}.h5")
        return cache[sample_id]

    return get


def evaluate_loose_recall(image_embeds, cad_embeds, ids, get_vec, ks=(1, 5, 10), threshold=0.9, sim_cfg=DEFAULT_CONFIG):
    """Recall@K, relaxed with GenCAD-macro-aware CAD similarity (cad_vec_similarity.cad_similarity).

    A retrieval within the top K counts as correct either when it's the exact
    true id (as in evaluate_recall) or, when it isn't, its raw CAD command
    sequence is similar enough (cad_similarity >= threshold) to the true id's.
    Each candidate's similarity is computed once and reused across all ks,
    rather than recomputed per k.
    """
    similarities = image_embeds @ cad_embeds.t()
    order = similarities.argsort(dim=1, descending=True)

    n = image_embeds.shape[0]
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    for i in range(n):
        candidates = order[i, :max_k].tolist()
        gt_vec = get_vec(ids[i])
        is_match = [
            j == i or cad_similarity(gt_vec, get_vec(ids[j]), sim_cfg) >= threshold
            for j in candidates
        ]
        for k in ks:
            if any(is_match[:k]):
                hits[k] += 1
    return {k: v / n for k, v in hits.items()}


def main():
    parser = argparse.ArgumentParser(description="Post-training loose recall using CAD command-sequence similarity")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Evaluate one specific checkpoint file only")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Sweep every epoch_XXX.pt checkpoint in this directory (default: CheckpointConfig.default())",
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--splits", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PathConfig.default()
    view_cfg = ViewConfig.default()

    if args.checkpoint is not None:
        checkpoints = [(None, args.checkpoint)]
    else:
        checkpoint_dir = args.checkpoint_dir or CheckpointConfig.default().checkpoint_dir
        checkpoints = discover_epoch_checkpoints(checkpoint_dir)
        if not checkpoints:
            raise SystemExit(f"no epoch_*.pt checkpoints found in {checkpoint_dir}")
        logger.info("found %d checkpoints to evaluate in %s", len(checkpoints), checkpoint_dir)

    _, val_ids = load_train_val_ids(cfg)
    test_ids = load_test_ids(cfg)

    splits = {}
    if args.splits in ("val", "both"):
        splits["val"] = val_ids
    if args.splits in ("test", "both"):
        splits["test"] = test_ids
    # ground-truth vectors don't change across checkpoints, so this is built once and reused
    vec_caches = {name: cad_vector_cache(cfg) for name in splits}

    run_dir = (EVAL_RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    writer = SummaryWriter(log_dir=str(run_dir))
    logger.info("logging eval curves to %s (tensorboard: tensorboard --logdir %s)", run_dir, run_dir.parent)

    model = CADClipModel(image_model_name="RN50", image_pretrained="openai").to(device)

    for epoch, ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        step = epoch if epoch is not None else ckpt.get("epoch", 0)
        logger.info("evaluating checkpoint %s (epoch %s)", ckpt_path, step)

        for name, ids in splits.items():
            image_embeds, cad_embeds, ordered_ids = embed_split(model, cfg, ids, device)
            exact_recalls, _ = evaluate_recall(image_embeds, cad_embeds)
            loose_recalls = evaluate_loose_recall(
                image_embeds, cad_embeds, ordered_ids, vec_caches[name],
                ks=tuple(args.ks), threshold=args.threshold,
            )
            # same fixed seed on every checkpoint: all checkpoints in a sweep are scored
            # against the identical sampled views, so results stay comparable across epochs.
            multiview_recalls, _ = evaluate_multiview_recall(
                model, cfg, ordered_ids, cad_embeds, device,
                n_views=view_cfg.views_per_query_inference, seed=RANDOM_SEED, ks=tuple(args.ks),
            )

            for k, v in exact_recalls.items():
                writer.add_scalar(f"{name}/recall_top{k}", v, step)
            for k, v in loose_recalls.items():
                writer.add_scalar(f"{name}/loose_recall_top{k}", v, step)
            for k, v in multiview_recalls.items():
                writer.add_scalar(f"{name}/multiview_recall_top{k}", v, step)

            logger.info(
                "[%s] epoch %s  n=%d  exact_recall=%s  loose_recall=%s  multiview_recall(n=%d)=%s",
                name, step, len(ordered_ids),
                {k: round(v, 4) for k, v in exact_recalls.items()},
                {k: round(v, 4) for k, v in loose_recalls.items()},
                view_cfg.views_per_query_inference,
                {k: round(v, 4) for k, v in multiview_recalls.items()},
            )
        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()

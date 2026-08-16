import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cad_vec_similarity import DEFAULT_CONFIG, cad_similarity, load_cad_vector
from config import CheckpointConfig, PathConfig
from dataset import CADImagePairDataset
from metrics import evaluate_recall
from model import CADClipModel
from splits import load_test_ids, load_train_val_ids

# avoids passing worker batches through /dev/shm, which is too small/full on some
# shared machines ("unable to allocate shared memory... No space left on device")
torch.multiprocessing.set_sharing_strategy("file_system")

BATCH_SIZE = 512
NUM_WORKERS = 8
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

            for k, v in exact_recalls.items():
                writer.add_scalar(f"{name}/recall_top{k}", v, step)
            for k, v in loose_recalls.items():
                writer.add_scalar(f"{name}/loose_recall_top{k}", v, step)

            logger.info(
                "[%s] epoch %s  n=%d  exact_recall=%s  loose_recall=%s",
                name, step, len(ordered_ids),
                {k: round(v, 4) for k, v in exact_recalls.items()},
                {k: round(v, 4) for k, v in loose_recalls.items()},
            )
        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()

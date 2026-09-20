import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    import psutil
except ImportError:
    psutil = None

from cad_vec_similarity import pairwise_similarity_matrix
from config import RANDOM_SEED, CheckpointConfig, LossConfig, PathConfig
from dataset import CADImagePairDataset
from losses import cad_similarity_matching_loss, clip_contrastive_loss
from metrics import evaluate_recall
from model import CADClipModel
from splits import load_train_val_ids

# cad_guided's similarity matrix is O(batch_size^2) pure-Python edit-distance calls — past
# this, per-step cost balloons (see cad_vec_similarity.pairwise_similarity_matrix).
CAD_GUIDED_BATCH_SIZE_WARNING_THRESHOLD = 64

NUM_EPOCHS = 150
# Sized against BOTH GPU and host RAM (back to single-view: N = BATCH_SIZE images go through
# the image tower together each step, one view per sample, no multiview batching).
#   - GPU: gradient-checkpointing the visual tower (see model.py) still needs each of its 5
#     checkpointed segments' boundary-input tensors held live for recomputation during backward,
#     plus backward-pass gradients of similar magnitude. Empirically calibrated against the one
#     production data point we have (BATCH_SIZE=64, 42 images/sample, N=2688 -> ~78GB used of
#     85GB total, confirmed via a CUDA OOM warning whose byte count matched the stem-output
#     tensor size exactly) gives ~0.029GB used per image-in-batch, all-inclusive. At one view per
#     sample, BATCH_SIZE=512 (N=512) extrapolates to a very safe ~15GB.
#   - Host RAM: one collated image batch is BATCH_SIZE x 3 x 224 x 224 x 4B, and the DataLoader
#     can buffer up to NUM_WORKERS x TRAIN_PREFETCH_FACTOR batches ahead of the training loop —
#     comfortably under train.slurm's --mem-per-gpu=128G (a cluster-imposed ceiling, not
#     adjustable further) even at TRAIN_PREFETCH_FACTOR=2, the PyTorch default.
BATCH_SIZE = 512
TRAIN_PREFETCH_FACTOR = 2
LR = 1e-4
RUNS_DIR = Path("runs")
LOG_EVERY_N_STEPS = 20
PERSISTENT_WORKERS = True

NUM_WORKERS = 32


def _worker_init_fn(_worker_id):
    torch.set_num_threads(1)
    # torch derives a unique-per-worker seed from the base seed set via torch.manual_seed(),
    # so this is deterministic across runs but distinct across workers, avoiding every
    # worker replaying the same "random" view choices after being forked from the parent.
    worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
    random.seed(worker_seed)
    dataset = torch.utils.data.get_worker_info().dataset
    if hasattr(dataset, "rng"):
        dataset.rng = random.Random(worker_seed)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("cad_clipper")


def _save_checkpoint(path, epoch, model, val_loss):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}, path)
    logger.info("saved checkpoint to %s", path)


def _host_rss_gb():
    """Total host RSS (this process + every live child, e.g. DataLoader workers), in GB.

    This is the number SLURM's cgroup OOM killer is actually watching — logged periodically
    so a future OOM shows the growth curve leading up to it instead of just the final kill.
    Returns None if psutil isn't installed rather than raising, since this is diagnostic only.
    """
    if psutil is None:
        return None
    try:
        proc = psutil.Process()
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return rss / 1e9
    except Exception:
        return None


def run_epoch(model, loader, device, epoch, phase, loss_cfg, optimizer=None, writer=None, collect_embeddings=False):
    is_train = optimizer is not None
    model.train(is_train)

    n_batches_total = len(loader)
    total_loss, n_batches = 0.0, 0
    image_embeds, cad_embeds = [], []

    # Separately timed so a slow step's cause is visible in the log: data_time is time spent
    # waiting on the DataLoader (workers fetching/decoding images from the network-mounted
    # store — the suspected bottleneck given how slow shard access has been elsewhere in this
    # pipeline); compute_time is the forward/backward/optimizer step. Windowed (reset every
    # LOG_EVERY_N_STEPS) rather than a whole-epoch running average, so a transient stall shows
    # up at the step where it happened instead of being smeared across the whole epoch.
    window_data_time, window_compute_time = 0.0, 0.0
    loader_iter = iter(loader)
    step = 0
    while True:
        t0 = time.monotonic()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        t1 = time.monotonic()
        window_data_time += t1 - t0

        image = batch["image"].to(device, non_blocking=True)
        command = batch["command"].to(device, non_blocking=True)
        args = batch["args"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            image_emb = model.encode_image(image)
            cad_emb = model.encode_cad(command, args)
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_emb @ cad_emb.t()
            logits_per_cad = logits_per_image.t()
            # clip_contrastive_loss is always the base term — cad_guided only ADDS a
            # decoupled auxiliary term on top (cad_guided_beta=0 is exactly plain CLIP).
            clip_loss = clip_contrastive_loss(logits_per_image, logits_per_cad)
            matching_loss = None
            if loss_cfg.loss_type == "cad_guided":
                sim = pairwise_similarity_matrix(batch["raw_vec"].numpy())
                sim = torch.as_tensor(sim, device=device, dtype=image_emb.dtype)
                matching_loss = cad_similarity_matching_loss(image_emb, cad_emb, sim)
                loss = clip_loss + loss_cfg.cad_guided_beta * matching_loss
            else:
                loss = clip_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if writer is not None:
                global_step = epoch * n_batches_total + step
                writer.add_scalar("loss/train_step", loss.item(), global_step)
                writer.add_scalar("loss/train_step_clip", clip_loss.item(), global_step)
                if matching_loss is not None:
                    writer.add_scalar("loss/train_step_cad_matching", matching_loss.item(), global_step)

        if collect_embeddings:
            image_embeds.append(image_emb.detach().float().cpu())
            cad_embeds.append(cad_emb.detach().float().cpu())

        # loss.item() above (or below, if not is_train) already forces a CUDA sync, so this
        # captures true GPU completion time, not just kernel-launch time.
        total_loss += loss.item()
        n_batches += 1
        t2 = time.monotonic()
        window_compute_time += t2 - t1

        if step % LOG_EVERY_N_STEPS == 0:
            host_rss_gb = _host_rss_gb()
            rss_str = f"  host_rss {host_rss_gb:.1f}GB" if host_rss_gb is not None else ""
            n_since_log = min(step, LOG_EVERY_N_STEPS) or 1
            logger.info(
                "epoch %03d  %s  step %d/%d  loss %.4f  running_avg %.4f  "
                "data_time %.2fs/step  compute_time %.2fs/step%s",
                epoch, phase, step, n_batches_total, loss.item(), total_loss / n_batches,
                window_data_time / n_since_log, window_compute_time / n_since_log, rss_str,
            )
            window_data_time, window_compute_time = 0.0, 0.0

        step += 1

    avg_loss = total_loss / n_batches
    if writer is not None:
        writer.add_scalar(f"loss/{phase}_epoch", avg_loss, epoch)

    if collect_embeddings:
        return avg_loss, torch.cat(image_embeds), torch.cat(cad_embeds)
    return avg_loss


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # leave the rest of the cores for the NUM_WORKERS dataloader processes
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - NUM_WORKERS))

    run_dir = (RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    writer = SummaryWriter(log_dir=str(run_dir))
    logger.info("logging to %s (tensorboard: tensorboard --logdir %s)", run_dir, run_dir.parent)

    cfg = PathConfig.default()
    loss_cfg = LossConfig.default()
    logger.info("loss_type: %s  cad_guided_beta: %.3f", loss_cfg.loss_type, loss_cfg.cad_guided_beta)
    if loss_cfg.loss_type == "cad_guided" and BATCH_SIZE > CAD_GUIDED_BATCH_SIZE_WARNING_THRESHOLD:
        logger.warning(
            "loss_type=cad_guided with BATCH_SIZE=%d: its similarity matrix is O(batch_size^2) "
            "pure-Python edit-distance calls (see cad_vec_similarity.pairwise_similarity_matrix) "
            "and will likely dominate step time above ~%d — consider a much smaller BATCH_SIZE.",
            BATCH_SIZE, CAD_GUIDED_BATCH_SIZE_WARNING_THRESHOLD,
        )
    model = CADClipModel(image_model_name="RN50", image_pretrained="openai").to(device)

    train_ids, val_ids = load_train_val_ids(cfg)
    logger.info("train ids: %d  val ids: %d", len(train_ids), len(val_ids))
    train_set = CADImagePairDataset(cfg, ids=train_ids, image_transform=model.preprocess)
    # deterministic=True: always the same rendered view per id, so val loss/recall are
    # comparable across epochs instead of jittering with a randomly chosen view.
    val_set = CADImagePairDataset(cfg, ids=val_ids, image_transform=model.preprocess, deterministic=True)

    pin_memory = device.type == "cuda"
    persistent_workers = PERSISTENT_WORKERS and NUM_WORKERS > 0
    train_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, generator=train_generator,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, worker_init_fn=_worker_init_fn,
        persistent_workers=persistent_workers, prefetch_factor=TRAIN_PREFETCH_FACTOR,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, worker_init_fn=_worker_init_fn,
        persistent_workers=persistent_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    ckpt_cfg = CheckpointConfig.default()
    ckpt_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_cfg.checkpoint_dir / "latest.pt"
    best_path = ckpt_cfg.checkpoint_dir / "best.pt"
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_loss = run_epoch(
            model, train_loader, device, epoch, "train", loss_cfg, optimizer=optimizer, writer=writer,
        )
        val_loss, val_image_embeds, val_cad_embeds = run_epoch(
            model, val_loader, device, epoch, "val", loss_cfg, writer=writer, collect_embeddings=True,
        )

        recalls, _ = evaluate_recall(val_image_embeds, val_cad_embeds)
        for k, recall in recalls.items():
            writer.add_scalar(f"recall/val_top{k}", recall, epoch)

        writer.add_scalar("logit_scale", model.logit_scale.exp().item(), epoch)
        logger.info(
            "epoch %03d  train_loss %.4f  val_loss %.4f  val_recall %s",
            epoch, train_loss, val_loss, {k: round(v, 4) for k, v in recalls.items()},
        )
        writer.flush()

        _save_checkpoint(latest_path, epoch, model, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(best_path, epoch, model, val_loss)

        if ckpt_cfg.save_every_n_epochs and (epoch + 1) % ckpt_cfg.every_n == 0:
            periodic_path = ckpt_cfg.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            _save_checkpoint(periodic_path, epoch, model, val_loss)

    writer.close()


if __name__ == "__main__":
    main()

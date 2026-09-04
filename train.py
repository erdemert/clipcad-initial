import logging
import os
import random
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import RANDOM_SEED, CheckpointConfig, PathConfig, ViewConfig
from dataset import CADImagePairDataset
from losses import clip_contrastive_loss, multiview_contrastive_loss
from metrics import evaluate_recall
from model import CADClipModel
from splits import load_train_val_ids

NUM_EPOCHS = 150
# Host-RAM sizing, not GPU: with views_per_sample_train=42, one collated image batch is
# BATCH_SIZE x 42 x 3 x 224 x 224 x 4B ~= BATCH_SIZE x 25.3MB. The DataLoader can buffer up to
# NUM_WORKERS x TRAIN_PREFETCH_FACTOR batches ahead of the training loop, so worst case this job
# holds NUM_WORKERS x TRAIN_PREFETCH_FACTOR x BATCH_SIZE x 25.3MB of host RAM. train.slurm's
# --mem-per-gpu is capped at 128G (cluster limit, not adjustable) — at NUM_WORKERS=32, 128/2
# (BATCH_SIZE/TRAIN_PREFETCH_FACTOR) needs ~208GB and reliably OOMs; 64/1 keeps worst case to
# ~52GB, leaving real headroom under the 128G ceiling.
BATCH_SIZE = 64
TRAIN_PREFETCH_FACTOR = 1
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


def run_epoch(model, loader, device, epoch, phase, optimizer=None, writer=None, collect_embeddings=False):
    is_train = optimizer is not None
    model.train(is_train)

    n_batches_total = len(loader)
    total_loss, n_batches = 0.0, 0
    image_embeds, cad_embeds = [], []
    for step, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        command = batch["command"].to(device, non_blocking=True)
        args = batch["args"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cad_emb = model.encode_cad(command, args)
            logit_scale = model.logit_scale.exp()
            if image.dim() == 5:
                # (B, K, C, H, W): K views per sample, each encoded and scored independently
                # against the shared CAD embedding — no pooling, matching the single-image
                # query used at retrieval time (see retrieve.py).
                batch_size, n_views = image.shape[:2]
                flat = image.reshape(batch_size * n_views, *image.shape[2:])
                image_emb = model.encode_image(flat)
                loss = multiview_contrastive_loss(cad_emb, image_emb, n_views, logit_scale)
            else:
                image_emb = model.encode_image(image)
                logits_per_image = logit_scale * image_emb @ cad_emb.t()
                logits_per_cad = logits_per_image.t()
                loss = clip_contrastive_loss(logits_per_image, logits_per_cad)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalar("loss/train_step", loss.item(), epoch * n_batches_total + step)

        if collect_embeddings:
            image_embeds.append(image_emb.detach().float().cpu())
            cad_embeds.append(cad_emb.detach().float().cpu())

        total_loss += loss.item()
        n_batches += 1

        if step % LOG_EVERY_N_STEPS == 0:
            logger.info(
                "epoch %03d  %s  step %d/%d  loss %.4f  running_avg %.4f",
                epoch, phase, step, n_batches_total, loss.item(), total_loss / n_batches,
            )

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
    view_cfg = ViewConfig.default()
    model = CADClipModel(image_model_name="RN50", image_pretrained="openai").to(device)

    train_ids, val_ids = load_train_val_ids(cfg)
    logger.info(
        "train ids: %d  val ids: %d  views_per_sample_train: %d",
        len(train_ids), len(val_ids), view_cfg.views_per_sample_train,
    )
    train_set = CADImagePairDataset(
        cfg, ids=train_ids, image_transform=model.preprocess,
        views_per_sample=view_cfg.views_per_sample_train, seed=RANDOM_SEED,
    )
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
        train_loss = run_epoch(model, train_loader, device, epoch, "train", optimizer=optimizer, writer=writer)
        val_loss, val_image_embeds, val_cad_embeds = run_epoch(
            model, val_loader, device, epoch, "val", writer=writer, collect_embeddings=True,
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

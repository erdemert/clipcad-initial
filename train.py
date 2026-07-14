import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import PathConfig
from dataset import CADImagePairDataset
from losses import clip_contrastive_loss
from metrics import evaluate_recall
from model import CADClipModel
from splits import load_train_val_ids

NUM_EPOCHS = 100
BATCH_SIZE = 512
LR = 1e-4
RUNS_DIR = Path("runs")
CHECKPOINT_PATH = Path("checkpoints") / "latest.pt"
LOG_EVERY_N_STEPS = 20
PERSISTENT_WORKERS = True

# Data loading for images at this scale plateaus well before using every core;
# cap workers rather than spawning one per core.
NUM_WORKERS = min(32, os.cpu_count() or 8)


def _worker_init_fn(_worker_id):
    torch.set_num_threads(1)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("cad_clipper")


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
            image_emb = model.encode_image(image)
            cad_emb = model.encode_cad(command, args)
            logit_scale = model.logit_scale.exp()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # leave the rest of the cores for the NUM_WORKERS dataloader processes
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - NUM_WORKERS))

    run_dir = (RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    writer = SummaryWriter(log_dir=str(run_dir))
    logger.info("logging to %s (tensorboard: tensorboard --logdir %s)", run_dir, run_dir.parent)

    cfg = PathConfig.default()
    model = CADClipModel(image_model_name="RN50", image_pretrained="openai").to(device)

    train_ids, val_ids = load_train_val_ids(cfg)
    logger.info("train ids: %d  val ids: %d", len(train_ids), len(val_ids))
    train_set = CADImagePairDataset(cfg, ids=train_ids, image_transform=model.preprocess)
    # deterministic=True: always the same rendered view per id, so val loss/recall are
    # comparable across epochs instead of jittering with a randomly chosen view.
    val_set = CADImagePairDataset(cfg, ids=val_ids, image_transform=model.preprocess, deterministic=True)

    pin_memory = device.type == "cuda"
    persistent_workers = PERSISTENT_WORKERS and NUM_WORKERS > 0
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, worker_init_fn=_worker_init_fn,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, worker_init_fn=_worker_init_fn,
        persistent_workers=persistent_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

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

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, CHECKPOINT_PATH)
        logger.info("saved checkpoint to %s", CHECKPOINT_PATH)

    writer.close()


if __name__ == "__main__":
    main()

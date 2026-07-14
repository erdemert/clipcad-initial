import argparse
import logging
import random

import torch
from torch.utils.data import DataLoader

from config import PathConfig
from dataset import CADImagePairDataset
from model import CADClipModel
from splits import load_train_val_ids
from train import CHECKPOINT_PATH, NUM_WORKERS, _worker_init_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("cad_clipper.retrieve")

BATCH_SIZE = 32
TOP_K = 5
DEFAULT_MAX_SAMPLES = 2000  # keep the candidate pool bounded unless the user asks for more


@torch.no_grad()
def encode_split(model, dataset, device):
    """Encode every sample in a dataset once, returning (ids, image_embeddings, cad_embeddings)."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=_worker_init_fn,
    )

    ids, image_embeds, cad_embeds = [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        command = batch["command"].to(device)
        args = batch["args"].to(device)

        image_embeds.append(model.encode_image(image).cpu())
        cad_embeds.append(model.encode_cad(command, args).cpu())
        ids.extend(batch["id"])

    return ids, torch.cat(image_embeds), torch.cat(cad_embeds)


def retrieve_topk(query_image_embed, cad_embeds, ids, top_k=TOP_K):
    """Rank all CAD candidates by similarity to one query image embedding."""
    similarities = query_image_embed @ cad_embeds.t()
    ranked = torch.argsort(similarities, descending=True)
    return [(ids[i], similarities[i].item()) for i in ranked[:top_k]]


def evaluate_recall(image_embeds, cad_embeds, ks=(1, 5, 10)):
    """Recall@K for image->CAD retrieval: is each image's true CAD match within its top K?"""
    similarities = image_embeds @ cad_embeds.t()
    order = similarities.argsort(dim=1, descending=True)
    rank_of_candidate = order.argsort(dim=1)  # invert the permutation

    n = image_embeds.shape[0]
    true_rank = rank_of_candidate[torch.arange(n), torch.arange(n)]
    recalls = {k: (true_rank < k).float().mean().item() for k in ks}
    return recalls, true_rank


def main():
    parser = argparse.ArgumentParser(description="Image -> CAD retrieval demo on the validation split")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--query-id", default=None, help="Specific validation id to query with, e.g. 0073/00732275")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
        help="Cap the candidate pool size (0 = use the full validation split)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = PathConfig.default()
    model = CADClipModel(image_model_name="RN50", image_pretrained=None).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("loaded checkpoint %s (epoch %d)", args.checkpoint, checkpoint["epoch"])

    _, val_ids = load_train_val_ids(cfg)
    if args.max_samples:
        val_ids = val_ids[: args.max_samples]
    val_set = CADImagePairDataset(cfg, ids=val_ids, image_transform=model.preprocess, deterministic=True)

    logger.info("encoding %d validation samples...", len(val_ids))
    ids, image_embeds, cad_embeds = encode_split(model, val_set, device)

    recalls, true_rank = evaluate_recall(image_embeds, cad_embeds)
    logger.info("image->CAD retrieval recall over %d candidates: %s", len(ids), recalls)

    query_id = args.query_id or random.choice(ids)
    query_index = ids.index(query_id)
    logger.info("query image id: %s", query_id)

    top_matches = retrieve_topk(image_embeds[query_index], cad_embeds, ids, top_k=args.top_k)
    for rank, (candidate_id, score) in enumerate(top_matches):
        marker = " <-- ground truth" if candidate_id == query_id else ""
        logger.info("  #%d  %s  sim=%.4f%s", rank + 1, candidate_id, score, marker)

    logger.info("ground truth rank: %d (0 = top match)", true_rank[query_index].item())


if __name__ == "__main__":
    main()

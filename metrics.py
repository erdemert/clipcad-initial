import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from config import PathConfig
from data_io import load_image


def _load_gray(sample_id: str, view: int, cfg: PathConfig) -> np.ndarray:
    return np.asarray(load_image(sample_id, view, cfg).convert("L"), dtype=np.float64)


def visual_similarity(id_a: str, id_b: str, cfg: PathConfig) -> float:
    """SSIM between the 0-indexed rendered view of two CAD models."""
    img_a = _load_gray(id_a, 0, cfg)
    img_b = _load_gray(id_b, 0, cfg)
    return ssim(img_a, img_b, data_range=255)


def evaluate_visual_recall(image_embeds, cad_embeds, ids, cfg: PathConfig, ks=(1, 5, 10), threshold=0.9):
    """Recall@K, relaxed with visual similarity.

    A retrieval within the top K counts as correct either when it's the exact
    true id (as in evaluate_recall) or, when it isn't, its 0-indexed rendered
    view is visually close enough (SSIM >= threshold) to the true id's — so a
    visually-identical but differently-labeled CAD model isn't scored as a miss.
    """
    similarities = image_embeds @ cad_embeds.t()
    order = similarities.argsort(dim=1, descending=True)

    n = image_embeds.shape[0]
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    for i in range(n):
        candidates = order[i, :max_k].tolist()
        is_match = [j == i or visual_similarity(ids[i], ids[j], cfg) >= threshold for j in candidates]
        for k in ks:
            if any(is_match[:k]):
                hits[k] += 1
    return {k: v / n for k, v in hits.items()}


def retrieve_topk(query_image_embed, cad_embeds, ids, top_k=5):
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

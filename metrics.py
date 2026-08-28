import torch


def retrieve_topk(query_image_embed, cad_embeds, ids, top_k=5):
    """Rank all CAD candidates by similarity to one query image embedding."""
    similarities = query_image_embed @ cad_embeds.t()
    ranked = torch.argsort(similarities, descending=True)
    return [(ids[i], similarities[i].item()) for i in ranked[:top_k]]


def evaluate_recall(image_embeds, cad_embeds, gt_indices=None, ks=(1, 5, 10)):
    """Recall@K for image->CAD retrieval: is each image's true CAD match within its top K?

    gt_indices[i] gives the row of cad_embeds that is the correct match for image_embeds[i].
    Defaults to the identity (row i matches row i), for the common case of one image per
    CAD id; pass an explicit mapping when several image rows (e.g. different rendered
    views of the same model) share one CAD gallery entry.
    """
    similarities = image_embeds @ cad_embeds.t()
    order = similarities.argsort(dim=1, descending=True)
    rank_of_candidate = order.argsort(dim=1)  # invert the permutation

    n = image_embeds.shape[0]
    if gt_indices is None:
        gt_indices = torch.arange(n)
    true_rank = rank_of_candidate[torch.arange(n), gt_indices]
    recalls = {k: (true_rank < k).float().mean().item() for k in ks}
    return recalls, true_rank

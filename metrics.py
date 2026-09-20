import torch


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

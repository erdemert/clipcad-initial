import torch
import torch.nn.functional as F


def clip_contrastive_loss(logits_per_image, logits_per_cad):
    """Symmetric InfoNCE loss (CLIP loss) over a batch of matched image/CAD pairs."""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_c = F.cross_entropy(logits_per_cad, labels)
    return (loss_i + loss_c) / 2


def cad_similarity_matching_loss(image_emb, cad_emb, cad_similarity_matrix):
    """Auxiliary regularizer: predicted embedding similarity should track real CAD similarity.

    Decoupled from clip_contrastive_loss entirely — meant to be ADDED to it with a small
    weight, not used alone: total = clip_contrastive_loss(...) + beta * this(...). At beta=0
    training is exactly plain CLIP, since this term is fully additive; it never touches
    clip_contrastive_loss's targets or gradients directly, only adds its own.

    image_emb, cad_emb: (N, D), L2-normalized, so image_emb @ cad_emb.t() is already cosine
    similarity in [-1, 1]. Remapped to [0, 1] to match cad_similarity_matrix's range before
    comparing, since cad_similarity (see cad_vec_similarity.py) is defined on [0, 1] (1.0 =
    identical CAD, -> 0 = maximally different) and is never negative.

    cad_similarity_matrix: (N, N), symmetric, 1.0 on the diagonal, same device/dtype as the
    embeddings (build via cad_vec_similarity.pairwise_similarity_matrix, then
    torch.as_tensor(..., device=..., dtype=...)). No separate image->cad / cad->image split
    needed (unlike the cross-entropy losses above) since this is a plain elementwise
    regression over the full matrix, which already covers both directions.
    """
    predicted_similarity = image_emb @ cad_emb.t()
    predicted_similarity_01 = (predicted_similarity + 1) / 2
    return F.mse_loss(predicted_similarity_01, cad_similarity_matrix)

import torch
import torch.nn.functional as F


def clip_contrastive_loss(logits_per_image, logits_per_cad):
    """Symmetric InfoNCE loss (CLIP loss) over a batch of matched image/CAD pairs."""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_c = F.cross_entropy(logits_per_cad, labels)
    return (loss_i + loss_c) / 2


def multiview_contrastive_loss(cad_emb, image_emb, n_views, logit_scale):
    """Symmetric InfoNCE generalized to n_views independently-encoded images per CAD model.

    cad_emb: (N, D) one embedding per sample. image_emb: (N * n_views, D), the n_views images
    for sample i occupying rows [i*n_views, (i+1)*n_views) (see how `flat` is built in train.py) —
    no pooling, each view is scored on its own. CAD->image has n_views positives per row, so that
    direction is averaged over per-positive cross-entropies (equivalent to SupCon's multi-positive
    loss); image->CAD has exactly one positive per row, so it's a plain single-label cross-entropy.
    """
    n = cad_emb.shape[0]
    logits_per_cad = logit_scale * cad_emb @ image_emb.t()  # (N, N * n_views)

    match_mask = torch.repeat_interleave(torch.eye(n, device=cad_emb.device), n_views, dim=1)
    targets_cad_to_image = match_mask / match_mask.sum(dim=1, keepdim=True)
    loss_cad = F.cross_entropy(logits_per_cad, targets_cad_to_image)

    labels_image_to_cad = torch.arange(n, device=cad_emb.device).repeat_interleave(n_views)
    loss_image = F.cross_entropy(logits_per_cad.t(), labels_image_to_cad)

    return (loss_cad + loss_image) / 2

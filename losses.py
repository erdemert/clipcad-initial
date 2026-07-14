import torch
import torch.nn.functional as F


def clip_contrastive_loss(logits_per_image, logits_per_cad):
    """Symmetric InfoNCE loss (CLIP loss) over a batch of matched image/CAD pairs."""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_c = F.cross_entropy(logits_per_cad, labels)
    return (loss_i + loss_c) / 2

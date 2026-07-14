import math

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from cad_format import ALL_COMMANDS, ARGS_DIM, DEFAULT_MAX_LEN, EOS_IDX


class CADSequenceEncoder(nn.Module):
    """Vanilla transformer encoder over CAD command-vector rows, trained from scratch."""

    def __init__(self, d_model=512, n_layers=4, n_heads=8, max_len=DEFAULT_MAX_LEN):
        super().__init__()
        self.cmd_embed = nn.Embedding(len(ALL_COMMANDS), d_model)
        self.arg_embed = nn.Embedding(ARGS_DIM + 2, d_model)  # +1 shift for PAD_VAL(-1), +1 buffer
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Embedding(max_len + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_dim = d_model

    def forward(self, command, args):
        batch_size, seq_len = command.shape

        tokens = self.cmd_embed(command) + self.arg_embed(args + 1).sum(dim=2)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        positions = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.pos_embed(positions).unsqueeze(0)

        padding_mask = self._padding_mask(command)
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return encoded[:, 0]  # pooled CLS token

    @staticmethod
    def _padding_mask(command):
        """True for positions strictly after the first EOS row (plus the CLS slot at index 0)."""
        is_eos = (command == EOS_IDX).long()
        first_eos = is_eos.argmax(dim=1)
        positions = torch.arange(command.shape[1], device=command.device).unsqueeze(0)
        seq_mask = positions > first_eos.unsqueeze(1)
        cls_mask = torch.zeros(command.shape[0], 1, dtype=torch.bool, device=command.device)
        return torch.cat([cls_mask, seq_mask], dim=1)


class CADClipModel(nn.Module):
    """CLIP-style dual encoder: pretrained open_clip image tower + from-scratch CAD transformer."""

    def __init__(
        self,
        embed_dim=512,
        image_model_name="RN50",
        image_pretrained="openai",
        cad_d_model=512,
        cad_layers=4,
        cad_heads=8,
        max_len=DEFAULT_MAX_LEN,
    ):
        super().__init__()
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            image_model_name, pretrained=image_pretrained,
        )
        self.visual = clip_model.visual
        image_dim = clip_model.visual.output_dim

        self.cad_encoder = CADSequenceEncoder(
            d_model=cad_d_model, n_layers=cad_layers, n_heads=cad_heads, max_len=max_len,
        )

        self.image_proj = nn.Identity() if image_dim == embed_dim else nn.Linear(image_dim, embed_dim)
        self.cad_proj = nn.Identity() if cad_d_model == embed_dim else nn.Linear(cad_d_model, embed_dim)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, image):
        return F.normalize(self.image_proj(self.visual(image)), dim=-1)

    def encode_cad(self, command, args):
        return F.normalize(self.cad_proj(self.cad_encoder(command, args)), dim=-1)

    def forward(self, image, command, args):
        image_emb = self.encode_image(image)
        cad_emb = self.encode_cad(command, args)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_emb @ cad_emb.t()
        logits_per_cad = logits_per_image.t()
        return logits_per_image, logits_per_cad

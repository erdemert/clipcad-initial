import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent

# Seeds every random operation (view sampling during training) so runs are reproducible
# given the same seed.
RANDOM_SEED = int(os.environ.get("CAD_CLIPPER_SEED", "42"))


@dataclass(frozen=True)
class PathConfig:
    cad_vec_root: Path
    images_root: Path
    split_path: Path

    @classmethod
    def default(cls) -> "PathConfig":
        cad_vec_root = Path(os.environ.get(
            "CAD_CLIPPER_CAD_VEC_ROOT",
            "/mnt/amax2_drive/cad_retrieval_2/erdem_denemeler/data/cad_vec",
        ))
        images_root = Path(os.environ.get(
            "CAD_CLIPPER_IMAGES_ROOT",
            "/mnt/amax2_drive/cad_retrieval_2/erdem_denemeler/data/images",
        ))
        split_path = Path(os.environ.get(
            "CAD_CLIPPER_SPLIT_PATH",
            str(PACKAGE_ROOT / "splits" / "filtered_data.json"),
        ))
        return cls(cad_vec_root=cad_vec_root, images_root=images_root, split_path=split_path)


@dataclass(frozen=True)
class LossConfig:
    loss_type: str  # "clip" or "cad_guided"
    cad_guided_beta: float

    @classmethod
    def default(cls) -> "LossConfig":
        # "cad_guided" ADDS an auxiliary CAD-similarity-matching regularizer on top of the
        # unchanged clip_contrastive_loss (see losses.cad_similarity_matching_loss): total =
        # clip_loss + cad_guided_beta * matching_loss. Fully decoupled/additive, not a
        # replacement for "clip" — cad_guided_beta=0 is exactly plain CLIP training. Building
        # the similarity matrix is O(batch_size^2) pure-Python edit-distance calls
        # (cad_vec_similarity.cad_distance isn't vectorized), so only select "cad_guided" with
        # a small BATCH_SIZE in train.py.
        loss_type = os.environ.get("CAD_CLIPPER_LOSS_TYPE", "clip")
        if loss_type not in ("clip", "cad_guided"):
            raise ValueError(f"CAD_CLIPPER_LOSS_TYPE must be 'clip' or 'cad_guided', got {loss_type!r}")
        cad_guided_beta = float(os.environ.get("CAD_CLIPPER_CAD_GUIDED_BETA", "1.0"))
        return cls(loss_type=loss_type, cad_guided_beta=cad_guided_beta)


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint_dir: Path
    save_every_n_epochs: bool
    every_n: int

    @classmethod
    def default(cls) -> "CheckpointConfig":
        checkpoint_dir = Path(os.environ.get("CAD_CLIPPER_CHECKPOINT_DIR", "checkpoints"))
        save_every_n_epochs = os.environ.get(
            "CAD_CLIPPER_SAVE_EVERY_N_EPOCHS", "true"
        ).lower() in ("1", "true", "yes")
        every_n = int(os.environ.get("CAD_CLIPPER_CHECKPOINT_EVERY_N", "5"))
        return cls(checkpoint_dir=checkpoint_dir, save_every_n_epochs=save_every_n_epochs, every_n=every_n)

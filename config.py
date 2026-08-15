import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PathConfig:
    cad_vec_root: Path
    images_root: Path
    split_path: Path

    @classmethod
    def default(cls) -> "PathConfig":
        cad_vec_root = Path(os.environ.get(
            "CAD_CLIPPER_CAD_VEC_ROOT",
            "/mnt/amax5_drive/erdem_erturk_0/cad_data/data/cad_vec",
        ))
        images_root = Path(os.environ.get(
            "CAD_CLIPPER_IMAGES_ROOT",
            "/mnt/amax5_drive/erdem_erturk_0/cad_data/images",
        ))
        split_path = Path(os.environ.get(
            "CAD_CLIPPER_SPLIT_PATH",
            str(PACKAGE_ROOT / "splits" / "filtered_data.json"),
        ))
        return cls(cad_vec_root=cad_vec_root, images_root=images_root, split_path=split_path)


@dataclass(frozen=True)
class CheckpointConfig:
    checkpoint_dir: Path
    save_every_n_epochs: bool
    every_n: int

    @classmethod
    def default(cls) -> "CheckpointConfig":
        checkpoint_dir = Path(os.environ.get("CAD_CLIPPER_CHECKPOINT_DIR", "checkpoints"))
        save_every_n_epochs = os.environ.get(
            "CAD_CLIPPER_SAVE_EVERY_N_EPOCHS", "false"
        ).lower() in ("1", "true", "yes")
        every_n = int(os.environ.get("CAD_CLIPPER_CHECKPOINT_EVERY_N", "5"))
        return cls(checkpoint_dir=checkpoint_dir, save_every_n_epochs=save_every_n_epochs, every_n=every_n)

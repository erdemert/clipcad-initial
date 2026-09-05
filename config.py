import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent

# Seeds every random operation (view sampling during training and the multi-view
# inference sweep) so runs are reproducible given the same seed.
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
            "/mnt/amax5_drive/erdem_erturk_0/cad_data/data/cad_vec",
        ))
        images_root = Path(os.environ.get(
            "CAD_CLIPPER_IMAGES_ROOT",
            "/mnt/DTX_AI_lab/cad_retrieval_2/rendering/renders",
        ))
        split_path = Path(os.environ.get(
            "CAD_CLIPPER_SPLIT_PATH",
            str(PACKAGE_ROOT / "splits" / "filtered_data.json"),
        ))
        return cls(cad_vec_root=cad_vec_root, images_root=images_root, split_path=split_path)


@dataclass(frozen=True)
class ViewConfig:
    views_per_sample_train: int
    views_per_query_inference: int

    @classmethod
    def default(cls) -> "ViewConfig":
        # 21 of 42 rendered angles per training sample (half) — N = BATCH_SIZE x
        # views_per_sample_train images go through the image tower together each training step,
        # and that N drove the GPU/host OOMs fixed by cutting BATCH_SIZE; halving this too gives
        # a further safety margin (N=32*21=672, vs 32*42=1344) while still training on every
        # sample's views across enough steps to see the full angular spread. Eval's multi-view
        # sweep (views_per_query_inference) is unrelated to that per-step OOM and stays at 42.
        views_per_sample_train = int(os.environ.get("CAD_CLIPPER_VIEWS_PER_SAMPLE_TRAIN", "21"))
        views_per_query_inference = int(os.environ.get("CAD_CLIPPER_VIEWS_PER_QUERY_INFERENCE", "42"))
        return cls(
            views_per_sample_train=views_per_sample_train,
            views_per_query_inference=views_per_query_inference,
        )


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

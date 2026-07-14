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
            "/media/erdem/Backup/THESIS/cad_vec/data/cad_vec",
        ))
        images_root = Path(os.environ.get(
            "CAD_CLIPPER_IMAGES_ROOT",
            "/media/erdem/Backup/THESIS/images",
        ))
        split_path = Path(os.environ.get(
            "CAD_CLIPPER_SPLIT_PATH",
            str(PACKAGE_ROOT / "splits" / "filtered_data.json"),
        ))
        return cls(cad_vec_root=cad_vec_root, images_root=images_root, split_path=split_path)

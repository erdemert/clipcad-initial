from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


# ============================================================
# GenCAD / DeepCAD command vocabulary
# ============================================================

LINE = 0
ARC = 1
CIRCLE = 2
EOS = 3
SOL = 4
EXTRUDE = 5

PAD = -1
N_ARGS = 16
VECTOR_DIM = 17
QUANTIZATION = 256


COMMAND_NAMES = {
    LINE: "Line",
    ARC: "Arc",
    CIRCLE: "Circle",
    EOS: "EOS",
    SOL: "SOL",
    EXTRUDE: "Extrude",
}


# ============================================================
# Parameter layout
#
# vector = [command, arg0, ..., arg15]
#
# Based on GenCAD/DeepCAD macro.py:
#
# Line:
#   [x, y]
#
# Arc:
#   [x, y, sweep_angle, clock_sign]
#
# Circle:
#   [x, y, radius]
#
# Extrude:
#   [theta, phi, gamma,
#    px, py, pz, sketch_size,
#    extent_one, extent_two,
#    operation, extent_type]
# ============================================================

PARAMS = {
    LINE: [1, 2],
    ARC: [1, 2, 3, 4],
    CIRCLE: [1, 2, 5],
    EXTRUDE: list(range(6, 17)),
}


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class CADSimilarityConfig:
    # Cost for inserting/deleting a command.
    gap_cost: float = 1.0

    # Cost of changing one command type into another.
    type_mismatch_cost: float = 1.0

    # Relative importance of parameter differences.
    parameter_weight: float = 1.0

    # Relative importance of command-type mismatch.
    type_weight: float = 1.0

    # Weights for individual parameters.
    #
    # These are multiplied by normalized parameter distances.
    #
    # Line:
    #   x, y
    #
    # Arc:
    #   x, y, sweep, clock_sign
    #
    # Circle:
    #   x, y, radius
    #
    # Extrude:
    #   theta, phi, gamma,
    #   px, py, pz, size,
    #   extent1, extent2,
    #   operation, extent_type
    line_weights: tuple[float, float] = (1.0, 1.0)

    arc_weights: tuple[float, float, float, float] = (
        1.0, 1.0, 1.0, 0.5
    )

    circle_weights: tuple[float, float, float] = (
        1.0, 1.0, 1.0
    )

    extrude_weights: tuple[float, ...] = (
        1.0,  # theta
        1.0,  # phi
        1.0,  # gamma
        1.0,  # px
        1.0,  # py
        1.0,  # pz
        1.0,  # sketch size
        1.0,  # extent one
        1.0,  # extent two
        1.0,  # operation
        1.0,  # extent type
    )

    # Lambda in S = exp(-lambda * normalized_distance)
    similarity_lambda: float = 3.0


DEFAULT_CONFIG = CADSimilarityConfig()


# ============================================================
# H5 loading
# ============================================================

def load_cad_vector(path: str | Path) -> np.ndarray:
    """
    Load a GenCAD .h5 file.

    Expected structure:
        file["vec"] -> (T, 17)
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        vec = np.asarray(f["vec"][:], dtype=np.int16)

    if vec.ndim != 2 or vec.shape[1] != VECTOR_DIM:
        raise ValueError(
            f"{path}: expected (T, 17), got {vec.shape}"
        )

    return vec


def remove_padding(vec: np.ndarray) -> np.ndarray:
    """
    Remove EOS padding after the real CAD sequence.

    GenCAD uses EOS (command 3) for both the actual end of the
    sequence and subsequent padding.

    Therefore we keep the first EOS and discard everything after it.
    """
    commands = vec[:, 0]

    eos_positions = np.flatnonzero(commands == EOS)

    if len(eos_positions) == 0:
        return vec

    first_eos = int(eos_positions[0])

    return vec[:first_eos + 1]


# ============================================================
# Utility functions
# ============================================================

def normalized_abs_difference(a: float, b: float) -> float:
    """
    Difference between two quantized continuous values.

    GenCAD numericalization uses approximately [0, 255].
    """
    return min(abs(float(a) - float(b)) / 255.0, 1.0)


def circular_difference(a: float, b: float) -> float:
    """
    Circular difference for parameters encoded over a 2*pi range.

    Used for phi and gamma.

    Values are quantized into 256 bins.
    """
    d = abs(float(a) - float(b))
    d = min(d, QUANTIZATION - d)

    return min(d / (QUANTIZATION / 2.0), 1.0)


def binary_difference(a: float, b: float) -> float:
    """
    Difference for binary categorical parameters.
    """
    return 0.0 if int(a) == int(b) else 1.0


def categorical_difference(a: float, b: float, n_classes: int) -> float:
    """
    Difference for small categorical parameters.

    We deliberately do NOT treat operation=0 vs operation=1 as
    a tiny continuous difference. They are different operations.
    """
    return 0.0 if int(a) == int(b) else 1.0


# ============================================================
# Command parameter distances
# ============================================================

def line_parameter_distance(
    a: np.ndarray,
    b: np.ndarray,
    cfg: CADSimilarityConfig,
) -> float:
    """
    Line:
        [x, y]
    """
    diffs = np.array([
        normalized_abs_difference(a[1], b[1]),
        normalized_abs_difference(a[2], b[2]),
    ])

    weights = np.asarray(cfg.line_weights)

    return float(np.average(diffs, weights=weights))


def arc_parameter_distance(
    a: np.ndarray,
    b: np.ndarray,
    cfg: CADSimilarityConfig,
) -> float:
    """
    Arc:
        [x, y, sweep_angle, clock_sign]

    sweep_angle:
        0..255 corresponds to 0..2*pi.

    clock_sign:
        binary categorical variable.
    """
    diffs = np.array([
        normalized_abs_difference(a[1], b[1]),
        normalized_abs_difference(a[2], b[2]),

        # Sweep angle is NOT circular:
        # 5 degrees and 355 degrees are not equivalent here.
        normalized_abs_difference(a[3], b[3]),

        binary_difference(a[4], b[4]),
    ])

    weights = np.asarray(cfg.arc_weights)

    return float(np.average(diffs, weights=weights))


def circle_parameter_distance(
    a: np.ndarray,
    b: np.ndarray,
    cfg: CADSimilarityConfig,
) -> float:
    """
    Circle:
        [x, y, radius]
    """
    diffs = np.array([
        normalized_abs_difference(a[1], b[1]),
        normalized_abs_difference(a[2], b[2]),
        normalized_abs_difference(a[5], b[5]),
    ])

    weights = np.asarray(cfg.circle_weights)

    return float(np.average(diffs, weights=weights))


def extrude_parameter_distance(
    a: np.ndarray,
    b: np.ndarray,
    cfg: CADSimilarityConfig,
) -> float:
    """
    Extrude:

        [theta, phi, gamma,
         px, py, pz, sketch_size,
         extent_one, extent_two,
         operation, extent_type]
    """

    # theta: 0..pi -> ordinary normalized distance
    theta_diff = normalized_abs_difference(a[6], b[6])

    # phi: -pi..pi -> circular
    phi_diff = circular_difference(a[7], b[7])

    # gamma: -pi..pi -> circular
    gamma_diff = circular_difference(a[8], b[8])

    # Sketch position
    px_diff = normalized_abs_difference(a[9], b[9])
    py_diff = normalized_abs_difference(a[10], b[10])
    pz_diff = normalized_abs_difference(a[11], b[11])

    # Sketch bounding size
    size_diff = normalized_abs_difference(a[12], b[12])

    # Extrusion distances
    extent1_diff = normalized_abs_difference(a[13], b[13])
    extent2_diff = normalized_abs_difference(a[14], b[14])

    # These are categorical, NOT continuous.
    operation_diff = categorical_difference(
        a[15], b[15], n_classes=4
    )

    extent_type_diff = categorical_difference(
        a[16], b[16], n_classes=3
    )

    diffs = np.array([
        theta_diff,
        phi_diff,
        gamma_diff,
        px_diff,
        py_diff,
        pz_diff,
        size_diff,
        extent1_diff,
        extent2_diff,
        operation_diff,
        extent_type_diff,
    ])

    weights = np.asarray(cfg.extrude_weights)

    return float(np.average(diffs, weights=weights))


def command_parameter_distance(
    a: np.ndarray,
    b: np.ndarray,
    cfg: CADSimilarityConfig,
) -> float:
    """
    Parameter distance assuming both commands have the same type.
    """
    command = int(a[0])

    if command == LINE:
        return line_parameter_distance(a, b, cfg)

    if command == ARC:
        return arc_parameter_distance(a, b, cfg)

    if command == CIRCLE:
        return circle_parameter_distance(a, b, cfg)

    if command == EXTRUDE:
        return extrude_parameter_distance(a, b, cfg)

    # SOL / EOS have no parameters.
    if command in (SOL, EOS):
        return 0.0

    raise ValueError(f"Unknown command ID: {command}")


# ============================================================
# Individual command cost
# ============================================================

def command_substitution_cost(
    a: np.ndarray,
    b: np.ndarray,
    cfg: CADSimilarityConfig,
) -> float:
    """
    Cost of matching command a against command b.
    """
    ta = int(a[0])
    tb = int(b[0])

    # Same command type:
    if ta == tb:
        return (
            cfg.parameter_weight
            * command_parameter_distance(a, b, cfg)
        )

    # Different command types:
    return cfg.type_weight * cfg.type_mismatch_cost


# ============================================================
# Sequence alignment
# ============================================================

def cad_distance(
    cad_a: np.ndarray,
    cad_b: np.ndarray,
    cfg: CADSimilarityConfig = DEFAULT_CONFIG,
) -> float:
    """
    Parameter-aware normalized sequence edit distance.

    Uses global sequence alignment:

        insertion
        deletion
        substitution

    with operation-aware substitution costs.

    Returns a value approximately in [0, 1].
    """

    a = remove_padding(cad_a)
    b = remove_padding(cad_b)

    m = len(a)
    n = len(b)

    if m == 0 and n == 0:
        return 0.0

    # DP matrix.
    dp = np.empty((m + 1, n + 1), dtype=np.float64)

    dp[0, 0] = 0.0

    for i in range(1, m + 1):
        dp[i, 0] = dp[i - 1, 0] + cfg.gap_cost

    for j in range(1, n + 1):
        dp[0, j] = dp[0, j - 1] + cfg.gap_cost

    for i in range(1, m + 1):
        ai = a[i - 1]

        for j in range(1, n + 1):
            bj = b[j - 1]

            deletion = dp[i - 1, j] + cfg.gap_cost

            insertion = dp[i, j - 1] + cfg.gap_cost

            substitution = (
                dp[i - 1, j - 1]
                + command_substitution_cost(ai, bj, cfg)
            )

            dp[i, j] = min(
                deletion,
                insertion,
                substitution,
            )

    # Normalize by maximum sequence length so the result is
    # comparable across CADs of different complexity.
    normalized = dp[m, n] / max(m, n)

    return float(np.clip(normalized, 0.0, 1.0))


def cad_similarity(
    cad_a: np.ndarray,
    cad_b: np.ndarray,
    cfg: CADSimilarityConfig = DEFAULT_CONFIG,
) -> float:
    """
    Convert normalized CAD distance to [0, 1] similarity.

    Identical CAD:
        1.0

    Increasingly different CADs:
        -> 0
    """
    distance = cad_distance(cad_a, cad_b, cfg)

    similarity = math.exp(
        -cfg.similarity_lambda * distance
    )

    return float(np.clip(similarity, 0.0, 1.0))


# ============================================================
# H5 convenience wrapper
# ============================================================

def cad_file_similarity(
    path_a: str | Path,
    path_b: str | Path,
    cfg: CADSimilarityConfig = DEFAULT_CONFIG,
) -> float:
    """
    Directly compare two .h5 CAD files.
    """
    a = load_cad_vector(path_a)
    b = load_cad_vector(path_b)

    return cad_similarity(a, b, cfg)


# ============================================================
# Batch retrieval relevance
# ============================================================

def soft_recall_score(
    gt_vec: np.ndarray,
    retrieved_vecs: list[np.ndarray],
    k: int,
    cfg: CADSimilarityConfig = DEFAULT_CONFIG,
) -> float:
    """
    Similarity-aware Recall@K for ONE query.

    Returns the maximum CAD similarity among the top-k
    retrieved CADs.

    Thus:

        exact GT retrieved -> 1.0

        very similar CAD retrieved -> e.g. 0.91

        only weakly similar CADs -> e.g. 0.25
    """
    top_k = retrieved_vecs[:k]

    if not top_k:
        return 0.0

    similarities = [
        cad_similarity(gt_vec, candidate, cfg)
        for candidate in top_k
    ]

    return float(max(similarities))


def threshold_recall_score(
    gt_vec: np.ndarray,
    retrieved_vecs: list[np.ndarray],
    k: int,
    threshold: float,
    cfg: CADSimilarityConfig = DEFAULT_CONFIG,
) -> float:
    """
    Binary similarity-aware Recall@K.

    Returns 1 if at least one of the top-k retrieved CADs
    has similarity >= threshold.
    """
    top_k = retrieved_vecs[:k]

    for candidate in top_k:
        if cad_similarity(gt_vec, candidate, cfg) >= threshold:
            return 1.0

    return 0.0


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Similarity between GenCAD CAD-vector files"
    )

    parser.add_argument(
        "cad_a",
        type=Path,
    )

    parser.add_argument(
        "cad_b",
        type=Path,
    )

    parser.add_argument(
        "--lambda",
        dest="similarity_lambda",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--gap-cost",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    cfg = CADSimilarityConfig(
        gap_cost=args.gap_cost,
        similarity_lambda=args.similarity_lambda,
    )

    vec_a = load_cad_vector(args.cad_a)
    vec_b = load_cad_vector(args.cad_b)

    distance = cad_distance(vec_a, vec_b, cfg)
    similarity = cad_similarity(vec_a, vec_b, cfg)

    print(f"A: {args.cad_a}")
    print(f"B: {args.cad_b}")
    print()
    print(f"CAD distance:   {distance:.6f}")
    print(f"CAD similarity: {similarity:.6f}")


if __name__ == "__main__":
    main()

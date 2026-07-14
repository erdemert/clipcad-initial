ALL_COMMANDS = ["Line", "Arc", "Circle", "EOS", "SOL", "Ext"]
LINE_IDX, ARC_IDX, CIRCLE_IDX, EOS_IDX, SOL_IDX, EXT_IDX = range(len(ALL_COMMANDS))

PAD_VAL = -1

N_ARGS_SKETCH = 5    # x, y, alpha, f, r
N_ARGS_PLANE = 3     # theta, phi, gamma
N_ARGS_TRANS = 4     # p_x, p_y, p_z, s
N_ARGS_EXT_PARAM = 4  # e1, e2, b, u
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_EXT_PARAM
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT

ARGS_DIM = 256  # quantization levels used for each arg value
DEFAULT_MAX_LEN = 64  # sequences observed so far max out well under this

EOS_VEC = [EOS_IDX] + [PAD_VAL] * N_ARGS


def split_command_args(vec):
    """Split a raw (seq_len, 1 + N_ARGS) command vector into command indices and args."""
    return vec[:, 0], vec[:, 1:]


def pad_vec(vec, max_len=DEFAULT_MAX_LEN):
    """Right-pad a (seq_len, 1 + N_ARGS) command vector with EOS rows up to max_len."""
    import numpy as np

    pad_len = max_len - vec.shape[0]
    if pad_len <= 0:
        return vec[:max_len]
    pad_rows = np.tile(np.array(EOS_VEC, dtype=vec.dtype), (pad_len, 1))
    return np.concatenate([vec, pad_rows], axis=0)

import numpy as np
from numba import float32, types
from numba.typed import Dict


def to_typed_f32_dict(src=None, *, strict=False):
    """
    Convert a Python dict-like object to:
      Dict[unicode_type, float32]

    Parameters
    ----------
    src : mapping | None
        Input dict (e.g., {"SI": 0.0008, "Gb": 110}).
    strict : bool
        If True, raise on non-numeric values; otherwise skip them.

    Returns
    -------
    numba.typed.Dict
    """
    out = Dict.empty(key_type=types.unicode_type, value_type=float32)
    if src is None:
        return out

    for k, v in src.items():
        try:
            out[str(k)] = np.float32(v)
        except Exception:
            if strict:
                raise TypeError(f"Cannot cast key={k!r}, value={v!r} to float32")
            # non-strict: ignore invalid entries
    return out

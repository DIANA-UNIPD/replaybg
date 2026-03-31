import numpy as np
from numba import float32, types
from numba.typed import Dict


def to_typed_f32_dict(src=None, *, strict=False) -> Dict:
    """Convert a mapping to a Numba typed dictionary with float32 values.

    This helper normalizes a Python dict-like object into a
    ``numba.typed.Dict`` with keys of type ``unicode_type`` and values of type
    ``float32``. It is useful when passing parameter dictionaries into Numba-
    compiled code or APIs that expect typed containers.

    Args:
        src: A mapping or mapping-like object to convert. If ``None``, an empty
            typed dictionary is returned.
        strict: If ``True``, raise ``TypeError`` when a value cannot be cast to
            ``float32``. If ``False``, invalid entries are skipped silently.

    Returns:
        numba.typed.Dict: A typed dictionary with ``str`` keys and ``float32``
        values.

    Raises:
        TypeError: If ``strict`` is ``True`` and a key/value pair cannot be
            converted to ``str``/``float32``.
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
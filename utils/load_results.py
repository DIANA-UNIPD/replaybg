import os
import pickle


def load_results(path: str, name: str, prefix: str) -> dict:
    """Load results previously pickled by :func:`save_results`.

    Args:
        path: Directory where the file lives.
        name: File name (with or without the ``.pkl`` extension), matching the
            ``name`` passed to :func:`save_results`.
        prefix: Prefix used when the file was saved (e.g. ``"twin"`` or
            ``"replay"``).

    Returns:
        dict: The unpickled results dictionary.
    """
    if not name.endswith('.pkl'):
        name += '.pkl'
    name = prefix + '_' + name
    file_path = os.path.join(path, name)
    with open(file_path, 'rb') as f:
        return pickle.load(f)

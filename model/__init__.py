from numba import float32
from numba.typed import Dict
from numba import types


class Model:
    def __init__(self, x0, theta=Dict.empty(
        key_type=types.unicode_type,
        value_type=float32,
    )):
        pass

    def step(self, u):
        pass

    def output(self):
        pass

from numba import njit
from numba.experimental import jitclass

DEBUG = True

def identity(x=None, **kwargs):
    if callable(x):
        return x
    def wrapper(func_or_cls):
        return func_or_cls
    return wrapper

njit_ = identity if DEBUG else njit
jitclass_ = identity if DEBUG else jitclass
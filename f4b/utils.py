import time
from contextlib import contextmanager

import numpy as np


class Timer:
    """Accumulates wall times per label."""

    def __init__(self):
        self.t = {}

    @contextmanager
    def __call__(self, label):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[label] = self.t.get(label, 0.0) + time.perf_counter() - t0

    def __repr__(self):
        return "\n".join(f"{k:>20s}: {v:8.2f} s" for k, v in self.t.items())


def symmetrize(m):
    return 0.5 * (m + m.T)


def log(verbose, level, *args):
    if verbose >= level:
        print(*args, flush=True)


def swap_spin(Da, Db):
    return Db, Da


def as_float_tuple(x):
    return tuple(float(v) for v in np.atleast_1d(x))

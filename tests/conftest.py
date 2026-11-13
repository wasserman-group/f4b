import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def h2():
    from f4b import fgto
    return fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="cc-pvdz", fragments={0: [0], 1: [1]},
                  ensembles={"spin": [0, 1]})


@pytest.fixture(scope="session")
def h2_ks(h2):
    """KS-PBE reference for H2 on the same grid level as f4b."""
    from pyscf import dft
    ks = dft.RKS(h2.mol)
    ks.xc = "PBE"
    ks.grids.level = 6
    ks.kernel()
    return ks


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slower tests (deselect with -m 'not slow')")

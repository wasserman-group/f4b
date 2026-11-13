"""frag_symm: symmetry-equivalent fragments, evaluation grid."""
import numpy as np
import pytest

from f4b import fgto, pdft
from f4b.fragment import FragmentSystem
from f4b.symmetry import FragSymmetry

WATER = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
WATER_ENS = {"charge": {0: {"charges": (0,), "weights": (1.0,), "spins": (2,)}}, "spin": [0, 1, 2]}


def test_detection_h2_and_h2o():
    m = fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="cc-pvdz", fragments={0: [0], 1: [1]}, ensembles={"spin": [0, 1]})
    s = FragSymmetry(m)
    assert len(s.ops) == 2 and s.reps == [0] and s.orbits[1][0] == 0
    w = fgto.M(atom=WATER, basis="sto-3g", fragments={0: [0], 1: [1], 2: [2]}, ensembles=WATER_ENS)
    s = FragSymmetry(w)
    assert len(s.ops) == 4 and s.reps == [0, 1] and s.orbits[2][0] == 1
    S = w.mol.intor("int1e_ovlp")
    for g in s.ops:
        assert np.allclose(g["U"].T @ S @ g["U"], S, atol=1e-8)
    V = np.random.RandomState(1).normal(size=S.shape)
    Vs = s.project(V + V.T)
    assert all(np.allclose(g["U"].T @ Vs @ g["U"], Vs, atol=1e-10) for g in s.ops)


def test_ensemble_mismatch_breaks_symmetry():
    """H and H+ fragments of H3+ (different members) are not equivalent: only the H-H operations remain."""
    m = fgto.M(atom="H 0 0 0; H 0 0 0.9; H 0.7794 0 0.45", basis="sto-3g", fragments={0: [0], 1: [1], 2: [2]}, charge=1,
               fcharge={2: 1}, ensembles={"spin": [0, 1]})
    s = FragSymmetry(m)
    assert len(s.ops) == 4 and s.orbits[1][0] == 0 and s.orbits[2][0] == 2


def test_h2_dimer_symmetry_same_result(h2):
    a = pdft.RKS(h2, method="inversion", verbose=0, maxiter=30, inv_reg=1e-4)
    a.kernel()
    b = pdft.RKS(h2, method="inversion", verbose=0, maxiter=30, inv_reg=1e-4, frag_symm="auto")
    b.kernel()
    assert b.sym is not None and b.frags.sym is b.sym
    assert b.e_tot == pytest.approx(a.e_tot, abs=3e-5)
    assert np.allclose(b.Da, a.Da, atol=2e-3)
    assert np.allclose(b.frag_result.Da[1], a.frag_result.Da[1], atol=2e-3)


@pytest.mark.slow
def test_water_h_atoms_equivalent():
    """For a C2v-invariant Vp the two H atom densities are mirror images (O, H, H fragments)."""
    m = fgto.M(atom=WATER, basis="cc-pvdz", fragments={0: [0], 1: [1], 2: [2]}, ensembles=WATER_ENS)
    mf = pdft.RKS(m, method="inversion", verbose=0, maxiter=2, lam=0.2, frag_symm="auto")
    mf.kernel()
    Vp = mf.Vp
    assert max(np.abs(g["U"].T @ Vp @ g["U"] - Vp).max() for g in mf.sym.ops) < 1e-10
    fs = FragmentSystem(m, fxc="PBE", grid_level=6, grid=(mf.grid.coords, mf.grid.weights))
    r = fs.solve(Vp)                                         # independent solves of all fragments
    rm = mf.frag_result
    assert np.allclose(rm.Da[2], r.Da[2], atol=1e-7) and np.allclose(rm.Da[1], r.Da[1], atol=1e-7)
    assert r.E[1] == pytest.approx(r.E[2], abs=1e-8)
    assert rm.E[2] == pytest.approx(r.E[2], abs=1e-7)


def test_eval_grid(h2, h2_ks):
    mf = pdft.RKS(h2, method="reference", verbose=0, ref_dm=h2_ks.make_rdm1(), maxiter=10, step=5.0)
    mf.kernel()
    from pyscf.dft import gen_grid
    g = gen_grid.Grids(h2.mol)
    g.level = 8
    g.build()
    assert mf.L1() == pytest.approx(mf.L1(grid=(g.coords, g.weights)), abs=2e-4)
    assert np.dot(g.weights, mf.eval_density(g.coords)) == pytest.approx(2.0, abs=1e-4)
    v = mf.eval_Vp(g.coords[:50])
    assert v.shape == (50,) and np.all(np.isfinite(v))

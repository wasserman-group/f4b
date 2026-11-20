"""Saving a result and resuming from the stored partition potential and fragment densities."""
import numpy as np
import pytest

from f4b import fgto, load_state, pdft
from test_config import hf_mol

KW = dict(grid_level=3, verbose=0)


def run(mol, method, **kw):
    mf = pdft.RKS(mol, method=method, **{**KW, **kw})
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def h2_ref(h2):
    from pyscf import scf
    return scf.RHF(h2.mol).run().make_rdm1()


@pytest.fixture(scope="module")
def saved(h2, h2_ref, tmp_path_factory):
    mf = run(h2, "reference", ref_dm=h2_ref, step=5.0, maxiter=6)
    path = mf.save(tmp_path_factory.mktemp("restart") / "state")
    return mf, path


def test_save_adds_extension_and_roundtrips(saved):
    mf, path = saved
    assert path.endswith(".npz")
    st = load_state(path)
    assert np.allclose(st["Vp"], mf.Vp)
    assert np.allclose(st["frag_dm"][0][0], mf.make_rdm1(0)[0])
    assert st["info"]["method"] == "reference" and st["info"]["niter"] == mf.niter
    assert len(st["members"]) == 2            # one spin-flip member per H fragment


def test_resume_starts_where_the_run_stopped(h2, h2_ref, saved):
    mf, path = saved
    best = min(h["L1"] for h in mf.history)
    res = run(h2, "reference", ref_dm=h2_ref, step=5.0, maxiter=3, restart=path)
    assert res.history[0]["L1"] == pytest.approx(best, abs=1e-6)
    assert res.L1() <= best + 1e-9
    fresh = run(h2, "reference", ref_dm=h2_ref, step=5.0, maxiter=1)
    assert res.history[0]["L1"] < fresh.history[0]["L1"]


def test_vp0_and_dm0_from_file_work_with_every_method(h2, saved):
    _, path = saved
    for method in ("functional-T", "inversion"):
        mf = run(h2, method, Vp0=path, dm0=path, maxiter=1)
        assert np.isfinite(mf.e_tot)


def test_dm0_arrays_and_vp0_arrays(h2, saved):
    mf, _ = saved
    dm0 = {k: mf.make_rdm1(k) for k in (0, 1)}
    res = run(h2, "functional-T", Vp0=mf.Vp, dm0=dm0, maxiter=1)
    assert res.Vp.shape == mf.Vp.shape and np.allclose(res.Vp, mf.Vp)      # maxiter=1: the best iterate is Vp0 itself
    res2 = run(h2, "functional-T", dm0=[mf.make_rdm1(0).sum(0), mf.make_rdm1(1).sum(0)], maxiter=1)   # spin-summed
    assert np.isfinite(res2.e_tot)


def test_density_guess_reaches_the_converged_scf_faster(h2, saved, monkeypatch):
    mf, _ = saved
    from f4b.fragment import FragmentSystem, VpUKS
    calls = [0]
    orig = VpUKS.get_veff

    def counting(self, *a, **k):
        calls[0] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(VpUKS, "get_veff", counting)
    cold = FragmentSystem(h2, grid_level=3).solve(mf.Vp)
    n_cold, calls[0] = calls[0], 0
    fs = FragmentSystem(h2, grid_level=3)
    fs.guess_frags = {k: np.array([cold.Da[k], cold.Db[k]]) for k in (0, 1)}
    warm = fs.solve(mf.Vp)
    assert calls[0] < n_cold
    assert warm.Ef == pytest.approx(cold.Ef, abs=1e-8)


def test_fractional_charge_fragment_densities_are_rescaled_per_member(saved):
    ens = {"charge": {0: {"charges": (0, -1), "weights": (0.3, 0.7)}, 1: {"charges": (None, 0), "weights": (0.7, 0.3)}}}
    mol = fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="cc-pvdz", fragments={0: [0], 1: [1]}, ensembles=ens)
    first = run(mol, "functional-T", maxiter=1)
    dm0 = [first.make_rdm1(0), first.make_rdm1(1)]            # fractional electron numbers (1.7 and 0.3)
    again = run(mol, "functional-T", dm0=dm0, maxiter=1)
    assert again.e_f == pytest.approx(first.e_f, abs=1e-7)


def test_wrong_shapes_and_fragments_raise(h2, saved):
    _, path = saved
    with pytest.raises(ValueError, match="fragment 5"):
        run(h2, "functional-T", dm0={5: np.zeros((2, 10, 10))}, maxiter=1)
    with pytest.raises(ValueError, match="shape"):
        run(h2, "functional-T", dm0={0: np.zeros((2, 3, 3))}, maxiter=1)
    with pytest.raises(ValueError, match="Vp0 has shape"):
        run(h2, "functional-T", Vp0=np.zeros((3, 3)), maxiter=1)
    other = fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0], 1: [1]}, ensembles={"spin": [0, 1]})
    with pytest.raises(ValueError, match="same basis"):
        run(other, "functional-T", restart=path, maxiter=1)


def test_configuration_weights_are_restored(tmp_path):
    mol = hf_mol("sto-3g")
    a = run(mol, "inversion", maxiter=2, optimize_omega=True)
    om = a._omega_get().copy()
    path = a.save(tmp_path / "hf")
    assert np.allclose(load_state(path)["omega"], om)
    b = pdft.RKS(mol, method="inversion", restart=path, **KW).build()
    for m in (m for ms in b._cfg_groups.values() for m in ms):
        m.omega = 1.0 / 3
        m.refresh()
    b._load_start()
    assert np.allclose(b._omega_get(), om)

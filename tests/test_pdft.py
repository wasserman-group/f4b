"""Fast end-to-end checks (cc-pVDZ H2, H2+).  Tolerances reflect the finite basis (see README)."""
import numpy as np
import pytest

from f4b import fgto, pdft


def run(mol, method, **kw):
    mf = pdft.RKS(mol, method=method, verbose=0, **kw)
    mf.kernel()
    return mf


def test_fragments_vp_zero_and_shift(h2):
    """Vp = 0 gives two spin-averaged H atoms; Vp = c*S leaves E_f and the densities unchanged."""
    from pyscf import gto, dft
    from f4b.fragment import FragmentSystem
    fs = FragmentSystem(h2, grid_level=6)
    nao = h2.mol.nao
    r0 = fs.solve(np.zeros((nao, nao)))
    atom = gto.M(atom="H 0 0 0", basis="cc-pvdz", spin=1, verbose=0)
    ks = dft.UKS(atom)
    ks.xc = "PBE"
    ks.grids.level = 6
    ks.kernel()
    assert r0.Ef == pytest.approx(2 * ks.e_tot, abs=2e-3)       # dimer basis lowers it slightly
    S = h2.mol.intor("int1e_ovlp")
    r1 = fs.solve(0.3 * S)
    assert r1.Ef == pytest.approx(r0.Ef, abs=1e-8)
    assert np.allclose(r1.Dfa, r0.Dfa, atol=1e-6)


def test_inversion_matches_ks(h2, h2_ks):
    mf = run(h2, "inversion", maxiter=30, inv_reg=1e-4)     # small reg: best accuracy for tiny systems
    assert mf.e_tot == pytest.approx(h2_ks.e_tot, abs=5e-5)
    assert mf.Ep < 0


def test_functional_matches_ks(h2, h2_ks):
    mf = run(h2, "functional-T", maxiter=30)
    assert mf.e_tot == pytest.approx(h2_ks.e_tot, abs=2e-3)
    assert mf.scf_summary["e_functional"] == pytest.approx(h2_ks.e_tot, abs=1e-2)


def test_reference_reduces_l1(h2, h2_ks):
    mf = run(h2, "reference", ref_dm=h2_ks.make_rdm1(), maxiter=40, step=5.0)
    assert mf.L1() < 0.06
    assert mf.e_tot == pytest.approx(h2_ks.e_tot, abs=3e-3)


def test_reference_residual_vanishes_at_reference(h2, h2_ks):
    mf = pdft.RKS(h2, method="reference", ref_dm=h2_ks.make_rdm1(), verbose=0)
    mf.build()
    D = h2_ks.make_rdm1() / 2
    dVp, info = mf._dVp_reference(D, D)
    assert np.abs(dVp).max() < 1e-10
    assert mf._L1_ref(D, D) < 1e-10


def test_h2plus_charge_ensemble_inversion():
    """One-electron system: PDFT with charge ensembles reproduces the exact (HF) energy for the PBE-free case
    only approximately, so we compare with UKS-PBE (self-interaction is shared by both)."""
    from pyscf import dft
    mol = fgto.M(atom="H 0 0 0; H 0 0 2.0", unit="bohr", basis="cc-pvdz", charge=1, spin=1,
                 fragments={0: [0], 1: [1]},
                 ensembles={"charge": {0: {"charges": (0, 1), "weights": (0.5, 0.5)}, 1: {"charges": (0, 1), "weights": (0.5, 0.5)}}})
    ks = dft.UKS(mol.mol)
    ks.xc = "PBE"
    ks.grids.level = 6
    ks.kernel()
    mf = run(mol, "inversion", maxiter=30, one_electron=False)
    assert mf.e_tot == pytest.approx(ks.e_tot, abs=5e-4)
    assert mf.fmol.frags[0].nelec_avg == pytest.approx(0.5)


def test_h2plus_one_electron_mode_is_exact():
    """One electron (auto mode): UHF fragments, dVp = vW' + Vext, E = Ts_vW + Eext + Enn -> exact (UHF) energy, free of the
    PBE self-interaction error (6.7 mEh at this geometry)."""
    from pyscf import scf
    ens = {"charge": {i: {"charges": (0, None), "weights": (0.5, 0.5)} for i in (0, 1)}}
    mol = fgto.M(atom=[["H", (-1.0, 0, 0)], ["H", (1.0, 0, 0)]], unit="bohr", basis="cc-pvqz", charge=1, spin=1,
                 fragments={0: [0], 1: [1]}, ensembles=ens)
    uhf = scf.UHF(mol.mol).run()
    mf = run(mol, "functional", accel="diis", maxiter=60)
    assert mf._1e
    assert mf.e_tot == pytest.approx(uhf.e_tot, abs=1e-4)
    with pytest.raises(ValueError):
        pdft.RKS(fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="cc-pvdz", fragments={0: [0], 1: [1]}), one_electron=True).build()


def test_unknown_option_and_method(h2):
    with pytest.raises(TypeError):
        pdft.RKS(h2, bogus=1)
    with pytest.raises(ValueError):
        pdft.RKS(h2, method="nonsense")
    with pytest.raises(ValueError):
        pdft.RKS(h2, frag_symm="reflection")


@pytest.mark.slow
def test_lih_charge_spin_ensembles_inversion():
    """LiH, R=3.014 bohr: Li 3e/4e and H 0e/1e charge ensembles + spin ensembles; E_total approaches KS-PBE."""
    from pyscf import dft
    mol = fgto.M(atom="Li 0 0 0; H 0 0 3.014", unit="bohr", basis="cc-pvdz", fragments={0: [0], 1: [1]},
                 ensembles={"charge": {0: {"charges": (0, -1), "weights": (0.56826, 0.43174)},
                                       1: {"charges": (1, 0), "weights": (0.43174, 0.56826)}},
                            "spin": [0, 1]})
    assert sum(f.nelec_avg for f in mol.frags) == pytest.approx(4.0)
    ks = dft.RKS(mol.mol)
    ks.xc = "PBE"
    ks.grids.level = 6
    ks.kernel()
    mf = run(mol, "inversion", maxiter=25)
    # the loop does not converge in 25 iterations and inversion noise (~1e-5, not reproducible run to run) is amplified:
    # observed E - E_KS between 0 and 4e-2
    assert mf.e_tot == pytest.approx(ks.e_tot, abs=5e-2)
    assert mf.Ep < 0


@pytest.mark.slow
def test_h3plus_three_fragments_inversion():
    """N_f = 3: H3+ with three H fragments (0e weight 1/3, 1e weight 2/3 with spin ensembles)."""
    from pyscf import dft
    s1, s2 = 1.7 / np.sqrt(3), 0.85
    s3 = s1 / 2
    atom = f"H {s1} 0 0; H {-s3} {s2} 0; H {-s3} {-s2} 0"
    ens = {"charge": {i: {"charges": (1, 0), "weights": (1 / 3, 2 / 3)} for i in range(3)}, "spin": [0, 1, 2]}
    mol = fgto.M(atom=atom, unit="bohr", basis="cc-pvdz", charge=1, fragments={0: [0], 1: [1], 2: [2]}, ensembles=ens)
    ks = dft.RKS(mol.mol)
    ks.xc = "PBE"
    ks.grids.level = 6
    ks.kernel()
    mf = run(mol, "inversion", inv_reg=1e-4, maxiter=30)
    assert mf.e_tot == pytest.approx(ks.e_tot, abs=2e-3)


def test_ensemble_electron_numbers_from_integrated_densities():
    """H2 with fractional charges (left 1.7 e, right 0.3 e): w_l = 0.3 on the 1e geometry (split 0.15/0.15 by the spin
    ensemble) and 1 - w_l on the 2e geometry; integrated fragment densities carry 1.7 / 0.3 electrons."""
    from f4b.fragment import FragmentSystem
    wl = 0.3
    mol = fgto.M(atom="H -0.7 0 0; H 0.7 0 0", unit="bohr", basis="cc-pvdz", fragments={0: [0], 1: [1]},
                 ensembles={"charge": {0: {"charges": (0, -1), "weights": (wl, 1 - wl)},
                                       1: {"charges": (1, 0), "weights": (1 - wl, wl)}},
                            "spin": [0, 1]})
    S = mol.mol.intor("int1e_ovlp")
    fs = FragmentSystem(mol, grid_level=4)
    nao = mol.mol.nao
    for Vp in (np.zeros((nao, nao)), 0.2 * S):
        r = fs.solve(Vp)
        assert np.einsum("ij,ij", r.Da[0] + r.Db[0], S) == pytest.approx(1.7, abs=1e-8)
        assert np.einsum("ij,ij", r.Da[1] + r.Db[1], S) == pytest.approx(0.3, abs=1e-8)
    # a wrong weight (here: the 1e member of fragment 1 with the weight of a different member) is caught
    mol.frags[0].members[0].weight *= 0.5
    with pytest.raises(RuntimeError, match="fragment 0"):
        fs.assemble(r)
    with pytest.raises(ValueError):                  # weights that do not sum to one are rejected at build
        fgto.M(atom="H -0.7 0 0; H 0.7 0 0", unit="bohr", basis="cc-pvdz", fragments={0: [0], 1: [1]},
               ensembles={"charge": {0: {"charges": (0, -1), "weights": (0.35, 0.35)},
                                     1: {"charges": (1, 0), "weights": (0.7, 0.3)}}, "spin": [0, 1]})


def test_spin_channel_counts_may_differ_from_supermolecule():
    """Only the total electron number is checked against the supermolecule: a charge ensemble without a spin ensemble
    (fragment spin channels 1.3/0.7) runs, and its inversion target equals the one of the spin-ensemble run."""
    from f4b.fragment import FragmentSystem
    ch = {0: {"charges": (0, -1), "weights": (0.3, 0.7)}, 1: {"charges": (None, 0), "weights": (0.7, 0.3)}}
    targets = []
    for ens in ({"charge": ch}, {"charge": ch, "spin": [0, 1]}):
        mol = fgto.M(atom="H -0.7 0 0; H 0.7 0 0", unit="bohr", basis="cc-pvdz", fragments={0: [0], 1: [1]}, ensembles=ens)
        mf = pdft.RKS(mol, method="inversion", verbose=0, maxiter=2, inv_reg=1e-4)
        S = mol.mol.intor("int1e_ovlp")
        r = FragmentSystem(mol, grid_level=4).solve(0.2 * S)
        Da, Db = r.Dfa, r.Dfb
        if "spin" not in ens:
            assert np.einsum("ij,ij", Da, S) == pytest.approx(1.3, abs=1e-6)
            assert np.einsum("ij,ij", Db, S) == pytest.approx(0.7, abs=1e-6)
        mf.kernel()                                       # used to raise ValueError for the unequal spin channels
        ta, tb = mf._target(Da, Db)
        assert np.einsum("ij,ij", ta, S) == pytest.approx(1.0, abs=1e-6)
        assert np.einsum("ij,ij", tb, S) == pytest.approx(1.0, abs=1e-6)
        targets.append(ta + tb)
    assert np.allclose(targets[0], targets[1], atol=1e-8)
    with pytest.raises(ValueError, match="in total"):             # a wrong total is still rejected
        mf._target(r.Dfa + 0.5 * np.linalg.inv(S), r.Dfb)


def test_none_means_zero_electrons():
    kw = dict(atom="Li 0 0 0; H 0 0 3.014", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]})
    a = fgto.M(ensembles={"charge": {0: {"charges": (0, -1), "weights": (0.56826, 0.43174)},
                                     1: {"charges": (None, 0), "weights": (0.43174, 0.56826)}}, "spin": [0, 1]}, **kw)
    b = fgto.M(ensembles={"charge": {0: {"charges": (0, -1), "weights": (0.56826, 0.43174)},
                                     1: {"charges": (1, 0), "weights": (0.43174, 0.56826)}}, "spin": [0, 1]}, **kw)
    assert repr(a) == repr(b)
    assert (a.frags[1].members[0].na, a.frags[1].members[0].nb) == (0, 0)

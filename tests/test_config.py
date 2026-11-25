"""Configuration ensembles (constrained-DFT states): HF with F 2p-hole configurations."""
import numpy as np
import pytest

from f4b import fgto, pdft

NHY = 0.8
STATES = [(("F 2p" + k, "beta", 0.0),) for k in "xyz"]


def hf_mol(basis="6-31g", nhy=NHY):
    return fgto.M(atom="F 0 0 0; H 0 0 1.7328", unit="bohr", basis=basis, fragments={0: [0], 1: [1]},
                  ensembles={"charge": {0: {"charges": (0, -1), "weights": (nhy, 1 - nhy), "spins": (1, 0)},
                                        1: {"charges": (0, 1), "weights": (nhy, 1 - nhy), "spins": (1, 0)}},
                             "config": {0: {"charge": 0, "states": STATES}},
                             "spin": [0, 1]})


def ks_density(mol):
    from pyscf import dft
    ks = dft.RKS(mol.mol)
    ks.xc = "PBE"
    ks.grids.level = 6
    ks.kernel()
    return ks


def test_config_members_and_weights():
    mol = hf_mol("sto-3g")
    f1 = mol.frags[0]
    cfg = [m for m in f1.members if m.config is not None]
    assert len(cfg) == 3
    assert sum(m.weight + m.flip for m in cfg) == pytest.approx(NHY)       # 9-electron configurations
    assert sum(m.omega for m in cfg) == pytest.approx(1.0)
    assert f1.nelec_avg == pytest.approx(9 * NHY + 10 * (1 - NHY))
    assert sum(f.nelec_avg for f in mol.frags) == pytest.approx(10.0)
    assert "config" in repr(mol)


def test_constraint_must_select_one_ao():
    with pytest.raises(ValueError):
        fgto.M(atom="F 0 0 0; H 0 0 1.7328", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]},
               ensembles={"charge": {0: {"charges": (0,), "weights": (1.0,), "spins": (1,)}},
                          "config": {0: {"charge": 0, "states": [(("F 2p", "beta", 0.0),)]}},
                          "spin": [0, 1]})


def test_cdft_hole_populations():
    """Each configuration state has the constrained beta 2p population at zero and an integer electron count."""
    from f4b.fragment import FragmentSystem
    mol = hf_mol("sto-3g")
    fs = FragmentSystem(mol, fxc="PBE", grid_level=4)
    r = fs.solve(np.zeros((mol.mol.nao,) * 2))
    S = mol.mol.intor("int1e_ovlp")
    n = 0
    for it in r.raw[0]:
        m = it["m"]
        if m.config is None:
            continue
        n += 1
        pops = m.config.populations((it["Da"], it["Db"]))
        assert np.abs(pops).max() < 1e-4
        assert np.trace(it["Da"] @ S) == pytest.approx(5.0, abs=1e-6)
        assert np.trace(it["Db"] @ S) == pytest.approx(4.0, abs=1e-6)
    assert n == 3
    assert sum(np.trace((a + b) @ S) for a, b in zip(r.Da, r.Db)) == pytest.approx(10.0, abs=1e-6)


@pytest.mark.parametrize("method", ["reference", "inversion"])
def test_omega_optimization_lowers_objective(method):
    """One fragment solve (Vp = 0): optimizing omega lowers L1 (reference) / the dVp norm (inversion)."""
    mol = hf_mol("6-31g")
    kw = {}
    if method == "reference":
        kw["ref_dm"] = ks_density(mol).make_rdm1()
    mf = pdft.RKS(mol, method=method, verbose=0, optimize_omega=False, **kw)
    mf.build()
    mf._Sinv = np.linalg.inv(mf.S)
    fr = mf.frags.solve(np.zeros_like(mf.S))
    x0 = mf._omega_get()
    f0 = mf._omega_objective(x0, fr)
    mf._optimize_omega(fr)
    x1 = mf._omega_get()
    f1 = mf._omega_objective(x1, fr)
    assert f1 < f0 - 1e-6
    assert x1.sum() == pytest.approx(1.0) and x1.min() >= -1e-12
    # the optimized weights give a consistent electron count
    S = mf.S
    assert np.einsum("ij,ij", fr.Dfa + fr.Dfb, S) == pytest.approx(10.0, abs=1e-6)


@pytest.mark.slow
def test_hf_reference_6_31g():
    """HF (R = 1.7328 bohr), Reference method, configuration weights optimized: L1 drops, energy approaches KS."""
    mol = hf_mol("6-31g")
    ks = ks_density(mol)
    mf = pdft.RKS(mol, method="reference", ref_dm=ks.make_rdm1(), step=5.0, maxiter=40, verbose=0)
    mf.kernel()
    assert mf.history[0]["L1"] > 0.5
    assert mf.L1() < 0.15
    assert mf.e_tot == pytest.approx(ks.e_tot, abs=1e-2)
    w = mf.omega[(0, 0)]
    assert w.sum() == pytest.approx(1.0) and w[2] > w[0]          # sigma-hole configuration dominates


def o2(states, fragments=None, **kw):
    ens = {"charge": {0: {"charges": (0,), "weights": (1.0,), "spins": (2,)}},
           "config": {0: {"charge": 0, "states": states}}}
    return fgto.M(atom="O 0 0 0; O 0 0 2.28", unit="bohr", basis="sto-3g", fragments=fragments or {0: [0], 1: [1]},
                  ensembles=ens, **kw)


def test_ao_label_is_resolved_within_the_fragment():
    """'O 2px' also matches the other O atom of O2; the constraint must pick the AO of the fragment's own atom."""
    mol = o2(tuple(((f"O 2p{k}", "beta", 0.0),) for k in "xyz"))
    idx = [m.config.constraints[0].index for m in mol.frags[0].members]
    assert idx == [2, 3, 4]                                    # 2px 2py 2pz of atom 0 (sto-3g AOs 0-4: 1s 2s 2px 2py 2pz)
    ens = {"charge": {1: {"charges": (0,), "weights": (1.0,), "spins": (2,)}},
           "config": {1: {"charge": 0, "states": ((("O 2px", "beta", 0.0),),)}}}
    mol1 = fgto.M(atom="O 0 0 0; O 0 0 2.28", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]}, ensembles=ens)
    assert mol1.frags[1].members[0].config.constraints[0].index == 7        # second O, not the first


def test_ao_label_on_another_fragment_raises():
    with pytest.raises(ValueError, match="not on the atoms"):
        o2(((("1 O 2px", "beta", 0.0),),))


def test_ambiguous_label_without_fragment_still_raises():
    with pytest.raises(ValueError, match="exactly one"):
        o2(((("O 2p", "beta", 0.0),),))


def test_full_occupation_and_hole_constraints_are_solved():
    """A beta population of 1 (attractive multiplier) is reached like a hole; two holes select the doubly occupied p."""
    from f4b.fragment import FragmentSystem
    one = ((("O 2px", "beta", 1.0),),)
    two = ((("O 2py", "beta", 0.0), ("O 2pz", "beta", 0.0)),)
    for states, target in [(one, 1.0), (two, 0.0)]:
        mol = o2(states)
        r = FragmentSystem(mol, grid_level=3).solve(np.zeros((mol.mol.nao,) * 2))
        it = r.raw[0][0]
        pops = it["m"].config.populations((it["Da"], it["Db"]))
        assert np.allclose(pops, target, atol=2e-4)


def test_same_density_states_trigger_a_warning():
    """For the triplet O atom a single beta hole is ambiguous: two of the three states coincide."""
    from f4b.fragment import FragmentSystem
    mol = o2(tuple(((f"O 2p{k}", "beta", 0.0),) for k in "xyz"))
    with pytest.warns(UserWarning, match="same density"):
        FragmentSystem(mol, grid_level=3).solve(np.zeros((mol.mol.nao,) * 2))


def test_configuration_ensembles_for_two_charges_of_one_fragment():
    """HF with F between F+ (2p^4: two beta holes) and F (2p^5: one beta hole), each with its own configuration group."""
    nhy = 0.8
    one = tuple(((f"F 2p{k}", "beta", 0.0),) for k in "xyz")
    two = tuple(((f"F 2p{a}", "beta", 0.0), (f"F 2p{b}", "beta", 0.0)) for a, b in (("y", "z"), ("x", "z"), ("x", "y")))
    ens = {"charge": {0: {"charges": (1, 0), "weights": (1 - nhy, nhy), "spins": (2, 1)},
                      1: {"charges": (-1, 0), "weights": (1 - nhy, nhy), "spins": (0, 1)}},
           "config": {0: [{"charge": 1, "states": two}, {"charge": 0, "states": one}]}}
    mol = fgto.M(atom="F 0 0 0; H 0 0 1.7328", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]}, ensembles=ens)
    ms = mol.frags[0].members
    assert [(m.na, m.nb, m.charge) for m in ms] == [(5, 3, 1)] * 3 + [(5, 4, 0)] * 3
    assert sum(m.total_weight for m in ms if m.charge == 1) == pytest.approx(1 - nhy)
    assert sum(m.total_weight for m in ms if m.charge == 0) == pytest.approx(nhy)
    assert mol.frags[0].nelec_avg == pytest.approx(9 - (1 - nhy))
    mf = pdft.RKS(mol, method="functional-T", verbose=0, grid_level=3).build()
    assert set(mf._cfg_groups) == {(0, 1), (0, 0)}                      # one optimized weight group per charge


# ---- constraints on combinations of AOs (manual states of fragments with several atoms)
def co2_manual():
    pi_y = {"C 2py": 1.0, "2 O 2py": -1.0}
    pi_x = {"C 2px": 1.0, "2 O 2px": -1.0}
    ens = {"charge": {0: {"charges": (0, -1), "weights": (0.5, 0.5), "spins": (0, 1)},
                      1: {"charges": (0, 1), "weights": (0.5, 0.5), "spins": (2, 3)}},
           "config": {0: {"charge": -1, "states": (((pi_y, "alpha", 0.0),), ((pi_x, "alpha", 0.0),))}}}
    return fgto.M(atom="O 0 0 -2.2; C 0 0 0; O 0 0 2.2", unit="bohr", basis="sto-3g", fragments={0: [1, 2], 1: [0]},
                  ensembles=ens)


def test_combination_orbital_is_normalized_and_resolved_in_the_fragment():
    mol = co2_manual()
    cs = [m.config.constraints[0] for m in mol.frags[0].members if m.config is not None]
    S = mol.mol.intor("int1e_ovlp")
    for c in cs:
        assert c.P is not None and c.index is None                    # a combination, not a single AO
    assert abs(np.trace(cs[0].P @ np.linalg.inv(S) @ cs[1].P @ np.linalg.inv(S))) < 1e-10     # pi_y and pi_x are orthogonal
    assert np.trace(cs[0].P @ np.linalg.inv(S)) == pytest.approx(1.0)                          # unit-norm orbital


def test_atom_index_in_label_is_anchored():
    """'1 H 1s' must not match '11 H 1s'."""
    from f4b.cdft import Constraint
    mol = fgto.M(atom="; ".join(f"H 0 0 {1.5 * i}" for i in range(12)), basis="sto-3g", unit="bohr", charge=0, spin=0,
                 fragments={i: [i] for i in range(12)})
    c = Constraint("1 H 1s", "alpha", 0.0).bind(mol.mol, mol.mol.intor("int1e_ovlp"), np.eye(mol.mol.nao))
    assert c.index == 1

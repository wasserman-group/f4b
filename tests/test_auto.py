"""Automatic ensembles: fractional electron numbers, Hund-rule spins and degenerate-shell configuration states."""
import numpy as np
import pytest
from pyscf import gto

from f4b import fgto, pdft
from f4b.fragment import FragmentSystem

CO = "C 0 0 0; O 0 0 2.132"
O2 = "O 0 0 0; O 0 0 2.3"


def whole(atom, N=None, basis="sto-3g"):
    """One fragment with all the atoms and ``N`` electrons (default: neutral), to look at its shells."""
    Z = int(sum(gto.mole.charge(a.split()[0]) for a in atom.split(";")))
    N = Z if N is None else N
    return fgto.M(atom=atom, unit="bohr", basis=basis, fragments={0: list(range(atom.count(";") + 1))}, charge=Z - N, spin=N % 2,
                  fcharge={0: Z - N})


def spins(mol, k=0):
    return [(m.na, m.nb) for m in mol.frags[k].members]


def solve_vp0(mol, level=3):
    return FragmentSystem(mol, grid_level=level).solve(np.zeros((mol.mol.nao,) * 2))


# ---------------------------------------------------------------- charge ensemble from a number
def test_charge_number_gives_the_neighbouring_integers():
    mol = fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0], 1: [1]}, ensembles={"charge": {0: 1.4, 1: 0.6}})
    f0, f1 = mol.frags
    assert [(m.na, m.nb, round(m.weight, 12)) for m in f0.members] == [(1, 0, 0.6), (1, 1, 0.4)]       # H (0.6) + H- (0.4)
    assert [(m.na, m.nb, round(m.weight, 12)) for m in f1.members] == [(0, 0, 0.4), (1, 0, 0.6)]       # no electrons + H
    assert f0.nelec_avg == pytest.approx(1.4) and f1.nelec_avg == pytest.approx(0.6)


def test_integer_number_is_no_ensemble_and_fragments_without_entry_keep_fcharge():
    mol = fgto.M(atom="He 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g", charge=1, fragments={0: [0], 1: [1]}, fcharge={0: 1},
                 ensembles={"charge": {1: 1}})
    assert spins(mol, 0) == [(1, 0)] and spins(mol, 1) == [(1, 0)]
    assert mol.frags[0].members[0].weight == 1.0


def test_charge_number_totals_are_checked_and_validated():
    with pytest.raises(ValueError, match="electron count"):
        fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0], 1: [1]}, ensembles={"charge": {0: 1.4, 1: 0.5}})
    with pytest.raises(ValueError, match="negative"):
        fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0], 1: [1]}, ensembles={"charge": {0: -0.4, 1: 2.4}})


def test_fcharge_is_ignored_with_a_warning_when_the_ensemble_gives_the_electrons():
    with pytest.warns(UserWarning, match="fcharge is ignored"):
        mol = fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0], 1: [1]}, fcharge={0: 5},
                     ensembles={"charge": {0: 1.4, 1: 0.6}})
    assert mol.frags[0].nelec_avg == pytest.approx(1.4)


def test_explicit_charge_form_still_works():
    ens = {"charge": {0: {"charges": (0, -1), "weights": (0.6, 0.4)}, 1: {"charges": (None, 0), "weights": (0.4, 0.6)}}}
    a = fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0], 1: [1]}, ensembles=ens)
    b = fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0], 1: [1]}, ensembles={"charge": {0: 1.4, 1: 0.6}})
    for fa, fb in zip(a.frags, b.frags):
        assert [(m.na, m.nb) for m in fa.members] == [(m.na, m.nb) for m in fb.members]
        assert [m.weight for m in fa.members] == pytest.approx([m.weight for m in fb.members])


# ---------------------------------------------------------------- Hund's rule
@pytest.mark.parametrize("atom,N,expected", [
    ("F 0 0 0", 9, (5, 4)), ("F 0 0 0", 8, (5, 3)), ("O 0 0 0", 8, (5, 3)), ("O 0 0 0", 7, (5, 2)), ("O 0 0 0", 9, (5, 4)),
    ("C 0 0 0", 6, (4, 2)), ("N 0 0 0", 7, (5, 2)), ("Ne 0 0 0", 10, (5, 5)), ("Li 0 0 0", 3, (2, 1)), ("H 0 0 0", 1, (1, 0)),
    (CO, 14, (7, 7)), (CO, 15, (8, 7)), (CO, 13, (7, 6)), (O2, 16, (9, 7)), (O2, 15, (8, 7)), (O2, 17, (9, 8)),
    ("N 0 0 0; O 0 0 2.17", 15, (8, 7)),
])
def test_hund_spin(atom, N, expected):
    assert spins(whole(atom, N)) == [expected]


def test_default_spin_applies_to_every_fragment_and_fspin_overrides():
    mol = fgto.M(atom=O2, unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]})
    assert spins(mol, 0) == [(5, 3)] and spins(mol, 1) == [(5, 3)]                 # triplet O atoms, not singlets
    mol = fgto.M(atom=O2, unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]}, fspin={0: 0})
    assert spins(mol, 0) == [(4, 4)] and spins(mol, 1) == [(5, 3)]


def test_ions_of_an_ensemble_follow_hund_too():
    mol = fgto.M(atom="F 0 0 0; H 0 0 1.733", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]},
                 ensembles={"charge": {0: 8.7, 1: 1.3}})
    assert spins(mol, 0) == [(5, 3), (5, 4)]                                      # F+ triplet, F doublet
    assert spins(mol, 1) == [(1, 0), (1, 1)]                                      # H, H-


# ---------------------------------------------------------------- shells and automatic states
@pytest.mark.parametrize("atom,N,nstates", [
    ("F 0 0 0", 9, 3), ("F 0 0 0", 8, 3), ("O 0 0 0", 8, 3), ("O 0 0 0", 9, 3), ("O 0 0 0", 7, 0), ("C 0 0 0", 6, 3), ("Ne 0 0 0", 10, 0),
    ("H 0 0 0", 1, 0), (CO, 14, 0), (CO, 15, 2), (CO, 13, 0), (O2, 16, 0), (O2, 15, 2), (O2, 17, 2),
])
def test_number_of_automatic_states(atom, N, nstates):
    ref = whole(atom, N).shells(0, N)
    assert len(ref.states(*ref.hund())) == nstates


def test_spin_forced_to_singlet_gives_product_states():
    ref = whole("O 0 0 0").shells(0, 8)
    assert len(ref.states(4, 4)) == 9                                              # alpha and beta both partly filled


def test_too_many_states_is_an_error():
    ref = whole("O 0 0 0").shells(0, 8)
    with pytest.raises(ValueError, match="states"):
        ref.states(4, 4, max_states=5)


def benzene_atoms():
    atoms = []
    for k in range(6):
        a = np.pi / 3 * k
        atoms += [f"C {2.6355 * np.cos(a):.6f} {2.6355 * np.sin(a):.6f} 0", f"H {4.6355 * np.cos(a):.6f} {4.6355 * np.sin(a):.6f} 0"]
    return "; ".join(atoms)


def test_benzene_ions_have_degenerate_configurations():
    mol = whole(benzene_atoms(), N=41)
    ref = mol.shells(0, 41)
    assert ref.hund() == (21, 20)
    assert [d for d, _ in ref.states(21, 20)] == ["b-hole0", "b-hole1"]            # cation: e1g pair, one beta electron, two orientations
    anion = mol.shells(0, 43)
    assert anion.hund() == (22, 21) and [d for d, _ in anion.states(22, 21)] == ["a-hole0", "a-hole1"]     # anion: e2u pair
    for N, hund in ((42, (21, 21)), (40, (21, 19)), (44, (23, 21))):                # closed shell, dication / dianion (triplets): unique
        r = mol.shells(0, N)
        assert r.hund() == hund and r.states(*hund) == []


@pytest.mark.parametrize("atom,Ns", [("O 0 0 0; H 1.43 1.11 0; H -1.43 1.11 0", (10, 9, 11)),
                                      ("O 0 0 0; O 2.3 0 0; O 2.3 1.0 3.5", (24, 23, 25))])
def test_h2o_and_o3_have_no_degenerate_shells(atom, Ns):
    for N in Ns:
        ref = whole(atom, N).shells(0, N)
        assert ref.hund() == ((N + N % 2) // 2, (N - N % 2) // 2)                  # closed shell or doublet
        assert ref.states(*ref.hund()) == [] and all(len(g) == 1 for g in ref.groups)


def test_shells_report():
    mol = whole(CO)
    text = str(mol.shells(0, 14))
    assert "degeneracy" in text and "Hund" in text and "2S=0" in text
    assert mol.shells(0, 14) is mol.shells(0)                                      # cached; default N = neutral fragment


# ---------------------------------------------------------------- 'config' in ensembles
def hf(**cfg):
    ens = {"charge": {0: 8.7, 1: 1.3}, **cfg}
    return fgto.M(atom="F 0 0 0; H 0 0 1.733", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]}, ensembles=ens)


def test_automatic_configuration_ensemble_for_hf():
    mol = hf(config=[0])
    ms = mol.frags[0].members
    assert [(m.na, m.nb, m.charge) for m in ms] == [(5, 3, 1)] * 3 + [(5, 4, 0)] * 3
    assert [m.config.desc for m in ms[:3]] == ["b-hole0 b-hole1", "b-hole0 b-hole2", "b-hole1 b-hole2"]
    assert [m.config.desc for m in ms[3:]] == ["b-hole0", "b-hole1", "b-hole2"]
    assert all(m.omega == pytest.approx(1 / 3) for m in ms)
    assert sum(m.total_weight for m in ms if m.charge == 1) == pytest.approx(0.3)
    assert sum(m.total_weight for m in ms if m.charge == 0) == pytest.approx(0.7)
    assert mol.frags[0].nelec_avg == pytest.approx(8.7)
    assert all(m.group is None for m in mol.frags[1].members)                       # H: not requested


@pytest.mark.parametrize("cfg", [{"config": [0]}, {"config": {0: {}}}, {"config": {0: True}}, {"config": "all"}, {"config": True}])
def test_config_input_forms_are_equivalent(cfg):
    ms = hf(**cfg).frags[0].members
    assert len(ms) == 6 and all(m.group is not None for m in ms)


def test_config_all_skips_fragments_without_degenerate_shell_silently_when_another_has_one():
    mol = hf(config="all")
    assert all(m.group is None for m in mol.frags[1].members)                       # H has no degenerate shell


def test_no_degenerate_shell_warns():
    with pytest.warns(UserWarning, match="not needed"):
        mol = fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0], 1: [1]}, ensembles={"config": [0]})
    assert all(m.group is None for m in mol.frags[0].members)


def test_members_filter_and_default_weights_and_custom_omega():
    only_f = hf(config={0: {"members": [9]}}).frags[0].members
    assert [m.group is not None for m in only_f] == [False, True, True, True]       # F+ (8) unchanged, F (9) has the ensemble
    ms = hf(config={0: {"omega": {9: (0.5, 0.3, 0.2)}}}).frags[0].members
    assert [round(m.omega, 12) for m in ms if m.nelec == 9] == [0.5, 0.3, 0.2]
    assert [round(m.omega, 12) for m in ms if m.nelec == 8] == [round(1 / 3, 12)] * 3      # others: equal weights
    ms = hf(config={0: {"omega": (0.2, 0.3, 0.5)}}).frags[0].members
    assert [m.omega for m in ms] == [0.2, 0.3, 0.5] * 2
    with pytest.raises(ValueError, match="initial omegas"):
        hf(config={0: {"omega": (0.5, 0.5)}})
    with pytest.raises(ValueError, match="initial omegas"):
        hf(config={0: {"omega": (0.5, 0.4, 0.2)}})


def test_fill_occupied_uses_population_one_constraints():
    ms = hf(config={0: {"fill": "occupied", "members": [9]}}).frags[0].members
    cfg = [m for m in ms if m.group is not None]
    assert [m.config.desc for m in cfg] == ["b-occ0 b-occ1", "b-occ0 b-occ2", "b-occ1 b-occ2"]
    assert all(c.nelec == 1.0 for m in cfg for c in m.config.constraints)
    with pytest.raises(ValueError, match="fill"):
        hf(config={0: {"fill": "both"}})


def test_manual_and_automatic_configurations_cannot_be_mixed_and_unknown_keys_fail():
    with pytest.raises(ValueError, match="not a mixture"):
        hf(config={0: [{"charge": 0, "states": ((("F 2px", "beta", 0.0),),)}, {"fill": "holes"}]})
    with pytest.raises(ValueError, match="unknown key"):
        hf(config={0: {"fil": "holes"}})
    with pytest.raises(ValueError, match="fragment 7"):
        hf(config=[7])


def test_spin_ensemble_combines_with_automatic_configurations():
    mol = hf(config=[0], spin=[0, 1])
    ms = mol.frags[0].members
    assert all(m.spin_sym and m.flip == pytest.approx(m.weight) for m in ms)       # open-shell members are flip-symmetrized
    assert mol.frags[0].nelec_avg == pytest.approx(8.7)


def test_fspin_changes_the_automatic_states():
    mol = fgto.M(atom=O2, unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]}, fspin={0: 0, 1: 0}, ensembles={"config": [0]})
    assert len(mol.frags[0].members) == 9                                           # singlet O: 3 x 3 product states
    assert len(mol.frags[1].members) == 1


# ---------------------------------------------------------------- the states are the intended ones
def test_hole_constraints_are_satisfied_and_states_degenerate_for_f():
    mol = fgto.M(atom="F 0 0 0; H 0 0 4.0", unit="bohr", basis="6-31g", fragments={0: [0], 1: [1]},
                 ensembles={"charge": {0: 9, 1: 1}, "config": [0]})
    r = solve_vp0(mol)
    cfg = [it for it in r.raw[0] if it["m"].config is not None]
    assert len(cfg) == 3
    for it in cfg:
        assert it["m"].config.populations((it["Da"], it["Db"]))[0] == pytest.approx(0.0, abs=1e-4)
    e = [it["E"] for it in cfg]
    assert abs(e[0] - e[1]) < 2e-4                                                  # px and py holes: degenerate
    assert abs(e[2] - e[0]) < 5e-3                                                  # pz: split a little by the ghost H along z
    lab = [x.strip() for x in mol.mol.ao_labels()]
    D = sum(it["Db"] for it in cfg) / 3                                             # ensemble-averaged beta density: spherical
    d = [D[lab.index(f"0 F 2p{k}"), lab.index(f"0 F 2p{k}")] for k in "xyz"]
    assert max(d) - min(d) < 5e-3                                                   # (z differs slightly: ghost H basis along z)


def test_occupied_constraints_are_satisfied():
    mol = fgto.M(atom="F 0 0 0; H 0 0 4.0", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]},
                 ensembles={"charge": {0: 9, 1: 1}, "config": {0: {"fill": "occupied"}}})
    r = solve_vp0(mol)
    for it in r.raw[0]:
        assert it["m"].config.populations((it["Da"], it["Db"])) == pytest.approx([1.0, 1.0], abs=2e-3)


def co2(axis=(0.0, 0.0, 1.0), basis="sto-3g"):
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    atoms = "; ".join(f"{s} {' '.join(f'{x:.10f}' for x in p * ax)}" for s, p in (("O", -2.2), ("C", 0.0), ("O", 2.2)))
    return fgto.M(atom=atoms, unit="bohr", basis=basis, fragments={0: [1, 2], 1: [0]},
                  ensembles={"charge": {0: 14.5, 1: 7.5}, "config": "all"})


def test_co2_fragments_carbon_monoxide_pi_star_states():
    mol = co2()
    f0, f1 = mol.frags
    assert [(m.na, m.nb) for m in f0.members] == [(7, 7), (8, 7), (8, 7)]           # CO, CO- (pi* electron in x or y)
    assert [(m.na, m.nb) for m in f1.members] == [(5, 2), (5, 3), (5, 3), (5, 3)]  # O+ (quartet, unique), O (3 states)
    r = solve_vp0(mol)
    cfg = [it for it in r.raw[0] if it["m"].config is not None]
    assert abs(cfg[0]["E"] - cfg[1]["E"]) < 1e-5
    for it in cfg:
        assert it["m"].config.populations((it["Da"], it["Db"]))[0] == pytest.approx(0.0, abs=1e-4)


def test_orientation_of_the_molecule_does_not_matter():
    """A tilted CO2 gives the same state energies relative to the closed-shell CO (the shell orbitals follow the molecular
    axis; the absolute energies differ by the orientation dependence of the integration grid)."""
    e0 = np.array([it["E"] for it in solve_vp0(co2(), 4).raw[0]])
    e1 = np.array([it["E"] for it in solve_vp0(co2((1.0, 2.0, 3.0)), 4).raw[0]])
    assert e1[1:] - e1[0] == pytest.approx(e0[1:] - e0[0], abs=3e-4)
    assert abs(e1[1] - e1[2]) < 2e-4                                                # pi*_x / pi*_y states stay degenerate


# ---------------------------------------------------------------- end to end
def test_pdft_runs_with_automatic_ensembles():
    mol = hf(config=[0])
    mf = pdft.RKS(mol, method="functional-T", maxiter=2, grid_level=3, verbose=0)
    mf.kernel()
    assert set(mf.omega) == {(0, 1), (0, 0)}
    assert all(abs(o.sum() - 1.0) < 1e-12 for o in mf.omega.values())
    assert np.isfinite(mf.e_tot)

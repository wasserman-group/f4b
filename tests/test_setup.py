import numpy as np
import pytest

from f4b import fgto


def test_ghost_atoms_keep_ao_ordering(h2):
    f1, f2 = h2.frags
    m1 = h2.member_mole(f1, 1, 0)
    assert m1.nao == h2.mol.nao
    assert m1.atom_symbol(1).lower().startswith("ghost")
    assert m1.atom_charges()[1] == 0
    assert np.allclose(m1.atom_coords(), h2.mol.atom_coords())


def test_spin_ensemble_members(h2):
    for f in h2.frags:
        assert sum(m.total_weight for m in f.members) == pytest.approx(1.0)
        assert f.nelec_avg == pytest.approx(1.0)


def test_charge_ensemble_electron_count():
    mol = fgto.M(atom="H 0 0 0; H 0 0 2.0", unit="bohr", basis="cc-pvdz", charge=1, spin=1,
                 fragments={0: [0], 1: [1]},
                 ensembles={"charge": {0: {"charges": (0, 1), "weights": (0.5, 0.5)}, 1: {"charges": (0, 1), "weights": (0.5, 0.5)}}})
    assert sum(f.nelec_avg for f in mol.frags) == pytest.approx(1.0)


def test_wrong_electron_count_raises():
    with pytest.raises(ValueError):
        fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0], 1: [1]},
               ensembles={"charge": {0: {"charges": (0, -1), "weights": (0.5, 0.5)}}})


def test_incomplete_fragments_raise():
    with pytest.raises(ValueError):
        fgto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0]})


def test_dict_interface_is_zero_based():
    kw = dict(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    m = fgto.M(fragments={0: [0], 1: [1]}, ensembles={"spin": [0, 1]}, **kw)
    assert [f.index for f in m.frags] == [0, 1] and [f.atoms for f in m.frags] == [(0,), (1,)]
    assert all(mm.spin_sym for f in m.frags for mm in f.members)
    for bad in (dict(fragments=((1,), (2,))),                                  # old tuple format
                dict(fragments={1: [0], 2: [1]}),                              # fragment indices must start at 0
                dict(fragments={0: [1], 1: [2]}),                              # atom indices must start at 0
                dict(fragments={0: [0], 1: [1]}, ensembles=(("spin", (0,)),)),
                dict(fragments={0: [0], 1: [1]}, ensembles={"charge": {0: {"charges": (0, -1)}}}),            # no 'weights'
                dict(fragments={0: [0], 1: [1]}, ensembles={"charge": {0: {"charges": (0,), "weight": (1,)}}})):  # typo'd key
        with pytest.raises((TypeError, ValueError)):
            fgto.M(**bad, **kw)


def test_charges_are_relative_to_neutral():
    """+1 = one electron less, -1 = one more, None = no electrons (same as +1 for H)."""
    kw = dict(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", fragments={0: [0], 1: [1]})
    m = fgto.M(ensembles={"charge": {0: {"charges": (1, 0, -1), "weights": (0.25, 0.5, 0.25)},
                                     1: {"charges": (None, 1, 0, -1), "weights": (0.25, 0.0, 0.5, 0.25)}}}, **kw)
    assert [(x.na + x.nb, x.charge) for x in m.frags[0].members] == [(0, 1), (1, 0), (2, -1)]
    assert [(x.na + x.nb, x.charge) for x in m.frags[1].members] == [(0, 1), (0, 1), (1, 0), (2, -1)]
    with pytest.raises(ValueError):                                       # H+ cannot lose two electrons
        fgto.M(ensembles={"charge": {0: {"charges": (2,), "weights": (1.0,)}}}, **kw)


ZMAT_H3P = "X; X 1 1.0; H 1 1.0 2 90.0; H 1 1.0 2 90.0 3 120.0; H 1 1.0 2 90.0 3 -120.0"


def test_zmatrix_dummy_atoms_not_in_fragments():
    """Rows of dummy centres ('X') are skipped: fragments list the real atoms by their row in ``atom``."""
    from f4b.fragment import FragmentSystem
    ens = {"charge": {i: {"charges": (None, 0), "weights": (1 / 3, 2 / 3)} for i in range(3)}, "spin": [0, 1, 2]}
    kw = dict(unit="bohr", basis="sto-3g", charge=1, ensembles=ens)
    mz = fgto.M(atom=ZMAT_H3P, fragments={0: [2], 1: [3], 2: [4]}, **kw)
    assert mz.mol.natm == 3 and [f.rows for f in mz.frags] == [(2,), (3,), (4,)]
    assert [f.index for f in fgto.M(atom=ZMAT_H3P, **kw).frags] == [0, 1, 2]            # default: one per real atom
    mc = fgto.M(atom="; ".join("H %.15f %.15f %.15f" % tuple(c) for c in mz.mol.atom_coords()),
                fragments={0: [0], 1: [1], 2: [2]}, **kw)
    V = 0.1 * mz.mol.intor("int1e_ovlp")
    rz, rc = FragmentSystem(mz, grid_level=3).solve(V), FragmentSystem(mc, grid_level=3).solve(V)
    assert np.allclose(rz.E, rc.E, atol=1e-9) and np.allclose(rz.Da[0], rc.Da[0], atol=1e-9)
    for bad in ({0: [0], 1: [3], 2: [4], 3: [2]},       # a dummy row inside a fragment
                {0: [2], 1: [3]},                        # real atom 4 missing
                {0: [2], 1: [3], 2: [9]}):               # out of range
        with pytest.raises(ValueError):
            fgto.M(atom=ZMAT_H3P, fragments=bad, **kw)


def _o2(**kw):
    return fgto.M(atom="O 0 0 0; O 0 0 2.28", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]}, **kw)


def _spins(mol, k):
    return [(m.na, m.nb) for m in mol.frags[k].members]


def test_fspin_default_is_lowest_spin():
    mol = _o2()
    assert _spins(mol, 0) == [(4, 4)] and _spins(mol, 1) == [(4, 4)]
    assert mol.frags[0].twoS0 is None


def test_fspin_sets_the_fragment_spin():
    mol = _o2(fspin={0: 2, 1: 2})
    assert _spins(mol, 0) == [(5, 3)] and _spins(mol, 1) == [(5, 3)]
    mol = _o2(fspin={1: 2})                                             # only fragment 1; fragment 0 keeps the default
    assert _spins(mol, 0) == [(4, 4)] and _spins(mol, 1) == [(5, 3)]
    assert _spins(_o2(fspin=[2, 0]), 0) == [(5, 3)]                      # one value per fragment, in order


def test_fspin_with_fcharge():
    mol = fgto.M(atom="O 0 0 0; O 0 0 2.28", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]},
                 fcharge={0: 1, 1: -1}, fspin={0: 3, 1: 1})            # O+ (7 e-, 2S=3) and O- (9 e-, 2S=1)
    assert _spins(mol, 0) == [(5, 2)] and _spins(mol, 1) == [(5, 4)]


def test_ensemble_spins_override_fspin_and_fspin_applies_to_the_neutral_member_only():
    ens = {"charge": {0: {"charges": (0, -1), "weights": (0.5, 0.5), "spins": (0, 3)},
                      1: {"charges": (0, -1), "weights": (0.5, 0.5)}}}
    mol = fgto.M(atom="O 0 0 0; Mg 0 0 2.28", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]},
                 charge=-1, spin=1, ensembles=ens, fspin={0: 2, 1: 2})
    assert _spins(mol, 0) == [(4, 4), (6, 3)]          # 'spins' beat fspin=2: O (8 e-, 2S=0) and O- (9 e-, 2S=3)
    assert _spins(mol, 1) == [(7, 5), (7, 6)]          # Mg neutral member: fspin=2; Mg- (13 e-) has no 'spins': lowest spin


def test_none_in_spins_means_default():
    def li_h(spins_li, fspin):
        ens = {"charge": {0: {"charges": (0, 1), "weights": (0.5, 0.5), "spins": spins_li},
                          1: {"charges": (-1, 0), "weights": (0.5, 0.5)}}}
        return fgto.M(atom="Li 0 0 0; H 0 0 3.0", unit="bohr", basis="sto-3g", fragments={0: [0], 1: [1]},
                      ensembles=ens, fspin=fspin)
    assert _spins(li_h((None, None), {0: 3}), 0) == [(3, 0), (1, 1)]      # neutral Li: fspin (quartet); Li+: default
    assert _spins(li_h((1, None), {0: 3}), 0) == [(2, 1), (1, 1)]         # explicit 'spins' wins for that member
    assert _spins(li_h((None, None), None), 0) == [(2, 1), (1, 1)]        # no fspin: lowest spin everywhere


def test_fspin_errors():
    with pytest.raises(ValueError, match="fragment 0.*incompatible"):
        _o2(fspin={0: 1})                                                # 8 electrons cannot have 2S=1
    with pytest.raises(ValueError, match="incompatible"):
        _o2(fspin={0: 10})
    with pytest.raises(ValueError, match="incompatible"):
        _o2(fspin={0: -2})
    with pytest.raises(ValueError, match="unknown fragment"):
        _o2(fspin={3: 2})
    with pytest.raises(ValueError, match="fragment 0, charge -1"):
        _o2(ensembles={"charge": {0: {"charges": (0, -1), "weights": (0.5, 0.5), "spins": (2, 2)}}})


def test_fspin_breaks_fragment_symmetry():
    from f4b.symmetry import FragSymmetry
    assert FragSymmetry(_o2(), tol=1e-4).nontrivial
    assert not FragSymmetry(_o2(fspin={0: 2}), tol=1e-4).nontrivial
    assert FragSymmetry(_o2(fspin={0: 2, 1: 2}), tol=1e-4).nontrivial


def test_fspin_with_configuration_ensemble_needs_no_charge_ensemble():
    states = tuple(((f"O 2p{k}", "beta", 0.0), ("O 2p" + {"x": "y", "y": "z", "z": "x"}[k], "beta", 0.0)) for k in "xyz")
    mol = _o2(fspin={0: 2}, ensembles={"config": {0: {"charge": 0, "states": states}}})
    assert _spins(mol, 0) == [(5, 3)] * 3

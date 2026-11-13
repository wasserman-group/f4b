"""Molecule with fragments (``fgto`` = fragment GTO).

Example::

    from f4b import fgto
    mol = fgto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz',
                 fragments={0: [0], 1: [1]},                  # fragment index -> atom indices
                 ensembles={'spin': [0, 1]})                  # see f4b.ensemble for 'charge' / 'config'

Atoms and fragments are numbered from 0, atoms in the order written in ``atom`` (rows of a Z-matrix included).
Dummy centres (``X`` in a Z-matrix: no nucleus, no basis) and ghost atoms (``ghost-H``: basis functions, no nucleus)
are not part of any fragment: list only the real atoms in ``fragments``::

    fgto.M(atom='X; X 1 1.0; H 1 1.0 2 90.0; H 1 1.0 2 90.0 3 120.0; H 1 1.0 2 90.0 3 -120.0', unit='bohr',
           charge=1, fragments={0: [2], 1: [3], 2: [4]}, ...)      # rows 0 and 1 are dummies

The supermolecule is a regular
``pyscf.gto.Mole`` (``mol.mol``).  For every fragment calculation the Mole contains *all* atoms of the
supermolecule, those that do not belong to the fragment being ghost atoms, so that every matrix
(density, partition potential) lives in the AO basis of the supermolecule.
"""
import numpy as np
from pyscf import gto

from .ensemble import FragmentSpec, build_members


def _symbol(sym):
    return "".join(ch for ch in str(sym).strip() if ch.isalpha() or ch == "-").upper()


def _is_dummy(sym):
    """Z-matrix dummy centre: a bare ``X`` (no nucleus and no basis functions)."""
    return _symbol(sym) == "X"


def _is_ghost(sym):
    """Atom with basis functions but no nucleus (``ghost-H``, ``X-H``)."""
    return _symbol(sym).startswith(("GHOST-", "X-"))


class FragMole:
    def __init__(self, atom=None, basis="cc-pvdz", charge=0, spin=0, unit="angstrom", fragments=None,
                 ensembles=None, fcharge=None, fbasis="same", verbose=0, **mole_kwargs):
        self.atom, self.basis, self.charge, self.spin, self.unit = atom, basis, charge, spin, unit
        self.fragments, self.ensembles, self.fcharge, self.fbasis = fragments, ensembles, fcharge, fbasis
        self.verbose = verbose
        self.mole_kwargs = mole_kwargs
        self.mol = None
        self.frags = None

    def build(self):
        if self.fbasis is not None and self.fbasis != "same" and self.fbasis != self.basis:
            raise NotImplementedError("fbasis different from basis is not implemented yet")
        rows = gto.mole.format_atom(self.atom, unit=self.unit)            # Z-matrices -> Cartesian (bohr)
        syms = [sym for sym, _ in rows]
        kept = [i for i, sym in enumerate(syms) if not _is_dummy(sym)]
        if len(kept) < len(rows):                                         # drop dummy centres, keep the rest
            self.mol = gto.M(atom=[rows[i] for i in kept], basis=self.basis, charge=self.charge, spin=self.spin,
                             unit="Bohr", verbose=self.verbose, **self.mole_kwargs)
        else:
            self.mol = gto.M(atom=self.atom, basis=self.basis, charge=self.charge, spin=self.spin, unit=self.unit,
                             verbose=self.verbose, **self.mole_kwargs)
        mol = self.mol
        row2atom = {row: k for k, row in enumerate(kept)}                 # input row -> atom index of ``self.mol``
        real = [row for row in kept if not _is_ghost(syms[row])]
        self.atom_symbols, self.row2atom = syms, row2atom
        fragments = self.fragments
        if fragments is None:
            fragments = {i: (row,) for i, row in enumerate(real)}
        if not isinstance(fragments, dict):
            raise TypeError("fragments must be a dict {fragment index: atom indices}, e.g. {0: [0], 1: [1, 2]} "
                            "(fragments and atoms are numbered from 0)")
        if sorted(fragments) != list(range(len(fragments))):
            raise ValueError(f"fragment indices must be 0..{len(fragments) - 1}, got {sorted(fragments)}")
        fragments = {k: tuple(int(a) for a in np.atleast_1d(fragments[k])) for k in sorted(fragments)}
        used = sorted(a for f in fragments.values() for a in f)
        listed = [a for a in used if a in row2atom and a in real]
        if listed != used:
            bad = [f"{a} ({syms[a]})" if 0 <= a < len(syms) else f"{a} (out of range)" for a in used if a not in listed]
            raise ValueError(f"fragments list atoms that are not real atoms: {bad}; dummy ('X') and ghost atoms "
                             f"belong to no fragment (atoms: {list(enumerate(syms))})")
        if used != real:
            raise ValueError(f"fragments {fragments} must contain every real atom {real} exactly once")
        fcharge = self.fcharge
        if fcharge is None:
            fcharge = {}
        if not isinstance(fcharge, dict):                      # also accept one charge per fragment, in order
            fcharge = dict(enumerate(np.atleast_1d(fcharge)))
        bad = set(fcharge) - set(fragments)
        if bad:
            raise ValueError(f"fcharge refers to unknown fragment(s) {sorted(bad)}")
        Zs = mol.atom_charges()
        frags = []
        for i, idx in fragments.items():
            idx = tuple(row2atom[a] for a in idx)
            Z = float(sum(Zs[a] for a in idx))
            frags.append(FragmentSpec(index=i, atoms=idx, Z=Z, nelec0=int(round(Z - fcharge.get(i, 0))), rows=fragments[i]))
        self.frags = build_members(frags, self.ensembles, mol=self.mol)
        nel = sum(f.nelec_avg for f in self.frags)
        if abs(nel - mol.nelectron) > 1e-8:
            raise ValueError(
                f"Fragment/ensemble electron count ({nel:g}) does not match the electron number of the "
                f"molecule ({mol.nelectron}). Use 'fcharge' or a charge ensemble to set the fragment "
                f"electron numbers (e.g. fcharge={0: 1} for HeH+ with fragments He+ and H).")
        self._fmoles = {}
        return self

    @property
    def nfrag(self):
        return len(self.frags)

    def member_mole(self, frag, na, nb):
        """Mole with all atoms (those outside the fragment are ghosts) for given (na, nb) of a fragment."""
        key = (frag.index, na, nb)
        if key not in self._fmoles:
            mol = self.mol
            atoms = []
            for i in range(mol.natm):
                sym = mol.atom_symbol(i)
                atoms.append((sym if i in frag.atoms or _is_ghost(sym) else "ghost-" + sym, tuple(mol.atom_coord(i))))
            self._fmoles[key] = gto.M(atom=atoms, basis=self.basis, unit="Bohr",
                                      charge=int(round(frag.Z - na - nb)), spin=na - nb,
                                      verbose=self.verbose, **self.mole_kwargs)
        return self._fmoles[key]

    def __repr__(self):
        if self.frags is None:
            return "FragMole(not built)"
        return "FragMole:\n  " + "\n  ".join(repr(f) for f in self.frags)


def M(**kwargs):
    """Build and return a :class:`FragMole` (analogous to ``pyscf.gto.M``)."""
    return FragMole(**kwargs).build()

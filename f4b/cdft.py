"""Configuration states through constrained DFT (population constraints on orbitals).

A *configuration state* of a fragment is a Slater-determinant (UKS) state selected by constraining the population of
one or more orbitals, e.g. a beta 2p_z hole of the F atom::

    ConfigState([Constraint('F 2pz', 'beta', 0.0)])

The orbital is a meta-Loewdin localized AO, a normalized combination of them (``{'C 2py': 1, 'O 2py': -1}``) or, for the
automatic ensembles (:mod:`f4b.shells`), a molecular orbital of the isolated fragment given as an AO vector.

The constraint is enforced by a Lagrange multiplier ``vc``: the Fock matrix of spin ``s`` gets ``vc * P`` with
``P = S c c^T S`` (``c`` = orbital).  The multipliers are found once, on the isolated fragment (Vp = 0), and
then held fixed during the P-DFT iterations (the constraint is only used to *select* the state; Vp is free to move the
population afterwards).  The energies of the states are the plain KS energies (the constraint term is not included).
"""
import re

import numpy as np
from pyscf.lo import orth

_SPIN = {0: 0, 1: 1, "a": 0, "alpha": 0, "up": 0, "b": 1, "beta": 1, "down": 1}


class Constraint:
    """Population ``nelec`` of an orbital for spin ``spin``.

    ``ao`` selects the orbital:

    * an AO index of the supermolecule, or a regular expression matched against ``mol.ao_labels()`` (``'F 2pz'``, ``'0 F 2pz'``);
    * a combination of such AOs, ``{'C 2py': 1.0, 'O 2py': -1.0}`` or ``[('C 2py', 1.0), ('O 2py', -1.0)]``;
    * an array with one coefficient per localized orbital (length ``nao``).

    A label is looked up among the AOs of the atoms of the fragment the state belongs to (so ``'O 2px'`` is enough in O2
    when the fragment is one O atom) and must select exactly one AO; use the atom index (``'1 O 2px'``) to tell the atoms
    of a fragment apart.  The meta-Loewdin orbitals are orthonormal, so the coefficients of a combination are normalized.
    """

    def __init__(self, ao, spin, nelec=0.0):
        self.ao = ao
        if spin not in _SPIN:
            raise ValueError(f"spin must be one of {sorted(map(str, _SPIN))}, got {spin!r}")
        self.spin = _SPIN[spin]
        self.nelec = float(nelec)
        self.index = None
        self.vector = None                # AO-basis coefficients of the orbital (automatic ensembles), instead of ``ao``
        self.P = None                     # AO-basis projector S c c^T S (set by bind)

    @classmethod
    def from_vector(cls, vector, spin, nelec=0.0, label=None):
        """Constraint on the orbital with AO coefficients ``vector`` (normalized: c^T S c = 1)."""
        c = cls(label, spin, nelec)
        c.vector = np.asarray(vector, float)
        return c

    @staticmethod
    def _find(label, mol, atoms):
        if isinstance(label, (int, np.integer)):
            return int(label)
        pattern = label.strip()
        if re.match(r"\d+\s", pattern):                 # '1 O 2px': the atom index must not match '11 O 2px'
            pattern = "^" + pattern
        idx = list(mol.search_ao_label(pattern))
        if atoms is not None:
            owner = np.empty(mol.nao, int)
            for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
                owner[p0:p1] = a
            allmatch = idx
            idx = [i for i in idx if owner[i] in atoms]
            if not idx and allmatch:
                raise ValueError(f"constraint AO '{label}' only matches orbitals {[mol.ao_labels()[i] for i in allmatch]}, "
                                 f"which are not on the atoms {sorted(atoms)} of the fragment")
        if len(idx) != 1:
            labels = [mol.ao_labels()[i] for i in idx]
            raise ValueError(f"constraint AO '{label}' matches {len(idx)} orbitals {labels}; it must match exactly one "
                             "(use the atom index, e.g. '0 O 2px')")
        return idx[0]

    def _coefficients(self, mol, atoms):
        """Coefficients of the orbital in the (orthonormal) localized basis, normalized."""
        ao, a = self.ao, np.zeros(mol.nao)
        if isinstance(ao, np.ndarray):
            if ao.shape != (mol.nao,):
                raise ValueError(f"constraint orbital array must have one coefficient per orbital ({mol.nao}), got shape {ao.shape}")
            a = ao.astype(float)
        elif isinstance(ao, dict) or (isinstance(ao, (list, tuple)) and ao and isinstance(ao[0], (list, tuple))):
            for label, coef in (ao.items() if isinstance(ao, dict) else ao):
                a[self._find(label, mol, atoms)] += float(coef)
        else:
            a[self._find(ao, mol, atoms)] = 1.0
        norm = np.linalg.norm(a)
        if norm < 1e-12:
            raise ValueError(f"constraint orbital {self.ao!r} has zero norm")
        return a / norm

    def bind(self, mol, S, C, atoms=None):
        if self.vector is not None:
            c = self.vector
        else:
            a = self._coefficients(mol, atoms)
            nz = np.flatnonzero(np.abs(a) > 1e-12)
            self.index = int(nz[0]) if len(nz) == 1 else None
            c = C @ a
        self.P = np.outer(S @ c, S @ c)
        return self

    def population(self, D):
        """Population of the constrained orbital in the (spin-resolved) density matrices ``D = (Da, Db)``."""
        return float(np.einsum("ij,ij", self.P, D[self.spin]))

    def __repr__(self):
        lab = self.ao if self.index is None or not isinstance(self.ao, str) else f"{self.ao}[{self.index}]"
        if self.vector is not None:
            lab = self.ao or "orbital"
        return f"Constraint({lab}, spin={'ab'[self.spin]}, nelec={self.nelec:g})"


class ConfigState:
    """A configuration: a list of :class:`Constraint` and their (fixed) multipliers."""

    def __init__(self, constraints, name=None):
        if isinstance(constraints, Constraint):
            constraints = [constraints]
        elif isinstance(constraints, (tuple, list)) and constraints and not isinstance(constraints[0], (Constraint, tuple, list)):
            constraints = [Constraint(*constraints)]                 # a bare (ao, spin[, nelec]) tuple
        self.constraints = [c if isinstance(c, Constraint) else Constraint(*c) for c in constraints]
        self.name = name
        self.vc = None
        self.desc = None                  # short text describing the state (automatic ensembles)

    def bind(self, mol, atoms=None):
        S = mol.intor("int1e_ovlp")
        C = None if all(c.vector is not None for c in self.constraints) else orth.orth_ao(mol, "meta_lowdin", s=S)
        for c in self.constraints:
            c.bind(mol, S, C, atoms)
        return self

    def fock_add(self, vc=None):
        """(2, nao, nao) addition to the Fock matrices."""
        vc = self.vc if vc is None else vc
        nao = self.constraints[0].P.shape[0]
        out = np.zeros((2, nao, nao))
        for v, c in zip(vc, self.constraints):
            out[c.spin] += v * c.P
        return out

    def populations(self, D):
        return np.array([c.population(D) for c in self.constraints])

    def __repr__(self):
        return f"ConfigState({self.name or self.constraints}, vc={self.vc})"


def solve_multipliers(state, run, vstart=0.2, vmax=50.0, ptol=1e-5, rtol=1e-6, sweeps=3, log=None):
    """Find the multipliers of ``state``.

    ``run(vc) -> (Da, Db)`` performs the constrained SCF.  A constraint with target population 0 (a hole) is
    satisfied by a penalty: the multiplier is doubled until the population drops below ``ptol`` (the population
    only vanishes asymptotically in general); a full spin orbital (population 1) is the same with an attractive multiplier (tolerance 1e-4).
    Other targets are found by bracketing (positive or negative multiplier) + Brent.
    Several constraints are solved by cyclic sweeps over the individual multipliers.
    """
    from scipy.optimize import brentq
    cons = state.constraints
    vc = [0.0] * len(cons)
    for sweep in range(sweeps if len(cons) > 1 else 1):
        for i, c in enumerate(cons):
            def pop(v, i=i):
                trial = list(vc)
                trial[i] = v
                return state.constraints[i].population(run(trial))
            if c.nelec <= 1e-12:
                v = vstart if vc[i] == 0.0 else vc[i]
                while pop(v) > ptol:
                    v *= 2.0
                    if v > vmax:
                        raise RuntimeError(f"CDFT: population of {c} cannot be driven below {ptol:g} with a multiplier "
                                           f"<= {vmax:g} (is the state reachable?)")
                vc[i] = v
            elif c.nelec >= 1.0 - 1e-12:
                # a fully occupied spin orbital is reached only asymptotically too: attractive multiplier
                # (the deficit decays only like 1/|v|, so the tolerance is looser than for holes)
                ptol_full = max(ptol, 1e-4)
                v = -vstart if vc[i] == 0.0 else vc[i]
                while 1.0 - pop(v) > ptol_full:
                    v *= 2.0
                    if abs(v) > 10 * vmax:
                        raise RuntimeError(f"CDFT: population of {c} cannot be driven above {1.0 - ptol_full:g} with a "
                                           f"multiplier |v| <= {10 * vmax:g} (is the state reachable?)")
                vc[i] = v
            else:
                f = lambda v: pop(v) - c.nelec                       # noqa: E731
                f0 = f(0.0)
                if abs(f0) < rtol:
                    vc[i] = 0.0
                    continue
                sign = np.sign(f0)
                # population above the target: repulsive (positive) multiplier; below it: attractive (negative)
                lo, hi = 0.0, vstart * sign
                while np.sign(f(hi)) == sign:
                    lo, hi = hi, hi * 2.0
                    if abs(hi) > vmax:
                        raise RuntimeError(f"CDFT: could not bracket the multiplier of {c}")
                vc[i] = brentq(f, lo, hi, xtol=1e-8, rtol=1e-8)
        if log:
            log(f"  CDFT sweep {sweep + 1}: vc = {vc}")
    return vc


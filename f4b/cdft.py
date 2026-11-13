"""Configuration states through constrained DFT (population constraints on localized AOs).

A *configuration state* of a fragment is a Slater-determinant (UKS) state selected by constraining the population of
one or more meta-Loewdin localized atomic orbitals, e.g. a beta 2p_z hole of the F atom::

    ConfigState([Constraint('F 2pz', 'beta', 0.0)])

The constraint is enforced by a Lagrange multiplier ``vc``: the Fock matrix of spin ``s`` gets ``vc * P`` with
``P = S c c^T S`` (``c`` = localized orbital).  The multipliers are found once, on the isolated fragment (Vp = 0), and
then held fixed during the P-DFT iterations (the constraint is only used to *select* the state; Vp is free to move the
population afterwards).  The energies of the states are the plain KS energies (the constraint term is not included).
"""
import numpy as np
from pyscf.lo import orth

_SPIN = {0: 0, 1: 1, "a": 0, "alpha": 0, "up": 0, "b": 1, "beta": 1, "down": 1}


class Constraint:
    """Population ``nelec`` of the localized AO ``ao`` for spin ``spin``.

    ``ao`` is an AO index of the supermolecule or a regular expression matched against ``mol.ao_labels()``
    (e.g. ``'F 2pz'`` or ``'0 F 2pz'``); it must select exactly one AO.
    """

    def __init__(self, ao, spin, nelec=0.0):
        self.ao = ao
        if spin not in _SPIN:
            raise ValueError(f"spin must be one of {sorted(map(str, _SPIN))}, got {spin!r}")
        self.spin = _SPIN[spin]
        self.nelec = float(nelec)
        self.index = None
        self.P = None                     # AO-basis projector S c c^T S (set by bind)

    def bind(self, mol, S, C):
        if isinstance(self.ao, (int, np.integer)):
            idx = [int(self.ao)]
        else:
            idx = list(mol.search_ao_label(self.ao))
        if len(idx) != 1:
            labels = [mol.ao_labels()[i] for i in idx]
            raise ValueError(f"constraint AO '{self.ao}' matches {len(idx)} orbitals {labels}; it must match exactly one")
        self.index = idx[0]
        c = C[:, self.index]
        self.P = np.outer(S @ c, S @ c)
        return self

    def population(self, D):
        """Population of the constrained orbital in the (spin-resolved) density matrices ``D = (Da, Db)``."""
        return float(np.einsum("ij,ij", self.P, D[self.spin]))

    def __repr__(self):
        lab = self.ao if self.index is None else f"{self.ao}[{self.index}]"
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

    def bind(self, mol):
        S = mol.intor("int1e_ovlp")
        C = orth.orth_ao(mol, "meta_lowdin", s=S)
        for c in self.constraints:
            c.bind(mol, S, C)
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
    only vanishes asymptotically in general).  Other targets are found by bracketing + Brent.
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
            else:
                f = lambda v: pop(v) - c.nelec                       # noqa: E731
                lo, hi = 0.0, vstart
                f0 = f(lo)
                if abs(f0) < rtol:
                    vc[i] = 0.0
                    continue
                sign = np.sign(f0)
                while np.sign(f(hi)) == sign:
                    lo, hi = hi, hi * 2.0
                    if hi > vmax:
                        raise RuntimeError(f"CDFT: could not bracket the multiplier of {c}")
                vc[i] = brentq(f, lo, hi, xtol=1e-8, rtol=1e-8)
        if log:
            log(f"  CDFT sweep {sweep + 1}: vc = {vc}")
    return vc

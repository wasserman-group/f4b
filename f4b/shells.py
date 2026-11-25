"""Orbital shells of isolated fragments: Hund-rule spins and automatic configuration states.

For a fragment with ``N`` electrons a spin-restricted *fractional-occupation* Kohn-Sham calculation of the isolated
fragment (no ghost atoms, ``N/2`` electrons per spin shared equally inside each group of degenerate orbitals) gives a
symmetric reference: the degenerate orbitals of atoms, linear molecules and symmetric molecules (benzene, ...) are exactly
degenerate and can be grouped by their energies.  From the groups:

* **Hund's rule** gives the spin: the orbitals are filled in order of energy; in the first group that is not full (``d``
  degenerate orbitals, ``k`` electrons) the electrons go parallel first, ``N_alpha - N_beta = min(k, 2d - k)``.
* **Configuration states**: in the partly filled group of a spin channel (``n`` of the ``d`` orbitals occupied) the
  ``C(d, n)`` ways of choosing the occupied orbitals are the degenerate determinants.  Each state is selected with population
  constraints (:mod:`f4b.cdft`): holes (population 0) in the ``d - n`` empty orbitals (default) or population 1 in the ``n``
  occupied ones.  The orbitals of the group are rotated to the principal axes of a generic quadratic form of the position
  (x^2 + 1.7 y^2 + 2.9 z^2), so the p orbitals of an atom are px, py, pz and the pi orbitals of a linear molecule are
  oriented along the axes, independently of the arbitrary rotation inside the degenerate eigenspace.
"""
import itertools

import numpy as np
from pyscf import dft, gto

DEG_TOL = 5e-5                 # Eh: orbital energies closer than this are one degenerate group
PROBE = (1.0, 1.7, 2.9)        # weights of the quadratic form that orients the orbitals of a group
MAX_STATES = 60


class _FracUKS(dft.uks.UKS):
    """UKS with the same fractional occupations for alpha and beta: ``N/2`` electrons per spin, equally shared in each
    group of (nearly) degenerate orbitals."""
    deg_tol = DEG_TOL

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        e = 0.5 * (np.asarray(mo_energy[0]) + np.asarray(mo_energy[1]))
        order = np.argsort(e, kind="stable")
        occ = np.zeros(len(e))
        rem = self.mol.nelectron / 2.0
        i = 0
        while i < len(order) and rem > 1e-12:
            j = i + 1
            while j < len(order) and e[order[j]] - e[order[j - 1]] < self.deg_tol:
                j += 1
            grp = order[i:j]
            take = min(rem, len(grp))
            occ[grp] = take / len(grp)
            rem -= take
            i = j
        return np.array([occ, occ])


def _groups(e, tol):
    order = np.argsort(e, kind="stable")
    out, cur = [], [order[0]]
    for k in order[1:]:
        if e[k] - e[cur[-1]] < tol:
            cur.append(k)
        else:
            out.append(cur)
            cur = [k]
    out.append(cur)
    return out


class Reference:
    """Symmetric fractional-occupation reference of the isolated fragment ``frag`` with ``N`` electrons."""

    def __init__(self, mol, frag, N, basis, mole_kwargs=None, xc="lda,vwn", grid_level=3, deg_tol=DEG_TOL):
        self.N, self.frag, self.deg_tol = int(N), frag, deg_tol
        kw = {k: v for k, v in (mole_kwargs or {}).items() if k not in ("symmetry", "charge", "spin")}
        atoms = [(mol.atom_symbol(i), tuple(mol.atom_coord(i))) for i in frag.atoms]
        self.mol = gto.M(atom=atoms, basis=basis, unit="Bohr", charge=int(round(frag.Z - N)), spin=self.N % 2, verbose=0, **kw)
        sl = mol.aoslice_by_atom()
        self.ao = np.concatenate([np.arange(sl[i][2], sl[i][3]) for i in frag.atoms])      # AOs of the fragment in the supermolecule
        self.nao_super = mol.nao
        self.S = self.mol.intor("int1e_ovlp")
        mf = _FracUKS(self.mol)
        mf.xc, mf.deg_tol = xc, deg_tol
        mf.grids.level = grid_level
        mf.conv_tol = 1e-8
        mf.max_cycle = 100
        mf.kernel()
        self.converged = bool(mf.converged)
        self.energy = np.asarray(mf.mo_energy[0])
        self.C = np.asarray(mf.mo_coeff[0])
        self.groups = _groups(self.energy, deg_tol)
        self._shell = {}

    # ------------------------------------------------------------------ occupations
    def hund(self):
        """``(N_alpha, N_beta)`` of the ground state by Hund's rule (orbitals in order of energy)."""
        na = nb = 0
        rem = self.N
        for g in self.groups:
            d = len(g)
            if rem >= 2 * d:
                na, nb, rem = na + d, nb + d, rem - 2 * d
            else:
                a = min(rem, d)
                na, nb, rem = na + a, nb + rem - a, 0
            if rem == 0:
                break
        return na, nb

    def _channel(self, n):
        """Fill ``n`` electrons of one spin: ``(group index, number of occupied orbitals)`` of the partly filled group, or None."""
        for gi, g in enumerate(self.groups):
            d = len(g)
            if n >= d:
                n -= d
            else:
                return (gi, n) if n > 0 else None
            if n == 0:
                return None
        return None

    # ------------------------------------------------------------------ orbitals of a group
    def shell_orbitals(self, gi):
        """AO coefficients (fragment AOs x d) of group ``gi`` rotated to the principal axes of the probe form."""
        if gi not in self._shell:
            g = self.groups[gi]
            C = self.C[:, g]
            Z, X = self.mol.atom_charges(), self.mol.atom_coords()
            center = (Z[:, None] * X).sum(0) / Z.sum() if Z.sum() > 0 else X.mean(0)
            with self.mol.with_common_orig(center):
                rr = self.mol.intor("int1e_rr").reshape(3, 3, *self.S.shape)
            Q = sum(w * rr[a, a] for a, w in enumerate(PROBE))
            w, U = np.linalg.eigh(C.T @ Q @ C)
            U = U * np.where(U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])] < 0, -1.0, 1.0)   # sign convention
            self._shell[gi] = C @ U
        return self._shell[gi]

    def full_vector(self, c):
        v = np.zeros(self.nao_super)
        v[self.ao] = c
        return v

    # ------------------------------------------------------------------ states
    def states(self, na, nb, fill="holes", max_states=MAX_STATES):
        """Configuration states of the member with ``na``/``nb`` electrons: a list of ``(description, constraints)`` where
        ``constraints`` is a list of ``(AO vector of the supermolecule, spin, population)``.  Empty if the filling is unique."""
        fill = str(fill).lower()
        if fill not in ("holes", "occupied"):
            raise ValueError("fill must be 'holes' or 'occupied'")
        per_channel = []
        for s, n in enumerate((na, nb)):
            ch = self._channel(n)
            if ch is None:
                per_channel.append([()])
                continue
            gi, nocc = ch
            d = len(self.groups[gi])
            orbs = self.shell_orbitals(gi)
            if fill == "holes":
                combos = itertools.combinations(range(d), d - nocc)
                pop = 0.0
            else:
                combos = itertools.combinations(range(d), nocc)
                pop = 1.0
            per_channel.append([tuple((self.full_vector(orbs[:, k]), s, pop, f"{'ab'[s]}{'-hole' if pop == 0 else '-occ'}{k}")
                                      for k in combo) for combo in combos])
        n_states = len(per_channel[0]) * len(per_channel[1])
        if n_states <= 1:
            return []
        if n_states > max_states:
            raise ValueError(f"the automatic configuration ensemble would have {n_states} states (limit {max_states}); "
                             "give the spin of the fragment (fspin) or choose the states by hand")
        out = []
        for ca, cb in itertools.product(*per_channel):
            cons = list(ca) + list(cb)
            out.append((" ".join(c[3] for c in cons), [(c[0], c[1], c[2]) for c in cons]))
        return out

    # ------------------------------------------------------------------ report
    def ao_labels(self, gi, top=2):
        labels = [lab.strip() for lab in self.mol.ao_labels()]
        C = self.C[:, self.groups[gi]]
        w = np.einsum("ik,ik->i", C, self.S @ C) / len(self.groups[gi])
        idx = np.argsort(-np.abs(w))[:top]
        return ", ".join(f"{labels[i]} ({w[i]:.2f})" for i in idx)

    def __repr__(self):
        na, nb = self.hund()
        lines = [f"Shells of fragment {self.frag.index} with {self.N} electrons (Hund: N_alpha={na}, N_beta={nb}, 2S={na - nb}); "
                 "orbitals in order of energy",
                 "  group  degeneracy  energy (Eh)   electrons   main AO components"]
        rem = self.N
        for gi, g in enumerate(self.groups):
            k = min(rem, 2 * len(g))
            rem -= k
            lines.append(f"  {gi:5d}  {len(g):10d}  {self.energy[g[0]]:11.5f}  {k:6d}/{2 * len(g):<3d}   {self.ao_labels(gi)}")
            if k == 0:
                break
        return "\n".join(lines)

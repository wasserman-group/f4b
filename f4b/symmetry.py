"""Symmetry between fragments (``frag_symm``).

An operation ``R`` (orthogonal 3x3 matrix about the centroid of the atoms) is *admissible* when it maps the set of
atoms (same species) onto itself and every fragment onto a fragment with identical ensemble members.  If the
partition potential is invariant under the admissible operations, the density of the fragment ``R(alpha)`` is obtained
from the one of fragment ``alpha`` by an AO transformation, so only one fragment per orbit has to be solved.

With ``A(r)`` the (npts x nao) AO values, the transformation ``U`` is defined by ``A(R r) = A(r) U``
(exact for complete shells, found by least squares).  Then::

    D_beta = U^-1 D_alpha U^-T,        Vp invariant  <=>  U^T Vp U = Vp.
"""
import itertools
import warnings

import numpy as np
from pyscf.dft import numint


def _members_key(frag):
    return tuple((m.na, m.nb, round(m.weight, 12), round(m.flip, 12)) for m in frag.members), \
        round(frag.Z, 8), frag.nelec0


class FragSymmetry:
    def __init__(self, fmol, tol=1e-4):
        self.fmol = fmol
        self.mol = fmol.mol
        self.tol = tol
        self.ops = []            # dicts: R, perm (atoms), fperm (fragments), U, Uinv
        self.orbits = {}         # fragment index (0-based) -> (representative, op index)
        self.reps = []
        self._build()

    # ------------------------------------------------------------------ operations
    def _admissible_rotations(self):
        mol, tol = self.mol, self.tol
        X = mol.atom_coords()
        sym = [mol.atom_symbol(i) for i in range(mol.natm)]
        c = X.mean(axis=0)
        Y = X - c
        n = len(Y)
        norms = np.linalg.norm(Y, axis=1)
        if norms.max() < tol:
            return [(np.eye(3), list(range(n)))]
        i0 = int(np.argmax(norms))
        u0 = Y[i0] / norms[i0]
        perp = Y - np.outer(Y @ u0, u0)
        pn = np.linalg.norm(perp, axis=1)
        basis = [i0]
        if pn.max() > tol:
            i1 = int(np.argmax(pn))
            u1 = perp[i1] / pn[i1]
            basis.append(i1)
            perp2 = perp - np.outer(perp @ u1, u1)
            pn2 = np.linalg.norm(perp2, axis=1)
            if pn2.max() > tol:
                basis.append(int(np.argmax(pn2)))
        k = len(basis)
        B = Y[basis].T                                       # 3 x k

        def images(i, prev):
            out = []
            for j in range(n):
                if sym[j] != sym[i] or abs(norms[j] - norms[i]) > tol:
                    continue
                if all(abs(Y[j] @ Y[jp] - Y[i] @ Y[ip]) < tol for ip, jp in zip(basis, prev)):
                    out.append(j)
            return out

        cands = []
        for j0 in images(basis[0], []):
            if k == 1:
                cands.append(self._householder(Y[basis[0]], Y[j0]))
                continue
            for j1 in images(basis[1], [j0]):
                if k == 2:
                    for s in (1.0, -1.0):
                        Bm = np.column_stack([B, np.cross(B[:, 0], B[:, 1])])
                        Ym = np.column_stack([Y[j0], Y[j1], s * np.cross(Y[j0], Y[j1])])
                        cands.append(Ym @ np.linalg.inv(Bm))
                else:
                    for j2 in images(basis[2], [j0, j1]):
                        Ym = np.column_stack([Y[j0], Y[j1], Y[j2]])
                        cands.append(Ym @ np.linalg.inv(B))
        out = []
        for R in cands:
            if np.abs(R.T @ R - np.eye(3)).max() > 10 * tol:
                continue
            Z = Y @ R.T
            perm = []
            ok = True
            for i in range(n):
                d = np.linalg.norm(Y - Z[i], axis=1)
                j = int(np.argmin(d))
                if d[j] > 10 * tol or sym[j] != sym[i]:
                    ok = False
                    break
                perm.append(j)
            if ok and len(set(perm)) == n:
                out.append((R, perm))
        return out

    @staticmethod
    def _householder(a, b):
        w = a - b
        if np.linalg.norm(w) < 1e-12:
            return np.eye(3)
        w = w / np.linalg.norm(w)
        return np.eye(3) - 2.0 * np.outer(w, w)

    def _build(self):
        fmol, mol = self.fmol, self.mol
        frags = fmol.frags
        atom2frag = {a: k for k, f in enumerate(frags) for a in f.atoms}
        keys = [_members_key(f) for f in frags]
        plain = [not any(getattr(m, "config", None) is not None for m in f.members) for f in frags]
        seen = []
        cand = []
        for R, perm in self._admissible_rotations():
            if any(np.abs(R - R2).max() < 1e-6 for R2 in seen):
                continue
            seen.append(R)
            fperm = []
            ok = True
            for k, f in enumerate(frags):
                imgs = {atom2frag[perm[a]] for a in f.atoms}
                if len(imgs) != 1:
                    ok = False
                    break
                kb = imgs.pop()
                if len(frags[kb].atoms) != len(f.atoms) or keys[kb] != keys[k] or not (plain[k] and plain[kb]):
                    ok = False
                    break
                fperm.append(kb)
            if ok:
                cand.append((R, perm, fperm))
        rng = np.random.RandomState(7)
        X = mol.atom_coords()
        npts = max(6 * mol.nao, 300)
        pts = X[rng.randint(0, len(X), npts)] + rng.normal(scale=1.2, size=(npts, 3))
        A0 = numint.eval_ao(mol, pts)
        S = mol.intor("int1e_ovlp")
        c = X.mean(axis=0)
        for R, perm, fperm in cand:
            A1 = numint.eval_ao(mol, (pts - c) @ R.T + c)
            U = np.linalg.lstsq(A0, A1, rcond=None)[0]
            res = np.abs(A0 @ U - A1).max()
            if res > 1e-7 or np.abs(U.T @ S @ U - S).max() > 1e-6:
                warnings.warn(f"frag_symm: an AO transformation could not be found (residual {res:.1e}); "
                              "an operation is ignored")
                continue
            self.ops.append(dict(R=R, perm=perm, fperm=fperm, U=U, Uinv=np.linalg.inv(U)))
        if not self.ops:                                     # identity is always present unless U failed
            self.ops = [dict(R=np.eye(3), perm=list(range(mol.natm)), fperm=list(range(len(frags))),
                             U=np.eye(mol.nao), Uinv=np.eye(mol.nao))]
        nf = len(frags)
        for k in range(nf):
            if k in self.orbits:
                continue
            self.orbits[k] = (k, None)
            self.reps.append(k)
            for gi, g in enumerate(self.ops):
                kb = g["fperm"][k]
                if kb not in self.orbits:
                    self.orbits[kb] = (k, gi)
        self.stabilizer = {k: [g for g in self.ops if g["fperm"][k] == k and not np.allclose(g["R"], np.eye(3))]
                           for k in self.reps}

    # ------------------------------------------------------------------ use
    @property
    def nontrivial(self):
        return len(self.ops) > 1

    def map_density(self, D, gi):
        Ui = self.ops[gi]["Uinv"]
        return Ui @ D @ Ui.T

    def project(self, V):
        """Average of ``V`` (nao x nao, or (2, nao, nao)) over the admissible operations (invariant part)."""
        out = np.zeros_like(V)
        for g in self.ops:
            U = g["U"]
            out += np.swapaxes(U, 0, 1) @ V @ U
        return out / len(self.ops)

    def asymmetry(self, k, D):
        """Max deviation of D (fragment k, a representative) from invariance under its stabilizer."""
        dev = 0.0
        for g in self.stabilizer.get(k, []):
            dev = max(dev, np.abs(g["Uinv"] @ D @ g["Uinv"].T - D).max())
        return dev

    def __repr__(self):
        return (f"<FragSymmetry {len(self.ops)} operations, representatives: "
                f"{list(self.reps)}, orbits: "
                f"{ {k: v[0] for k, v in self.orbits.items()} }>")

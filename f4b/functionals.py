"""Density functionals evaluated on the supermolecule grid and assembled into AO matrices.

Only LDA/GGA functionals are supported (kinetic ones included, through libxc as shipped with PySCF).
A functional can be a libxc name (``'PBE'``, ``'GGA_K_VW'``...), one of the kinetic aliases ``'vW'``, ``'TF'``,
``'TFvW'`` or a user callable ``f(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb) -> (e, vrho, vsigma)`` with an
attribute ``xctype`` (``'LDA'`` or ``'GGA'``) where ``e`` is the energy density per volume, ``vrho`` has shape
(ng, 2) and ``vsigma`` (ng, 3) (None for LDA).
"""
import numpy as np
from pyscf import scf
from pyscf.dft import gen_grid, libxc, numint

TS_ALIASES = {"vw": "GGA_K_VW", "tf": "LDA_K_TF", "tfvw": "GGA_K_TFVW"}


class Functional:
    def __init__(self, spec, kind="xc"):
        self.spec = spec
        if callable(spec):
            self.fn = spec
            self.name = getattr(spec, "__name__", "user")
            self.xctype = getattr(spec, "xctype", "GGA")
        else:
            self.fn = None
            name = TS_ALIASES.get(spec.lower(), spec) if kind == "ts" else spec
            self.name = name
            self.xctype = libxc.xc_type(name)
            if self.xctype not in ("LDA", "GGA"):
                raise NotImplementedError(f"functional '{spec}' is {self.xctype}; only LDA/GGA are supported")
            if libxc.is_hybrid_xc(name):
                raise NotImplementedError(f"hybrid functional '{spec}' is not supported for the supermolecule")

    def eval(self, ra, rb, ga=None, gb=None):
        """Energy density per volume, vrho (ng, 2), vsigma (ng, 3) or None."""
        ra = np.maximum(ra, 0.0)
        rb = np.maximum(rb, 0.0)
        if self.xctype == "GGA":
            saa = np.einsum("xp,xp->p", ga, ga)
            sab = np.einsum("xp,xp->p", ga, gb)
            sbb = np.einsum("xp,xp->p", gb, gb)
        if self.fn is not None:
            if self.xctype == "GGA":
                return self.fn(ra, rb, saa, sab, sbb)
            return self.fn(ra, rb, None, None, None)
        if self.xctype == "GGA":
            rho = np.array([np.vstack([ra, ga]), np.vstack([rb, gb])])
        else:
            rho = np.array([ra[None, :], rb[None, :]])
        exc, vxc = libxc.eval_xc(self.name, rho, spin=1, deriv=1)[:2]
        e = exc * (ra + rb)
        vrho = np.nan_to_num(vxc[0])
        vsig = np.nan_to_num(vxc[1]) if self.xctype == "GGA" else None
        return np.nan_to_num(e), vrho, vsig


def rho_grad(ao, D):
    """Density and gradient (3, ng) for a symmetric density matrix."""
    c0 = ao[0] @ D
    rho = np.einsum("pi,pi->p", c0, ao[0])
    g = 2.0 * np.einsum("pi,xpi->xp", c0, ao[1:4])
    return rho, g


class GridData:
    """Integration grid of the supermolecule split in blocks, with cached AO values (value + gradient)."""

    def __init__(self, mol, level=6, block_size=20000, max_cache_mb=2000):
        self.mol = mol
        g = gen_grid.Grids(mol)
        g.level = level
        g.build(with_non0tab=False)
        self.coords, self.weights = g.coords, g.weights
        self.ngrid = len(self.weights)
        self.slices = [slice(i, min(i + block_size, self.ngrid)) for i in range(0, self.ngrid, block_size)]
        self.nao = mol.nao
        self._cache = None
        if 4 * self.nao * self.ngrid * 8 / 1e6 < max_cache_mb:
            self._cache = [self._ao(s) for s in self.slices]

    def _ao(self, s):
        return numint.eval_ao(self.mol, self.coords[s], deriv=1)

    def blocks(self):
        """Yield (coords, weights, ao[4, ng, nao]) blocks."""
        for i, s in enumerate(self.slices):
            ao = self._cache[i] if self._cache is not None else self._ao(s)
            yield self.coords[s], self.weights[s], ao

    # ------------------------------------------------------------------
    def integrate_density(self, D):
        return sum(np.dot(w, rho_grad(ao, D)[0]) for _, w, ao in self.blocks())

    def eval_density(self, D, coords):
        """Density of ``D`` (AO matrix of the supermolecule) on arbitrary points."""
        ao = numint.eval_ao(self.mol, coords)
        return np.einsum("pi,pi->p", ao @ D, ao)

    def xc(self, func, Da, Db, Dq=None, Dsum=None, Qeps=1e-8):
        """Energy and spin AO potentials of ``func`` on the densities (Da, Db).

        If ``Dq=(Dqa, Dqb)`` and ``Dsum=(Dsa, Dsb)`` are given, every spin potential is multiplied by the
        local weight n_q,s / (n_sum,s + Qeps) ("local-Q"; used by the non-additive approximations).
        """
        nao = self.nao
        Va, Vb = np.zeros((nao, nao)), np.zeros((nao, nao))
        E = 0.0
        gga = func.xctype == "GGA"
        for _, w, ao in self.blocks():
            ra, ga = rho_grad(ao, Da)
            rb, gb = rho_grad(ao, Db)
            e, vr, vs = func.eval(ra, rb, ga, gb)
            E += np.dot(w, e)
            if Dq is None:
                qa = qb = 1.0
            else:
                qa = rho_grad(ao, Dq[0])[0] / (rho_grad(ao, Dsum[0])[0] + Qeps)
                qb = rho_grad(ao, Dq[1])[0] / (rho_grad(ao, Dsum[1])[0] + Qeps)
            for V, q, v0, g, go, vss, vso in ((Va, qa, vr[:, 0], ga, gb, vs[:, 0] if gga else 0, vs[:, 1] if gga else 0),
                                              (Vb, qb, vr[:, 1], gb, ga, vs[:, 2] if gga else 0, vs[:, 1] if gga else 0)):
                aow = ao[0] * (0.5 * q * v0 * w)[:, None]
                if gga:
                    gv = q * (2.0 * vss * g + vso * go) * w
                    aow += np.einsum("xp,xpi->pi", gv, ao[1:4])
                V += ao[0].T @ aow
        return E, Va + Va.T, Vb + Vb.T


class Coulomb:
    """Hartree matrix builder (ERIs stored in core when small)."""

    def __init__(self, mol, max_incore_nao=120):
        self.mol = mol
        self.mf = scf.RHF(mol)
        if mol.nao <= max_incore_nao:
            self.mf._eri = mol.intor("int2e", aosym="s8")

    def J(self, D):
        return self.mf.get_j(self.mol, D)

    def EH(self, D):
        return 0.5 * np.einsum("ij,ij", self.J(D), D)

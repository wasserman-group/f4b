"""Kohn-Sham inversion of the fragment density on the supermolecule (Wu-Yang through n2v).

The n2v PySCF engine is replaced by one that (i) builds the three-orbital overlap tensor S3 analytically
(libcint ``int3c1e``) instead of on a coarse grid, (ii) builds the potential basis set correctly
(coordinates in Bohr), (iii) reuses the Hartree builder of the package.
"""
import contextlib
import io

import numpy as np
from opt_einsum import contract
from pyscf import df, gto
from n2v import Inverter
from n2v.engines import PySCFEngine


class _Engine(PySCFEngine):
    coulomb = None

    def set_system(self, molecule, basis, ref=1, pbs="same"):
        self.mol = molecule
        self.basis_str = basis
        self.pbs_str = pbs
        self.ref = ref
        self.pbs = None
        if pbs != "same":
            self.pbs = gto.M(atom=molecule._atom, unit="Bohr", basis=pbs, verbose=0)
        self.nalpha, self.nbeta = molecule.nelec

    def initialize(self):
        self.nbf = self.mol.nao
        self.npbs = self.nbf if self.pbs is None else self.pbs.nao

    def get_S3(self):
        aux = self.mol if self.pbs is None else self.pbs
        return df.incore.aux_e2(self.mol, aux, intor="int3c1e", aosym="s1")

    def compute_hartree(self, Da, Db=None):
        Db = Da if Db is None else Db
        return [self.coulomb.J(Da), self.coulomb.J(Db)]


class _Inverter(Inverter):
    """n2v Inverter with a consistent restricted Lagrangian.

    In n2v (<= 0.5) the restricted Lagrangian is multiplied by 2 while its gradient and Hessian are not, which
    makes the trust-region optimizer reject steps for densities that are not trivially v-representable.
    """

    @property
    def ct(self):
        # stock n2v (PyPI) builds the guide Hartree potential from ``ct``; f4b's engine takes the target densities
        return self.Dt

    @ct.setter
    def ct(self, value):
        pass

    def lagrangian_wy(self, v):
        val = super().lagrangian_wy(v)
        if self.ref != 1:
            return val
        # n2v's restricted gradient/Hessian contain the regularization as 4*lambda*T*v (and no factor 2 on the
        # main term); with the factor 2 removed from the function this requires -2*lambda*v.T.v (n2v: -lambda*norm)
        val *= 0.5
        if self.lambda_reg is not None:
            val += 0.5 * self.lambda_reg * self.regul_norm
        return val


class InversionResult:
    pass


class KSInverter:
    """Reusable inverter for a fixed supermolecule (warm-starts the potential coefficients)."""

    accept_gtol = 1e-5

    def __init__(self, mol, coulomb, pbs="same", guide="hartree", method="WuYang", gtol=1e-6, maxiter=200,
                 opt_method="trust-krylov", reg=None, auto_reg=(1e-4, 1e-3), verbose=0):
        if method.lower() != "wuyang":
            raise NotImplementedError("only the Wu-Yang KS inversion is available (KSinv='WuYang')")
        self.mol, self.coulomb = mol, coulomb
        self.pbs, self.guide, self.gtol, self.maxiter = pbs, guide, gtol, maxiter
        self.opt_method, self.reg, self.auto_reg, self.verbose = opt_method, reg, auto_reg, verbose
        self._inv = {}
        self._v0 = {}

    def _get_inverter(self, ref, nel):
        key = (ref, nel)
        if key not in self._inv:
            inv = _Inverter(engine="pyscf")
            eng = _Engine()
            eng.coulomb = self.coulomb
            inv.eng = eng
            mol = self.mol
            if nel != tuple(mol.nelec):
                mol = mol.copy()
                mol.spin = nel[0] - nel[1]
            inv.set_system(mol, "same", ref=ref, pbs=self.pbs)
            if not hasattr(inv, "T_pbs"):
                inv.T_pbs = inv.T
            self._inv[key] = inv
        return self._inv[key]

    def invert(self, Da, Db, guide=None, nelec=None):
        """Invert spin densities (AO matrices). ``nelec=(na, nb)`` defaults to the electron numbers of ``mol``.

        ``reg='auto'``: plain Wu-Yang first; if the optimizer fails (typical for densities that are not
        v-representable in the basis) retry with lambda-regularization, walking through the ladder ``auto_reg``.
        """
        regs = [None, *np.atleast_1d(self.auto_reg)] if self.reg == "auto" else [self.reg]
        best = None
        for i, reg in enumerate(regs):
            try:
                res = self._invert(Da, Db, guide, nelec, reg)
            except np.linalg.LinAlgError:
                if i == len(regs) - 1 and best is None:
                    raise
                continue
            if res.success:
                return res
            if best is None or res.grad_norm < best.grad_norm:
                best = res
        return best

    def _invert(self, Da, Db, guide, nelec, reg):
        guide = guide or self.guide
        nelec = tuple(nelec if nelec is not None else self.mol.nelec)
        ref = 1 if (nelec[0] == nelec[1] and np.allclose(Da, Db, atol=1e-10)) else 2
        inv = self._get_inverter(ref, nelec)
        inv.Dt = [np.asarray(Da), np.asarray(Db)]
        key = (ref, nelec, reg)
        inv.v_pbs = np.array(self._v0[key]) if key in self._v0 else np.zeros(inv.npbs if ref == 1 else 2 * inv.npbs)
        if guide != "none" and guide not in ("hartree", "fermi_amaldi"):
            raise ValueError("Vguide must be 'hartree' or 'fermi_amaldi'")
        for k, opt_method in enumerate((self.opt_method, "trust-exact")):
            try:
                with contextlib.redirect_stdout(io.StringIO()):         # n2v prints at every call
                    inv.invert("wuyang", guide_components=guide, opt_max_iter=self.maxiter, gtol=self.gtol, reg=reg,
                               opt_method=opt_method)
                break
            except ValueError as err:       # stock n2v raises when scipy reports failure; v_pbs/opt_info are already set
                if "unsucessful" not in str(err):
                    raise
                break
            except np.linalg.LinAlgError:          # trust-krylov occasionally takes a wild step
                if k == 1 or opt_method == "trust-exact":
                    raise
                inv.v_pbs = np.zeros_like(inv.v_pbs)
        inv.v_pbs = np.real(inv.v_pbs)
        inv._diagonalize_with_potential_pbs(inv.v_pbs)
        grad_norm = float(np.linalg.norm(inv.grad)) if hasattr(inv, "grad") else np.nan
        # scipy's ``success`` is unreliable here (often False although |grad| ~ 1e-8): also accept a small gradient
        success = bool(getattr(inv.opt_info, "success", True)) or grad_norm < self.accept_gtol
        if success:
            self._v0[key] = np.array(inv.v_pbs)
        else:
            self._v0.pop(key, None)         # do not warm-start from a failed optimization

        n = inv.npbs
        vr_a = contract("ijk,k->ij", inv.S3, inv.v_pbs[:n])
        vr_b = vr_a if ref == 1 else contract("ijk,k->ij", inv.S3, inv.v_pbs[n:])
        J = inv.J0[0] + inv.J0[1]
        res = InversionResult()
        res.vxc = (inv.va + vr_a - J, inv.vb + vr_b - J)
        res.vrest = (vr_a, vr_b)
        res.vguide = (inv.va, inv.vb)
        res.D = (np.real(inv.Da), np.real(inv.Db))
        res.eps = (inv.eigvecs_a, inv.eigvecs_b)
        res.nelec = nelec
        res.v_pbs = np.array(inv.v_pbs)
        res.grad_norm = grad_norm
        res.success = success
        res.dD = float(np.abs(res.D[0] + res.D[1] - Da - Db).sum())
        res.C = (np.array(inv.Ca), np.array(inv.Cb))
        res.T = inv.T
        res.V = inv.V
        return res

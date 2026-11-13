"""Partition-DFT drivers: Functional (Eq. 16), Inversion (Eq. 18) and Reference (Eq. 22) methods.

The outer loop is a gradient-like update of the partition potential (AO matrix of the supermolecule)

    Vp <- Vp + lam * sym(dVp),     sym(A) = (A + A^T)/2

where ``dVp`` is the method-dependent residual.  Fragment densities come from :mod:`f4b.fragment`.
"""
import warnings

import numpy as np
from pyscf import dft
from pyscf.dft import numint

from . import fragment
from .functionals import Coulomb, Functional, GridData
from .inversion import KSInverter
from .utils import Timer, log

_METHODS = {
    "functional": "functional", "functional-t": "functional", "-t": "functional",
    "inversion": "inversion",
    "reference": "reference",
}
_PLANNED = ("direct", "functional-txcnad", "-txcnad", "xcnad-t", "-tnad-t", "functional-tnad-t",
            "inversion-xcnad", "inversion-tnad-t")


class PDFT:
    # defaults (see PLAN.md)
    xc = "PBE"
    fxc = None
    Ts = "vW"
    KSinv = "WuYang"
    Vguide = "hartree"
    pbs = "same"
    frag_symm = "none"          # 'none' or 'auto' (True): solve one fragment per symmetry orbit, symmetrize Vp
    frag_symm_tol = 1e-4        # bohr, tolerance of the symmetry detection
    dEf_min = 1e-6
    L1_min = 1e-3
    maxiter = None
    lam = 0.5
    step = None                 # None -> 'adaptive' for reference, lam otherwise
    grid_level = 6
    fgrid_level = None
    inv_gtol = 1e-6
    inv_reg = 1e-3              # lambda-regularization of the Wu-Yang inversions in the loop; None; 'auto' (plain, then 1e-4, 1e-3); or a float
    diverge_factor = 20.0
    final_gtol = 1e-8
    final_reg = "auto"          # same options as inv_reg, for the final inversion of n_f (accuracy of E_total)
    mu_shift = False
    accel = None                # None or 'diis': Pulay extrapolation of Vp from the gauge-free dVp residuals
    Vp0 = None                  # initial guess of Vp (AO matrix, or (2,nao,nao) when spin-resolved)
    optimize_omega = None       # None -> True when the ensembles contain configuration groups
    omega_maxiter = 20
    diis_space = 8
    diis_start = 2
    spin_vp = None              # None -> spin-resolved Vp only if the supermolecule is spin-polarized
    verbose = 1
    nref = None
    ref_dm = None
    one_electron = None         # None -> True when the supermolecule has one electron: exact treatment (UHF fragments,
                                # dVp = dTs_vW/dn + Vext, E = Ts_vW + Eext + Enn), no Hartree/xc approximation

    def __init__(self, fmol, method="inversion", **kwargs):
        if fmol.frags is None:
            fmol.build()
        self.fmol = fmol
        self.mol = fmol.mol
        key = method.lower().replace(" ", "")
        if key in _PLANNED:
            raise NotImplementedError(f"method '{method}' is planned but not implemented yet (see PLAN.md)")
        if key not in _METHODS:
            raise ValueError(f"unknown method '{method}'. Available: inversion, functional-T, reference")
        self.method = _METHODS[key]
        for k, v in kwargs.items():
            if not hasattr(type(self), k):
                raise TypeError(f"unknown option '{k}'")
            setattr(self, k, v)
        fs = self.frag_symm
        self._use_symm = (fs.lower() not in ("none", "false", "no", "off")) if isinstance(fs, str) else bool(fs)
        if isinstance(fs, str) and fs.lower() not in ("none", "false", "no", "off", "auto", "dimer", "true", "yes", "on"):
            raise ValueError("frag_symm must be 'none' or 'auto' (equivalently True/False)")
        self.timer = Timer()
        self.Vp = None
        self._spin_vp = False
        self.e_tot = self.Ep = self.e_f = None
        self.scf_summary = {}
        self.converged = False
        self.history = []
        self._built = False

    # ------------------------------------------------------------------ setup
    def build(self):
        if self._built:
            return self
        mol = self.mol
        self.Vguide = str(self.Vguide).lower().replace("-", "_")
        if self.Vguide not in ("hartree", "fermi_amaldi"):
            raise ValueError("Vguide must be 'hartree' (default; then vxc = vrest) or 'fermi_amaldi'")
        self.fxc_name = self.fxc if self.fxc is not None else self.xc
        self.xcf = Functional(self.xc, "xc")
        if self.one_electron is None:
            self._1e = mol.nelectron == 1
        else:
            self._1e = bool(self.one_electron)
            if self._1e and mol.nelectron != 1:
                raise ValueError("one_electron=True needs a supermolecule with exactly one electron")
        if self._1e:
            if any(m.na + m.nb > 1 for f in self.fmol.frags for m in f.members):
                raise ValueError("one-electron mode: every fragment member must have at most one electron")
            if any(m.group is not None for f in self.fmol.frags for m in f.members):
                raise ValueError("one-electron mode does not support configuration ensembles")
        self.tsf = Functional("vW" if self._1e else self.Ts, "ts") if (self.method == "functional" or self._1e) else None
        with self.timer("setup"):
            self.grid = GridData(mol, level=self.grid_level)
            self.coulomb = Coulomb(mol)
            self.S = mol.intor("int1e_ovlp")
            self.Vext = mol.intor("int1e_nuc")
            self.frags = fragment.FragmentSystem(
                self.fmol, fxc=self.fxc_name, grid_level=self.fgrid_level or self.grid_level,
                grid=None if self.fgrid_level else (self.grid.coords, self.grid.weights), hf=self._1e)
            self.inverter = KSInverter(mol, self.coulomb, pbs=self.pbs, guide=self.Vguide, method=self.KSinv,
                                       gtol=self.inv_gtol, reg=self.inv_reg)
            if self.method == "reference" or self.nref is not None or self.ref_dm is not None:
                self._setup_reference()
        if self.maxiter is None:
            self.maxiter = 5000 if self.method == "reference" else 100
        self._cfg_groups = {}
        for f in self.fmol.frags:
            for m in f.members:
                if m.group is not None:
                    self._cfg_groups.setdefault(m.group, []).append(m)
        if self._cfg_groups and self._use_symm:
            warnings.warn("frag_symm is ignored for configuration ensembles")
            self._use_symm = False
        self.sym = None
        if self._use_symm:
            from .symmetry import FragSymmetry
            self.sym = FragSymmetry(self.fmol, tol=self.frag_symm_tol)
            if self.sym.nontrivial:
                self.frags.sym = self.sym
                log(self.verbose, 1, f"frag_symm: {len(self.sym.ops)} operations; fragments solved: "
                                     f"{list(self.sym.reps)} of {self.fmol.nfrag}")
            else:
                log(self.verbose, 1, "frag_symm: no symmetry operation relates the fragments; nothing to do")
                self.sym = None
        self._built = True
        return self

    def _setup_reference(self):
        if self.nref is None and self.ref_dm is None:
            raise ValueError("a reference density is needed (method='reference' or L1()): set nref (callable of coordinates, or "
                             "(mol, dm)) or ref_dm (density matrix in the supermolecule AO basis)")
        nref = self.nref
        if nref is None:
            dm = self.ref_dm
            dm = dm if np.ndim(dm) == 2 else np.sum(dm, axis=0)
            nref = lambda coords: self.grid.eval_density(dm, coords)           # noqa: E731
        elif isinstance(nref, (tuple, list)):
            rmol, dm = nref
            dm = dm if np.ndim(dm) == 2 else np.sum(dm, axis=0)
            from pyscf.dft import numint
            nref = lambda coords: numint.eval_rho(rmol, numint.eval_ao(rmol, coords), dm)    # noqa: E731
        self._nref_fn = lambda coords: np.asarray(nref(coords)).reshape(-1)       # noqa: E731
        self._nref_blocks = [self._nref_fn(c) for c, _, _ in self.grid.blocks()]

    # ------------------------------------------------------------ target spin densities
    def _target(self, Da, Db):
        """Inversion target: only the total electron number must match the supermolecule; a different fragment spin split is
        symmetrized (closed shell) or rescaled per channel (open shell)."""
        na, nb = self.mol.nelec
        Na, Nb = np.einsum("ij,ij", Da, self.S), np.einsum("ij,ij", Db, self.S)
        if abs(Na - na) < 1e-4 and abs(Nb - nb) < 1e-4:
            return Da, Db
        if abs(Na + Nb - (na + nb)) > 1e-4:
            raise ValueError(f"fragment densities integrate to {Na + Nb:.4f} electrons in total but the supermolecule has "
                             f"{na + nb}; check the ensemble weights")
        # Only the total electron number is fixed; the spin split of the fragments may differ from the supermolecule's.
        if na == nb:                              # closed shell: spin-unpolarized target
            return 0.5 * (Da + Db), 0.5 * (Da + Db)
        return Da * (na / Na if Na > 1e-8 else 0.0), Db * (nb / Nb if Nb > 1e-8 else 0.0)

    # ------------------------------------------------------------ method residuals
    def _dVp_functional(self, Da, Db):
        _, Vka, Vkb = self.grid.xc(self.tsf, Da, Db)
        if self._1e:                              # Hartree and xc cancel exactly for one electron
            if self._spin_vp:
                return np.array([Vka + self.Vext, Vkb + self.Vext]), {}
            return 0.5 * (Vka + Vkb) + self.Vext, {}
        J = self.coulomb.J(Da + Db)
        _, Vxa, Vxb = self.grid.xc(self.xcf, Da, Db)
        base = self.Vext + J
        if self._spin_vp:
            return np.array([Vka + Vxa + base, Vkb + Vxb + base]), {}
        return 0.5 * (Vka + Vkb + Vxa + Vxb) + base, {}

    def _dVp_inversion(self, Da, Db):
        ta, tb = self._target(Da, Db)
        res = self.inverter.invert(ta, tb, nelec=self.mol.nelec)
        if self._1e:                              # exact xc potential of one electron: -v_H
            Jt = self.coulomb.J(ta + tb)
            Vxa = Vxb = -Jt
        else:
            _, Vxa, Vxb = self.grid.xc(self.xcf, ta, tb)
        info = {"inv_dD": res.dD, "inv_ok": res.success}
        da, db = Vxa - res.vxc[0], Vxb - res.vxc[1]
        if self._spin_vp:
            return np.array([da, db]), info
        return 0.5 * (da + db), info

    def _gauge_free(self, dVp, Da, Db):
        """Remove the constant shift of dVp (one per spin channel if Vp is spin-resolved)."""
        if dVp.ndim == 2:
            return dVp - (np.einsum("ij,ij", dVp, Da + Db) / self.mol.nelectron) * self.S
        out = dVp.copy()
        for s, D in enumerate((Da, Db)):
            n = np.einsum("ij,ij", D, self.S)
            if n > 1e-8:
                out[s] -= (np.einsum("ij,ij", dVp[s], D) / n) * self.S
        return out

    def _L1_ref(self, Da, Db, smooth=0.0):
        L1 = 0.0
        for (c, w, ao), nref in zip(self.grid.blocks(), self._nref_blocks):
            nf = np.einsum("pi,pi->p", ao[0] @ (Da + Db), ao[0])
            d = nf - nref
            L1 += np.dot(w, np.sqrt(d * d + smooth * smooth) if smooth else np.abs(d))
        return L1

    def _dVp_reference(self, Da, Db):
        nao = self.mol.nao
        M = np.zeros((nao, nao))
        L1 = 0.0
        for (c, w, ao), nref in zip(self.grid.blocks(), self._nref_blocks):
            c0a = ao[0] @ Da
            c0b = ao[0] @ Db
            nf = np.einsum("pi,pi->p", c0a + c0b, ao[0])
            d = nf - nref
            L1 += np.dot(w, np.abs(d))
            M += ao[0].T @ (ao[0] * (w * d)[:, None])
        return M, {"L1": L1}

    # ------------------------------------------------------------ configuration weights
    def _omega_get(self):
        return np.concatenate([[m.omega for m in ms] for ms in self._cfg_groups.values()])

    def _omega_set(self, x):
        i = 0
        for ms in self._cfg_groups.values():
            for m in ms:
                m.omega = float(x[i])
                m.refresh()
                i += 1

    def _omega_normalize(self, x):
        x = np.clip(np.asarray(x, float), 0.0, None)
        i = 0
        for ms in self._cfg_groups.values():
            n = len(ms)
            tot = x[i:i + n].sum()
            x[i:i + n] = x[i:i + n] / tot if tot > 0 else 1.0 / n
            i += n
        return x

    def _omega_objective(self, x, fr):
        self._omega_set(self._omega_normalize(x))
        self.frags.assemble(fr)
        Da, Db = fr.Dfa, fr.Dfb
        if self.method == "reference":
            return self._L1_ref(Da, Db, smooth=1e-7)
        dVp, _ = getattr(self, "_dVp_" + self.method)(Da, Db)
        dVp = 0.5 * (dVp + np.swapaxes(dVp, -1, -2))
        if self._spin_vp and dVp.ndim == 2:
            dVp = np.array([dVp, dVp])
        dg = self._gauge_free(dVp, Da, Db)
        return float(np.sqrt(np.sum(dg * (self._Sinv @ dg @ self._Sinv))))

    def _optimize_omega(self, fr):
        """Choose the configuration weights: minimize the L1 error (reference) or the gauge-free norm of dVp."""
        from scipy.optimize import minimize
        x0 = self._omega_get()
        sizes = [len(ms) for ms in self._cfg_groups.values()]
        if sum(sizes) == len(sizes):
            return
        cons, i = [], 0
        for n in sizes:
            idx = slice(i, i + n)
            cons.append(dict(type="eq", fun=lambda x, idx=idx: np.sum(x[idx]) - 1.0))
            i += n
        eps = 1e-4 if self.method == "reference" else 2e-3
        f0 = self._omega_objective(x0, fr)
        res = minimize(self._omega_objective, x0, args=(fr,), method="SLSQP", bounds=[(0.0, 1.0)] * len(x0),
                       constraints=cons, options=dict(maxiter=self.omega_maxiter, ftol=1e-10, eps=eps))
        x = np.clip(res.x, 0.0, 1.0)
        i = 0
        for n in sizes:
            x[i:i + n] /= x[i:i + n].sum()
            i += n
        f1 = self._omega_objective(x, fr)
        if f1 > f0:                                   # keep the better of old/new
            self._omega_objective(x0, fr)

    # ------------------------------------------------------------ main loop
    def kernel(self):
        self.build()
        nao = self.mol.nao
        if self.spin_vp is None:
            has_flip = any(m.flips for f in self.fmol.frags for m in f.members)
            self._spin_vp = bool(self.mol.spin != 0) and not has_flip
        else:
            self._spin_vp = bool(self.spin_vp)
        Vp = np.zeros((2, nao, nao)) if self._spin_vp else np.zeros_like(self.S)
        if self.Vp0 is not None:
            V0 = np.asarray(self.Vp0, float)
            Vp = (np.array([V0, V0]) if V0.ndim == 2 else V0.copy()) if self._spin_vp else (V0 if V0.ndim == 2 else V0.mean(axis=0)).copy()
        Ef_prev = None
        best = None
        Sinv = self._Sinv = np.linalg.inv(self.S)
        w_, v_ = np.linalg.eigh(self.S)
        Sh_inv = (v_ / np.sqrt(w_)) @ v_.T
        self._diis_hist = []
        self.history = []
        log(self.verbose, 1, f"P-DFT method={self.method} xc={self.xc} fxc={self.fxc_name} nao={nao} "
                             f"grid={self.grid.ngrid}")
        converged = False
        for it in range(1, self.maxiter + 1):
            with self.timer("fragments"):
                try:
                    fr = self.frags.solve(Vp)
                except np.linalg.LinAlgError:
                    if best is None:
                        raise
                    warnings.warn(f"fragment SCF failed at P-DFT iteration {it} (Vp too large); "
                                  "using the best iterate. Try a smaller step (lam).")
                    break
            if self._cfg_groups and (self.optimize_omega is None or self.optimize_omega):
                with self.timer("omega"):
                    self._optimize_omega(fr)
            Da, Db = fr.Dfa, fr.Dfb
            with self.timer("residual"):
                dVp, info = getattr(self, "_dVp_" + self.method)(Da, Db)
            dVp = 0.5 * (dVp + np.swapaxes(dVp, -1, -2))
            if self.sym is not None:
                dVp = self.sym.project(dVp)
            if self._spin_vp:
                if dVp.ndim == 2:
                    dVp = np.array([dVp, dVp])
                for s_, n_ in enumerate(self.mol.nelec):
                    if n_ == 0:                 # no electrons of this spin: Vp of that channel is meaningless
                        dVp[s_] = 0.0
            if "L1" not in info and hasattr(self, "_nref_blocks"):
                info["L1"] = self._L1_ref(Da, Db)
            Ef = fr.Ef
            dEf = np.nan if Ef_prev is None else Ef - Ef_prev
            Ef_prev = Ef
            dg = self._gauge_free(dVp, Da, Db)
            gnorm = float(np.sqrt(np.sum(dg * (Sinv @ dg @ Sinv))))
            rec = dict(it=it, Ef=Ef, dEf=dEf, gnorm=gnorm, **info)
            if self._cfg_groups:
                rec["omega"] = self._omega_get().copy()
            if self.method == "reference":
                metric = info["L1"]
                done = metric < self.L1_min
            else:
                metric = gnorm
                done = it > 1 and abs(dEf) < self.dEf_min
            self.history.append(rec)
            if best is None or metric < best[0]:
                best = (metric, Vp.copy(), fr, dVp.copy(), rec)
            log(self.verbose, 1, f"  it {it:4d}  Ef = {Ef:16.10f}  dEf = {dEf:11.3e}  |dVp| = {gnorm:9.3e}"
                + (f"  L1 = {info['L1']:.3e}" if "L1" in info else "")
                + (f"  omega = {np.array2string(self._omega_get(), precision=4)}" if self._cfg_groups else ""))
            if not done and it > 5 and metric > self.diverge_factor * best[0]:
                warnings.warn(f"P-DFT iterations are diverging (it {it}); stopping and using the best iterate")
                break
            if done:
                converged = True
                break
            if self.mu_shift:
                dVp = self._gauge_free(dVp, Da, Db)
            Vnew = Vp + self._step(info) * dVp
            if self.accel:
                Vp = self._diis(Vnew, dg, Sh_inv)
            else:
                Vp = Vnew
        self.converged = converged
        if not converged:
            warnings.warn(f"P-DFT did not converge in {self.maxiter} iterations; using the best iterate")
        _, self.Vp, fr, self._dVp, rec = best
        if self._cfg_groups:
            self._omega_set(rec["omega"])
            self.omega = {g: np.array([m.omega for m in ms]) for g, ms in self._cfg_groups.items()}
        self.niter = len(self.history)
        self.frag_result = fr
        self.e_f = fr.Ef
        self.Da, self.Db = fr.Dfa, fr.Dfb
        with self.timer("final energy"):
            self._final_energy()
        return self.e_tot

    def _diis(self, Vnew, dg, Sh_inv):
        if str(self.accel).lower() != "diis":
            raise ValueError("accel must be None or 'diis'")
        err = (Sh_inv @ dg @ Sh_inv).ravel()
        self._diis_hist.append((Vnew.copy(), err))
        del self._diis_hist[:-self.diis_space]
        n = len(self._diis_hist)
        if n < self.diis_start:
            return Vnew
        B = -np.ones((n + 1, n + 1))
        B[n, n] = 0.0
        for i in range(n):
            for j in range(n):
                B[i, j] = np.dot(self._diis_hist[i][1], self._diis_hist[j][1])
        scale = np.abs(np.diag(B)[:n]).max()
        B[:n, :n] /= scale
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        try:
            c = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            return Vnew
        if not np.all(np.isfinite(c)) or np.abs(c).max() > 1e3:
            self._diis_hist = self._diis_hist[-1:]
            return Vnew
        return sum(ci * h[0] for ci, h in zip(c, self._diis_hist))

    def _step(self, info):
        step = self.step
        if step is None:
            step = "adaptive" if self.method == "reference" else self.lam
        if isinstance(step, str):
            if step.lower() != "adaptive":
                raise ValueError("step must be a number or 'adaptive' (2*L1, reference)")
            return 2.0 * info["L1"] if "L1" in info else self.lam
        return step

    # ------------------------------------------------------------ energies
    def _final_energy_1e(self, ta, tb):
        """One electron: Ts is exactly von Weizsaecker and E_H + E_xc = 0, so no inversion is needed."""
        D = ta + tb
        Ts = self.grid.xc(self.tsf, ta, tb)[0]
        Eext = np.einsum("ij,ij", self.Vext, D)
        Enn = self.mol.energy_nuc()
        self.e_tot = Ts + Eext + Enn
        self.Ep = self.e_tot - self.e_f
        self.scf_summary = dict(e_tot=self.e_tot, e_f=self.e_f, Ep=self.Ep, Ts=Ts, E_ext=Eext, E_nn=Enn,
                                E_frag=self.frag_result.E.copy(), converged=self.converged, niter=self.niter)
        if self.method == "reference" or self.nref is not None or self.ref_dm is not None:
            self.scf_summary["L1"] = self.L1()

    def _final_energy(self):
        mol = self.mol
        ta, tb = self._target(self.Da, self.Db)
        if self._1e:
            return self._final_energy_1e(ta, tb)
        reg0 = self.inverter.reg
        try:
            self.inverter.gtol = self.final_gtol
            for reg in ([self.final_reg, None] if self.final_reg is not None else [None]):
                self.inverter.reg = reg
                try:
                    res = self.inverter.invert(ta, tb, nelec=mol.nelec)
                    break
                except np.linalg.LinAlgError:
                    if reg is None:
                        raise
                    warnings.warn("regularized final KS inversion failed; retrying without regularization")
        finally:
            self.inverter.gtol, self.inverter.reg = self.inv_gtol, reg0
        self.inv_result = res
        Da, Db = res.D
        D = Da + Db
        Ts = np.einsum("ij,ij", res.T, D)
        Eext = np.einsum("ij,ij", self.Vext, D)
        EH = self.coulomb.EH(D)
        Exc = self.grid.xc(self.xcf, Da, Db)[0]
        Enn = mol.energy_nuc()
        self.e_tot = Ts + Eext + EH + Exc + Enn
        self.Ep = self.e_tot - self.e_f
        self.scf_summary = dict(e_tot=self.e_tot, e_f=self.e_f, Ep=self.Ep, Ts=Ts, E_ext=Eext, E_H=EH, E_xc=Exc,
                                E_nn=Enn, E_frag=self.frag_result.E.copy(), converged=self.converged,
                                niter=self.niter, inv_dD=res.dD)
        if self.method == "functional":
            Ts_f = self.grid.xc(self.tsf, ta, tb)[0]
            Ex_f = self.grid.xc(self.xcf, ta, tb)[0]
            Dt = ta + tb
            self.scf_summary["e_functional"] = (Ts_f + np.einsum("ij,ij", self.Vext, Dt) + self.coulomb.EH(Dt)
                                                + Ex_f + Enn)
        if self.method == "reference" or self.nref is not None or self.ref_dm is not None:
            self.scf_summary["L1"] = self.L1()

    # ------------------------------------------------------------ API
    def L1(self, grid=None):
        """L1 error of the fragment density with respect to the reference density.

        ``grid=(coords, weights)`` evaluates the integral on a user grid (e.g. a finer or a cubic grid) instead of
        the molecular integration grid.
        """
        if not hasattr(self, "_nref_blocks"):
            self._setup_reference()
        if grid is None:
            return self._dVp_reference(self.Da, self.Db)[1]["L1"]
        coords, w = np.asarray(grid[0], float), np.asarray(grid[1], float)
        return float(np.dot(w, np.abs(self.eval_density(coords) - self._nref_fn(coords))))

    def eval_density(self, coords, frag=None):
        """Density n_f (sum of the fragments, or fragment ``frag``, 0-based) on arbitrary points (n, 3) in bohr."""
        coords = np.asarray(coords, float).reshape(-1, 3)
        D = self.make_rdm1(frag)
        return self.grid.eval_density(D[0] + D[1], coords)

    def eval_Vp(self, coords, spin=None):
        """Partition potential on arbitrary points (n, 3) in bohr, from the AO matrix ``Vp``.

        ``Vp`` is an operator in the AO basis, not a local potential; its local representation is taken as
        ``v(r) = sum_mn phi_m(r) phi_n(r) (S^-1 Vp S^-1)_mn / sum_mn phi_m(r) phi_n(r) S^-1_mn`` (the diagonal of the
        projected operator divided by that of the projector), which becomes exact for a local potential in the
        complete-basis limit.  Only meaningful where the basis is well described (not far from the molecule) and
        up to a constant.  For a spin-resolved ``Vp`` the spin-averaged potential is returned unless ``spin`` (0/1) is given.
        """
        coords = np.asarray(coords, float).reshape(-1, 3)
        V = self.Vp
        if V.ndim == 3:
            V = 0.5 * (V[0] + V[1]) if spin is None else V[spin]
        Si = np.linalg.inv(self.S)
        ao = numint.eval_ao(self.mol, coords)
        num = np.einsum("pi,ij,pj->p", ao, Si @ V @ Si, ao)
        den = np.einsum("pi,ij,pj->p", ao, Si, ao)
        return num / np.maximum(den, 1e-12)

    def make_rdm1(self, frag=None):
        """Spin density matrices (2, nao, nao) of the sum of fragments, or of fragment ``frag`` (0-based)."""
        if frag is None:
            return np.array([self.Da, self.Db])
        r = self.frag_result
        return np.array([r.Da[frag], r.Db[frag]])

    @property
    def mo_energy(self):
        return self._mo("eps")

    @property
    def mo_coeff(self):
        return self._mo("C")

    @property
    def mo_occ(self):
        return self._mo("occ")

    def _mo(self, what):
        res = self.inv_result
        na, nb = res.nelec
        nmo = res.C[0].shape[1]
        occ_a = np.array([1.0] * na + [0.0] * (nmo - na))
        occ_b = np.array([1.0] * nb + [0.0] * (nmo - nb))
        if isinstance(self, RKS) and na == nb and np.allclose(res.D[0], res.D[1]):
            return {"eps": res.eps[0], "C": res.C[0], "occ": occ_a + occ_b}[what]
        return {"eps": np.array(res.eps), "C": np.array(res.C), "occ": np.array([occ_a, occ_b])}[what]

    def analyze(self):
        s = self.scf_summary
        print("---- P-DFT analysis ----")
        print(f"method {self.method}   xc {self.xc}   fxc {self.fxc_name}   converged {self.converged} "
              f"({self.niter} iterations)")
        for k in ("e_tot", "e_f", "Ep", "Ts", "E_ext", "E_H", "E_xc", "E_nn"):
            print(f"  {k:8s} = {s[k]: .10f}")
        if "L1" in s:
            print(f"  L1       = {s['L1']: .6f}")
        for i, e in enumerate(s["E_frag"]):
            print(f"  E_fragment {i} = {e: .10f}")
        print(self.timer)

    def __repr__(self):
        return f"<f4b.pdft.{type(self).__name__} method={self.method} xc={self.xc} e_tot={self.e_tot}>"


class RKS(PDFT):
    pass


class UKS(PDFT):
    pass

"""Fragment calculations with the partition potential (UKS + Vp) and ensemble handling."""
import warnings

import numpy as np
from pyscf import dft, scf


class _VpMixin:
    """Adds the partition potential ``Vp`` (AO matrix; or a (2, nao, nao) array for spin-resolved Vp) to the core Hamiltonian."""

    # not named ``Vp``: a PySCF patched as in the notebooks (SCF.Vp added in get_hcore) would count it twice
    vpart = None
    vc_term = None          # (2, nao, nao) CDFT multiplier terms added to the Fock matrices only (not to the energy)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None, diis_start_cycle=None,
                 level_shift_factor=None, damp_factor=None, fock_last=None):
        if self.vc_term is not None:
            if h1e is None:
                h1e = self.get_hcore()
            h1e = h1e + self.vc_term
        return super().get_fock(h1e, s1e, vhf, dm, cycle, diis, diis_start_cycle, level_shift_factor, damp_factor,
                                fock_last)

    def get_hcore(self, mol=None):
        h = super().get_hcore(mol)
        return h if self.vpart is None else h + self.vpart

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if self.vpart is None or self.vpart.ndim == 2:
            return super().energy_elec(dm, h1e, vhf)
        # spin-resolved Vp: PySCF's energy expression expects a spin-independent h1e
        dm = self.make_rdm1() if dm is None else dm
        e, ec = super().energy_elec(dm, scf.hf.SCF.get_hcore(self), vhf)
        return e + np.einsum("ij,ij", self.vpart[0], dm[0]) + np.einsum("ij,ij", self.vpart[1], dm[1]), ec


class VpUKS(_VpMixin, dft.uks.UKS):
    """UKS whose core Hamiltonian includes the partition potential."""


class VpUHF(_VpMixin, scf.uhf.UHF):
    """UHF (exact for one electron) whose core Hamiltonian includes the partition potential."""


class FragmentResult:
    def __init__(self, nfrag, nao):
        self.Da = [np.zeros((nao, nao)) for _ in range(nfrag)]      # ensemble-averaged spin densities per fragment
        self.Db = [np.zeros((nao, nao)) for _ in range(nfrag)]
        self.E = np.zeros(nfrag)                                    # fragment energies E_alpha = E'_alpha - Tr[D Vp]
        self.Eprime = np.zeros(nfrag)                               # energies including Vp
        self.converged = True
        self.raw = [[] for _ in range(nfrag)]                       # per fragment: dicts of the member results

    @property
    def Ef(self):
        return float(self.E.sum())

    @property
    def Dfa(self):
        return sum(self.Da)

    @property
    def Dfb(self):
        return sum(self.Db)


class FragmentSystem:
    def __init__(self, fmol, fxc="PBE", grid_level=6, conv_tol=1e-10, max_cycle=200, verbose=0, grid=None, hf=False):
        self.fmol = fmol
        self.hf = hf                    # solve the fragments with UHF instead of UKS (one-electron systems)
        self.fxc = fxc
        self.grid_level = grid_level
        self.grid = grid
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.verbose = verbose
        self._mf = {}
        self.nao = fmol.mol.nao
        self._S = None
        self.sym = None                 # FragSymmetry: solve only one fragment per orbit
        self._warned_sym = set()

    def _get_mf(self, frag, m):
        key = (frag.index, m.na, m.nb, None if m.config is None else m.config.name)
        if key not in self._mf:
            if self.hf:
                mf = VpUHF(self.fmol.member_mole(frag, m.na, m.nb))
            else:
                mf = VpUKS(self.fmol.member_mole(frag, m.na, m.nb))
                mf.xc = self.fxc
                mf.grids.level = self.grid_level
                if self.grid is not None:
                    mf.grids.coords, mf.grids.weights = self.grid
            mf.conv_tol = self.conv_tol
            mf.max_cycle = self.max_cycle
            mf.verbose = self.verbose
            mf._dm_prev = None
            if m.config is not None:
                self._set_multipliers(mf, m)
            self._mf[key] = mf
        return self._mf[key]

    def _set_multipliers(self, mf, m):
        """Constrained SCF on the isolated fragment (Vp = 0) to get the fixed multipliers of a configuration."""
        from .cdft import solve_multipliers
        st = m.config
        if st.vc is None:
            def run(vc):
                mf.vc_term = st.fock_add(vc)
                mf.kernel(dm0=mf._dm_prev)
                if not mf.converged:
                    warnings.warn(f"CDFT SCF for {st.name} did not converge")
                mf._dm_prev = mf.make_rdm1()
                return mf._dm_prev
            mf.vpart = None
            st.vc = solve_multipliers(st, run)
            mf._dm_prev = None
        mf.vc_term = st.fock_add()

    def solve(self, Vp):
        res = FragmentResult(self.fmol.nfrag, self.nao)
        sym = self.sym
        for k, frag in enumerate(self.fmol.frags):
            if sym is not None and sym.orbits[k][0] != k:
                continue
            for m in frag.members:
                if m.nelec == 0:
                    e = self.fmol.member_mole(frag, 0, 0).energy_nuc()
                    res.raw[k].append(dict(m=m, Da=0.0, Db=0.0, E=e, Ep=e))
                    continue
                mf = self._get_mf(frag, m)
                if Vp.ndim == 3 and m.flips:
                    raise ValueError("a spin-resolved Vp cannot be combined with spin-flip (ensemble 'spin') members")
                mf.vpart = Vp
                mf.kernel(dm0=mf._dm_prev)
                if not mf.converged:
                    res.converged = False
                    warnings.warn(f"fragment {frag.index} member ({m.na},{m.nb}) SCF did not converge")
                D = mf.make_rdm1()
                mf._dm_prev = D
                Vpa, Vpb = (Vp[0], Vp[1]) if Vp.ndim == 3 else (Vp, Vp)
                E = mf.e_tot - np.einsum("ij,ij", D[0], Vpa) - np.einsum("ij,ij", D[1], Vpb)
                res.raw[k].append(dict(m=m, Da=D[0], Db=D[1], E=E, Ep=mf.e_tot))
        self.assemble(res)
        for k, frag in enumerate(self.fmol.frags):
            if sym is not None and sym.orbits[k][0] == k and k not in self._warned_sym:
                dev = sym.asymmetry(k, res.Da[k] + res.Db[k])
                if dev > 1e-3:
                    self._warned_sym.add(k)
                    warnings.warn(f"frag_symm: the density of fragment {frag.index} is not invariant under its own "
                                  f"symmetry operations (max deviation {dev:.1e}); its state breaks the symmetry "
                                  "(open-shell fragment?), so the symmetrized Vp is only approximate")
        return res

    def assemble(self, res):
        """(Re)compute fragment densities and energies from the member results with the current member weights
        (changing the configuration weights ``omega`` needs no new SCF)."""
        sym = self.sym
        for k, items in enumerate(res.raw):
            if not items:
                continue
            res.Da[k] = np.zeros((self.nao, self.nao))
            res.Db[k] = np.zeros((self.nao, self.nao))
            res.E[k] = res.Eprime[k] = 0.0
            for it in items:
                m = it["m"]
                wt = m.total_weight
                res.E[k] += wt * it["E"]
                res.Eprime[k] += wt * it["Ep"]
                res.Da[k] = res.Da[k] + m.weight * it["Da"] + m.flip * it["Db"]
                res.Db[k] = res.Db[k] + m.weight * it["Db"] + m.flip * it["Da"]
        if sym is not None:
            for k in range(self.fmol.nfrag):
                rep, gi = sym.orbits[k]
                if rep != k:
                    res.Da[k] = sym.map_density(res.Da[rep], gi)
                    res.Db[k] = sym.map_density(res.Db[rep], gi)
                    res.E[k], res.Eprime[k] = res.E[rep], res.Eprime[rep]
        self.check_electrons(res)
        return res

    def check_electrons(self, res, tol=1e-6):
        """Integrated densities must carry the expected electron numbers: every member's SCF density has its (na, nb),
        the weights sum to one, and each fragment density has sum_m (w_m n_a + flip_m n_b) alpha electrons (beta alike)."""
        if self._S is None:
            self._S = self.fmol.mol.intor("int1e_ovlp")
        S = self._S
        for k, frag in enumerate(self.fmol.frags):
            wsum = sum(m.total_weight for m in frag.members)
            if abs(wsum - 1.0) > tol:
                raise RuntimeError(f"fragment {frag.index}: ensemble weights sum to {wsum:.6f}, not 1")
            for it in res.raw[k] or []:
                m = it["m"]
                na, nb = (np.einsum("ij,ij", D, S) if np.ndim(D) else 0.0 for D in (it["Da"], it["Db"]))
                if abs(na - m.na) > tol or abs(nb - m.nb) > tol:
                    raise RuntimeError(f"fragment {frag.index}: member ({m.na},{m.nb}) has integrated density "
                                       f"({na:.6f}, {nb:.6f}) electrons")
            exp_a = sum(m.weight * m.na + m.flip * m.nb for m in frag.members)
            exp_b = sum(m.weight * m.nb + m.flip * m.na for m in frag.members)
            Na, Nb = np.einsum("ij,ij", res.Da[k], S), np.einsum("ij,ij", res.Db[k], S)
            if abs(Na - exp_a) > tol or abs(Nb - exp_b) > tol:
                raise RuntimeError(
                    f"fragment {frag.index}: integrated density has (N_alpha, N_beta) = ({Na:.6f}, {Nb:.6f}) but the "
                    f"ensemble weights require ({exp_a:.6f}, {exp_b:.6f}); check the ensemble weights")

"""Reproduce the H2 / LiH runs of the Method-*.ipynb notebooks (resources/codes) with f4b and compare to KS-DFT.

    python scripts/validate_notebooks.py                  # all cases (several minutes)
    python scripts/validate_notebooks.py inversion        # one case: inversion | functional | reference | lih
    python scripts/validate_notebooks.py reference --basis cc-pvtz

Notebook numbers (for orientation, different grids/guesses/n2v version):
  Inversion  H2 R=1.4   cc-pvtz / pbs aug-cc-pvqz : E_pdft - E_KS = -7.2e-05
  Functional H2 R=1.424 cc-pvqz                   : E_pdft - E_KS =  1.4e-06   (Ts = vW)
  Reference  H2 R=1.4   aug-cc-pvqz               : E_pdft - E_KS = -7.1e-04, L1 ~ 0.05 after 100 steps
"""
import argparse
import time
import warnings

from pyscf import dft

from f4b import fgto, pdft

warnings.filterwarnings("ignore")

H2_ENS = {"spin": [0, 1]}
LIH_ENS = {"charge": {0: {"charges": (0, -1), "weights": (0.56826, 0.43174)},
                      1: {"charges": (1, 0), "weights": (0.43174, 0.56826)}},
           "spin": [0, 1]}

CASES = {
    "inversion": dict(atom="H 0 0 0; H 0 0 1.4", basis="cc-pvtz", pbs="aug-cc-pvqz", ens=H2_ENS,
                      opts=dict(method="inversion", inv_reg=1e-4, lam=1.0, maxiter=40, grid_level=5)),
    "functional": dict(atom="H 0 0 0; H 0 0 1.424", basis="cc-pvqz", pbs="same", ens=H2_ENS,
                       opts=dict(method="functional-T", Ts="vW", maxiter=40, grid_level=6)),
    "reference": dict(atom="H 0 0 0; H 0 0 1.4", basis="aug-cc-pvqz", pbs="same", ens=H2_ENS,
                      opts=dict(method="reference", step=5.0, maxiter=100, grid_level=5)),
    "lih": dict(atom="Li 0 0 0; H 0 0 3.014", basis="cc-pvdz", pbs="same", ens=LIH_ENS,
                opts=dict(method="inversion", inv_reg=1e-3, maxiter=40, grid_level=6)),
}


def run(name, basis=None):
    c = CASES[name]
    basis = basis or c["basis"]
    mol = fgto.M(atom=c["atom"], unit="bohr", basis=basis, fragments={0: [0], 1: [1]}, ensembles=c["ens"])
    ks = dft.RKS(mol.mol)
    ks.xc = "PBE"
    ks.grids.level = c["opts"]["grid_level"]
    ks.kernel()
    opts = dict(c["opts"])
    opts["ref_dm"] = ks.make_rdm1()     # reference for the method, and for the L1 report otherwise
    t = time.time()
    mf = pdft.RKS(mol, pbs=c["pbs"], verbose=0, **opts)
    mf.kernel()
    print(f"{name:11s} basis={basis:11s} E_KS={ks.e_tot:.7f} E_pdft={mf.e_tot:.7f} "
          f"dE={mf.e_tot - ks.e_tot:+.2e} Ep={mf.Ep:+.5f} L1={mf.L1():.4f} "
          f"iters={mf.niter} converged={mf.converged} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="*", help="one or more of: " + ", ".join(CASES))
    ap.add_argument("--basis", default=None, help="override the basis set of the selected case(s)")
    a = ap.parse_args()
    bad = [c for c in a.case if c not in CASES]
    if bad:
        ap.error(f"unknown case {bad}")
    bad = [c for c in a.case if c not in CASES]
    if bad:
        ap.error(f"unknown case {bad}")
    for name in a.case or ["inversion", "functional", "reference"]:
        run(name, a.basis)

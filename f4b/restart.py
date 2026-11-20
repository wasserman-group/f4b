"""Saving and loading the state of a P-DFT calculation, to resume it from the stored partition potential and densities."""
import os

import numpy as np

_KEYS = ("Vp", "members", "frag_dm", "omega")


def member_key(frag, na, nb, config=None):
    return f"{frag}|{na}|{nb}|{'' if config is None else config}"


def save_state(mf, path):
    """Write the result of ``mf.kernel()`` to ``path`` (an ``.npz`` file; the extension is added if missing).

    Stored: the partition potential ``Vp`` (best iterate), the density of every fragment-ensemble member (these seed the
    fragment SCFs), the ensemble-averaged fragment densities, the configuration weights and the final energies."""
    if getattr(mf, "frag_result", None) is None:
        raise RuntimeError("nothing to save: run kernel() first")
    fr = mf.frag_result
    keys, dms = [], []
    for k, items in enumerate(fr.raw):
        for it in items:
            m = it["m"]
            if np.ndim(it["Da"]) == 0:          # member without electrons
                continue
            keys.append(member_key(k, m.na, m.nb, None if m.config is None else m.config.name))
            dms.append(np.array([it["Da"], it["Db"]]))
    nao = mf.mol.nao
    data = dict(
        Vp=np.asarray(mf.Vp),
        member_keys=np.array(keys, dtype="U"),
        member_dm=np.array(dms).reshape(len(dms), 2, nao, nao),
        frag_dm=np.array([[a, b] for a, b in zip(fr.Da, fr.Db)]),
        omega=mf._omega_get() if mf._cfg_groups else np.zeros(0),
        e_tot=mf.e_tot, e_f=mf.e_f, Ep=mf.Ep, niter=mf.niter, converged=mf.converged, method=mf.method,
    )
    if not str(path).endswith(".npz"):
        path = str(path) + ".npz"
    np.savez(path, **data)
    return os.fspath(path)


def load_state(path):
    """Read a file written by :func:`save_state` into a dict with ``Vp``, ``members``, ``frag_dm`` and ``omega``."""
    if not str(path).endswith(".npz") and not os.path.exists(path):
        path = str(path) + ".npz"
    with np.load(path, allow_pickle=False) as f:
        st = dict(
            Vp=f["Vp"],
            members={str(k): d for k, d in zip(f["member_keys"], f["member_dm"])},
            frag_dm=f["frag_dm"],
            omega=f["omega"],
            info={k: f[k].item() for k in ("e_tot", "e_f", "Ep", "niter", "converged", "method") if k in f.files},
        )
    return st


def as_state(x):
    """A state dict from a file name or an already loaded state; ``None`` if ``x`` is neither."""
    if isinstance(x, (str, os.PathLike)):
        return load_state(x)
    if isinstance(x, dict) and any(k in x for k in ("Vp", "members")):
        return x
    return None


def fragment_guess(dm0, nfrag, nao):
    """Normalize user-given fragment densities (``dm0``) into ``(members, frag_dm)``.

    Accepted: a file name or loaded state; a dict ``{fragment: dm}``; or a list with one ``dm`` per fragment.  Each ``dm`` is
    ``(2, nao, nao)`` spin densities or a spin-summed ``(nao, nao)`` matrix (split equally between the spins)."""
    st = as_state(dm0)
    if st is not None:
        members = st.get("members", {})
        fd = st.get("frag_dm")
        frag_dm = {} if fd is None else {k: np.asarray(d) for k, d in enumerate(fd)}
    else:
        members = {}
        items = dm0.items() if isinstance(dm0, dict) else enumerate(dm0)
        frag_dm = {}
        for k, d in items:
            if d is None:
                continue
            d = np.asarray(d, float)
            frag_dm[int(k)] = np.array([0.5 * d, 0.5 * d]) if d.ndim == 2 else d
    for k, d in frag_dm.items():
        if not 0 <= k < nfrag:
            raise ValueError(f"dm0: fragment {k} does not exist (the system has {nfrag} fragments, numbered from 0)")
        if d.shape != (2, nao, nao):
            raise ValueError(f"dm0: fragment {k} density has shape {d.shape}, expected (2, {nao}, {nao}) or ({nao}, {nao}); "
                             "was it computed with the same basis and atoms?")
    for key, d in members.items():
        if np.shape(d) != (2, nao, nao):
            raise ValueError(f"dm0: stored density {key} has shape {np.shape(d)}, expected (2, {nao}, {nao}); "
                             "was it computed with the same basis and atoms?")
    return members, frag_dm


def scaled_guess(dm, na, nb, S):
    """Rescale a (2, nao, nao) density to (na, nb) electrons (a fragment-averaged density has fractional counts)."""
    out = np.array(dm, float)
    for s, n in enumerate((na, nb)):
        t = np.einsum("ij,ij", out[s], S)
        out[s] = out[s] * (n / t) if (n > 0 and t > 1e-8) else out[s] * 0.0
    return out

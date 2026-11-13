"""Fragment ensembles (fixed weights).

A fragment is described by a list of :class:`Member` objects.  Member ``(na, nb, weight, flip)``
contributes ``weight*D + flip*swap_spin(D)`` to the fragment density and ``(weight+flip)*E`` to the fragment energy,
where ``D, E`` come from a single UKS calculation with ``na`` alpha and ``nb`` beta electrons.

Grammar for ``ensembles`` (a dict keyed by ensemble type; fragments and atoms are numbered from 0)::

    ensembles = {
        'charge': {0: {'charges': (0, -1), 'weights': (0.3, 0.7)[, 'spins': (0, 1)]},   # one entry per fragment
                   1: {'charges': (None, 0), 'weights': (0.7, 0.3)}},
        'spin':   [0, 1],                                                               # fragments (or a single int)
        'config': {1: {'charge': 0, 'states': (state, ...)[, 'omega': (omega0, ...)]}}, # dict, or list of dicts per fragment
    }

The keys of ``'charge'``/``'config'`` and the items of ``'spin'`` are fragment indices.  ``charges`` are the charges of
the members relative to the neutral fragment (``fcharge``): ``+1`` = one electron less (cation), ``-1`` = one electron
more (anion), ``0`` = neutral; ``None`` = a member with no electrons at all (zero density, no calculation).  ``weights``
are the total weight of each member and ``spins`` an optional 2S (multiplicity-1) per member
(default ``N % 2``).  A ``'spin'`` entry symmetrises every open-shell member of the fragment with its spin-flipped
partner (weights 1/2 - 1/2).  ``'config'`` ``charge`` defaults to 0.

A ``'config'`` entry replaces the member of the fragment with that ``charge`` by several same-N states
(:class:`f4b.cdft.ConfigState`, e.g. ``(('F 2px', 'beta', 0),)``) whose weights ``omega_i`` (sum 1, optional start
values ``omega``) are optimized by the P-DFT loop; the member weight ``w`` is shared among them (``w*omega_i``).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Member:
    na: int
    nb: int
    weight: float
    flip: float = 0.0
    charge: int = 0                  # charge relative to the neutral fragment (+1: one electron less)
    config: object = None            # ConfigState (constraint selecting the state) or None
    group: object = None             # (fragment index, charge) of the configuration group
    omega: float = 1.0               # weight inside the group
    gweight: float = 1.0             # total weight of the group
    spin_sym: bool = False           # spin-flip symmetrized

    def refresh(self):
        """Recompute weight/flip of a configuration member from (gweight, omega)."""
        w = self.gweight * self.omega
        if self.spin_sym and self.na != self.nb:
            self.weight, self.flip = 0.5 * w, 0.5 * w
        else:
            self.weight, self.flip = w, 0.0

    @property
    def flips(self):
        return bool(self.flip) or (self.spin_sym and self.na != self.nb)

    @property
    def nelec(self):
        return self.na + self.nb

    @property
    def total_weight(self):
        return self.weight + self.flip


@dataclass
class FragmentSpec:
    index: int                       # 0-based
    atoms: tuple                     # 0-based atom indices in the supermolecule
    Z: float                         # sum of nuclear charges
    nelec0: int                      # electron number of the fragment before ensemble shifts (Z - fcharge)
    members: list = field(default_factory=list)
    rows: tuple = ()                 # atom indices as the user wrote them (differ from ``atoms`` if dummies are dropped)

    @property
    def nelec_avg(self):
        return sum(m.total_weight * m.nelec for m in self.members)

    def __repr__(self):
        ms = ", ".join(f"({m.na},{m.nb};w={m.weight:g},flip={m.flip:g}" + (f",config={m.config.name},omega={m.omega:.4f}" if m.config else "") + ")"
                       for m in self.members)
        return f"Fragment {self.index}: atoms={tuple(self.rows or self.atoms)} Z={self.Z:g} members=[{ms}]"


def _spin_split(n, twoS=None):
    twoS = n % 2 if twoS is None else int(twoS)
    if (n - twoS) % 2 or twoS > n:
        raise ValueError(f"Spin 2S={twoS} is incompatible with {n} electrons")
    return (n + twoS) // 2, (n - twoS) // 2


_CHARGE_KEYS = {"charges", "weights", "spins"}
_CONFIG_KEYS = {"charge", "states", "omega"}


def _check_keys(entry, allowed, what, fi):
    if not isinstance(entry, dict):
        raise TypeError(f"{what} entry of fragment {fi} must be a dict like {{'charges': ..., 'weights': ...}}, got {entry!r}")
    bad = set(entry) - allowed
    if bad:
        raise ValueError(f"{what} entry of fragment {fi}: unknown key(s) {sorted(bad)} (allowed: {sorted(allowed)})")


def _normalize(ensembles):
    """Return a list of (kind, entries); ``entries`` are tuples ``(frag, ...)`` consumed by :func:`build_members`."""
    if ensembles is None or len(ensembles) == 0:
        return []
    if not isinstance(ensembles, dict):
        raise TypeError("ensembles must be a dict such as {'charge': {0: {'charges': (0, -1), 'weights': (0.3, 0.7)}}, "
                        "'spin': [0, 1]} (fragments numbered from 0)")
    out = []
    for kind, spec in ensembles.items():
        kind = str(kind).lower()
        if kind == "spin":
            frs = [spec] if np.isscalar(spec) else list(spec)
            entries = [(int(f),) for f in frs]
        elif kind in ("charge", "config"):
            if not isinstance(spec, dict):
                raise TypeError(f"ensembles['{kind}'] must be a dict {{fragment: {{...}}}}")
            entries = []
            for fi, ent in spec.items():
                if kind == "charge":
                    _check_keys(ent, _CHARGE_KEYS, "charge", fi)
                    for k in ("charges", "weights"):
                        if k not in ent:
                            raise ValueError(f"charge entry of fragment {fi} needs the key '{k}'")
                    entries.append((fi, tuple(ent["charges"]), tuple(ent["weights"]))
                                   + ((tuple(ent["spins"]),) if "spins" in ent else ()))
                else:
                    for e in (ent if isinstance(ent, (list, tuple)) else [ent]):
                        _check_keys(e, _CONFIG_KEYS, "config", fi)
                        if "states" not in e:
                            raise ValueError(f"config entry of fragment {fi} needs the key 'states'")
                        entries.append((fi, e.get("charge", 0), tuple(e["states"])) + ((tuple(e["omega"]),) if "omega" in e else ()))
        else:
            raise NotImplementedError(f"ensemble type '{kind}' (available: charge, spin, config)")
        out.append((kind, entries))
    return out


def build_members(frags, ensembles, mol=None):
    """Fill ``frags[i].members`` according to the ``ensembles`` specification."""
    by_index = {f.index: f for f in frags}
    charge = {}
    spin = set()
    config = {}
    for kind, entries in _normalize(ensembles):
        for e in entries:
            if np.isscalar(e):
                e = (e,)
            fi = int(e[0])
            if fi not in by_index:
                raise ValueError(f"Ensemble refers to fragment {fi}, but there are only {len(frags)} fragments")
            if kind == "charge":
                if len(e) < 3:
                    raise ValueError("charge ensemble entry needs 'charges' and 'weights'")
                qs, w = tuple(e[1]), tuple(float(x) for x in e[2])
                twoS = tuple(e[3]) if len(e) > 3 else (None,) * len(qs)
                if len(qs) != len(w) or len(twoS) != len(qs):
                    raise ValueError(f"fragment {fi}: charges, weights (and spins) must have the same length")
                if abs(sum(w) - 1.0) > 1e-10:
                    raise ValueError(f"fragment {fi}: ensemble weights sum to {sum(w)}, not 1")
                if fi in charge:
                    raise ValueError(f"fragment {fi} has more than one charge ensemble")
                charge[fi] = (qs, w, twoS)
            elif kind == "spin":
                spin.add(fi)
            elif kind == "config":
                if len(e) < 3:
                    raise ValueError("config ensemble entry needs 'states'")
                if (fi, int(e[1])) in config:
                    raise ValueError(f"fragment {fi}: more than one config ensemble for charge={e[1]}")
                config[(fi, int(e[1]))] = (tuple(e[2]), tuple(e[3]) if len(e) > 3 else None)
            else:
                raise NotImplementedError(f"ensemble type '{kind}' (available: charge, spin, config)")
    for f in frags:
        if f.index in charge:
            qs, w, twoS = charge[f.index]
            members = []
            for q, wi, s in zip(qs, w, twoS):
                q = f.nelec0 if q is None else int(q)              # None = no electrons at all
                n = f.nelec0 - q
                if n < 0:
                    raise ValueError(f"fragment {f.index}: charge {q} leaves a negative electron number")
                na, nb = _spin_split(n, s)
                members.append(Member(na, nb, wi, charge=q))
        else:
            na, nb = _spin_split(f.nelec0)
            members = [Member(na, nb, 1.0)]
        expanded = []
        for m in members:
            key = (f.index, m.charge)
            if key not in config:
                expanded.append(m)
                continue
            states, om0 = config[key]
            if mol is None:
                raise ValueError("config ensembles need the molecule (internal error)")
            if len(states) < 1:
                raise ValueError(f"fragment {f.index}: empty config ensemble")
            om0 = np.full(len(states), 1.0 / len(states)) if om0 is None else np.asarray(om0, float)
            if len(om0) != len(states) or abs(om0.sum() - 1.0) > 1e-8:
                raise ValueError(f"fragment {f.index}: initial omegas must match the states and sum to 1")
            from copy import deepcopy
            from .cdft import ConfigState
            for k, (st, o) in enumerate(zip(states, om0)):
                st = deepcopy(st) if isinstance(st, ConfigState) else ConfigState(st)
                st.name = st.name or f"{f.index}:{m.charge}:{k}"
                st.bind(mol)
                expanded.append(Member(m.na, m.nb, 0.0, charge=m.charge, config=st, group=key, omega=float(o), gweight=m.weight))
        used = {m.charge for m in members}
        for (fi, q) in config:
            if fi == f.index and q not in used:
                raise ValueError(f"fragment {f.index}: no member with charge={q} for the config ensemble")
        members = expanded
        for m in members:
            m.spin_sym = f.index in spin
            if m.group is not None:
                m.refresh()
            elif f.index in spin and m.na != m.nb:
                m.flip, m.weight = 0.5 * m.weight, 0.5 * m.weight
        f.members = members
    return frags

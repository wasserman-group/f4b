"""Fragment ensembles (fixed weights, except configuration weights that are optimized).

A fragment is described by a list of :class:`Member` objects.  Member ``(na, nb, weight, flip)``
contributes ``weight*D + flip*swap_spin(D)`` to the fragment density and ``(weight+flip)*E`` to the fragment energy,
where ``D, E`` come from a single UKS calculation with ``na`` alpha and ``nb`` beta electrons.

Grammar for ``ensembles`` (a dict keyed by ensemble type; fragments and atoms are numbered from 0)::

    ensembles = {
        'charge': {0: 1.4, 1: 0.6},       # electrons of the fragment (fractional): the two neighbouring integers are used
        'spin':   [0, 1],                 # fragments whose open-shell members are averaged with their spin-flipped partner
        'config': [0, 1],                 # degenerate configurations: filled automatically (see f4b.shells)
    }

``'charge'``: a number is the (average) electron number of the fragment, e.g. 1.4 -> H (weight 0.6) + H- (0.4); an integer
means no ensemble.  The advanced form ``{0: {'charges': (0, -1), 'weights': (0.3, 0.7)[, 'spins': (0, 1)]}}`` lists the members
by hand: ``charges`` relative to the neutral fragment (``fcharge``; ``+1`` = one electron less, ``None`` = no electrons).

Spins (2S = N_alpha - N_beta): the ground-state spin of every member follows Hund's rule (:mod:`f4b.shells`) unless it is
given: ``fspin`` sets the spin of the neutral member (the one with ``Z - fcharge`` electrons) and ``'spins'`` of the
advanced charge form overrides ``fspin`` member by member.

``'config'``: a list of fragments, ``'all'``, or ``{fragment: options}`` with the options ``fill`` (``'holes'`` default or
``'occupied'``), ``omega`` (initial weights: a sequence, or ``{N electrons: sequence}``) and ``members`` (electron numbers
that get the ensemble; default: all that have a partly filled degenerate shell).  Equal weights are used if no ``omega`` is
given; the weights are optimized by the P-DFT loop and the member weight ``w`` is shared among the states (``w*omega_i``).
States chosen by hand are also possible: ``{fragment: {'charge': 0, 'states': (state, ...)[, 'omega': (...)]}}`` (or a
list of such dicts, one per charge), with states such as ``(('F 2px', 'beta', 0),)`` (see :class:`f4b.cdft.ConfigState`).
"""
import numbers
import warnings
from copy import deepcopy
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
    twoS0: object = None             # 2S (= N_alpha - N_beta) of the neutral member (``fspin``); None -> Hund's rule
    fcharge_given: bool = False      # the user gave ``fcharge`` for this fragment
    members: list = field(default_factory=list)
    rows: tuple = ()                 # atom indices as the user wrote them (differ from ``atoms`` if dummies are dropped)

    @property
    def nelec_avg(self):
        return sum(m.total_weight * m.nelec for m in self.members)

    def __repr__(self):
        ms = ", ".join(f"({m.na},{m.nb};w={m.weight:g},flip={m.flip:g}" + (f",config={m.config.name},omega={m.omega:.4f}" if m.config else "") + ")"
                       for m in self.members)
        return f"Fragment {self.index}: atoms={tuple(self.rows or self.atoms)} Z={self.Z:g} members=[{ms}]"


def _spin_split(n, twoS=None, where=""):
    twoS = n % 2 if twoS is None else int(twoS)
    if twoS < 0 or (n - twoS) % 2 or twoS > n:
        raise ValueError(f"{where}spin 2S={twoS} (N_alpha - N_beta) is incompatible with {n} electrons: it must be between 0 "
                         f"and {n} with the parity of {n}")
    return (n + twoS) // 2, (n - twoS) // 2


_CHARGE_KEYS = {"charges", "weights", "spins"}
_CONFIG_MANUAL_KEYS = {"charge", "states", "omega"}
_CONFIG_AUTO_KEYS = {"fill", "omega", "members"}


def _check_keys(entry, allowed, what, fi):
    if not isinstance(entry, dict):
        raise TypeError(f"{what} entry of fragment {fi} must be a dict, got {entry!r}")
    bad = set(entry) - allowed
    if bad:
        raise ValueError(f"{what} entry of fragment {fi}: unknown key(s) {sorted(bad)} (allowed: {sorted(allowed)})")


def _is_number(x):
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _normalize(ensembles):
    """Return ``(charge, spin, config_auto, config_manual)``: charge ``{frag: number | (qs, weights, spins)}``, spin: set of
    fragments, config_auto ``{frag: options}`` (``frag`` = ``'all'`` for every fragment), config_manual ``{(frag, charge): ...}``."""
    charge, spin, auto, manual = {}, set(), {}, {}
    if ensembles is None or len(ensembles) == 0:
        return charge, spin, auto, manual
    if not isinstance(ensembles, dict):
        raise TypeError("ensembles must be a dict such as {'charge': {0: 1.4, 1: 0.6}, 'spin': [0, 1], 'config': [0]} "
                        "(fragments numbered from 0)")
    for kind, spec in ensembles.items():
        kind = str(kind).lower()
        if kind == "spin":
            spin.update(int(f) for f in ([spec] if np.isscalar(spec) else spec))
        elif kind == "charge":
            if not isinstance(spec, dict):
                raise TypeError("ensembles['charge'] must be a dict {fragment: electrons}, e.g. {0: 1.4, 1: 0.6}")
            for fi, ent in spec.items():
                if fi in charge:
                    raise ValueError(f"fragment {fi} has more than one charge ensemble")
                if _is_number(ent):
                    if ent < 0:
                        raise ValueError(f"fragment {fi}: a negative electron number ({ent})")
                    charge[fi] = float(ent)
                else:
                    _check_keys(ent, _CHARGE_KEYS, "charge", fi)
                    for k in ("charges", "weights"):
                        if k not in ent:
                            raise ValueError(f"charge entry of fragment {fi} needs the key '{k}' (or give the number of electrons)")
                    qs, w = tuple(ent["charges"]), tuple(float(x) for x in ent["weights"])
                    twoS = tuple(ent["spins"]) if "spins" in ent else (None,) * len(qs)
                    if len(qs) != len(w) or len(twoS) != len(qs):
                        raise ValueError(f"fragment {fi}: charges, weights (and spins) must have the same length")
                    if abs(sum(w) - 1.0) > 1e-10:
                        raise ValueError(f"fragment {fi}: ensemble weights sum to {sum(w)}, not 1")
                    charge[fi] = (qs, w, twoS)
        elif kind == "config":
            if spec is True or (isinstance(spec, str) and spec.lower() == "all"):
                auto["all"] = {}
            elif isinstance(spec, (list, tuple, set)):
                auto.update({int(f): {} for f in spec})
            elif isinstance(spec, dict):
                for fi, ent in spec.items():
                    if ent is True or (isinstance(ent, dict) and not ent):
                        auto[fi] = {}
                    elif isinstance(ent, dict) and "states" not in ent:
                        _check_keys(ent, _CONFIG_AUTO_KEYS, "config", fi)
                        fill = str(ent.get("fill", "holes")).lower()
                        if fill not in ("holes", "occupied"):
                            raise ValueError(f"config entry of fragment {fi}: fill must be 'holes' or 'occupied', got {ent['fill']!r}")
                        auto[fi] = dict(ent, fill=fill)
                    else:
                        for e in (ent if isinstance(ent, (list, tuple)) else [ent]):
                            if isinstance(e, dict) and "states" not in e:
                                raise ValueError(f"config entry of fragment {fi}: give either automatic options (fill, omega, members) "
                                                 "or the 'states' to use, not a mixture")
                            _check_keys(e, _CONFIG_MANUAL_KEYS, "config", fi)
                            key = (fi, int(e.get("charge", 0)))
                            if key in manual:
                                raise ValueError(f"fragment {fi}: more than one config ensemble for charge={key[1]}")
                            manual[key] = (tuple(e["states"]), tuple(e["omega"]) if "omega" in e else None)
            else:
                raise TypeError("ensembles['config'] must be a list of fragments, 'all' or a dict {fragment: options}")
        else:
            raise NotImplementedError(f"ensemble type '{kind}' (available: charge, spin, config)")
    return charge, spin, auto, manual


def _hund(f, N, shells):
    if N <= 2 or shells is None:
        return _spin_split(N)
    return shells(f, N).hund()


def _charge_members(f, spec, shells):
    """Members of fragment ``f`` from the charge ensemble ``spec`` (None: no ensemble)."""
    where = lambda n: f"fragment {f.index} ({n} electrons): "                           # noqa: E731

    def member(n, w, s=None, none=False):
        if s is None and n == f.nelec0 and not none:
            s = f.twoS0                                                                    # fspin: the neutral member
        na, nb = _spin_split(n, s, where(n)) if s is not None else _hund(f, n, shells)
        return Member(na, nb, w, charge=f.nelec0 - n)

    if spec is None:
        return [member(f.nelec0, 1.0)]
    if isinstance(spec, float):
        if f.fcharge_given:
            warnings.warn(f"fragment {f.index}: fcharge is ignored for the electrons of a charge ensemble (the electron number "
                          f"{spec:g} is used); it only defines which member is 'neutral' for fspin")
        lo = int(np.floor(spec + 1e-9))
        if abs(spec - round(spec)) < 1e-9:
            return [member(int(round(spec)), 1.0)]
        return [member(lo, lo + 1 - spec), member(lo + 1, spec - lo)]
    qs, w, twoS = spec
    members = []
    for q, wi, s in zip(qs, w, twoS):
        none = q is None                                                                   # None = no electrons at all
        q = f.nelec0 if none else int(q)
        n = f.nelec0 - q
        if n < 0:
            raise ValueError(f"fragment {f.index}: charge {q} leaves a negative electron number")
        members.append(member(n, wi, s, none))
    return members


def build_members(frags, ensembles, mol=None, shells=None):
    """Fill ``frags[i].members`` according to the ``ensembles`` specification.

    ``shells(frag, N)`` returns the :class:`f4b.shells.Reference` of a fragment with ``N`` electrons (Hund's rule spins and
    the automatic configuration states)."""
    from .cdft import ConfigState, Constraint
    by_index = {f.index: f for f in frags}
    charge, spin, auto, manual = _normalize(ensembles)
    for name, keys in (("charge", charge), ("config", [k for k in auto if k != "all"] + [k[0] for k in manual])):
        for fi in keys:
            if fi not in by_index:
                raise ValueError(f"Ensemble '{name}' refers to fragment {fi}, but there are only {len(frags)} fragments")
    for fi in spin:
        if fi not in by_index:
            raise ValueError(f"Ensemble 'spin' refers to fragment {fi}, but there are only {len(frags)} fragments")
    if "all" in auto:
        for f in frags:
            auto.setdefault(f.index, auto["all"])
        del auto["all"]
    both = set(auto) & {k[0] for k in manual}
    if both:
        raise ValueError(f"fragment(s) {sorted(both)}: the configuration ensemble is either automatic or given by hand, not both")
    for f in frags:
        members = _charge_members(f, charge.get(f.index), shells)
        used = {m.charge for m in members}
        for (fi, q) in manual:
            if fi == f.index and q not in used:
                raise ValueError(f"fragment {f.index}: no member with charge={q} for the config ensemble")
        expanded = []
        opts = auto.get(f.index)
        n_auto = 0
        for m in members:
            key = (f.index, m.charge)
            if opts is not None and m.nelec > 0 and (opts.get("members") is None or m.nelec in opts["members"]):
                if shells is None:
                    raise ValueError("automatic configuration ensembles need the molecule (internal error)")
                found = shells(f, m.nelec).states(m.na, m.nb, opts.get("fill", "holes"))
                if found:
                    om = _initial_omega(f, m, opts.get("omega"), len(found))
                    for k, (desc, cons) in enumerate(found):
                        st = ConfigState([Constraint.from_vector(v, s, pop, label=desc) for v, s, pop in cons])
                        st.name, st.desc = f"{f.index}:{m.charge}:{k}", desc
                        st.bind(mol)
                        expanded.append(Member(m.na, m.nb, 0.0, charge=m.charge, config=st, group=key, omega=float(om[k]),
                                               gweight=m.weight))
                    n_auto += 1
                    continue
            if key in manual:
                states, om0 = manual[key]
                if len(states) < 1:
                    raise ValueError(f"fragment {f.index}: empty config ensemble")
                om0 = np.full(len(states), 1.0 / len(states)) if om0 is None else np.asarray(om0, float)
                if len(om0) != len(states) or abs(om0.sum() - 1.0) > 1e-8:
                    raise ValueError(f"fragment {f.index}: initial omegas must match the states and sum to 1")
                for k, (st, o) in enumerate(zip(states, om0)):
                    st = deepcopy(st) if isinstance(st, ConfigState) else ConfigState(st)
                    st.name = st.name or f"{f.index}:{m.charge}:{k}"
                    st.bind(mol, atoms=set(f.atoms))
                    expanded.append(Member(m.na, m.nb, 0.0, charge=m.charge, config=st, group=key, omega=float(o), gweight=m.weight))
                continue
            expanded.append(m)
        if opts is not None and n_auto == 0:
            warnings.warn(f"fragment {f.index}: no member has a partly filled degenerate shell, so the configuration ensemble "
                          "is not needed (a single configuration is used)")
        members = expanded
        for m in members:
            m.spin_sym = f.index in spin
            if m.group is not None:
                m.refresh()
            elif f.index in spin and m.na != m.nb:
                m.flip, m.weight = 0.5 * m.weight, 0.5 * m.weight
        f.members = members
    return frags


def _initial_omega(f, m, omega, n):
    """Initial configuration weights of a member with ``n`` states: equal unless given."""
    if isinstance(omega, dict):
        omega = omega.get(m.nelec)
    if omega is None:
        return np.full(n, 1.0 / n)
    om = np.asarray(omega, float)
    if len(om) != n or abs(om.sum() - 1.0) > 1e-8 or (om < 0).any():
        raise ValueError(f"fragment {f.index} ({m.nelec} electrons): the initial omegas must be {n} non-negative numbers that sum "
                         f"to 1 (one per configuration state), got {omega!r}")
    return om

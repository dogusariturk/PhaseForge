"""Tests for bin/MLIPentropy. Run from the repository root:  python -m pytest -q tests

Needs numpy, ase and pytest. Units J/mol-atom/K.
"""
import importlib.machinery
import importlib.util
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "bin" / "MLIPentropy"
_loader = importlib.machinery.SourceFileLoader("mlipentropy", str(TOOL))
_spec = importlib.util.spec_from_loader("mlipentropy", _loader)
me = importlib.util.module_from_spec(_spec)
_loader.exec_module(me)

R = me.R
XS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
rng = np.random.default_rng(1)


def H(p):
    return me.shannon(p)


# ====================================================================== random labels give ideal mixing
@pytest.mark.parametrize("z", [2, 4, 12])
def test_random_labels_ideal_exact_binary(z):
    """Random labels (y_ij = x_i x_j): pair term equals -R sum x ln x exactly, and so does the quasi-chemical
    closed form (z/2) H(y) - (z-1) H(x)."""
    for x in XS:
        s, si, ex = me.pair_entropy_tree(z, x, 0.0)
        assert s == pytest.approx(me.ideal(x), abs=1e-12) and abs(ex) < 1e-12
        y = np.outer([1 - x, x], [1 - x, x]).ravel()
        qc = R * ((z / 2) * H(y) - (z - 1) * H([1 - x, x]))
        assert qc == pytest.approx(me.ideal(x), abs=1e-12)


@pytest.mark.parametrize("z", [4, 12])
@pytest.mark.parametrize("x", [(0.2, 0.3, 0.5), (1 / 3, 1 / 3, 1 / 3), (0.1, 0.2, 0.3, 0.4), (0.25, 0.25, 0.25, 0.25),
                               (0.05, 0.15, 0.8)])
def test_random_labels_ideal_exact_multicomponent(z, x):
    """Ternary and quaternary: random labels give exactly -R sum x ln x."""
    x = np.array(x)
    s, si, ex = me.pair_entropy(x, np.outer(x, x), z)
    assert s == pytest.approx(-R * (x * np.log(x)).sum(), abs=1e-12)
    assert abs(ex) < 1e-12


@pytest.mark.parametrize("kind,L", [("chain", 20000), ("square", 100), ("fcc", 10)])
@pytest.mark.parametrize("x", [(0.1, 0.9), (0.2, 0.3, 0.5), (0.1, 0.2, 0.3, 0.4)])
def test_random_labels_ideal_sampled_lattices(kind, L, x):
    """Random labels on periodic z = 2, 4, 12 lattices (k = 2, 3, 4), pooled samples: pair term = ideal to 0.01."""
    N, nb = me.lattice(kind, L)
    k = len(x)
    acc = me.Accumulator(k)
    for _ in range(3):
        lab = rng.choice(k, size=N, p=x)
        acc.add(*me.graph_statistics(lab, nb, k))
    r = acc.result()
    assert abs(r["pair"] - r["ideal"]) < 0.01
    assert r["ideal"] == pytest.approx(me.ideal(acc.sites / acc.sites.sum()))


def test_random_labels_irregular_graph():
    """Random labels on an irregular graph (degrees 10-16, ternary): pair term ideal (bond-end reference)."""
    N = 4000
    deg = rng.integers(10, 17, N)
    stubs = np.repeat(np.arange(N), deg)
    stubs = stubs[: len(stubs) // 2 * 2]
    rng.shuffle(stubs)
    nbr = [[] for _ in range(N)]
    for a, b in zip(stubs[::2], stubs[1::2]):
        nbr[a].append(b)
        nbr[b].append(a)
    nbr = [np.array(v, int) for v in nbr]
    acc = me.Accumulator(3)
    for _ in range(10):
        acc.add(*me.graph_statistics(rng.choice(3, size=N, p=(0.2, 0.3, 0.5)), nbr, 3))
    r = acc.result()
    assert abs(r["pair"] - r["ideal"]) < 0.01


# ====================================================================== exact enumeration
def _exact_enumeration(nb, xB):
    """All 2^N labellings of a small periodic graph with i.i.d. probability xB."""
    N, z = nb.shape
    probs, ybar = {}, np.zeros((2, 2))
    for bits in itertools.product((0, 1), repeat=N):
        lab = np.array(bits, int)
        p = xB ** lab.sum() * (1 - xB) ** (N - lab.sum())
        sig = tuple(lab) + tuple(lab[nb].sum(1))
        probs[sig] = probs.get(sig, 0) + p
        _, _, counts = me.graph_statistics(lab, nb, 2)
        for s in (0, 1):
            ybar[s] += p * counts[lab == s].sum(0)
    return H(list(probs.values())), ybar / ybar.sum()


@pytest.mark.parametrize("graph", ["ring12", "square4x4"])
@pytest.mark.parametrize("xB", [0.1, 0.5])
def test_exact_enumeration_equals_ideal(graph, xB):
    """Ring of 12 (z = 2) and 4x4 square torus (z = 4): the exact entropy of the full label + neighbour-count
    configuration equals the label entropy N H(x), and the pair term on the exact ensemble equals ideal."""
    nb = me.lattice("chain", 12)[1] if graph == "ring12" else me.lattice("square", 4)[1]
    joint, y = _exact_enumeration(nb, xB)
    assert R * joint / nb.shape[0] == pytest.approx(me.ideal(xB), abs=1e-10)
    s, si, ex = me.pair_entropy([1 - xB, xB], y, nb.shape[1])
    assert s == pytest.approx(me.ideal(xB), abs=1e-10)


# ====================================================================== exact cases with order
@pytest.mark.parametrize("x,alpha", [(0.3, -0.3), (0.5, 0.5), (0.1, 0.4), (0.25, -0.2)])
def test_1d_chain_exact(x, alpha):
    """1D stationary Markov chain with Warren-Cowley alpha: the exact entropy rate H(x) + H(next | current) - H(x)
    equals the pair term with z = 2; a sampled chain agrees within sampling error."""
    xA = 1 - x
    P = np.array([[1 - x * (1 - alpha), x * (1 - alpha)], [xA * (1 - alpha), 1 - xA * (1 - alpha)]])
    rate = R * (xA * H(P[0]) + x * H(P[1]))
    assert me.pair_entropy_tree(2, x, alpha)[0] == pytest.approx(rate, abs=1e-12)
    N = 200_000
    u = rng.random(N)
    lab = np.empty(N, int)
    lab[0] = 0
    for i in range(1, N):
        lab[i] = u[i] < P[lab[i - 1], 1]
    acc = me.Accumulator(2)
    acc.add(*me.graph_statistics(lab, me.lattice("chain", N)[1], 2))
    assert acc.result()["pair"] == pytest.approx(rate, abs=0.02)


@pytest.mark.parametrize("z", [3, 4, 12])
def test_bethe_lattice_ising_exact(z):
    """Zero-field Ising on a Bethe lattice (paramagnetic, tanh K < 1/(z-1)): exact entropy per site
    ln 2 + (z/2)(ln cosh K - K tanh K) with bond correlation tanh K equals the pair term."""
    Kc = math.atanh(1 / (z - 1))
    for K in np.linspace(0.05, 0.95, 7) * Kc:
        c = math.tanh(K)
        exact = R * (math.log(2) + z / 2 * (math.log(math.cosh(K)) - K * c))
        y = np.array([[(1 + c) / 4, (1 - c) / 4], [(1 - c) / 4, (1 + c) / 4]])
        assert me.pair_entropy([0.5, 0.5], y, z)[0] == pytest.approx(exact, abs=1e-12)
        # ferro- and antiferro-like order lower the entropy below ideal
        assert me.pair_entropy([0.5, 0.5], y, z)[2] < 0


def test_pair_excess_never_positive_multicomponent():
    for _ in range(200):
        k = rng.integers(2, 5)
        Y = rng.random((k, k))
        Y = Y + Y.T
        Y /= Y.sum()
        assert me.pair_entropy(Y.sum(1), Y, 12)[2] <= 1e-12


# ====================================================================== parity constraint
def _parity_entropy(P, N):
    """Exact entropy of N independent counts n_k ~ P conditioned on an even sum (renormalised)."""
    P = np.asarray(P, float)
    n = np.arange(len(P))
    lp = np.where(P > 0, np.log(np.where(P > 0, P, 1)), 0)
    sgn = (-1.0) ** n
    a, b = P.sum(), (P * sgn).sum()
    c, d = (P * lp).sum(), (P * lp * sgn).sum()
    peven = 0.5 * (a ** N + b ** N)
    e_ln = 0.5 * (N * c * a ** (N - 1) + N * d * b ** (N - 1))
    q = 1 / peven
    return -(q * e_ln + math.log(q))


def test_parity_constraint_removes_ln2():
    """An even-sum (parity) constraint on N independent neighbour counts lowers their total entropy by ln 2,
    not by a fraction of it: exact by enumeration (N = 6) and by generating functions (N up to 500)."""
    z, p, N = 4, 0.3, 6
    pn = me.binom_pmf(z, p)
    probs = np.array([np.prod(pn[list(ns)]) for ns in itertools.product(range(z + 1), repeat=N) if sum(ns) % 2 == 0])
    probs /= probs.sum()
    assert N * H(pn) - H(probs) == pytest.approx(math.log(2), abs=0.05)
    Pal = np.zeros(21)
    Pal[11:17] = [0.03, 0.33, 0.32, 0.22, 0.08, 0.02]
    Pal /= Pal.sum()
    for N in (10, 100, 500):
        assert N * H(Pal) - _parity_entropy(Pal, N) == pytest.approx(math.log(2), abs=1e-6)


# ====================================================================== comparison option
def test_hongliu_comparison_reference_values():
    """Hong-Liu Eqs. 15-16, for comparison: reproduces the reference values of the Cu-Zr campaign (z = 4, x = 0.1)."""
    p = me.hongliu_analytic_tree(4, 0.1)
    assert p["species_blind"] == 0.0
    assert p["same_species"] == pytest.approx(3.3424899997105593, abs=1e-10)
    assert p["cross_species"] == pytest.approx(6.6849799994211185, abs=1e-10)
    assert p["heavy_first"] == pytest.approx(6.350730999450064, abs=1e-10)


def test_calphad_mapping_binary_recovers_bk():
    xs = np.array([0.25, 0.5, 0.75])
    b = np.array([-3.0, 1.2, 0.4])
    X = np.stack([1 - xs, xs], 1)
    sex = -xs * (1 - xs) * sum(b[k] * (1 - 2 * xs) ** k for k in range(3))
    fit = me.map_to_bk(X, sex, np.eye(3) * 0.01, 2)
    assert np.allclose(fit["b"], b, atol=1e-10)


# ====================================================================== synthetic trajectories end to end
A0, NCELL, T, DT, NF = 3.6, 3, 1000.0, 5.0, 1000
MASS = {"Cu": 63.546, "Ni": 58.6934, "Zr": 91.224}


def _fcc():
    M = 2 * NCELL
    g = np.indices((M, M, M)).reshape(3, -1).T
    pts = g[g.sum(1) % 2 == 0]
    sub = {(0, 0, 0): 0, (1, 1, 0): 1, (1, 0, 1): 2, (0, 1, 1): 3}
    return pts * A0 / 2, np.array([sub[tuple(p % 2)] for p in pts]), NCELL * A0


def _write(path, symbols, nu_THz, seed):
    """Atoms on fcc sites, each Cartesian component a classical harmonic oscillator at its frequency with
    equipartition amplitude at T; positions and velocities consistent."""
    from ase import Atoms, units
    from ase.io import write
    r = np.random.default_rng(seed)
    x0, _, Lb = _fcc()
    N = len(x0)
    m = np.array([MASS[s] for s in symbols])
    w = 2 * np.pi * np.asarray(nu_THz, float)[:, None] * 1e-3            # rad/fs, per atom
    kT = me.kB_eV * T / me.EV_PER_AMU_A2_FS2                             # amu A^2/fs^2
    amp = np.sqrt(2 * kT / m)[:, None] * np.ones((N, 3))                 # A/fs
    ph = r.uniform(0, 2 * np.pi, (N, 3))
    frames = []
    for f in range(NF):
        t = f * DT
        at = Atoms(symbols, positions=x0 + amp / w * np.sin(w * t + ph), cell=[Lb] * 3, pbc=True)
        at.set_velocities(amp * np.cos(w * t + ph) / units.fs)
        frames.append(at)
    write(str(path), frames)
    return path


def _s_harm(nu):
    x = me.h_eV_s * nu * 1e12 / (2 * me.kB_eV * T)
    return 3 * R * (x / math.tanh(x) - math.log(2 * math.sinh(x)))


def _lattice_pair_excess(lab, k):
    """Pair excess computed directly on the fcc nearest-neighbour table (independent of the RDF/cutoff route)."""
    x0, _, Lb = _fcc()
    d = x0[:, None] - x0[None]
    d -= Lb * np.round(d / Lb)
    r = np.sqrt((d ** 2).sum(-1))
    nb = np.array([np.where(np.abs(row - A0 / math.sqrt(2)) < 1e-6)[0] for row in r])
    assert nb.shape[1] == 12
    acc = me.Accumulator(k)
    acc.add(*me.graph_statistics(lab, nb, k))
    return acc.result()["pair_excess"]


def _run(tmp_path, args):
    out = tmp_path / "entropy.json"
    cp = subprocess.run([sys.executable, str(TOOL), f"-temp={T}", f"-dt={DT}", "-tmax=0.5", "-stride=5", "-rdf_stride=20",
                         f"-o={out}"] + args, capture_output=True, text=True, cwd=tmp_path)
    assert cp.returncode == 0, cp.stderr
    print(cp.stdout)
    return json.loads(out.read_text()), cp


# Thermal displacements put a few nearest-neighbour pairs near the first empty RDF bin, so the exactness checks use
# a fixed cutoff in the middle of the gap between the first (a/sqrt 2) and second (a) fcc shells; the default
# total-RDF minimum is checked separately to lie in that gap.
RC_GAP = 0.5 * (A0 / math.sqrt(2) + A0)


def _check_default_cutoff_in_gap(alloy):
    rc = alloy["neighbour_cutoffs"]["rc_total_A"]
    assert A0 / math.sqrt(2) < rc < A0
    assert alloy["S_chem_excess_cutoff_spread"] < 1e-9


def test_end_to_end_binary(tmp_path):
    """Cu-Ni, L1_0 order at x = 0.5 on fcc: S_chem,ex equals the exact lattice pair value (-6 R I, I the bond mutual
    information), S_vib,ex matches the harmonic value, the ideal-mixing check passes and b_0 = -4 S_ex."""
    x0, sub, _ = _fcc()
    N = len(x0)
    nuA, nuB, nuM = 8.0, 10.0, 9.5
    alloy_sym = np.where(sub < 2, "Cu", "Ni")
    _write(tmp_path / "Cu.traj", ["Cu"] * N, [nuA] * N, 1)
    _write(tmp_path / "Ni.extxyz", ["Ni"] * N, [nuB] * N, 2)
    _write(tmp_path / "CuNi.traj", list(alloy_sym), [nuM] * N, 3)
    out, cp = _run(tmp_path, ["-ref=Cu.traj", "-ref=Ni.extxyz", "-alloy=CuNi.traj", "-hongliu", f"-rc={RC_GAP}"])
    a = out["alloys"]["CuNi.traj"]
    _check_default_cutoff_in_gap(a)
    assert out["ideal_mixing_check"]["passed"]
    assert abs(out["ideal_mixing_check"]["alloys"]["CuNi.traj"]["S_chem_excess_permuted"]) < 0.01
    y = np.array([[1 / 6, 1 / 3], [1 / 3, 1 / 6]])
    mi = (y * np.log(y / 0.25)).sum()
    assert a["S_chem_excess"] == pytest.approx(-R * 6 * mi, abs=1e-9)
    assert a["S_chem_excess"] == pytest.approx(_lattice_pair_excess((alloy_sym == "Ni").astype(int), 2), abs=1e-9)
    assert a["zbar"] == pytest.approx(12.0)
    assert a["bonds_per_pair"]["Cu-Ni"] == pytest.approx(a["frames_in_graph"] * N * 8 / 2)
    for el, nu in (("Cu", nuA), ("Ni", nuB)):
        assert out["references"][el]["S_vib"] == pytest.approx(_s_harm(nu), rel=0.01)
        assert out["references"][el]["dos_integral_per_atom"] == pytest.approx(3.0, rel=0.01)
    expect = _s_harm(nuM) - 0.5 * _s_harm(nuA) - 0.5 * _s_harm(nuB)
    assert a["S_vib_excess"] == pytest.approx(expect, abs=0.05)
    fit = out["calphad_mapping"]["fit"]
    assert fit["order"] == 0 and fit["b"][0] == pytest.approx(-a["S_excess"] / 0.25, rel=1e-9)
    hl = a["hongliu_eqs_15_16_for_comparison"]
    assert set(hl["S_conf"]) == set(me.HONGLIU_READINGS) and "comparison" in hl["note"]
    assert out["missing_terms"]["electronic"].startswith("not computed")
    # a check that cannot pass (tolerance 1e-12) must warn and hold the excess entropies
    out2, cp2 = _run(tmp_path, ["-ref=Cu.traj", "-ref=Ni.extxyz", "-alloy=CuNi.traj", f"-rc={RC_GAP}", "-ideal_tol=1e-12"])
    assert not out2["ideal_mixing_check"]["passed"]
    assert out2["alloys"]["CuNi.traj"]["status"].startswith("held")
    assert "ideal-mixing check FAILED" in cp2.stderr


def test_end_to_end_ternary(tmp_path):
    """Cu-Ni-Zr on fcc: Cu on two sublattices, Ni and Zr on one each; random-label check passes, S_chem,ex equals
    the lattice pair value, a 3x3 bond matrix and per-pair bond counts are reported."""
    x0, sub, _ = _fcc()
    N = len(x0)
    nu = {"Cu": 8.0, "Ni": 10.0, "Zr": 7.0}
    sym = np.array(["Cu", "Cu", "Ni", "Zr"])[sub]
    for seed, el in enumerate(nu):
        _write(tmp_path / f"{el}.traj", [el] * N, [nu[el]] * N, 10 + seed)
    _write(tmp_path / "CuNiZr.traj", list(sym), [nu[s] for s in sym], 7)
    out, cp = _run(tmp_path, ["-ref=Zr.traj", "-ref=Cu.traj", "-ref=Ni.traj", "-alloy=CuNiZr.traj", f"-rc={RC_GAP}"])
    _check_default_cutoff_in_gap(out["alloys"]["CuNiZr.traj"])
    assert out["species_order"] == ["Cu", "Ni", "Zr"]
    assert out["ideal_mixing_check"]["passed"]
    a = out["alloys"]["CuNiZr.traj"]
    assert a["x"] == pytest.approx({"Cu": 0.5, "Ni": 0.25, "Zr": 0.25})
    lab = np.array([{"Cu": 0, "Ni": 1, "Zr": 2}[s] for s in sym])
    assert a["S_chem_excess"] == pytest.approx(_lattice_pair_excess(lab, 3), abs=1e-9)
    assert a["S_chem_excess"] < -0.5
    assert np.array(a["y_bonds"]).shape == (3, 3) and np.array(a["bond_counts_ordered"]).shape == (3, 3)
    assert set(a["bonds_per_pair"]) == {"Cu-Cu", "Cu-Ni", "Cu-Zr", "Ni-Ni", "Ni-Zr", "Zr-Zr"}
    nfg = a["frames_in_graph"]
    assert sum(a["bonds_per_pair"].values()) == pytest.approx(nfg * N * 12 / 2)
    assert a["bonds_per_pair"]["Ni-Ni"] == 0 and a["bonds_per_pair"]["Zr-Zr"] == 0   # Ni, Zr sublattices: no like bonds
    assert not out["warnings"]
    # per-site vibrations equal the pure references: S_vib,ex ~ 0
    assert abs(a["S_vib_excess"]) < 0.05
    assert out["calphad_mapping"]["fit"] is None and "not determined" in out["calphad_mapping"]["note"]

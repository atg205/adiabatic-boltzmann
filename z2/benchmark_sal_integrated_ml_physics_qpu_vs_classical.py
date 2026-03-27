#!/usr/bin/env python3
"""
Integrated sampler-adaptive learning benchmark for machine-learning and physics datasets.

This script merges the earlier energy-based ML benchmark with two core ideas from
Kubo and Goto's sampler-adaptive learning (SAL) paper:

1. Temperature-aware positive-phase updates via an effective inverse temperature beta_eff.
2. Conditional expectation matching (CEM) to estimate beta_eff for non-MCMC samplers.

Model families
--------------
- rbm        : dense restricted Boltzmann machine.
- masked_rbm : sparse hidden-visible mask, still with no visible-visible couplings.
- srbm       : semi-restricted Boltzmann machine with visible-visible couplings.

Negative-phase backends
-----------------------
- exact       : exact marginalization over visible states.
- gibbs       : classical Gibbs sampler.
- sa          : classical simulated annealing sampler.
- lsb         : Langevin simulated bifurcation sampler inspired by the SAL paper.
- qpu_pegasus : D-Wave Pegasus QPU sampler.
- qpu_zephyr  : D-Wave Zephyr QPU sampler.

Dataset families
----------------
- bars_stripes, parity, planted_ising, three_spin
- physics_ground : samples drawn from |psi_0(s)|^2 of a stoquastic transverse-field model.

Physical Hamiltonians for physics_ground
----------------------------------------
- long_range_1d : long-range 1D TFIM with J(r) = 1/r^alpha on a ring
- j1j2_1d       : frustrated J1-J2 TFIM on a ring
- ea_2d         : 2D Edwards-Anderson spin glass in a transverse field

For the physics datasets the script also evaluates the trained BM as a positive
wavefunction psi_theta(s) = sqrt(Q_theta(s)), which lets it report a variational
energy under the chosen Hamiltonian in addition to generative-model metrics.

Quick classical example
-----------------------
python benchmark_sal_integrated_ml_physics_qpu_vs_classical.py \
    --models rbm masked_rbm srbm \
    --samplers exact gibbs sa lsb \
    --dataset-source three_spin \
    --n-visible 10 \
    --epochs 80 \
    --negative-samples 512 \
    --output-dir sal_benchmarks_three_spin

Physics example
---------------
python benchmark_sal_integrated_ml_physics_qpu_vs_classical.py \
    --models rbm srbm \
    --samplers exact gibbs sa \
    --dataset-source physics_ground \
    --physical-hamiltonian j1j2_1d \
    --j1j2-size 10 \
    --j1 1.0 --j2 0.5 \
    --transverse-field 1.0 \
    --epochs 60 \
    --negative-samples 512 \
    --output-dir sal_benchmarks_physics

QPU example
-----------
export DWAVE_API_TOKEN=...your token...
python benchmark_sal_integrated_ml_physics_qpu_vs_classical.py \
    --models rbm srbm \
    --samplers sa qpu_pegasus qpu_zephyr \
    --dataset-source physics_ground \
    --physical-hamiltonian long_range_1d \
    --long-range-size 8 \
    --transverse-field 0.8 \
    --epochs 40 \
    --negative-samples 256 \
    --cem-period 5 \
    --cem-samples 64 \
    --annealing-time 20 \
    --output-dir sal_benchmarks_qpu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.sparse.linalg import eigsh

from qpu_runtime_safe_sampler_extended import QPUAccessConfig, RuntimeSafeQPUIsingSampler

Array = np.ndarray


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def save_json(obj: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)



def save_history_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def spin_states(n: int) -> Array:
    if n < 1:
        raise ValueError("n must be positive")
    vals = np.arange(1 << n, dtype=np.uint32)[:, None]
    bits = ((vals >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)
    return (2 * bits - 1).astype(np.int8, copy=False)



def logistic_spin(field: Array, rng: np.random.Generator) -> Array:
    probs = 1.0 / (1.0 + np.exp(-2.0 * field))
    return np.where(rng.random(size=field.shape) < probs, 1, -1).astype(np.int8)



def periodic_ring_edges(n: int) -> List[Tuple[int, int]]:
    if n < 2:
        return []
    out = [(i, i + 1) for i in range(n - 1)]
    out.append((n - 1, 0))
    return out



def grid_edges(rows: int, cols: int, periodic: bool = False) -> List[Tuple[int, int]]:
    edges: set[Tuple[int, int]] = set()

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            here = idx(r, c)
            if c + 1 < cols:
                edges.add(tuple(sorted((here, idx(r, c + 1)))))
            elif periodic and cols > 2:
                edges.add(tuple(sorted((here, idx(r, 0)))))
            if r + 1 < rows:
                edges.add(tuple(sorted((here, idx(r + 1, c)))))
            elif periodic and rows > 2:
                edges.add(tuple(sorted((here, idx(0, c)))))
    return sorted(edges)



def ring_distance(i: int, j: int, n: int) -> int:
    d = abs(i - j)
    return min(d, n - d)



def field_from_condition(params: "EBMParameters", condition_v: Array) -> Array:
    return params.b + params.W @ condition_v.astype(np.float64)


# ---------------------------------------------------------------------------
# Physical problems and datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalProblem:
    name: str
    n_sites: int
    edge_u: Array
    edge_v: Array
    couplings: Array
    longitudinal_fields: Array
    metadata: Dict[str, object]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train_data: Array
    support_states: Optional[Array]
    support_probs: Optional[Array]
    metadata: Dict[str, object]
    physical_problem: Optional[PhysicalProblem] = None



def bars_and_stripes(rows: int, cols: int) -> Array:
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    patterns: List[Array] = []
    for mask in range(1 << rows):
        bits = np.array([(mask >> r) & 1 for r in range(rows)], dtype=np.int8)
        grid = np.repeat(bits[:, None], cols, axis=1)
        patterns.append(2 * grid.reshape(-1) - 1)
        patterns.append(1 - 2 * grid.reshape(-1))
    for mask in range(1 << cols):
        bits = np.array([(mask >> c) & 1 for c in range(cols)], dtype=np.int8)
        grid = np.repeat(bits[None, :], rows, axis=0)
        patterns.append(2 * grid.reshape(-1) - 1)
        patterns.append(1 - 2 * grid.reshape(-1))
    arr = np.unique(np.asarray(patterns, dtype=np.int8), axis=0)
    return arr



def parity_support(n_visible: int, parity: int = 0) -> Array:
    states = spin_states(n_visible)
    ones = ((states + 1) // 2).sum(axis=1)
    keep = (ones % 2) == int(parity)
    return states[keep]



def planted_ising_distribution(n_visible: int, seed: int) -> Tuple[Array, Array, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    states = spin_states(n_visible)
    a = 0.15 * rng.standard_normal(n_visible)
    edges = periodic_ring_edges(n_visible)
    J = np.zeros((n_visible, n_visible), dtype=np.float64)
    for i, j in edges:
        val = rng.choice([-1.0, 1.0]) * (0.5 + 0.25 * rng.random())
        J[i, j] = J[j, i] = val
    score = states @ a + 0.5 * np.einsum("si,ij,sj->s", states, J, states)
    logp = score - logsumexp(score)
    probs = np.exp(logp)
    meta = {
        "visible_biases": a.tolist(),
        "visible_ring_couplings": [(i, j, float(J[i, j])) for i, j in edges],
    }
    return states, probs, meta



def three_spin_distribution(n_visible: int, zeta: float, seed: int) -> Tuple[Array, Array, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    states = spin_states(n_visible).astype(np.int8, copy=False)
    triples: List[Tuple[int, int, int]] = []
    coeffs: List[float] = []
    std = math.sqrt(3.0 * float(zeta) / float(n_visible))
    for i in range(n_visible):
        for j in range(i + 1, n_visible):
            for k in range(j + 1, n_visible):
                triples.append((i, j, k))
                coeffs.append(float(rng.normal(0.0, std)))
    energy = np.zeros(states.shape[0], dtype=np.float64)
    for (i, j, k), t in zip(triples, coeffs):
        energy -= t * states[:, i] * states[:, j] * states[:, k]
    logp = -energy - logsumexp(-energy)
    probs = np.exp(logp)
    meta = {
        "three_spin_zeta": float(zeta),
        "three_spin_terms": [(i, j, k, float(t)) for (i, j, k), t in zip(triples, coeffs)],
    }
    return states, probs, meta



def build_long_range_1d(L: int, alpha: float) -> PhysicalProblem:
    edge_u: List[int] = []
    edge_v: List[int] = []
    J: List[float] = []
    for i in range(L):
        for j in range(i + 1, L):
            r = ring_distance(i, j, L)
            edge_u.append(i)
            edge_v.append(j)
            J.append(1.0 / (float(r) ** float(alpha)))
    return PhysicalProblem(
        name=f"long_range_1d_L{L}_alpha{alpha:g}",
        n_sites=L,
        edge_u=np.asarray(edge_u, dtype=np.int64),
        edge_v=np.asarray(edge_v, dtype=np.int64),
        couplings=np.asarray(J, dtype=np.float64),
        longitudinal_fields=np.zeros(L, dtype=np.float64),
        metadata={"kind": "long_range_1d", "L": L, "alpha": alpha},
    )



def build_j1j2_1d(L: int, J1: float, J2: float) -> PhysicalProblem:
    seen: Dict[Tuple[int, int], float] = {}
    for i in range(L):
        j1 = (i + 1) % L
        u, v = sorted((i, j1))
        seen[(u, v)] = seen.get((u, v), 0.0) + float(J1)
        j2 = (i + 2) % L
        u, v = sorted((i, j2))
        seen[(u, v)] = seen.get((u, v), 0.0) + float(J2)
    edges = sorted(seen)
    return PhysicalProblem(
        name=f"j1j2_1d_L{L}_J1{J1:g}_J2{J2:g}",
        n_sites=L,
        edge_u=np.asarray([u for u, _ in edges], dtype=np.int64),
        edge_v=np.asarray([v for _, v in edges], dtype=np.int64),
        couplings=np.asarray([seen[e] for e in edges], dtype=np.float64),
        longitudinal_fields=np.zeros(L, dtype=np.float64),
        metadata={"kind": "j1j2_1d", "L": L, "J1": J1, "J2": J2},
    )



def build_ea_2d(Lx: int, Ly: int, disorder_seed: int) -> PhysicalProblem:
    rng = np.random.default_rng(disorder_seed)
    edges: List[Tuple[int, int]] = []
    J: List[float] = []

    def idx(x: int, y: int) -> int:
        return y * Lx + x

    for y in range(Ly):
        for x in range(Lx):
            here = idx(x, y)
            if x + 1 < Lx:
                edges.append((here, idx(x + 1, y)))
                J.append(float(rng.choice([-1.0, 1.0])))
            if y + 1 < Ly:
                edges.append((here, idx(x, y + 1)))
                J.append(float(rng.choice([-1.0, 1.0])))
    return PhysicalProblem(
        name=f"ea_2d_{Lx}x{Ly}_seed{disorder_seed}",
        n_sites=Lx * Ly,
        edge_u=np.asarray([u for u, _ in edges], dtype=np.int64),
        edge_v=np.asarray([v for _, v in edges], dtype=np.int64),
        couplings=np.asarray(J, dtype=np.float64),
        longitudinal_fields=np.zeros(Lx * Ly, dtype=np.float64),
        metadata={"kind": "ea_2d", "Lx": Lx, "Ly": Ly, "disorder_seed": disorder_seed},
    )



def build_physical_problem(args: argparse.Namespace) -> PhysicalProblem:
    if args.physical_hamiltonian == "long_range_1d":
        return build_long_range_1d(args.long_range_size, args.long_range_alpha)
    if args.physical_hamiltonian == "j1j2_1d":
        return build_j1j2_1d(args.j1j2_size, args.j1, args.j2)
    if args.physical_hamiltonian == "ea_2d":
        return build_ea_2d(args.ea_Lx, args.ea_Ly, args.disorder_seed)
    raise ValueError(f"Unsupported physical Hamiltonian: {args.physical_hamiltonian}")



def exact_ground_distribution(problem: PhysicalProblem, gamma: float) -> Tuple[Array, Array, float]:
    n = problem.n_sites
    dim = 1 << n
    states = spin_states(n).astype(np.int8, copy=False)
    diag = -np.sum(problem.couplings[None, :] * states[:, problem.edge_u] * states[:, problem.edge_v], axis=1)
    diag -= states @ problem.longitudinal_fields

    rows = np.repeat(np.arange(dim, dtype=np.int64), n)
    cols = np.empty(dim * n, dtype=np.int64)
    data = np.full(dim * n, -float(gamma), dtype=np.float64)
    for basis in range(dim):
        base = basis * n
        for i in range(n):
            cols[base + i] = basis ^ (1 << i)

    H = sparse.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()
    H = H + sparse.diags(diag.astype(np.float64), offsets=0, shape=(dim, dim), format="csr")
    vals, vecs = eigsh(H, k=1, which="SA", tol=1.0e-10)
    psi0 = np.abs(np.asarray(vecs[:, 0], dtype=np.float64))
    psi0 /= np.linalg.norm(psi0)
    probs = psi0 ** 2
    return states, probs, float(vals[0])



def build_dataset(args: argparse.Namespace, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)

    if args.dataset_source == "bars_stripes":
        support = bars_and_stripes(args.rows, args.cols)
        probs = np.full(support.shape[0], 1.0 / support.shape[0], dtype=np.float64)
        idx = rng.choice(support.shape[0], size=args.train_samples, replace=True)
        return DatasetBundle(
            name=f"bars_stripes_{args.rows}x{args.cols}",
            train_data=support[idx].astype(np.int8, copy=False),
            support_states=support.astype(np.int8, copy=False),
            support_probs=probs,
            metadata={"rows": args.rows, "cols": args.cols},
        )

    if args.dataset_source == "parity":
        support = parity_support(args.n_visible, parity=args.parity)
        probs = np.full(support.shape[0], 1.0 / support.shape[0], dtype=np.float64)
        idx = rng.choice(support.shape[0], size=args.train_samples, replace=True)
        return DatasetBundle(
            name=f"parity_n{args.n_visible}_p{args.parity}",
            train_data=support[idx].astype(np.int8, copy=False),
            support_states=support.astype(np.int8, copy=False),
            support_probs=probs,
            metadata={"n_visible": args.n_visible, "parity": args.parity},
        )

    if args.dataset_source == "planted_ising":
        support, probs, meta = planted_ising_distribution(args.n_visible, seed=seed + 701)
        idx = rng.choice(support.shape[0], size=args.train_samples, replace=True, p=probs)
        return DatasetBundle(
            name=f"planted_ising_n{args.n_visible}",
            train_data=support[idx].astype(np.int8, copy=False),
            support_states=support.astype(np.int8, copy=False),
            support_probs=probs.astype(np.float64, copy=False),
            metadata=meta,
        )

    if args.dataset_source == "three_spin":
        support, probs, meta = three_spin_distribution(args.n_visible, args.three_spin_zeta, seed=seed + 1701)
        idx = rng.choice(support.shape[0], size=args.train_samples, replace=True, p=probs)
        return DatasetBundle(
            name=f"three_spin_n{args.n_visible}_z{args.three_spin_zeta:g}",
            train_data=support[idx].astype(np.int8, copy=False),
            support_states=support.astype(np.int8, copy=False),
            support_probs=probs.astype(np.float64, copy=False),
            metadata=meta,
        )

    if args.dataset_source == "physics_ground":
        problem = build_physical_problem(args)
        if problem.n_sites > args.max_exact_visible:
            raise ValueError(
                f"physics_ground requires exact reference states, but n_sites={problem.n_sites} exceeds "
                f"--max-exact-visible={args.max_exact_visible}."
            )
        support, probs, e0 = exact_ground_distribution(problem, gamma=args.transverse_field)
        idx = rng.choice(support.shape[0], size=args.train_samples, replace=True, p=probs)
        meta = {
            "physical_hamiltonian": problem.metadata,
            "transverse_field": float(args.transverse_field),
            "exact_ground_energy_total": float(e0),
            "exact_ground_energy_per_spin": float(e0 / problem.n_sites),
        }
        return DatasetBundle(
            name=f"physics_ground_{problem.name}_g{args.transverse_field:g}",
            train_data=support[idx].astype(np.int8, copy=False),
            support_states=support.astype(np.int8, copy=False),
            support_probs=probs.astype(np.float64, copy=False),
            metadata=meta,
            physical_problem=problem,
        )

    raise ValueError(f"Unsupported dataset source: {args.dataset_source}")


# ---------------------------------------------------------------------------
# Model specification and parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    n_visible: int
    n_hidden: int
    W_mask: Array
    L_mask: Array


@dataclass
class EBMParameters:
    a: Array
    b: Array
    W: Array
    L: Array

    def copy(self) -> "EBMParameters":
        return EBMParameters(self.a.copy(), self.b.copy(), self.W.copy(), self.L.copy())


@dataclass
class NegativePhase:
    mean_v: Array
    mean_h: Array
    mean_hv: Array
    mean_vv: Array
    samples_v: Optional[Array]
    samples_h: Optional[Array]
    meta: Dict[str, object]



def make_visible_graph_mask(
    kind: str,
    n_visible: int,
    rows: int,
    cols: int,
    physics_problem: Optional[PhysicalProblem] = None,
) -> Array:
    mask = np.zeros((n_visible, n_visible), dtype=bool)
    edges: List[Tuple[int, int]]
    if kind == "ring":
        edges = periodic_ring_edges(n_visible)
    elif kind == "dense":
        edges = [(i, j) for i in range(n_visible) for j in range(i + 1, n_visible)]
    elif kind == "grid":
        if rows * cols != n_visible:
            raise ValueError("grid visible graph requires rows * cols == n_visible")
        edges = grid_edges(rows, cols, periodic=False)
    elif kind == "physics":
        if physics_problem is None:
            raise ValueError("visible_graph=physics requires a physical dataset/problem")
        edges = list(zip(physics_problem.edge_u.tolist(), physics_problem.edge_v.tolist()))
    else:
        raise ValueError(f"Unsupported visible graph kind: {kind}")
    for i, j in edges:
        mask[i, j] = True
        mask[j, i] = True
    np.fill_diagonal(mask, False)
    return mask



def make_hidden_visible_mask(n_hidden: int, n_visible: int, kind: str, degree: int, rng: np.random.Generator) -> Array:
    if kind == "rbm":
        return np.ones((n_hidden, n_visible), dtype=bool)
    if kind == "masked_rbm":
        degree = max(1, min(int(degree), n_visible))
        mask = np.zeros((n_hidden, n_visible), dtype=bool)
        for j in range(n_hidden):
            picks = rng.choice(n_visible, size=degree, replace=False)
            mask[j, picks] = True
        lonely_visible = np.where(mask.sum(axis=0) == 0)[0]
        for i in lonely_visible:
            j = int(rng.integers(0, n_hidden))
            mask[j, i] = True
        lonely_hidden = np.where(mask.sum(axis=1) == 0)[0]
        for j in lonely_hidden:
            i = int(rng.integers(0, n_visible))
            mask[j, i] = True
        return mask
    if kind == "srbm":
        return np.ones((n_hidden, n_visible), dtype=bool)
    raise ValueError(f"Unsupported model kind: {kind}")



def build_model_spec(args: argparse.Namespace, dataset: DatasetBundle, seed: int) -> ModelSpec:
    rng = np.random.default_rng(seed)
    n_visible = dataset.train_data.shape[1]
    W_mask = make_hidden_visible_mask(args.n_hidden, n_visible, args.model_kind, args.masked_degree, rng)
    if args.model_kind == "srbm":
        L_mask = make_visible_graph_mask(
            args.visible_graph,
            n_visible,
            args.rows,
            args.cols,
            physics_problem=dataset.physical_problem,
        )
    else:
        L_mask = np.zeros((n_visible, n_visible), dtype=bool)
    return ModelSpec(kind=args.model_kind, n_visible=n_visible, n_hidden=args.n_hidden, W_mask=W_mask, L_mask=L_mask)



def init_parameters(spec: ModelSpec, scale: float, rng: np.random.Generator) -> EBMParameters:
    a = scale * rng.standard_normal(spec.n_visible)
    b = scale * rng.standard_normal(spec.n_hidden)
    W = scale * rng.standard_normal((spec.n_hidden, spec.n_visible)) * spec.W_mask
    L = scale * rng.standard_normal((spec.n_visible, spec.n_visible))
    L = 0.5 * (L + L.T)
    np.fill_diagonal(L, 0.0)
    L *= spec.L_mask
    return EBMParameters(a=a, b=b, W=W, L=L)



def enforce_parameter_constraints(params: EBMParameters, spec: ModelSpec, clip: float) -> None:
    np.clip(params.a, -clip, clip, out=params.a)
    np.clip(params.b, -clip, clip, out=params.b)
    np.clip(params.W, -clip, clip, out=params.W)
    params.W *= spec.W_mask
    params.L = 0.5 * (params.L + params.L.T)
    np.fill_diagonal(params.L, 0.0)
    params.L *= spec.L_mask
    np.clip(params.L, -clip, clip, out=params.L)


# ---------------------------------------------------------------------------
# Model log probabilities, metrics, and physics observables
# ---------------------------------------------------------------------------



def hidden_field_unscaled(V: Array, params: EBMParameters) -> Array:
    return params.b[None, :] + V @ params.W.T



def visible_log_unnorm_beta(V: Array, params: EBMParameters, beta: float) -> Array:
    beta = float(beta)
    theta = beta * hidden_field_unscaled(V, params)
    linear = beta * (V @ params.a)
    vv = 0.5 * beta * np.einsum("si,ij,sj->s", V, params.L, V)
    hidden = np.sum(np.logaddexp(theta, -theta), axis=1)
    return linear + vv + hidden



def exact_visible_distribution(params: EBMParameters, beta: float) -> Tuple[Array, Array, Array]:
    states = spin_states(params.a.shape[0]).astype(np.int8, copy=False)
    logu = visible_log_unnorm_beta(states, params, beta)
    logp = logu - logsumexp(logu)
    return states, logu, logp



def exact_target_kl(params: EBMParameters, target_states: Optional[Array], target_probs: Optional[Array], beta: float) -> Optional[float]:
    if target_states is None or target_probs is None:
        return None
    logu = visible_log_unnorm_beta(target_states, params, beta)
    log_model = logu - logsumexp(logu)
    return float(np.sum(target_probs * (np.log(target_probs + 1.0e-300) - log_model)))



def pseudo_log_likelihood(V: Array, params: EBMParameters, beta: float) -> float:
    n_samples, n_visible = V.shape
    base = visible_log_unnorm_beta(V, params, beta)
    total = 0.0
    for i in range(n_visible):
        flipped = V.copy()
        flipped[:, i] *= -1
        alt = visible_log_unnorm_beta(flipped, params, beta)
        total += float(np.mean(base - np.logaddexp(base, alt)))
    return total / n_visible



def reconstruction_error(V: Array, params: EBMParameters, beta: float, rng: np.random.Generator) -> float:
    theta = beta * hidden_field_unscaled(V, params)
    H = logistic_spin(theta, rng)
    Vrec = V.copy()
    for i in range(V.shape[1]):
        field = params.a[i] + H @ params.W[:, i] + Vrec @ params.L[:, i]
        Vrec[:, i] = logistic_spin(beta * field, rng)
    return float(np.mean(Vrec != V))



def positive_phase_stats(V: Array, params: EBMParameters, beta_eff: float) -> Tuple[Array, Array, Array, Array]:
    Hmean = np.tanh(beta_eff * hidden_field_unscaled(V, params))
    mean_v = V.mean(axis=0)
    mean_h = Hmean.mean(axis=0)
    mean_hv = (Hmean.T @ V) / V.shape[0]
    mean_vv = (V.T @ V) / V.shape[0]
    return mean_v, mean_h, mean_hv, mean_vv



def diagonal_energy_batch(V: Array, problem: PhysicalProblem) -> Array:
    diag = -np.sum(problem.couplings[None, :] * V[:, problem.edge_u] * V[:, problem.edge_v], axis=1)
    diag -= V @ problem.longitudinal_fields
    return diag



def exact_physics_variational_energy_per_spin(
    params: EBMParameters,
    beta_eff: float,
    problem: PhysicalProblem,
    gamma: float,
    support_states: Optional[Array] = None,
) -> float:
    states = spin_states(problem.n_sites).astype(np.int8, copy=False) if support_states is None else support_states.astype(np.int8, copy=False)
    if states.shape[0] != (1 << problem.n_sites):
        raise ValueError("Physical variational energy requires the full visible-state support.")
    logu = visible_log_unnorm_beta(states, params, beta_eff)
    logp = logu - logsumexp(logu)
    probs = np.exp(logp)
    dim = states.shape[0]
    indices = np.arange(dim, dtype=np.int64)
    diag = diagonal_energy_batch(states, problem)
    offdiag = np.zeros(dim, dtype=np.float64)
    for i in range(problem.n_sites):
        flip_idx = indices ^ (1 << i)
        logratio = 0.5 * (logp[flip_idx] - logp)
        offdiag += np.exp(logratio)
    energy = float(np.sum(probs * (diag - float(gamma) * offdiag)))
    return energy / float(problem.n_sites)



def exact_model_observables(
    params: EBMParameters,
    beta_eff: float,
    support_states: Array,
) -> Dict[str, float]:
    logu = visible_log_unnorm_beta(support_states, params, beta_eff)
    logp = logu - logsumexp(logu)
    p = np.exp(logp)
    mz = float(np.sum(p[:, None] * support_states, axis=0).mean())
    abs_mz = float(np.sum(p * np.abs(support_states.mean(axis=1))))
    entropy = float(-np.sum(p * logp))
    return {
        "model_mean_magnetization": mz,
        "model_abs_site_averaged_magnetization": abs_mz,
        "model_visible_entropy_nats": entropy,
    }


# ---------------------------------------------------------------------------
# Sampler backends
# ---------------------------------------------------------------------------


class BackendBase:
    def negative_phase(self, params: EBMParameters, spec: ModelSpec, n_samples: int, beta_submit: float) -> NegativePhase:
        raise NotImplementedError

    def conditional_hidden_means(self, params: EBMParameters, conditions: Array, n_samples: int, beta_submit: float) -> Tuple[Array, Dict[str, object]]:
        raise NotImplementedError

    def known_beta_eff(self, beta_submit: float) -> Optional[float]:
        return None

    def close(self) -> None:
        return None


class ExactBackend(BackendBase):
    def negative_phase(self, params: EBMParameters, spec: ModelSpec, n_samples: int, beta_submit: float) -> NegativePhase:
        states, _, logp = exact_visible_distribution(params, beta_submit)
        p = np.exp(logp)
        Hmean = np.tanh(beta_submit * hidden_field_unscaled(states, params))
        mean_v = p @ states
        mean_h = p @ Hmean
        mean_hv = Hmean.T @ (p[:, None] * states)
        mean_vv = states.T @ (p[:, None] * states)
        entropy = float(-np.sum(p * logp))
        return NegativePhase(
            mean_v=mean_v,
            mean_h=mean_h,
            mean_hv=mean_hv,
            mean_vv=mean_vv,
            samples_v=None,
            samples_h=None,
            meta={
                "backend": "exact",
                "beta_submit": float(beta_submit),
                "visible_state_count": int(states.shape[0]),
                "visible_entropy_nats": entropy,
            },
        )

    def conditional_hidden_means(self, params: EBMParameters, conditions: Array, n_samples: int, beta_submit: float) -> Tuple[Array, Dict[str, object]]:
        fields = params.b[None, :] + conditions @ params.W.T
        means = np.tanh(beta_submit * fields)
        return means, {"conditional_backend": "exact", "beta_submit": float(beta_submit)}

    def known_beta_eff(self, beta_submit: float) -> Optional[float]:
        return float(beta_submit)


class GibbsBackend(BackendBase):
    def __init__(self, n_visible: int, n_hidden: int, seed: int, sweeps: int = 2):
        self.rng = np.random.default_rng(seed)
        self.sweeps = max(1, int(sweeps))
        self.V = self.rng.choice([-1, 1], size=(1, n_visible)).astype(np.int8)
        self.H = self.rng.choice([-1, 1], size=(1, n_hidden)).astype(np.int8)

    def _ensure(self, n_samples: int) -> None:
        if self.V.shape[0] == n_samples:
            return
        reps = int(math.ceil(n_samples / self.V.shape[0]))
        self.V = np.tile(self.V, (reps, 1))[:n_samples].copy()
        self.H = np.tile(self.H, (reps, 1))[:n_samples].copy()

    def negative_phase(self, params: EBMParameters, spec: ModelSpec, n_samples: int, beta_submit: float) -> NegativePhase:
        self._ensure(n_samples)
        beta = float(beta_submit)
        for _ in range(self.sweeps):
            self.H = logistic_spin(beta * hidden_field_unscaled(self.V, params), self.rng)
            for i in range(spec.n_visible):
                field = params.a[i] + self.H @ params.W[:, i] + self.V @ params.L[:, i]
                self.V[:, i] = logistic_spin(beta * field, self.rng)
        mean_v = self.V.mean(axis=0)
        mean_h = self.H.mean(axis=0)
        mean_hv = (self.H.T @ self.V) / n_samples
        mean_vv = (self.V.T @ self.V) / n_samples
        return NegativePhase(
            mean_v=mean_v,
            mean_h=mean_h,
            mean_hv=mean_hv,
            mean_vv=mean_vv,
            samples_v=self.V.copy(),
            samples_h=self.H.copy(),
            meta={
                "backend": "gibbs",
                "beta_submit": float(beta_submit),
                "gibbs_sweeps": int(self.sweeps),
                "negative_samples": int(n_samples),
            },
        )

    def conditional_hidden_means(self, params: EBMParameters, conditions: Array, n_samples: int, beta_submit: float) -> Tuple[Array, Dict[str, object]]:
        fields = params.b[None, :] + conditions @ params.W.T
        means = np.tanh(beta_submit * fields)
        return means, {"conditional_backend": "gibbs_exact_conditional", "beta_submit": float(beta_submit)}

    def known_beta_eff(self, beta_submit: float) -> Optional[float]:
        return float(beta_submit)


class SimulatedAnnealingBackend(BackendBase):
    def __init__(self, n_visible: int, n_hidden: int, seed: int, sweeps: int = 30, temp_start: float = 3.0, temp_end: float = 0.35):
        self.rng = np.random.default_rng(seed)
        self.sweeps = max(2, int(sweeps))
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)
        self.V = self.rng.choice([-1, 1], size=(1, n_visible)).astype(np.int8)
        self.H = self.rng.choice([-1, 1], size=(1, n_hidden)).astype(np.int8)

    def _ensure(self, n_samples: int) -> None:
        if self.V.shape[0] == n_samples:
            return
        reps = int(math.ceil(n_samples / self.V.shape[0]))
        self.V = np.tile(self.V, (reps, 1))[:n_samples].copy()
        self.H = np.tile(self.H, (reps, 1))[:n_samples].copy()

    def _temps(self, beta_submit: float) -> Array:
        scale = 1.0 / max(1.0e-6, float(beta_submit))
        return np.geomspace(self.temp_start * scale, self.temp_end * scale, num=self.sweeps)

    def negative_phase(self, params: EBMParameters, spec: ModelSpec, n_samples: int, beta_submit: float) -> NegativePhase:
        self._ensure(n_samples)
        temps = self._temps(beta_submit)
        for temp in temps:
            inv_temp = 1.0 / float(temp)
            for j in range(spec.n_hidden):
                field = params.b[j] + self.V @ params.W[j, :]
                delta = 2.0 * self.H[:, j] * field
                accept = np.log(self.rng.random(n_samples)) < np.minimum(0.0, -inv_temp * delta)
                self.H[accept, j] *= -1
            for i in range(spec.n_visible):
                field = params.a[i] + self.H @ params.W[:, i] + self.V @ params.L[:, i]
                delta = 2.0 * self.V[:, i] * field
                accept = np.log(self.rng.random(n_samples)) < np.minimum(0.0, -inv_temp * delta)
                self.V[accept, i] *= -1
        mean_v = self.V.mean(axis=0)
        mean_h = self.H.mean(axis=0)
        mean_hv = (self.H.T @ self.V) / n_samples
        mean_vv = (self.V.T @ self.V) / n_samples
        return NegativePhase(
            mean_v=mean_v,
            mean_h=mean_h,
            mean_hv=mean_hv,
            mean_vv=mean_vv,
            samples_v=self.V.copy(),
            samples_h=self.H.copy(),
            meta={
                "backend": "sa",
                "beta_submit": float(beta_submit),
                "sa_sweeps": int(self.sweeps),
                "negative_samples": int(n_samples),
                "sa_temp_start": float(temps[0]),
                "sa_temp_end": float(temps[-1]),
            },
        )

    def conditional_hidden_means(self, params: EBMParameters, conditions: Array, n_samples: int, beta_submit: float) -> Tuple[Array, Dict[str, object]]:
        means = np.zeros((conditions.shape[0], params.b.shape[0]), dtype=np.float64)
        temps = self._temps(beta_submit)
        for cidx, cond in enumerate(conditions):
            H = self.rng.choice([-1, 1], size=(n_samples, params.b.shape[0])).astype(np.int8)
            field = params.b + params.W @ cond.astype(np.float64)
            for temp in temps:
                inv_temp = 1.0 / float(temp)
                for j in range(H.shape[1]):
                    delta = 2.0 * H[:, j] * field[j]
                    accept = np.log(self.rng.random(n_samples)) < np.minimum(0.0, -inv_temp * delta)
                    H[accept, j] *= -1
            means[cidx] = H.mean(axis=0)
        return means, {
            "conditional_backend": "sa",
            "conditional_sa_sweeps": int(self.sweeps),
            "conditional_sa_temp_start": float(temps[0]),
            "conditional_sa_temp_end": float(temps[-1]),
        }


class LangevinSBBackend(BackendBase):
    def __init__(self, n_visible: int, n_hidden: int, seed: int, steps: int = 100, delta: float = 1.0, sigma: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.n_visible = int(n_visible)
        self.n_hidden = int(n_hidden)
        self.steps = max(1, int(steps))
        self.delta = float(delta)
        self.sigma = float(sigma)

    @staticmethod
    def _sign01(x: Array) -> Array:
        out = np.sign(x)
        out[out == 0] = 1
        return out.astype(np.int8)

    def _joint_bias_and_coupler_matrix(self, params: EBMParameters) -> Tuple[Array, Array]:
        n = self.n_visible + self.n_hidden
        f = np.concatenate([params.a, params.b]).astype(np.float64)
        J = np.zeros((n, n), dtype=np.float64)
        if self.n_visible:
            J[: self.n_visible, : self.n_visible] = params.L
        if self.n_hidden:
            J[self.n_visible :, : self.n_visible] = params.W
            J[: self.n_visible, self.n_visible :] = params.W.T
        np.fill_diagonal(J, 0.0)
        return f, J

    def _sample_problem(self, f: Array, J: Array, n_samples: int) -> Array:
        S = self.rng.choice([-1, 1], size=(n_samples, f.shape[0])).astype(np.int8)
        Y = self.rng.normal(0.0, self.sigma, size=(n_samples, f.shape[0]))
        for _ in range(self.steps):
            force = S @ J.T + f[None, :]
            Y = Y + self.delta * force
            X = S.astype(np.float64) + self.delta * Y
            S = self._sign01(X)
            Y = self.rng.normal(0.0, self.sigma, size=Y.shape)
        return S

    def negative_phase(self, params: EBMParameters, spec: ModelSpec, n_samples: int, beta_submit: float) -> NegativePhase:
        f, J = self._joint_bias_and_coupler_matrix(params)
        S = self._sample_problem(f, J, n_samples)
        V = S[:, : self.n_visible]
        H = S[:, self.n_visible :]
        mean_v = V.mean(axis=0)
        mean_h = H.mean(axis=0)
        mean_hv = (H.T @ V) / n_samples
        mean_vv = (V.T @ V) / n_samples
        return NegativePhase(
            mean_v=mean_v,
            mean_h=mean_h,
            mean_hv=mean_hv,
            mean_vv=mean_vv,
            samples_v=V,
            samples_h=H,
            meta={
                "backend": "lsb",
                "lsb_steps": int(self.steps),
                "lsb_delta": float(self.delta),
                "lsb_sigma": float(self.sigma),
                "negative_samples": int(n_samples),
            },
        )

    def conditional_hidden_means(self, params: EBMParameters, conditions: Array, n_samples: int, beta_submit: float) -> Tuple[Array, Dict[str, object]]:
        means = np.zeros((conditions.shape[0], params.b.shape[0]), dtype=np.float64)
        for cidx, cond in enumerate(conditions):
            field = params.b + params.W @ cond.astype(np.float64)
            J = np.zeros((field.shape[0], field.shape[0]), dtype=np.float64)
            S = self._sample_problem(field, J, n_samples)
            means[cidx] = S.mean(axis=0)
        return means, {
            "conditional_backend": "lsb",
            "conditional_lsb_steps": int(self.steps),
            "conditional_lsb_delta": float(self.delta),
            "conditional_lsb_sigma": float(self.sigma),
        }


class QPUBackend(BackendBase):
    def __init__(self, topology: str, spec: ModelSpec, args: argparse.Namespace):
        self.topology = topology
        self.spec = spec
        self.annealing_time = float(args.annealing_time)
        self.requested_reads_per_child_call = args.requested_reads_per_child_call
        self.visible_labels = [f"v{i}" for i in range(spec.n_visible)]
        self.hidden_labels = [f"h{j}" for j in range(spec.n_hidden)]
        self.qpu = RuntimeSafeQPUIsingSampler(
            QPUAccessConfig(
                topology=topology,
                token=args.token,
                endpoint=args.endpoint,
                region=args.region,
                solver_name=(args.solver_pegasus if topology == "pegasus" else args.solver_zephyr) or args.solver_name,
                max_runtime_fraction=float(args.max_runtime_fraction),
                programming_thermalization=args.programming_thermalization,
                readout_thermalization=args.readout_thermalization,
                num_spin_reversal_transforms=int(args.num_spin_reversal_transforms),
                chain_strength_prefactor=float(args.chain_strength_prefactor),
            )
        )

    def _logical_problem(self, params: EBMParameters) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], Optional[Tuple[Sequence[str], Sequence[str]]]]:
        h_score: Dict[str, float] = {}
        J_score: Dict[Tuple[str, str], float] = {}
        for i, label in enumerate(self.visible_labels):
            h_score[label] = float(params.a[i])
        for j, label in enumerate(self.hidden_labels):
            h_score[label] = float(params.b[j])
        for j, h_label in enumerate(self.hidden_labels):
            for i, v_label in enumerate(self.visible_labels):
                if self.spec.W_mask[j, i]:
                    J_score[(h_label, v_label)] = float(params.W[j, i])
        for i in range(self.spec.n_visible):
            for j in range(i + 1, self.spec.n_visible):
                if self.spec.L_mask[i, j]:
                    J_score[(self.visible_labels[i], self.visible_labels[j])] = float(params.L[i, j])
        biclique_partition: Optional[Tuple[Sequence[str], Sequence[str]]] = None
        if self.spec.kind == "rbm" and np.all(self.spec.W_mask) and not np.any(self.spec.L_mask):
            biclique_partition = (self.visible_labels, self.hidden_labels)
        return h_score, J_score, biclique_partition

    def negative_phase(self, params: EBMParameters, spec: ModelSpec, n_samples: int, beta_submit: float) -> NegativePhase:
        if beta_submit <= 0:
            raise ValueError("beta_submit must be positive for QPU sampling")
        h_score, J_score, biclique = self._logical_problem(params)
        samples, variables, meta = self.qpu.sample_ising(
            h_score=h_score,
            J_score=J_score,
            target_samples=n_samples,
            annealing_time=self.annealing_time,
            beta_scale=1.0 / float(beta_submit),
            requested_reads_per_child_call=self.requested_reads_per_child_call,
            biclique_partition=biclique,
        )
        index = {name: i for i, name in enumerate(variables)}
        V = np.asarray(samples[:, [index[name] for name in self.visible_labels]], dtype=np.int8)
        H = np.asarray(samples[:, [index[name] for name in self.hidden_labels]], dtype=np.int8)
        mean_v = V.mean(axis=0)
        mean_h = H.mean(axis=0)
        mean_hv = (H.T @ V) / n_samples
        mean_vv = (V.T @ V) / n_samples
        meta = dict(meta)
        meta["backend"] = f"qpu_{self.topology}"
        meta["beta_submit"] = float(beta_submit)
        meta["sampler_metadata"] = self.qpu.metadata()
        return NegativePhase(mean_v, mean_h, mean_hv, mean_vv, V, H, meta)

    def conditional_hidden_means(self, params: EBMParameters, conditions: Array, n_samples: int, beta_submit: float) -> Tuple[Array, Dict[str, object]]:
        if beta_submit <= 0:
            raise ValueError("beta_submit must be positive for QPU sampling")
        means = np.zeros((conditions.shape[0], self.spec.n_hidden), dtype=np.float64)
        total_qpu_access = 0.0
        total_batches = 0
        max_chain_break = 0.0
        for cidx, cond in enumerate(conditions):
            fields = field_from_condition(params, cond)
            h_score = {label: float(fields[j]) for j, label in enumerate(self.hidden_labels)}
            J_score: Dict[Tuple[str, str], float] = {}
            samples, variables, meta = self.qpu.sample_ising(
                h_score=h_score,
                J_score=J_score,
                target_samples=n_samples,
                annealing_time=self.annealing_time,
                beta_scale=1.0 / float(beta_submit),
                requested_reads_per_child_call=self.requested_reads_per_child_call,
                biclique_partition=None,
            )
            index = {name: i for i, name in enumerate(variables)}
            H = np.asarray(samples[:, [index[name] for name in self.hidden_labels]], dtype=np.int8)
            means[cidx] = H.mean(axis=0)
            total_qpu_access += float(meta.get("avg_qpu_access_time_us", 0.0)) * int(meta.get("n_batches", 0))
            total_batches += int(meta.get("n_batches", 0))
            max_chain_break = max(max_chain_break, float(meta.get("avg_chain_break_fraction", 0.0)))
        return means, {
            "conditional_backend": f"qpu_{self.topology}",
            "conditional_qpu_access_time_us": float(total_qpu_access),
            "conditional_batches": int(total_batches),
            "conditional_max_chain_break_fraction": float(max_chain_break),
        }

    def close(self) -> None:
        self.qpu.close()



def make_backend(args: argparse.Namespace, spec: ModelSpec, backend_name: str, seed: int) -> BackendBase:
    if backend_name == "exact":
        return ExactBackend()
    if backend_name == "gibbs":
        return GibbsBackend(spec.n_visible, spec.n_hidden, seed=seed, sweeps=args.gibbs_sweeps)
    if backend_name == "sa":
        return SimulatedAnnealingBackend(
            spec.n_visible,
            spec.n_hidden,
            seed=seed,
            sweeps=args.sa_sweeps,
            temp_start=args.sa_temp_start,
            temp_end=args.sa_temp_end,
        )
    if backend_name == "lsb":
        return LangevinSBBackend(
            spec.n_visible,
            spec.n_hidden,
            seed=seed,
            steps=args.lsb_steps,
            delta=args.lsb_delta,
            sigma=args.lsb_sigma,
        )
    if backend_name == "qpu_pegasus":
        return QPUBackend("pegasus", spec, args)
    if backend_name == "qpu_zephyr":
        return QPUBackend("zephyr", spec, args)
    raise ValueError(f"Unsupported backend: {backend_name}")


# ---------------------------------------------------------------------------
# Effective-temperature estimation: analytic / CEM / fixed
# ---------------------------------------------------------------------------



def choose_cem_conditions(args: argparse.Namespace, dataset: DatasetBundle, rng: np.random.Generator) -> Array:
    k = max(1, int(args.cem_conditions))
    n_visible = dataset.train_data.shape[1]
    if args.cem_condition_source == "data":
        idx = rng.choice(dataset.train_data.shape[0], size=k, replace=dataset.train_data.shape[0] < k)
        return dataset.train_data[idx].astype(np.int8, copy=False)
    if args.cem_condition_source == "random":
        return rng.choice([-1, 1], size=(k, n_visible)).astype(np.int8)
    raise ValueError(f"Unsupported CEM condition source: {args.cem_condition_source}")



def fit_beta_eff_from_cem(fields: Array, sample_means: Array, beta_min: float, beta_max: float, x0: float) -> Tuple[float, float]:
    fields = np.asarray(fields, dtype=np.float64)
    sample_means = np.asarray(sample_means, dtype=np.float64)
    if np.max(np.abs(fields)) < 1.0e-12:
        return float(x0), float(np.mean(sample_means ** 2))

    def objective(beta: float) -> float:
        pred = np.tanh(beta * fields)
        return float(np.mean((sample_means - pred) ** 2))

    res = minimize_scalar(objective, bounds=(float(beta_min), float(beta_max)), method="bounded", options={"xatol": 1.0e-4})
    if not res.success or not np.isfinite(res.x):
        return float(x0), objective(float(x0))
    return float(res.x), float(res.fun)



def estimate_beta_eff(
    args: argparse.Namespace,
    dataset: DatasetBundle,
    params: EBMParameters,
    backend: BackendBase,
    beta_submit: float,
    epoch: int,
    previous_beta_eff: Optional[float],
    rng: np.random.Generator,
) -> Tuple[float, Dict[str, object]]:
    mode = args.temperature_estimator
    if mode == "fixed":
        return float(args.fixed_beta_eff), {"beta_eff_source": "fixed", "cem_loss": 0.0}

    known = None if mode == "cem" else backend.known_beta_eff(beta_submit)
    if known is not None:
        return float(known), {"beta_eff_source": "analytic", "cem_loss": 0.0}

    if previous_beta_eff is not None and epoch > 1 and (epoch - 1) % int(args.cem_period) != 0:
        return float(previous_beta_eff), {"beta_eff_source": "cached", "cem_loss": math.nan}

    conditions = choose_cem_conditions(args, dataset, rng)
    sample_means, cond_meta = backend.conditional_hidden_means(params, conditions, n_samples=args.cem_samples, beta_submit=beta_submit)
    fields = params.b[None, :] + conditions @ params.W.T
    beta0 = float(previous_beta_eff if previous_beta_eff is not None else max(1.0e-3, beta_submit))
    beta_eff, loss = fit_beta_eff_from_cem(fields, sample_means, args.cem_beta_min, args.cem_beta_max, beta0)
    meta = {
        "beta_eff_source": "cem",
        "cem_loss": float(loss),
        "cem_conditions": int(conditions.shape[0]),
        "cem_samples": int(args.cem_samples),
    }
    meta.update({k: v for k, v in cond_meta.items() if isinstance(v, (int, float, str, bool))})
    return float(beta_eff), meta


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    model_kind: str
    backend_name: str
    seed: int



def run_single_experiment(payload: Mapping[str, object]) -> Dict[str, object]:
    args = argparse.Namespace(**dict(payload["args"]))
    cfg = ExperimentConfig(**payload["config"])
    root = Path(str(payload["output_dir"]))
    rng = np.random.default_rng(cfg.seed)

    dataset = build_dataset(args, seed=args.seed)
    spec_args = argparse.Namespace(**vars(args))
    spec_args.model_kind = cfg.model_kind
    spec = build_model_spec(spec_args, dataset, seed=cfg.seed + 11)
    params = init_parameters(spec, scale=args.param_init_scale, rng=rng)
    backend = make_backend(args, spec, cfg.backend_name, seed=cfg.seed + 101)

    exp_dir = root / dataset.name / cfg.model_kind / cfg.backend_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, object]] = []
    best_params = params.copy()
    best_kl = math.inf
    best_pll = -math.inf
    best_phys = math.inf
    best_primary_name = 'pseudo_log_likelihood'
    best_primary_value = math.inf
    beta_eff: Optional[float] = None

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            beta_eff, beta_meta = estimate_beta_eff(
                args=args,
                dataset=dataset,
                params=params,
                backend=backend,
                beta_submit=args.beta_submit,
                epoch=epoch,
                previous_beta_eff=beta_eff,
                rng=rng,
            )

            if args.mini_batch_size > 0 and args.mini_batch_size < dataset.train_data.shape[0]:
                idx = rng.choice(dataset.train_data.shape[0], size=args.mini_batch_size, replace=False)
                batch_data = dataset.train_data[idx]
            else:
                batch_data = dataset.train_data

            pos_v, pos_h, pos_hv, pos_vv = positive_phase_stats(batch_data, params, beta_eff)
            neg = backend.negative_phase(params, spec, args.negative_samples, beta_submit=args.beta_submit)

            params.a += args.learning_rate * (pos_v - neg.mean_v) - args.l2 * params.a
            params.b += args.learning_rate * (pos_h - neg.mean_h) - args.l2 * params.b
            params.W += args.learning_rate * (pos_hv - neg.mean_hv) - args.l2 * params.W
            if np.any(spec.L_mask):
                params.L += args.learning_rate * (pos_vv - neg.mean_vv) - args.l2 * params.L
            enforce_parameter_constraints(params, spec, clip=args.param_clip)

            row: Dict[str, object] = {
                "epoch": int(epoch),
                "beta_submit": float(args.beta_submit),
                "beta_eff": float(beta_eff),
                "negative_samples": int(args.negative_samples),
                "n_visible": int(spec.n_visible),
                "n_hidden": int(spec.n_hidden),
                "n_visible_visible_edges": int(np.count_nonzero(np.triu(spec.L_mask, 1))),
                "n_hidden_visible_edges": int(np.count_nonzero(spec.W_mask)),
            }
            row.update({k: v for k, v in beta_meta.items() if isinstance(v, (int, float, str, bool))})
            row.update({k: v for k, v in neg.meta.items() if isinstance(v, (int, float, str, bool))})

            do_eval = (epoch == 1) or (epoch == args.epochs) or (epoch % int(args.eval_every) == 0)
            if do_eval:
                pll = pseudo_log_likelihood(dataset.train_data, params, beta_eff)
                recon = reconstruction_error(dataset.train_data[: min(256, len(dataset.train_data))], params, beta_eff, rng)
                row["pseudo_log_likelihood"] = float(pll)
                row["reconstruction_error"] = float(recon)
                best_pll = max(best_pll, float(pll))

                if dataset.support_states is not None and dataset.support_probs is not None:
                    kl = exact_target_kl(params, dataset.support_states, dataset.support_probs, beta_eff)
                    row["target_kl"] = float(kl) if kl is not None else math.nan
                    if kl is not None:
                        best_kl = min(best_kl, float(kl))
                    row.update(exact_model_observables(params, beta_eff, dataset.support_states))

                if dataset.physical_problem is not None:
                    evar = exact_physics_variational_energy_per_spin(
                        params,
                        beta_eff,
                        dataset.physical_problem,
                        gamma=float(dataset.metadata["transverse_field"]),
                        support_states=dataset.support_states,
                    )
                    row["physics_variational_energy_per_spin"] = float(evar)
                    row["physics_reference_ground_energy_per_spin"] = float(dataset.metadata["exact_ground_energy_per_spin"])
                    row["physics_relative_energy_error"] = abs(
                        (float(evar) - float(dataset.metadata["exact_ground_energy_per_spin"]))
                        / float(dataset.metadata["exact_ground_energy_per_spin"])
                    )
                    best_phys = min(best_phys, float(evar))

            row["iteration_seconds"] = float(time.time() - t0)

            if "physics_variational_energy_per_spin" in row:
                primary_name = "physics_variational_energy_per_spin"
                primary_value = float(row[primary_name])
            elif "target_kl" in row:
                primary_name = "target_kl"
                primary_value = float(row[primary_name])
            elif "pseudo_log_likelihood" in row:
                primary_name = "pseudo_log_likelihood"
                primary_value = -float(row[primary_name])
            else:
                primary_name = "beta_eff"
                primary_value = abs(float(beta_eff))

            if primary_value < best_primary_value:
                best_primary_value = float(primary_value)
                best_primary_name = str(primary_name)
                best_params = params.copy()

            history.append(row)

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                msg = f"[{dataset.name}/{cfg.model_kind}/{cfg.backend_name}] epoch {epoch:4d}/{args.epochs} beta_eff={beta_eff:.4f}"
                if "pseudo_log_likelihood" in row:
                    msg += f" pll={row['pseudo_log_likelihood']:+.5f} recon={row['reconstruction_error']:.4f}"
                if "target_kl" in row:
                    msg += f" KL={row['target_kl']:.5f}"
                if "physics_variational_energy_per_spin" in row:
                    msg += f" Evar/N={row['physics_variational_energy_per_spin']:+.6f}"
                print(msg, flush=True)

        save_history_csv(history, exp_dir / "history.csv")
        np.savez_compressed(exp_dir / "final_params.npz", a=params.a, b=params.b, W=params.W, L=params.L)
        np.savez_compressed(exp_dir / "best_params.npz", a=best_params.a, b=best_params.b, W=best_params.W, L=best_params.L)

        summary = {
            "config": asdict(cfg),
            "dataset": {
                "name": dataset.name,
                "metadata": dataset.metadata,
                "train_size": int(dataset.train_data.shape[0]),
            },
            "model": {
                "kind": spec.kind,
                "n_visible": int(spec.n_visible),
                "n_hidden": int(spec.n_hidden),
                "visible_visible_edges": int(np.count_nonzero(np.triu(spec.L_mask, 1))),
                "hidden_visible_edges": int(np.count_nonzero(spec.W_mask)),
            },
            "best_pseudo_log_likelihood": None if not np.isfinite(best_pll) else float(best_pll),
            "best_target_kl": None if not np.isfinite(best_kl) else float(best_kl),
            "best_physics_variational_energy_per_spin": None if not np.isfinite(best_phys) else float(best_phys),
            "best_primary_metric_name": best_primary_name,
            "best_primary_metric_value": float(best_primary_value),
            "final_row": history[-1],
            "output_dir": str(exp_dir),
        }
        save_json(summary, exp_dir / "summary.json")
        return summary
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------



def load_histories(root: Path, dataset_name: str, models: Sequence[str], backends: Sequence[str]) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    out: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for model in models:
        out[model] = {}
        for backend in backends:
            path = root / dataset_name / model / backend / "history.csv"
            if not path.exists():
                continue
            rows: List[Dict[str, object]] = []
            with path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    rows.append(dict(row))
            out[model][backend] = rows
    return out



def _metric_array(hist: Sequence[Mapping[str, object]], key: str) -> Optional[Array]:
    vals: List[float] = []
    seen = False
    for row in hist:
        if key in row and row[key] not in ("", None):
            try:
                vals.append(float(row[key]))
                seen = True
            except Exception:
                vals.append(np.nan)
        else:
            vals.append(np.nan)
    return np.asarray(vals, dtype=float) if seen else None



def plot_training_families(histories: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]], root: Path) -> None:
    metric_specs = [
        ("pseudo_log_likelihood", "Pseudo-log-likelihood"),
        ("reconstruction_error", "Reconstruction error"),
        ("target_kl", "KL to target"),
        ("beta_eff", "Effective inverse temperature"),
        ("physics_variational_energy_per_spin", "Variational energy per spin"),
        ("physics_relative_energy_error", "Relative energy error"),
    ]
    for model, per_backend in histories.items():
        if not per_backend:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0))
        axes_flat = axes.ravel()
        for ax, (key, title) in zip(axes_flat, metric_specs):
            have_any = False
            for backend, hist in per_backend.items():
                x = np.asarray([int(row["epoch"]) for row in hist], dtype=int)
                y = _metric_array(hist, key)
                if y is None:
                    continue
                have_any = True
                ax.plot(x, y, label=backend)
            if have_any:
                ax.set_title(title)
                ax.set_xlabel("epoch")
                ax.legend(fontsize=8)
            else:
                ax.axis("off")
        fig.suptitle(model)
        fig.tight_layout()
        fig.savefig(root / f"figure_{model}_training.png", dpi=220)
        plt.close(fig)



def plot_final_summary(histories: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]], root: Path) -> None:
    labels: List[str] = []
    final_kl: List[float] = []
    final_beta: List[float] = []
    final_phys: List[float] = []
    phys_labels: List[str] = []

    for model, per_backend in histories.items():
        for backend, hist in per_backend.items():
            last = hist[-1]
            labels.append(f"{model}\n{backend}")
            final_beta.append(float(last.get("beta_eff", np.nan)))
            final_kl.append(float(last.get("target_kl", np.nan)))
            if "physics_variational_energy_per_spin" in last and last["physics_variational_energy_per_spin"] not in ("", None):
                final_phys.append(float(last["physics_variational_energy_per_spin"]))
                phys_labels.append(f"{model}\n{backend}")

    if not labels:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    x = np.arange(len(labels))
    axes[0].bar(x, np.nan_to_num(np.asarray(final_beta, dtype=float), nan=0.0))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Final beta_eff")

    axes[1].bar(x, np.nan_to_num(np.asarray(final_kl, dtype=float), nan=0.0))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Final KL")

    if final_phys:
        x2 = np.arange(len(final_phys))
        axes[2].bar(x2, np.asarray(final_phys, dtype=float))
        axes[2].set_xticks(x2)
        axes[2].set_xticklabels(phys_labels, rotation=45, ha="right", fontsize=8)
        axes[2].set_title("Final variational energy / spin")
    else:
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(root / "figure_final_summary.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated SAL benchmark for classical and quantum samplers")

    parser.add_argument("--models", nargs="+", choices=["rbm", "masked_rbm", "srbm"], default=["rbm", "masked_rbm", "srbm"])
    parser.add_argument("--samplers", nargs="+", choices=["exact", "gibbs", "sa", "lsb", "qpu_pegasus", "qpu_zephyr"], default=["exact", "gibbs", "sa", "lsb"])
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="sal_integrated_benchmarks")

    parser.add_argument("--dataset-source", choices=["bars_stripes", "parity", "planted_ising", "three_spin", "physics_ground"], default="bars_stripes")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--n-visible", type=int, default=10)
    parser.add_argument("--parity", type=int, default=0, choices=[0, 1])
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--three-spin-zeta", type=float, default=2.0)

    parser.add_argument("--physical-hamiltonian", choices=["long_range_1d", "j1j2_1d", "ea_2d"], default="long_range_1d")
    parser.add_argument("--transverse-field", type=float, default=1.0)
    parser.add_argument("--long-range-size", type=int, default=8)
    parser.add_argument("--long-range-alpha", type=float, default=2.0)
    parser.add_argument("--j1j2-size", type=int, default=8)
    parser.add_argument("--j1", type=float, default=1.0)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--ea-Lx", type=int, default=3)
    parser.add_argument("--ea-Ly", type=int, default=3)
    parser.add_argument("--disorder-seed", type=int, default=11)
    parser.add_argument("--max-exact-visible", type=int, default=16)

    parser.add_argument("--n-hidden", type=int, default=8)
    parser.add_argument("--masked-degree", type=int, default=4)
    parser.add_argument("--visible-graph", choices=["ring", "dense", "grid", "physics"], default="ring")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--mini-batch-size", type=int, default=0)
    parser.add_argument("--negative-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1.0e-4)
    parser.add_argument("--param-init-scale", type=float, default=0.05)
    parser.add_argument("--param-clip", type=float, default=2.5)
    parser.add_argument("--eval-every", type=int, default=1)

    parser.add_argument("--beta-submit", type=float, default=1.0,
                        help="Control inverse temperature used by exact and Gibbs samplers, and the scale passed to SA/QPU backends.")
    parser.add_argument("--temperature-estimator", choices=["auto", "cem", "fixed"], default="auto")
    parser.add_argument("--fixed-beta-eff", type=float, default=1.0)
    parser.add_argument("--cem-period", type=int, default=1)
    parser.add_argument("--cem-samples", type=int, default=64)
    parser.add_argument("--cem-conditions", type=int, default=1)
    parser.add_argument("--cem-condition-source", choices=["data", "random"], default="data")
    parser.add_argument("--cem-beta-min", type=float, default=1.0e-3)
    parser.add_argument("--cem-beta-max", type=float, default=10.0)

    parser.add_argument("--gibbs-sweeps", type=int, default=2)
    parser.add_argument("--sa-sweeps", type=int, default=30)
    parser.add_argument("--sa-temp-start", type=float, default=3.0)
    parser.add_argument("--sa-temp-end", type=float, default=0.35)

    parser.add_argument("--lsb-steps", type=int, default=100)
    parser.add_argument("--lsb-delta", type=float, default=1.0)
    parser.add_argument("--lsb-sigma", type=float, default=1.0)

    parser.add_argument("--annealing-time", type=float, default=20.0)
    parser.add_argument("--requested-reads-per-child-call", type=int, default=None)
    parser.add_argument("--num-spin-reversal-transforms", type=int, default=4)
    parser.add_argument("--chain-strength-prefactor", type=float, default=1.5)
    parser.add_argument("--max-runtime-fraction", type=float, default=0.85)
    parser.add_argument("--programming-thermalization", type=float, default=None)
    parser.add_argument("--readout-thermalization", type=float, default=None)
    parser.add_argument("--solver-name", type=str, default=None)
    parser.add_argument("--solver-pegasus", type=str, default=None)
    parser.add_argument("--solver-zephyr", type=str, default=None)
    parser.add_argument("--token", type=str, default=os.getenv("DWAVE_API_TOKEN"))
    parser.add_argument("--endpoint", type=str, default=os.getenv("DWAVE_API_ENDPOINT"))
    parser.add_argument("--region", type=str, default=os.getenv("DWAVE_API_REGION"))

    parser.add_argument("--seed", type=int, default=1234)

    ns = parser.parse_args(argv)
    if ns.processes < 1:
        parser.error("--processes must be at least 1")
    if ns.eval_every < 1:
        parser.error("--eval-every must be at least 1")
    if ns.cem_period < 1:
        parser.error("--cem-period must be at least 1")
    if ns.beta_submit <= 0:
        parser.error("--beta-submit must be positive")
    if ns.dataset_source == "bars_stripes" and ns.rows * ns.cols > ns.max_exact_visible and "exact" in ns.samplers:
        parser.error("bars_stripes with exact backend requires rows*cols <= --max-exact-visible")
    return ns



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), root / "run_config.json")

    dataset = build_dataset(args, seed=args.seed)
    save_json({"dataset_name": dataset.name, "dataset_metadata": dataset.metadata}, root / "dataset_info.json")

    tasks = []
    for midx, model in enumerate(args.models):
        for bidx, backend in enumerate(args.samplers):
            seed = int(args.seed + 1000 * midx + 100 * bidx)
            tasks.append(
                {
                    "args": vars(args),
                    "config": {"model_kind": model, "backend_name": backend, "seed": seed},
                    "output_dir": str(root),
                }
            )

    if args.processes == 1:
        summaries = [run_single_experiment(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(processes=args.processes) as pool:
            summaries = pool.map(run_single_experiment, tasks)

    save_json(summaries, root / "all_summaries.json")

    histories = load_histories(root, dataset.name, args.models, args.samplers)
    plot_training_families(histories, root)
    plot_final_summary(histories, root)
    print(f"Saved benchmark outputs to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Runtime-safe generic Ising sampler for D-Wave Pegasus and Zephyr QPUs.

This version extends the earlier helper in two ways that matter for
sampler-adaptive learning (SAL) and conditional expectation matching (CEM):

1. It can embed disconnected logical graphs, including bias-only problems.
2. It keeps a fixed embedding cache and runtime-safe batching logic so the
   caller can reuse one helper for full BM sampling and hidden-only conditional
   sampling.

The logical problem is an Ising score model
    P(s) proportional to exp(sum_i g_i s_i + sum_{i<j} K_ij s_i s_j)
The QPU receives the corresponding minimized Ising energy with coefficients
negated and optionally divided by beta_scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class QPUAccessConfig:
    topology: str
    token: Optional[str] = None
    endpoint: Optional[str] = None
    region: Optional[str] = None
    solver_name: Optional[str] = None
    max_runtime_fraction: float = 0.85
    programming_thermalization: Optional[float] = None
    readout_thermalization: Optional[float] = None
    num_spin_reversal_transforms: int = 4
    chain_strength_prefactor: float = 1.5


class RuntimeSafeQPUIsingSampler:
    """Generic Pegasus/Zephyr Ising sampler with fixed embeddings and safe batching."""

    def __init__(self, access: QPUAccessConfig):
        self.access = access
        self._lazy_import_ocean()
        self.base_sampler = self._make_base_sampler()
        self.target_graph = self.base_sampler.to_networkx_graph()
        self.embedding_cache: Dict[Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]], Dict[str, Tuple[int, ...]]] = {}
        self.composite_cache: Dict[Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]], object] = {}
        self.runtime_cache: Dict[Tuple[int, float], Tuple[int, float]] = {}

    def _lazy_import_ocean(self) -> None:
        try:
            import dimod  # type: ignore
            from dwave.preprocessing import SpinReversalTransformComposite  # type: ignore
            from dwave.system import DWaveSampler, FixedEmbeddingComposite  # type: ignore
            from minorminer import find_embedding  # type: ignore
            from minorminer.busclique import busgraph_cache  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "QPU sampling requires the D-Wave Ocean SDK. Install with\n"
                "    pip install 'dwave-ocean-sdk>=9.0'\n"
                f"Original import error: {exc}"
            ) from exc

        self.dimod = dimod
        self.SpinReversalTransformComposite = SpinReversalTransformComposite
        self.DWaveSampler = DWaveSampler
        self.FixedEmbeddingComposite = FixedEmbeddingComposite
        self.find_embedding = find_embedding
        self.busgraph_cache = busgraph_cache

    def _make_base_sampler(self):
        solver_selector: object
        if self.access.solver_name:
            solver_selector = self.access.solver_name
        else:
            solver_selector = dict(topology__type=self.access.topology)

        kwargs: Dict[str, object] = {"solver": solver_selector}
        if self.access.token:
            kwargs["token"] = self.access.token
        if self.access.endpoint:
            kwargs["endpoint"] = self.access.endpoint
        if self.access.region:
            kwargs["region"] = self.access.region
        return self.DWaveSampler(**kwargs)

    @staticmethod
    def _canonical_edge(u: str, v: str) -> Tuple[str, str]:
        return (u, v) if u < v else (v, u)

    def _graph_key(self, variables: Sequence[str], couplers: Mapping[Tuple[str, str], float]) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
        edge_list = sorted(self._canonical_edge(str(u), str(v)) for (u, v) in couplers)
        return tuple(sorted(str(v) for v in variables)), tuple(edge_list)

    @staticmethod
    def _merge_embedding_object(obj: object) -> Dict[str, Tuple[int, ...]]:
        if isinstance(obj, dict):
            return {str(k): tuple(int(q) for q in chain) for k, chain in obj.items()}
        if isinstance(obj, tuple) and len(obj) == 2 and all(isinstance(x, dict) for x in obj):
            merged: Dict[str, Tuple[int, ...]] = {}
            for side in obj:
                merged.update({str(k): tuple(int(q) for q in chain) for k, chain in side.items()})
            return merged
        raise TypeError(f"Unsupported embedding object type: {type(obj)!r}")

    @staticmethod
    def _complete_bipartite_edges(left: Sequence[str], right: Sequence[str]) -> set[Tuple[str, str]]:
        return {
            (u, v) if u < v else (v, u)
            for u in left
            for v in right
        }

    @staticmethod
    def _complete_graph_edges(nodes: Sequence[str]) -> set[Tuple[str, str]]:
        out: set[Tuple[str, str]] = set()
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                out.add((u, v) if u < v else (v, u))
        return out

    def _assign_isolated_variables(
        self,
        embedding: Dict[str, Tuple[int, ...]],
        variables: Sequence[str],
    ) -> Dict[str, Tuple[int, ...]]:
        expected = [str(v) for v in variables]
        missing = [v for v in expected if v not in embedding]
        if not missing:
            return embedding
        used = {q for chain in embedding.values() for q in chain}
        available = [int(q) for q in self.target_graph.nodes if int(q) not in used]
        if len(available) < len(missing):
            raise RuntimeError(
                f"Not enough unused qubits to place {len(missing)} isolated variables on {self.access.topology}."
            )
        for name, qubit in zip(missing, available):
            embedding[name] = (int(qubit),)
        return embedding

    def get_embedding(
        self,
        variables: Sequence[str],
        couplers: Mapping[Tuple[str, str], float],
        biclique_partition: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
    ) -> Dict[str, Tuple[int, ...]]:
        key = self._graph_key(variables, couplers)
        if key in self.embedding_cache:
            return self.embedding_cache[key]

        source_edges = set(self._canonical_edge(str(u), str(v)) for (u, v) in couplers)
        nodes = [str(v) for v in variables]
        embedding: Dict[str, Tuple[int, ...]] = {}

        # Fast paths for connected clique/biclique cases.
        if source_edges:
            try:
                cache = self.busgraph_cache(self.target_graph)
                if biclique_partition is not None:
                    left = [str(v) for v in biclique_partition[0]]
                    right = [str(v) for v in biclique_partition[1]]
                    if set(nodes) == set(left) | set(right) and source_edges == self._complete_bipartite_edges(left, right):
                        raw = cache.find_biclique_embedding(left, right)
                        embedding = self._merge_embedding_object(raw)
                elif source_edges == self._complete_graph_edges(nodes):
                    raw = cache.find_clique_embedding(nodes)
                    embedding = self._merge_embedding_object(raw)
            except Exception:
                embedding = {}

        if not embedding and source_edges:
            raw = self.find_embedding(list(source_edges), list(self.target_graph.edges), random_seed=19)
            embedding = {str(k): tuple(int(q) for q in chain) for k, chain in raw.items()}

        # Handle disconnected graphs and pure-bias problems by placing any missing
        # logical variables on unused single qubits.
        embedding = self._assign_isolated_variables(embedding, nodes)

        expected = set(nodes)
        if not embedding or set(embedding) != expected:
            raise RuntimeError(
                f"Failed to embed a logical graph with {len(nodes)} variables on {self.access.topology}."
            )

        self.embedding_cache[key] = embedding
        return embedding

    def get_composite(
        self,
        variables: Sequence[str],
        couplers: Mapping[Tuple[str, str], float],
        biclique_partition: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
    ):
        key = self._graph_key(variables, couplers)
        if key not in self.composite_cache:
            embedding = self.get_embedding(variables, couplers, biclique_partition=biclique_partition)
            composite = self.FixedEmbeddingComposite(
                self.SpinReversalTransformComposite(self.base_sampler),
                embedding,
            )
            self.composite_cache[key] = composite
        return self.composite_cache[key]

    def physical_qubit_count(
        self,
        variables: Sequence[str],
        couplers: Mapping[Tuple[str, str], float],
        biclique_partition: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
    ) -> int:
        embedding = self.get_embedding(variables, couplers, biclique_partition=biclique_partition)
        return int(sum(len(chain) for chain in embedding.values()))

    def embedding_stats(
        self,
        variables: Sequence[str],
        couplers: Mapping[Tuple[str, str], float],
        biclique_partition: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
    ) -> Dict[str, float]:
        embedding = self.get_embedding(variables, couplers, biclique_partition=biclique_partition)
        lengths = np.asarray([len(chain) for chain in embedding.values()], dtype=np.float64)
        return {
            "embedding_chain_count": float(lengths.size),
            "embedding_avg_chain_length": float(lengths.mean()) if lengths.size else 0.0,
            "embedding_max_chain_length": float(lengths.max()) if lengths.size else 0.0,
        }

    def metadata(self) -> Dict[str, object]:
        props = getattr(self.base_sampler, "properties", {})
        return {
            "solver_name": getattr(getattr(self.base_sampler, "solver", None), "name", None),
            "topology_requested": self.access.topology,
            "chip_id": props.get("chip_id"),
            "topology": props.get("topology"),
            "num_qubits": props.get("num_qubits"),
            "problem_run_duration_range": props.get("problem_run_duration_range"),
            "num_reads_range": props.get("num_reads_range"),
            "annealing_time_range": props.get("annealing_time_range"),
        }

    def _estimate_runtime_us(self, n_physical: int, num_reads: int, annealing_time: float) -> float:
        kwargs: Dict[str, object] = {"num_reads": int(num_reads), "annealing_time": float(annealing_time)}
        if self.access.programming_thermalization is not None:
            kwargs["programming_thermalization"] = float(self.access.programming_thermalization)
        if self.access.readout_thermalization is not None:
            kwargs["readout_thermalization"] = float(self.access.readout_thermalization)
        return float(self.base_sampler.solver.estimate_qpu_access_time(n_physical, **kwargs))

    def safe_child_reads(self, n_physical: int, annealing_time: float) -> Tuple[int, float]:
        key = (int(n_physical), float(annealing_time))
        if key in self.runtime_cache:
            return self.runtime_cache[key]

        props = getattr(self.base_sampler, "properties", {})
        runtime_limit = float(props.get("problem_run_duration_range", [0.0, 1_000_000.0])[1])
        max_reads = int(props.get("num_reads_range", [1, 10_000])[1])
        target_limit = self.access.max_runtime_fraction * runtime_limit

        est_one = self._estimate_runtime_us(n_physical, 1, annealing_time)
        if est_one > target_limit:
            raise RuntimeError(
                "Even one read exceeds the chosen safe runtime budget. "
                f"estimated={est_one:.1f} us threshold={target_limit:.1f} us"
            )

        lo, hi = 1, max_reads
        while lo < hi:
            mid = (lo + hi + 1) // 2
            est = self._estimate_runtime_us(n_physical, mid, annealing_time)
            if est <= target_limit:
                lo = mid
            else:
                hi = mid - 1

        safe_reads = int(lo)
        safe_est = float(self._estimate_runtime_us(n_physical, safe_reads, annealing_time))
        self.runtime_cache[key] = (safe_reads, safe_est)
        return self.runtime_cache[key]

    @staticmethod
    def _heuristic_chain_strength(h: Mapping[str, float], J: Mapping[Tuple[str, str], float], prefactor: float) -> float:
        max_linear = max((abs(float(v)) for v in h.values()), default=0.0)
        max_quadratic = max((abs(float(v)) for v in J.values()), default=0.0)
        scale = max(1.0e-3, max_linear, max_quadratic)
        return float(prefactor * scale)

    @staticmethod
    def _expand_sampleset(sampleset, variables: Sequence[str]) -> Array:
        samples = np.asarray(sampleset.record.sample, dtype=np.int8)
        counts = np.asarray(sampleset.record.num_occurrences, dtype=np.int64)
        var_index = {str(v): i for i, v in enumerate(sampleset.variables)}
        cols = [var_index[str(v)] for v in variables]
        ordered = samples[:, cols]
        return np.repeat(ordered, counts, axis=0).astype(np.int8, copy=False)

    def sample_ising(
        self,
        h_score: Mapping[str, float],
        J_score: Mapping[Tuple[str, str], float],
        target_samples: int,
        annealing_time: float,
        beta_scale: float = 1.0,
        requested_reads_per_child_call: Optional[int] = None,
        biclique_partition: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
    ) -> Tuple[Array, List[str], Dict[str, object]]:
        if target_samples < 1:
            raise ValueError("target_samples must be positive")
        if beta_scale <= 0:
            raise ValueError("beta_scale must be positive")

        variables = sorted({str(k) for k in h_score} | {str(u) for uv in J_score for u in uv})
        if not variables:
            raise ValueError("The logical problem has no variables")

        h_ising = {str(v): -float(h_score.get(v, 0.0)) / beta_scale for v in variables}
        J_ising = {
            self._canonical_edge(str(u), str(v)): -float(val) / beta_scale
            for (u, v), val in J_score.items()
        }

        composite = self.get_composite(variables, J_ising, biclique_partition=biclique_partition)
        bqm = self.dimod.BinaryQuadraticModel.from_ising(h_ising, J_ising)
        chain_strength = self._heuristic_chain_strength(h_ising, J_ising, self.access.chain_strength_prefactor)

        n_physical = self.physical_qubit_count(variables, J_ising, biclique_partition=biclique_partition)
        safe_reads, safe_est = self.safe_child_reads(n_physical, annealing_time)
        child_reads = safe_reads if requested_reads_per_child_call is None else min(safe_reads, int(requested_reads_per_child_call))
        child_reads = max(1, int(child_reads))

        srt = max(1, int(self.access.num_spin_reversal_transforms))
        effective_per_batch = child_reads * srt
        n_batches = max(1, math.ceil(int(target_samples) / effective_per_batch))

        arrays: List[Array] = []
        qpu_access_times: List[float] = []
        chain_break_fracs: List[float] = []

        for _ in range(n_batches):
            kwargs: Dict[str, object] = {
                "chain_strength": chain_strength,
                "num_reads": child_reads,
                "num_spin_reversal_transforms": srt,
                "annealing_time": float(annealing_time),
                "auto_scale": True,
            }
            if self.access.programming_thermalization is not None:
                kwargs["programming_thermalization"] = float(self.access.programming_thermalization)
            if self.access.readout_thermalization is not None:
                kwargs["readout_thermalization"] = float(self.access.readout_thermalization)

            sampleset = composite.sample(bqm, **kwargs)
            arrays.append(self._expand_sampleset(sampleset, variables))

            timing = {}
            if isinstance(getattr(sampleset, "info", None), Mapping):
                timing = dict(sampleset.info.get("timing", {}))
            if "qpu_access_time" in timing:
                qpu_access_times.append(float(timing["qpu_access_time"]))

            dtype_names = set(sampleset.record.dtype.names)
            if "chain_break_fraction" in dtype_names:
                c = np.asarray(sampleset.record.chain_break_fraction, dtype=np.float64)
                occ = np.asarray(sampleset.record.num_occurrences, dtype=np.int64)
                c_expanded = np.repeat(c, occ)
                if c_expanded.size:
                    chain_break_fracs.append(float(np.mean(c_expanded)))

        stacked = np.concatenate(arrays, axis=0)[: int(target_samples)]
        props = getattr(self.base_sampler, "properties", {})
        meta = {
            "physical_qubits": int(n_physical),
            "chain_strength": float(chain_strength),
            "safe_reads_per_child_call": int(safe_reads),
            "reads_per_child_call": int(child_reads),
            "num_spin_reversal_transforms": int(srt),
            "requested_target_samples": int(target_samples),
            "effective_samples_per_batch": int(effective_per_batch),
            "n_batches": int(n_batches),
            "estimated_runtime_us_per_child": float(safe_est),
            "runtime_limit_us": float(props.get("problem_run_duration_range", [0.0, 1_000_000.0])[1]),
            "avg_qpu_access_time_us": float(np.mean(qpu_access_times)) if qpu_access_times else 0.0,
            "avg_chain_break_fraction": float(np.mean(chain_break_fracs)) if chain_break_fracs else 0.0,
        }
        meta.update(self.embedding_stats(variables, J_ising, biclique_partition=biclique_partition))
        return stacked, variables, meta

    def close(self) -> None:
        sampler = self.base_sampler
        try:
            if hasattr(sampler, "close"):
                sampler.close()  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        try:
            client = getattr(sampler, "client", None)
            if client is not None and hasattr(client, "close"):
                client.close()  # type: ignore[attr-defined]
        except Exception:
            pass

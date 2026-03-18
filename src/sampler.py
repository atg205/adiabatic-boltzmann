import numpy as np
from abc import ABC, abstractmethod
from model import RBM
import dimod
import neal
from dwave.samplers import TabuSampler
from veloxq_sdk import VeloxQSolver
from veloxq_sdk.config import load_config, VeloxQAPIConfig
from pathlib import Path
import json


class Sampler(ABC):
    """Abstract sampling interface."""

    def rbm_to_ising(self, rbm):
        """
        Convert RBM parameters to Ising model parameters (J, h).
        Args:
            rbm (RBM): An RBM instance
        """
        Nv = rbm.n_visible
        Nh = rbm.n_hidden

        linear = {}
        quadratic = {}

        # visible biases
        for i in range(Nv):
            linear[i] = -rbm.a[i]

        # hidden biases
        for j in range(Nh):
            linear[Nv + j] = -rbm.b[j]

        # RBM couplings
        for i in range(Nv):
            for j in range(Nh):
                quadratic[(i, Nv + j)] = -rbm.W[i, j]

        return quadratic, linear

    @abstractmethod
    def sample(self, rbm, n_samples: int, config: dict = None) -> np.ndarray:
        """
        Generate samples from the RBM distribution.

        rbm: the RBM instance (has log_psi, psi_ratio methods)
        n_samples: how many samples to draw
        config: optional configuration dict

        Returns: (n_samples, n_visible) array of spin configurations
        """
        pass


class ClassicalSampler(Sampler):
    """
    Classical sampling via Metropolis-Hastings or Simulated Annealing.
    """

    def __init__(self, method: str, n_warmup: int = 200, n_sweeps: int = 1):
        """
        method:   'metropolis' | 'simulated_annealing' | 'gibbs'
        n_warmup: equilibration sweeps before collecting samples
        n_sweeps: full sweeps (n_visible flip attempts) between each sample
                  increase to 2-3 if acceptance rate drops below 0.2
        """
        self.method = method
        self.n_warmup = n_warmup
        self.n_sweeps = n_sweeps

    def sample(self, rbm: RBM, n_samples: int, config: dict = None) -> np.ndarray:
        if config is None:
            config = {}

        if self.method == "metropolis":
            return self._metropolis_hastings(rbm, n_samples, config)
        elif self.method == "simulated_annealing":
            return self._simulated_annealing(rbm, n_samples, config)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _metropolis_hastings(
        self, rbm: RBM, n_samples: int, config: dict
    ) -> np.ndarray:
        """
        Metropolis-Hastings sampling targeting |Ψ(v)|².

        Proposal: flip a single random spin.
        Acceptance: min(1, |Ψ(v')/Ψ(v)|²)

        One sweep = n_visible attempted flips at randomly chosen sites.

        Parameters from config:
        - n_warmup: equilibration sweeps (overrides __init__ value)
        - n_sweeps: sweeps between collected samples (overrides __init__ value)
        """
        N = rbm.n_visible
        n_warmup = config.get("n_warmup", self.n_warmup)
        n_sweeps = config.get("n_sweeps", self.n_sweeps)
        rng = np.random.default_rng()

        v = rng.choice([-1.0, 1.0], size=N)

        n_accepted = 0
        n_proposed = 0

        def sweep(v):
            nonlocal n_accepted, n_proposed
            for flip_idx in rng.integers(0, N, size=N):
                ratio_sq = rbm.psi_ratio(v, flip_idx) ** 2
                n_proposed += 1
                if rng.random() < min(1.0, ratio_sq):
                    v[flip_idx] *= -1
                    n_accepted += 1
            return v

        # Warmup — equilibrate from random initial state
        for _ in range(n_warmup):
            sweep(v)

        # Reset counters so acceptance rate reflects collection phase only
        n_accepted = 0
        n_proposed = 0

        # Collect samples
        samples = []
        for _ in range(n_samples):
            for _ in range(n_sweeps):
                sweep(v)
            samples.append(v.copy())

        acceptance_rate = n_accepted / max(n_proposed, 1)
        print(
            f"  [MH]    acceptance={acceptance_rate:.3f}  "
            f"unique={len(set(map(tuple, samples)))}/{n_samples}"
        )

        return np.array(samples)

    def _simulated_annealing(self, rbm, n_samples: int, config: dict) -> np.ndarray:
        """
        Simulated Annealing: gradually lower temperature as you sample.

        Parameters from config:
        - T_initial: starting temperature (default ???)
        - T_final: final temperature (default ???)
        - n_steps: total annealing steps (default ???)
        """

        T_initial = config.get("T_initial", 10)
        T_final = config.get("T_final", 0.05)
        n_steps = config.get("n_steps", int(1e5))
        print("simulated annealing")
        n_visible = rbm.n_visible
        v = (2 * np.random.randint(0, 2, n_visible) - 1).astype(float)
        samples = []

        # Create temperature schedule
        def schedule(step: int) -> float:
            return T_initial * (T_final / T_initial) ** (step / n_steps)  # Geometric

        # Equilibrium
        spin_flip_array = np.random.randint(0, len(v), n_steps)
        for step, spin_flip_idx in enumerate(spin_flip_array):
            T = schedule(step)
            ratio_squared = rbm.psi_ratio(v, spin_flip_idx) ** 2
            if np.random.random() < min(1, ratio_squared ** (1 / T)):
                v[spin_flip_idx] *= -1

            if step % (n_steps // n_samples) == 0:
                samples.append(np.copy(v))

        return np.array(samples)


class VeloxSampler(Sampler):
    def __init__(self, method: str):
        self.method = method
        self.solver = VeloxQSolver()

        load_config("velox_api_config.py")
        api_config = VeloxQAPIConfig.instance()

        with open("velox_token.txt", "r") as file:
            api_config.token = file.read().strip()

    def sample(self, rbm, n_samples: int, config: dict = {}) -> np.ndarray:
        self.n_visible = rbm.n_visible
        J, h = self.rbm_to_ising(rbm)
        self.solver.parameters.num_rep = n_samples
        sampleset = self.solver.sample(h, J)

        df = sampleset.to_pandas_dataframe()
        df = df.loc[df.index.repeat(df["num_occurrences"])].reset_index(
            drop=True
        )  # expand
        # return visible only
        return df.loc[:, list(range(self.n_visible))].to_numpy()


class DimodSampler(Sampler):
    def __init__(self, method: str):
        self.method = method
        self.time_path = Path("time.json")
        if not self.time_path.exists():
            with self.time_path.open("w") as f:
                json.dump({"time_ms": 0}, f)

    def sample(self, rbm, n_samples: int, config: dict = {}) -> np.ndarray:
        """
        Sample from the RBM distribution using a classical/quantum sampler from the dimod library.
        Args:
            - rbm (RBM): An RBM instance
            - n_samples (int): Number of samples to draw
            - config (dict): Optional configuration for the sampler
        """
        J, h = self.rbm_to_ising(rbm)
        self.n_visible = rbm.n_visible
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, 0.0)

        if self.method == "simulated_annealing":
            return self.simulated_annealing(bqm, n_samples, config)
        elif self.method == "tabu":
            return self.tabu_search(bqm, n_samples, config)
        elif self.method == "pegasus":
            config["solver"] = "Advantage_system6.4"
            return self.dwave(bqm, n_samples, config)
        elif self.method == "zephyr":
            config["solver"] = "Advantage2_system4.3"
            return self.dwave(bqm, n_samples, config)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def simulated_annealing(self, bqm, n_samples: int, config: dict = {}) -> np.ndarray:
        """
        Run simulated annealing using the neal library.

        Args:
            - bqm (dimod.BinaryQuadraticModel): The Ising model to sample from
            - n_samples (int): Number of samples to draw
            - config (dict): Optional configuration for the annealing schedule
        """
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(
            bqm,
            num_reads=n_samples,
            beta_range=(0.01, 10.0),  # wider temperature range
            num_sweeps=1000,  # more sweeps per read
            beta_schedule_type="geometric",
        )
        samples = sampleset.record.sample
        unique_samples = len(set(map(tuple, samples)))
        print(f"  unique samples: {unique_samples}/{len(samples)}")
        # return visible spins only
        return samples[:, : self.n_visible]

    def tabu_search(self, bqm, n_samples: int, config: dict = {}) -> np.ndarray:
        """
        Run tabu search using the neal library.

        Args:
            - bqm (dimod.BinaryQuadraticModel): The Ising model to sample from
            - n_samples (int): Number of samples to draw
            - config (dict): Optional configuration for the tabu search
        """
        sampler = TabuSampler()
        sampleset = sampler.sample(bqm, num_reads=n_samples)

        samples = sampleset.record.sample

        # return visible spins only
        return samples[:, : self.n_visible]

    def dwave(self, bqm, n_samples: int, config: dict = {}) -> np.ndarray:
        """
        Sample using a real D-Wave QPU via dwave-system.

        Requires:
            pip install dwave-system
            dwave config create   # set up API token

        Config keys:
        - solver:          D-Wave solver name, e.g. "Advantage_system6.4"
                           defaults to the best available solver
        - annealing_time:  annealing time in µs (default 20)
        - num_reads:       overrides n_samples if set
        - chain_strength:  embedding chain strength (default auto)
        """
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
        except ImportError:
            raise ImportError(
                "dwave-system is required for D-Wave QPU sampling. "
                "Install with: pip install dwave-system"
            )

        solver_name = config.get("solver", None)
        annealing_time = config.get("annealing_time", 20)
        num_reads = config.get("num_reads", n_samples)
        chain_strength = config.get("chain_strength", None)

        dwave_sampler = DWaveSampler(solver=solver_name)
        sampler = EmbeddingComposite(dwave_sampler)

        sample_kwargs = dict(
            num_reads=num_reads,
            annealing_time=annealing_time,
        )
        if chain_strength is not None:
            sample_kwargs["chain_strength"] = chain_strength

        sampleset = sampler.sample(bqm, **sample_kwargs)

        samples = self._expand_sampleset(sampleset)

        # D-Wave may return fewer samples than requested if chains break —
        # pad by resampling with replacement if needed
        if len(samples) < n_samples:
            print(
                f"  [DWave] got {len(samples)}/{n_samples} samples "
                f"(chain breaks), padding by resampling"
            )
            idx = np.random.choice(len(samples), size=n_samples, replace=True)
            samples = samples[idx]

        return samples[:n_samples]

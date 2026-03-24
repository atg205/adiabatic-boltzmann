import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ising import TransverseFieldIsing2D
from model import FullyConnectedRBM


def make_rbm(N: int, seed: int) -> FullyConnectedRBM:
    rng = np.random.default_rng(seed)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 0.01, N)
    rbm.b = rng.normal(0, 0.01, N)
    rbm.W = rng.normal(0, 0.01, (N, N))
    return rbm


@pytest.mark.parametrize(
    "L,h",
    [
        (2, 0.5),
        (2, 1.0),
        (2, 2.0),
        (3, 0.5),
        (3, 1.0),
        (4, 0.5),
        (4, 1.0),
        (4, 2.0),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_local_energy_batch_matches_scalar(L: int, h: float, seed: int):
    """
    local_energy_batch must return the same values as local_energy
    for every configuration in the batch.
    """
    N = L * L
    rng = np.random.default_rng(seed)
    rbm = make_rbm(N, seed)
    ising = TransverseFieldIsing2D(L, h)

    n_configs = 10
    configs = rng.choice([-1.0, 1.0], size=(n_configs, N))

    # Scalar reference
    E_scalar = np.array([ising.local_energy(v.copy(), rbm.psi_ratio) for v in configs])

    # Batched
    E_batch = ising.local_energy_batch(configs.copy(), rbm)

    assert E_scalar.shape == E_batch.shape, (
        f"Shape mismatch: scalar={E_scalar.shape}, batch={E_batch.shape}"
    )

    for i, (es, eb) in enumerate(zip(E_scalar, E_batch)):
        assert np.isclose(es, eb, atol=1e-8), (
            f"L={L}, h={h}, seed={seed}, config {i}:\n"
            f"  scalar = {es:.10f}\n"
            f"  batch  = {eb:.10f}\n"
            f"  diff   = {abs(es - eb):.2e}"
        )


@pytest.mark.parametrize("L,h", [(2, 1.0), (3, 0.5), (4, 1.0)])
def test_local_energy_batch_all_configs(L: int, h: float):
    """
    Exhaustive check over all 2^N configurations for small L.
    Ensures no edge case is missed by random sampling.
    """
    import itertools

    N = L * L
    rbm = make_rbm(N, seed=0)
    ising = TransverseFieldIsing2D(L, h)

    all_configs = np.array(
        list(itertools.product([-1.0, 1.0], repeat=N)),
        dtype=float,
    )

    E_scalar = np.array(
        [ising.local_energy(v.copy(), rbm.psi_ratio) for v in all_configs]
    )
    E_batch = ising.local_energy_batch(all_configs.copy(), rbm)

    max_diff = np.max(np.abs(E_scalar - E_batch))
    assert max_diff < 1e-8, (
        f"L={L}, h={h}: max diff over all {len(all_configs)} configs = {max_diff:.2e}"
    )


@pytest.mark.parametrize("L,h", [(2, 0.5), (3, 1.0), (4, 2.0)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_local_energy_batch_does_not_mutate_input(L: int, h: float, seed: int):
    """
    local_energy_batch must not modify the input V array in place.
    """
    N = L * L
    rng = np.random.default_rng(seed)
    rbm = make_rbm(N, seed)
    ising = TransverseFieldIsing2D(L, h)

    V = rng.choice([-1.0, 1.0], size=(10, N))
    V_before = V.copy()

    ising.local_energy_batch(V, rbm)

    assert np.array_equal(V, V_before), "local_energy_batch mutated the input V array"


@pytest.mark.parametrize("L", [2, 3, 4])
def test_diagonal_term_only(L: int):
    """
    With h=0, off-diagonal term vanishes and E_loc = -Σ_{bonds} v_i v_j.
    Verify batch diagonal matches scalar diagonal exactly.
    """
    N = L * L
    rng = np.random.default_rng(0)
    rbm = make_rbm(N, seed=0)
    ising = TransverseFieldIsing2D(L, h=0.0)

    configs = rng.choice([-1.0, 1.0], size=(20, N))

    E_scalar = np.array([ising.local_energy(v.copy(), rbm.psi_ratio) for v in configs])
    E_batch = ising.local_energy_batch(configs.copy(), rbm)

    assert np.allclose(E_scalar, E_batch, atol=1e-10), (
        f"Diagonal-only mismatch for L={L}: "
        f"max diff={np.max(np.abs(E_scalar - E_batch)):.2e}"
    )


@pytest.mark.parametrize("L", [2, 3, 4])
def test_off_diagonal_term_only(L: int):
    """
    With J=0 (zero weights and biases), diagonal term vanishes and
    all psi_ratios = 1, so E_loc = -h * N exactly.
    Verify both scalar and batch return this.
    """
    N = L * L
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = np.zeros(N)
    rbm.b = np.zeros(N)
    rbm.W = np.zeros((N, N))
    h = 1.5
    ising = TransverseFieldIsing2D(L, h=h)

    rng = np.random.default_rng(0)
    configs = rng.choice([-1.0, 1.0], size=(20, N))

    expected = -h * N  # psi_ratio = 1 for all flips, diagonal = 0

    E_scalar = np.array([ising.local_energy(v.copy(), rbm.psi_ratio) for v in configs])
    E_batch = ising.local_energy_batch(configs.copy(), rbm)

    assert np.allclose(E_scalar, expected, atol=1e-10), (
        f"Scalar off-diagonal mismatch: expected {expected}, got {E_scalar}"
    )
    assert np.allclose(E_batch, expected, atol=1e-10), (
        f"Batch off-diagonal mismatch: expected {expected}, got {E_batch}"
    )

"""
End-to-end integration test for N=4 VMC.

For N=4 we can enumerate all 2^4=16 configurations exactly and compute:
  - exact <E> under the RBM distribution
  - exact <E> via the VMC estimator formula

If these disagree, the bug is in local_energy or psi_ratio.
If they agree but are below E_exact, the bug is in exact_ground_energy.
If the training loop computes a different value than both, the bug is in
the training loop (SR matrices, update step, sampler encoding).
"""

import itertools
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ising import TransverseFieldIsing1D
from model import FullyConnectedRBM


# ---------------------------------------------------------------------------
# Exact reference computations (no RBM, pure matrix algebra)
# ---------------------------------------------------------------------------

def all_configs(N: int) -> np.ndarray:
    """All 2^N spin configurations in {-1, +1}. Shape: (2^N, N)."""
    return np.array(list(itertools.product([-1, 1], repeat=N)), dtype=float)


def exact_hamiltonian_matrix(N: int, h: float) -> np.ndarray:
    """
    Build the full 2^N x 2^N Hamiltonian matrix explicitly.
    H = -sum_i sigma^z_i sigma^z_{i+1} - h * sum_i sigma^x_i
    In the sigma^z basis:
      - diagonal: -sum_i v_i * v_{i+1}  (Ising term)
      - off-diagonal: -h for each single spin flip (transverse field)
    """
    configs = all_configs(N)
    n_states = 2 ** N
    H = np.zeros((n_states, n_states))

    # Map config tuple -> index
    config_to_idx = {tuple(c): i for i, c in enumerate(configs)}

    for idx, v in enumerate(configs):
        # Diagonal: Ising bonds (periodic BC)
        H[idx, idx] = -sum(v[i] * v[(i + 1) % N] for i in range(N))

        # Off-diagonal: transverse field — flip each spin
        for flip_i in range(N):
            v_flip = v.copy()
            v_flip[flip_i] *= -1
            jdx = config_to_idx[tuple(v_flip)]
            H[idx, jdx] += -h

    return H


def exact_energy_expectation(rbm: FullyConnectedRBM, H: np.ndarray, N: int) -> float:
    """
    Compute <Psi|H|Psi> / <Psi|Psi> by direct matrix-vector multiplication.
    This is the ground truth — no sampling, no VMC estimator.
    """
    configs = all_configs(N)

    # Wave function amplitudes Psi(v) for all configs
    psi = np.array([np.exp(rbm.log_psi(v)) for v in configs])  # (2^N,)

    # <Psi|H|Psi> / <Psi|Psi>
    numerator   = psi @ H @ psi
    denominator = psi @ psi
    return numerator / denominator


def vmc_energy_exact_samples(rbm: FullyConnectedRBM,
                              ising: TransverseFieldIsing1D,
                              N: int) -> float:
    """
    Compute VMC energy estimator using ALL 2^N configs weighted by |Psi(v)|^2.
    E_vmc = sum_v |Psi(v)|^2 * E_loc(v) / sum_v |Psi(v)|^2

    This is what the training loop approximates via sampling.
    If this disagrees with exact_energy_expectation, the bug is in local_energy.
    """
    configs = all_configs(N)
    psi_sq  = np.array([np.exp(2 * rbm.log_psi(v)) for v in configs])
    e_loc   = np.array([ising.local_energy(v.copy(), rbm.psi_ratio) for v in configs])

    return np.sum(psi_sq * e_loc) / np.sum(psi_sq)


def vmc_energy_from_samples(rbm: FullyConnectedRBM,
                             ising: TransverseFieldIsing1D,
                             samples: np.ndarray) -> float:
    """
    Compute VMC energy the same way the training loop does:
    simple mean of local energies over samples.
    """
    return np.mean([
        ising.local_energy(v.copy(), rbm.psi_ratio)
        for v in samples
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_local_energy_consistent_with_hamiltonian(h: float, seed: int):
    """
    VMC energy via local_energy must equal <Psi|H|Psi>/<Psi|Psi> computed
    directly from the Hamiltonian matrix.

    This is the most fundamental check — if this fails, local_energy is wrong.
    If this passes, the bug is in sampling or the training loop.
    """
    N   = 4
    rng = np.random.default_rng(seed)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 0.1, N)
    rbm.b = rng.normal(0, 0.1, N)
    rbm.W = rng.normal(0, 0.1, (N, N))

    ising = TransverseFieldIsing1D(N, h)
    H     = exact_hamiltonian_matrix(N, h)

    E_matrix = exact_energy_expectation(rbm, H, N)
    E_vmc    = vmc_energy_exact_samples(rbm, ising, N)

    assert np.isclose(E_matrix, E_vmc, atol=1e-8), (
        f"h={h}, seed={seed}\n"
        f"  Matrix <E>  = {E_matrix:.10f}\n"
        f"  VMC <E>     = {E_vmc:.10f}\n"
        f"  diff        = {abs(E_matrix - E_vmc):.2e}\n"
        f"  → local_energy or psi_ratio is wrong"
    )


@pytest.mark.parametrize("h", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_variational_bound(h: float, seed: int):
    """
    For any RBM state, <E> >= E_exact (variational principle).
    If this fails, either local_energy is wrong or exact_ground_energy is wrong.
    """
    N   = 4
    rng = np.random.default_rng(seed)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 0.1, N)
    rbm.b = rng.normal(0, 0.1, N)
    rbm.W = rng.normal(0, 0.1, (N, N))

    ising   = TransverseFieldIsing1D(N, h)
    H       = exact_hamiltonian_matrix(N, h)
    E_exact = np.linalg.eigvalsh(H)[0]          # exact diagonalization, not integral

    E_vmc   = vmc_energy_exact_samples(rbm, ising, N)

    assert E_vmc >= E_exact - 1e-8, (
        f"Variational bound violated! h={h}, seed={seed}\n"
        f"  E_vmc   = {E_vmc:.10f}\n"
        f"  E_exact = {E_exact:.10f}  (exact diag)\n"
        f"  diff    = {E_vmc - E_exact:.2e}\n"
        f"  → fundamental bug in local_energy or psi_ratio"
    )


@pytest.mark.parametrize("h", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sample_mean_converges_to_exact_vmc(h: float, seed: int):
    """
    The sample mean of local energies (as computed in training loop)
    must converge to the exact VMC energy as n_samples -> infinity.

    Uses importance-weighted sampling from |Psi|^2 directly.
    If this fails with exact samples, the training loop mean is biased.
    """
    N   = 4
    rng = np.random.default_rng(seed)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 0.1, N)
    rbm.b = rng.normal(0, 0.1, N)
    rbm.W = rng.normal(0, 0.1, (N, N))

    ising   = TransverseFieldIsing1D(N, h)
    configs = all_configs(N)

    # Sample from exact |Psi(v)|^2 distribution
    psi_sq = np.array([np.exp(2 * rbm.log_psi(v)) for v in configs])
    probs  = psi_sq / psi_sq.sum()

    n_samples = 50000
    indices   = rng.choice(len(configs), size=n_samples, p=probs)
    samples   = configs[indices]

    E_sampled = vmc_energy_from_samples(rbm, ising, samples)
    E_exact_vmc = vmc_energy_exact_samples(rbm, ising, N)

    assert np.isclose(E_sampled, E_exact_vmc, atol=0.05), (
        f"Sample mean doesn't converge to exact VMC energy\n"
        f"  h={h}, seed={seed}, n_samples={n_samples}\n"
        f"  E_sampled    = {E_sampled:.6f}\n"
        f"  E_exact_vmc  = {E_exact_vmc:.6f}\n"
        f"  diff         = {abs(E_sampled - E_exact_vmc):.4f}\n"
        f"  → training loop mean is biased"
    )


@pytest.mark.parametrize("h", [0.5, 1.0, 2.0])
def test_psi_ratio_consistent_with_log_psi(h: float):
    """
    psi_ratio(v, i) must equal exp(log_psi(v_flip) - log_psi(v)).
    Tests consistency between the two implementations over all N=4 configs.
    """
    N   = 4
    rng = np.random.default_rng(0)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 0.1, N)
    rbm.b = rng.normal(0, 0.1, N)
    rbm.W = rng.normal(0, 0.1, (N, N))

    configs = all_configs(N)
    max_err = 0.0

    for v in configs:
        for flip_idx in range(N):
            v_flip = v.copy()
            v_flip[flip_idx] *= -1

            ratio_direct = np.exp(rbm.log_psi(v_flip) - rbm.log_psi(v))
            ratio_fast   = rbm.psi_ratio(v.copy(), flip_idx)

            err = abs(ratio_direct - ratio_fast)
            max_err = max(max_err, err)

            assert np.isclose(ratio_direct, ratio_fast, atol=1e-8), (
                f"psi_ratio mismatch at v={v}, flip={flip_idx}\n"
                f"  direct = {ratio_direct:.10f}\n"
                f"  fast   = {ratio_fast:.10f}\n"
                f"  diff   = {err:.2e}"
            )

    print(f"  max psi_ratio error over all configs: {max_err:.2e}")


@pytest.mark.parametrize("h", [0.5, 1.0, 2.0])
def test_e_loc_sum_equals_matrix_row(h: float):
    """
    For a single config v, E_loc(v) = sum_v' H(v,v') * Psi(v')/Psi(v).
    This tests local_energy directly against the Hamiltonian matrix row.
    """
    N   = 4
    rng = np.random.default_rng(0)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 0.1, N)
    rbm.b = rng.normal(0, 0.1, N)
    rbm.W = rng.normal(0, 0.1, (N, N))

    ising   = TransverseFieldIsing1D(N, h)
    H       = exact_hamiltonian_matrix(N, h)
    configs = all_configs(N)
    config_to_idx = {tuple(c): i for i, c in enumerate(configs)}

    psi = np.array([np.exp(rbm.log_psi(v)) for v in configs])

    for v in configs:
        idx   = config_to_idx[tuple(v)]
        # E_loc from matrix: sum_v' H[v, v'] * Psi(v') / Psi(v)
        E_matrix_row = np.sum(H[idx, :] * psi) / psi[idx]
        # E_loc from our implementation
        E_our = ising.local_energy(v.copy(), rbm.psi_ratio)

        assert np.isclose(E_matrix_row, E_our, atol=1e-8), (
            f"E_loc mismatch at v={v}, h={h}\n"
            f"  matrix row = {E_matrix_row:.10f}\n"
            f"  ours       = {E_our:.10f}\n"
            f"  diff       = {abs(E_matrix_row - E_our):.2e}"
        )


@pytest.mark.parametrize("h", [0.5, 1.0, 2.0])
def test_exact_ground_energy_matches_diagonalization(h: float):
    """
    exact_ground_energy() (integral formula) must match exact diagonalization
    for N=4. Checks whether the reference energy itself is correct.
    """
    N     = 4
    ising = TransverseFieldIsing1D(N, h)
    H     = exact_hamiltonian_matrix(N, h)

    E_integral = ising.exact_ground_energy()
    E_diag     = np.linalg.eigvalsh(H)[0]

    # Note: integral formula is thermodynamic limit, finite-size correction expected
    # but for N=4 should be within ~5%
    rel_diff = abs(E_integral - E_diag) / abs(E_diag)
    print(f"\n  h={h}: integral={E_integral:.6f}  diag={E_diag:.6f}  "
          f"rel_diff={rel_diff:.3%}")

    assert rel_diff < 0.10, (
        f"exact_ground_energy too far from diagonalization for N={h}\n"
        f"  integral = {E_integral:.6f}\n"
        f"  diag     = {E_diag:.6f}\n"
        f"  rel_diff = {rel_diff:.3%}"
    )


if __name__ == "__main__":
    # Run a quick diagnostic without pytest
    print("=" * 60)
    print("End-to-end VMC diagnostic for N=4")
    print("=" * 60)

    for h in [0.5, 1.0, 2.0]:
        print(f"\nh = {h}")
        N   = 4
        rng = np.random.default_rng(0)
        rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
        rbm.a = rng.normal(0, 0.1, N)
        rbm.b = rng.normal(0, 0.1, N)
        rbm.W = rng.normal(0, 0.1, (N, N))

        ising = TransverseFieldIsing1D(N, h)
        H     = exact_hamiltonian_matrix(N, h)

        E_exact_diag  = np.linalg.eigvalsh(H)[0]
        E_exact_integ = ising.exact_ground_energy()
        E_matrix      = exact_energy_expectation(rbm, H, N)
        E_vmc         = vmc_energy_exact_samples(rbm, ising, N)

        print(f"  E_exact (diag)     = {E_exact_diag:.8f}")
        print(f"  E_exact (integral) = {E_exact_integ:.8f}")
        print(f"  E_matrix <H>       = {E_matrix:.8f}  (direct matrix-vector)")
        print(f"  E_vmc (all configs)= {E_vmc:.8f}  (via local_energy)")
        print(f"  E_matrix == E_vmc? {np.isclose(E_matrix, E_vmc, atol=1e-8)}")
        print(f"  Bound satisfied?   {E_vmc >= E_exact_diag - 1e-8}")

        if not np.isclose(E_matrix, E_vmc, atol=1e-8):
            print(f"  *** BUG IN local_energy OR psi_ratio ***")
            print(f"  diff = {abs(E_matrix - E_vmc):.2e}")
        if E_vmc < E_exact_diag - 1e-8:
            print(f"  *** VARIATIONAL BOUND VIOLATED ***")
            print(f"  violation = {E_exact_diag - E_vmc:.6f}")

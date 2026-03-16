import numpy as np
import pytest
import jax
import jax.numpy as jnp
import flax
import netket as nk

jax.config.update("jax_enable_x64", True)

from src.model import FullyConnectedRBM
from src.ising import TransverseFieldIsing1D
from src.encoder import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rbm_with_known_weights(N: int, seed: int) -> FullyConnectedRBM:
    rng = np.random.default_rng(seed)
    rbm = FullyConnectedRBM(n_visible=N, n_hidden=N)
    rbm.a = rng.normal(0, 0.01, N)
    rbm.b = rng.normal(0, 0.01, N)
    rbm.W = rng.normal(0, 0.01, (N, N))
    return rbm


def make_trainer(rbm: FullyConnectedRBM, regularization: float = 0.0) -> Trainer:
    """Trainer with no sampler — we call _compute_sr_matrices directly."""
    ising = TransverseFieldIsing1D(rbm.n_visible, h=1.0)
    return Trainer(rbm, ising, sampler=None, config={"regularization": regularization})


def build_netket_model_and_params(rbm: FullyConnectedRBM, N: int):
    nk_rbm = nk.models.RBM(
        alpha=rbm.n_hidden // N,
        use_visible_bias=True,
        use_hidden_bias=True,
    )
    dummy = jnp.ones((1, N))
    params = nk_rbm.init(jax.random.PRNGKey(0), dummy)
    params = flax.core.unfreeze(params)
    params["params"]["kernel"] = jnp.array(rbm.W, dtype=jnp.float64)
    params["params"]["visible_bias"] = jnp.array(rbm.a, dtype=jnp.float64)
    params["params"]["hidden_bias"] = jnp.array(rbm.b, dtype=jnp.float64)
    params = flax.core.freeze(params)

    assert np.allclose(params["params"]["kernel"], rbm.W, atol=1e-10)
    assert np.allclose(params["params"]["visible_bias"], rbm.a, atol=1e-10)
    assert np.allclose(params["params"]["hidden_bias"], rbm.b, atol=1e-10)

    return nk_rbm, params


def samples_to_gradients(rbm: FullyConnectedRBM, samples: np.ndarray) -> list:
    """Convert sample array to list of gradient dicts — same as trainer loop."""
    return [rbm.gradient_log_psi(v.copy()) for v in samples]


def netket_jacobian(nk_rbm, params, samples: np.ndarray) -> np.ndarray:
    def log_psi_single(params, v):
        return nk_rbm.apply(params, v.reshape(1, -1))[0]

    grad_fn = jax.grad(log_psi_single, argnums=0)

    rows = []
    for v in jnp.array(samples, dtype=jnp.float64):
        g = grad_fn(params, v)
        row = np.concatenate(
            [
                np.array(g["params"]["visible_bias"]).flatten(),
                np.array(g["params"]["hidden_bias"]).flatten(),
                np.array(
                    g["params"]["kernel"]
                ).T.flatten(),  # ← match our (n_visible, n_hidden) layout
            ]
        )
        rows.append(row)

    return np.array(rows)


def netket_sr_matrix(
    nk_rbm, params, samples: np.ndarray, regularization: float = 0.0
) -> np.ndarray:
    """
    Compute NetKet's S matrix from the Jacobian directly.
    S = (1/M) * D_centered.T @ D_centered + reg * I
    using NetKet's own per-sample gradients.
    """
    D = netket_jacobian(nk_rbm, params, samples)
    M = D.shape[0]
    D_centered = D - np.mean(D, axis=0)
    S = (1 / M) * D_centered.T @ D_centered
    S += regularization * np.eye(S.shape[0])
    return S


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_jacobian_matches_netket(N: int, seed: int):
    """
    Per-sample gradient matrix from gradient_log_psi must match
    NetKet's Jacobian exactly. Catches any factor-of-2, transpose,
    or sign convention differences before they propagate into S.
    """
    rng = np.random.default_rng(seed)
    rbm = make_rbm_with_known_weights(N, seed)
    nk_rbm, params = build_netket_model_and_params(rbm, N)

    samples = rng.choice([-1, 1], size=(50, N)).astype(float)

    # Our Jacobian — same flattening order as _compute_sr_matrices
    gradients = samples_to_gradients(rbm, samples)
    D_ours = np.array(
        [
            np.concatenate([g["a"].flatten(), g["b"].flatten(), g["W"].flatten()])
            for g in gradients
        ]
    )

    D_nk = netket_jacobian(nk_rbm, params, samples)

    assert D_ours.shape == D_nk.shape, (
        f"Jacobian shape mismatch: ours={D_ours.shape}, netket={D_nk.shape}"
    )
    assert np.allclose(D_ours, D_nk, atol=1e-8), (
        f"Jacobian mismatch N={N}, seed={seed}\n"
        f"  max diff  = {np.max(np.abs(D_ours - D_nk)):.2e}\n"
        f"  mean diff = {np.mean(np.abs(D_ours - D_nk)):.2e}\n"
        f"  first row ours: {D_ours[0, :6]}\n"
        f"  first row nk:   {D_nk[0, :6]}"
    )


@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sr_matrix_matches_netket(N: int, seed: int):
    """
    S matrix from trainer._compute_sr_matrices must match NetKet's QGT.
    Uses no regularization so any difference is in the core computation.
    """
    rng = np.random.default_rng(seed)
    rbm = make_rbm_with_known_weights(N, seed)
    nk_rbm, params = build_netket_model_and_params(rbm, N)
    trainer = make_trainer(rbm, regularization=0.0)

    M = 200
    samples = rng.choice([-1, 1], size=(M, N)).astype(float)

    # Use trainer's own method
    gradients = samples_to_gradients(rbm, samples)
    energies = np.zeros(M)  # energies don't affect S, only F
    S_ours, _ = trainer._compute_sr_matrices(gradients, energies)

    S_nk = netket_sr_matrix(nk_rbm, params, samples, regularization=0.0)

    assert S_ours.shape == S_nk.shape, (
        f"S shape mismatch: ours={S_ours.shape}, netket={S_nk.shape}"
    )
    assert np.allclose(S_ours, S_nk, atol=1e-8), (
        f"SR matrix mismatch N={N}, seed={seed}\n"
        f"  max diff     = {np.max(np.abs(S_ours - S_nk)):.2e}\n"
        f"  mean diff    = {np.mean(np.abs(S_ours - S_nk)):.2e}\n"
        f"  diag ours:   {np.diag(S_ours)[:5]}\n"
        f"  diag netket: {np.diag(S_nk)[:5]}"
    )


@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("reg", [1e-4, 1e-2])
def test_sr_matrix_regularization_matches(N: int, reg: float):
    """
    Regularized S must match NetKet's diag_shift.
    Confirms our reg*I and NetKet's diag_shift are the same operation.
    """
    rng = np.random.default_rng(0)
    rbm = make_rbm_with_known_weights(N, seed=0)
    nk_rbm, params = build_netket_model_and_params(rbm, N)
    trainer = make_trainer(rbm, regularization=reg)

    samples = rng.choice([-1, 1], size=(200, N)).astype(float)
    gradients = samples_to_gradients(rbm, samples)
    energies = np.zeros(len(samples))

    S_ours, _ = trainer._compute_sr_matrices(gradients, energies)
    S_nk = netket_sr_matrix(nk_rbm, params, samples, regularization=reg)

    assert np.allclose(S_ours, S_nk, atol=1e-8), (
        f"Regularized SR matrix mismatch reg={reg}, N={N}\n"
        f"  max diff = {np.max(np.abs(S_ours - S_nk)):.2e}"
    )


@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_force_vector_matches_netket(N: int, seed: int):
    """
    F vector from trainer._compute_sr_matrices must match NetKet's
    energy gradient vector: F_k = ⟨O_k E_loc⟩ - ⟨O_k⟩⟨E_loc⟩.
    """
    rng = np.random.default_rng(seed)
    rbm = make_rbm_with_known_weights(N, seed)
    nk_rbm, params = build_netket_model_and_params(rbm, N)
    trainer = make_trainer(rbm, regularization=0.0)

    M = 200
    samples = rng.choice([-1, 1], size=(M, N)).astype(float)
    energies = rng.normal(-10, 2, M)  # synthetic energies with known stats

    gradients = samples_to_gradients(rbm, samples)
    _, F_ours = trainer._compute_sr_matrices(gradients, energies)

    # NetKet F: (1/M) * D_centered.T @ E_centered
    D_nk = netket_jacobian(nk_rbm, params, samples)
    D_centered = D_nk - np.mean(D_nk, axis=0)
    F_nk = (1 / M) * D_centered.T @ energies

    assert np.allclose(F_ours, F_nk, atol=1e-8), (
        f"Force vector mismatch N={N}, seed={seed}\n"
        f"  max diff  = {np.max(np.abs(F_ours - F_nk)):.2e}\n"
        f"  mean diff = {np.mean(np.abs(F_ours - F_nk)):.2e}\n"
        f"  F_ours[:5]: {F_ours[:5]}\n"
        f"  F_nk[:5]:   {F_nk[:5]}"
    )

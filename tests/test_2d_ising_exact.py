"""
Pytest for 2D Ising model exact ground state energy calculation.

Verifies that the _exact_diag_2d() method matches NetKet's exact diagonalization
exactly across different system sizes and transverse field strengths.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ising import TransverseFieldIsing2D


def compute_exact_energy_netket(L, h):
    """
    Compute exact ground state energy using NetKet directly.
    
    L: linear lattice size (total N = L²)
    h: transverse field strength
    
    Returns: ground state energy (lowest eigenvalue)
    """
    try:
        import netket as nk
        from scipy.sparse.linalg import eigsh
        
        N = L * L
        hilbert = nk.hilbert.Spin(s=0.5, N=N)
        ha = nk.operator.LocalOperator(hilbert)
        
        # Build 2D square lattice Hamiltonian
        for i in range(N):
            # Right neighbor (periodic)
            if (i + 1) % L != 0:
                right = i + 1
            else:
                right = i + 1 - L
            
            # Down neighbor (periodic)
            down = (i + L) % N
            
            # Ising couplings (avoid double-counting by only going right and down)
            ha += -1.0 * nk.operator.spin.sigmaz(hilbert, i) @ nk.operator.spin.sigmaz(hilbert, right)
            ha += -1.0 * nk.operator.spin.sigmaz(hilbert, i) @ nk.operator.spin.sigmaz(hilbert, down)
            
            # Transverse field
            ha += -h * nk.operator.spin.sigmax(hilbert, i)
        
        # Exact diagonalization
        H_sparse = ha.to_sparse()
        vals, _ = eigsh(H_sparse, k=1, which="SA")
        
        return float(vals[0])
    
    except ImportError:
        pytest.skip("NetKet not installed")


class TestIsing2DExactEnergy:
    """Test that 2D Ising model exact energy matches NetKet exactly."""
    
    @pytest.mark.parametrize("L,h", [
        (2, 0.5),   # 4 spins
        (2, 1.0),   # 4 spins
        (2, 2.0),   # 4 spins
        (3, 0.5),   # 9 spins
        (3, 1.0),   # 9 spins
        (3, 2.0),   # 9 spins
        (4, 0.5),   # 16 spins
        (4, 1.0),   # 16 spins
        (4, 2.0),   # 16 spins
    ])
    def test_exact_diag_2d_matches_netket(self, L, h):
        """
        Verify that TransverseFieldIsing2D._exact_diag_2d() matches NetKet 
        exact diagonalization to machine precision.
        """
        # Reference: NetKet exact diagonalization
        E_netket = compute_exact_energy_netket(L, h)
        
        # Method under test: _exact_diag_2d()
        model = TransverseFieldIsing2D(L, h=h)
        E_model = model._exact_diag_2d()
        
        # Should match to machine precision (rtol=1e-10)
        assert np.isclose(E_model, E_netket, rtol=1e-10), \
            f"L={L}, h={h}: _exact_diag_2d()={E_model:.10f} vs NetKet={E_netket:.10f}, " \
            f"diff={abs(E_model - E_netket):.2e}, rel_error={abs(E_model - E_netket)/abs(E_netket):.2e}"
        
        print(f"  ✓ L={L}, h={h}: E={E_netket:.8f} (match exact)")
    
    @pytest.mark.parametrize("L,h", [
        (2, 0.5),
        (2, 1.0),
        (2, 2.0),
        (3, 0.5),
        (3, 1.0),
        (3, 2.0),
        (6, 0.5),
        (5, 1.0),
        (5, 2.0),
    ])
    def test_exact_ground_energy_matches_netket(self, L, h):
        """
        Verify that TransverseFieldIsing2D.exact_ground_energy() 
        (which calls _exact_diag_2d() for small L) matches NetKet.
        """
        E_netket = compute_exact_energy_netket(L, h)
        
        model = TransverseFieldIsing2D(L, h=h)
        E_exact = model.exact_ground_energy()
        
        # Should match to machine precision
        assert np.isclose(E_exact, E_netket, rtol=1e-10), \
            f"L={L}, h={h}: exact_ground_energy()={E_exact:.10f} vs NetKet={E_netket:.10f}, " \
            f"rel_error={abs(E_exact - E_netket)/abs(E_netket):.2e}"
        
        print(f"  ✓ L={L}, h={h}: E={E_netket:.8f} (exact_ground_energy match)")


class TestIsing2DLocalEnergy:
    """Test local energy batch computation for 2D Ising model."""
    
    def test_local_energy_batch_shape(self):
        """Test that local_energy_batch returns correct shape."""
        L = 2
        h = 1.0
        ns = 10  # Number of samples
        N = L * L
        
        model = TransverseFieldIsing2D(L, h=h)
        
        # Create mock RBM-like object
        class MockRBM:
            def __init__(self, n_visible):
                self.a = np.zeros(n_visible)
                self.b = np.zeros(10)
                self.W = np.random.randn(n_visible, 10)
            
            def logcosh(self, x):
                return np.log(np.cosh(x))
        
        rbm = MockRBM(N)
        V = np.random.choice([-1, 1], size=(ns, N))
        
        E_batch = model.local_energy_batch(V, rbm)
        
        assert E_batch.shape == (ns,), f"Expected shape ({ns},), got {E_batch.shape}"
        assert np.all(np.isfinite(E_batch)), "Local energies contain non-finite values"
        print(f"  ✓ Batch shape correct: {E_batch.shape}")
    
    def test_local_energy_batch_values_reasonable(self):
        """Test that local energies are in reasonable range."""
        L = 2
        h = 1.0
        ns = 20
        N = L * L
        
        model = TransverseFieldIsing2D(L, h=h)
        
        class MockRBM:
            def __init__(self, n_visible):
                self.a = np.random.randn(n_visible) * 0.1
                self.b = np.random.randn(10) * 0.1
                self.W = np.random.randn(n_visible, 10) * 0.1
            
            def logcosh(self, x):
                return np.log(np.cosh(x))
        
        rbm = MockRBM(N)
        V = np.random.choice([-1, 1], size=(ns, N))
        
        E_batch = model.local_energy_batch(V, rbm)
        
        # For a 2x2 system, local energies should be finite and not extreme
        assert np.all(np.isfinite(E_batch)), "Local energies contain inf/nan"
        assert len(E_batch) == ns, f"Expected {ns} energies, got {len(E_batch)}"
        print(f"  ✓ Batch values reasonable: min={E_batch.min():.4f}, max={E_batch.max():.4f}, mean={E_batch.mean():.4f}")


class TestIsing2DComparison:
    """Comprehensive comparison of exact energies across parameter space."""
    
    def test_comparison_all_sizes_and_fields(self):
        """
        Comprehensive verification: _exact_diag_2d() vs NetKet 
        across all tested sizes and field strengths.
        Shows exact matching across parameter space.
        """
        results = []
        
        for L in [5,6,7]:
            for h in [0.5, 1.0, 2.0]:
                E_netket = compute_exact_energy_netket(L, h)
                
                model = TransverseFieldIsing2D(L, h=h)
                E_exact_diag = model._exact_diag_2d()
                
                abs_error = abs(E_exact_diag - E_netket)
                rel_error = abs_error / abs(E_netket) if E_netket != 0 else 0
                
                results.append({
                    'L': L,
                    'h': h,
                    'E_netket': E_netket,
                    'E_exact_diag': E_exact_diag,
                    'abs_error': abs_error,
                    'rel_error': rel_error,
                })
        
        # Print comprehensive comparison table
        print("\n" + "="*90)
        print("2D Ising Model: Exact Diagonalization vs NetKet Comparison")
        print("="*90)
        print(f"{'L':>3} {'h':>5} {'E_NetKet':>14} {'E_ExactDiag':>14} {'Abs Error':>14} {'Rel Error':>12}")
        print("-"*90)
        
        for r in results:
            print(f"{r['L']:3d} {r['h']:5.1f} {r['E_netket']:14.8f} {r['E_exact_diag']:14.8f} "
                  f"{r['abs_error']:14.2e} {r['rel_error']:11.2e}")
        
        print("="*90)
        
        # Verify all match to machine precision
        for r in results:
            assert r['abs_error'] < 1e-10, \
                f"L={r['L']}, h={r['h']}: Absolute error {r['abs_error']:.2e} too large"
            assert r['rel_error'] < 1e-10, \
                f"L={r['L']}, h={r['h']}: Relative error {r['rel_error']:.2e} too large"
        
        print("✓ All energies match NetKet to machine precision")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

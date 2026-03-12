import numpy as np
import pytest
import netket as nk


def netket_ground_energy(N, h):
    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    ha = nk.operator.LocalOperator(hilbert)
    for i in range(N):
        ha += (
            -1.0
            * nk.operator.spin.sigmaz(hilbert, i)
            @ nk.operator.spin.sigmaz(hilbert, (i + 1) % N)
        )
        ha += -h * nk.operator.spin.sigmax(hilbert, i)
    H_sparse = ha.to_sparse()
    from scipy.sparse.linalg import eigsh

    vals, _ = eigsh(H_sparse, k=1, which="SA")
    return vals


print(netket_ground_energy(16, 0.5))

"""Host grids must fail clearly under a recorded CUDA policy (NumPy 1/2)."""
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import pytest
from ciderpress.gpaw.new_backend import CiderNewMGGAFunctional


def test_numpy_grid_under_cuda_policy(monkeypatch):
    from ciderpress import gpu
    monkeypatch.setattr(gpu, 'policy', lambda device: nullcontext())
    f = object.__new__(CiderNewMGGAFunctional)
    f._device = 0
    grid = SimpleNamespace(data=np.zeros((1, 2, 2, 2)))
    with pytest.raises(ValueError, match='recorded CUDA device'):
        f.calculate(grid, grid)
    f._device = None
    f._calculate = lambda *args: 'host'
    assert f.calculate(grid, grid) == 'host'

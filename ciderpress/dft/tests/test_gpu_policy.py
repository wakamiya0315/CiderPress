"""CPU simulations of evaluation ownership and bounded diagnostics."""
from types import SimpleNamespace

import numpy as np
import pytest

from ciderpress import gpu


@pytest.fixture
def fake_cuda(monkeypatch):
    class Device:
        current = 3

        def __init__(self, device=None):
            self.id = self.current if device is None else device

        def __enter__(self):
            self.previous = Device.current
            Device.current = self.id
            return self

        def __exit__(self, *args):
            Device.current = self.previous

        def use(self):
            Device.current = self.id

    cp = SimpleNamespace(__name__='cupy', cuda=SimpleNamespace(Device=Device))
    monkeypatch.setattr(gpu, '_real_cupy', lambda: cp)
    gpu.disable()
    yield cp
    gpu.disable()


def test_interleaved_functionals_and_exceptions(fake_cuda):
    from ciderpress.gpaw.new_backend import CiderNewMGGAFunctional

    def functional(device):
        obj = object.__new__(CiderNewMGGAFunctional)
        obj._device = device
        obj._calculate = lambda *a: (gpu.use_gpu('rbf'), fake_cuda.cuda.Device().id)
        obj._calculate_paw_correction = obj._calculate
        return obj

    a, b = functional(0), functional(None)
    assert a.calculate(None) == (True, 0)
    assert not gpu.is_enabled() and fake_cuda.cuda.Device().id == 3
    gpu.enable(2)
    assert b.calculate(None) == (False, 2)
    assert a.calculate(None) == (True, 0)
    assert gpu.is_enabled() and fake_cuda.cuda.Device().id == 2
    assert a.calculate_paw_correction(None, None) == (True, 0)

    def fail(*args):
        assert gpu.is_enabled() and fake_cuda.cuda.Device().id == 0
        assert b.calculate(None) == (False, 0)
        raise RuntimeError('evaluation failed')

    a._calculate = a._calculate_paw_correction = fail
    for call in (lambda: a.calculate(None), lambda: a.calculate_paw_correction(None, None)):
        with pytest.raises(RuntimeError, match='evaluation failed'):
            call()
        assert gpu.is_enabled() and fake_cuda.cuda.Device().id == 2


def test_failed_constructor_leaves_policy_untouched(fake_cuda, monkeypatch):
    from ciderpress.gpaw.new_backend import CiderNewMGGAFunctional
    import gpaw.gpu
    monkeypatch.setattr(gpaw.gpu, 'cupy', fake_cuda)
    grid = SimpleNamespace(comm=SimpleNamespace(size=2))
    for enabled in (False, True):
        if enabled:
            gpu.enable(1)
        with pytest.raises(NotImplementedError, match='domain parallelism'):
            CiderNewMGGAFunctional(None, grid, xp=fake_cuda)
        assert gpu.is_enabled() is enabled
        with pytest.raises(NotImplementedError):
            CiderNewMGGAFunctional(None, grid, xp=np)
        assert gpu.is_enabled() is enabled


def test_transfer_history_is_bounded():
    from ciderpress.gpaw.new_backend import CiderNewMGGAFunctional
    obj = object.__new__(CiderNewMGGAFunctional)
    obj.transfer_timings = []
    array = SimpleNamespace(xp=np, data=np.zeros(2))
    for i in range(1000):
        assert obj._transfer(array, np, str(i)) is array
    assert len(obj.transfer_timings) == 256
    assert obj.transfer_timings[0]['operation'] == '744'

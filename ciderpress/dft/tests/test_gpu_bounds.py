"""CPU tests for CUDA map validation, cache limits and index arithmetic."""
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest


def test_descriptor_indices_and_cache(monkeypatch):
    from ciderpress.dft.transform_data import FeatureList, SLXMap, SLNMap, VMap
    from ciderpress.gpu import descriptors as d
    features = FeatureList([SLXMap(-1, -6, 1.), SLNMap(-1, 1.), VMap(-1, 1.)])
    rows = d.signature(features, 6)
    assert rows[0][1:3] == (5, 0)
    assert all(row[1] == 5 for row in rows)
    for bad in (-7, 6, 1.5, [0], True):
        features[0].i = bad
        assert d.signature(features, 6) is None
    monkeypatch.setattr(d, 'cupy', lambda: SimpleNamespace(RawKernel=lambda src, *a, **kw: src))
    d._kernel.cache_clear()
    source = d._kernel(rows, 6, True)
    assert 'x[-' not in source and 'out[-' not in source
    for i in range(200):
        d._kernel((('V', 0, 0, float(i), 1., 0.),), 6, False)
    assert d._kernel.cache_info().currsize == 128
    d._kernel.cache_clear()


def test_model_validation_uses_actual_nraw():
    from ciderpress.gpaw.new_backend import CiderXC
    from ciderpress.gpu.semilocal import supported
    model = CiderXC().mlfunc
    model.kernels[0].feature_list[0].i = -1
    assert supported(model, 6)
    model.kernels[0].feature_list[0].i = -7
    assert not supported(model, 6)


def test_pasdw_large_dimensions_and_bounded_slots(monkeypatch):
    from ciderpress.gpu import pasdw as p
    assert p._checked_size(70_000_000, 32) == 2_240_000_000
    for dims in ((-1,), (2**62, 8), (2**63,)):
        with pytest.raises(ValueError):
            p._checked_size(*dims)
    launches = []
    monkeypatch.setattr(p, '_kernel', lambda name: lambda *a: launches.append(a))
    p._launch('theta_grid', 2_240_000_000, (np.int64(70_000_000), np.int64(32)))
    assert launches[0][0] == (8_750_000,)
    with pytest.raises(ValueError, match='grid size'):
        p._launch('theta_grid', 2**40, ())
    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0)))
    monkeypatch.setattr(p, 'cupy', lambda: cp)
    owner = SimpleNamespace()
    for i in range(100):
        assert p._cached(owner, 'gradient', i, lambda cp: i) == i
    assert len(owner._pasdw_device_cache) == 1
    for i in range(100):
        p._cached(owner, ('radial', i), i, lambda cp: i)
    assert len(owner._pasdw_device_cache) == 32


def test_generated_libxc_arithmetic_and_whitespace():
    path = Path(__file__).parents[2] / 'gpu/libxc_7.cuh'
    source = path.read_text()
    assert all(line == line.rstrip() for line in source.splitlines())
    assert 'Copyright (C) 2006-2007 M.A.L. Marques' in source
    assert 'Copyright (C) 2019 X. Andrade' in source
    assert 'src/util.h SHA256 7679744982b3b944f3054b7d2b298612d7b4f3a1051f4f48f4fded2ef26f04b5' in source
    tokens = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)
    # Fingerprint of the reviewed header's exact non-whitespace code bytes.
    digest = hashlib.sha256(re.sub(r'\s+', '', tokens).encode()).hexdigest()
    assert digest == '6c9dae313fc5de35be933602ea105b6227c0a44d81ecb8ae725603389f0588dc'

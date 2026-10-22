"""Device mode-2 strain against native FP64, even/odd skew reciprocal grids."""
import numpy as np
import pytest

from ciderpress import gpu
from ciderpress.gpaw.cider_paw import odd_fft_stress_correction
from ciderpress.gpaw.tests.test_gpu_nldf_fft import (
    active_work, cuda, make_pw)  # noqa: F401


@pytest.mark.parametrize('shape,nq', [((8, 10, 12), 1), ((7, 9, 11), 3),
                                    ((8, 6, 10), 21)])
@pytest.mark.parametrize('corrected', [False, True])
def test_strain_native_and_corrected(cuda, monkeypatch, record_property, shape, nq, corrected):
    rng = np.random.default_rng(823)
    theta, adjoint = [rng.normal(size=(*shape, nq))*.01 for _ in range(2)]
    rows = []
    for selection in ('none', 'nldf_fft', 'nldf_fft'):
        monkeypatch.setenv('CIDER_GPU_KERNELS', selection)
        pw = make_pw(shape, nq)
        pw.set_work(np.ones(shape), theta)
        saved = pw.copy_work_array()
        pw.fftv2(saved)
        # Native reciprocal layout and in-place semantics are part of the API.
        spectrum = saved[:int(pw.get_work_size())].copy()
        pw.set_work(np.ones(shape), adjoint)
        pw._stress_full_rfft = corrected
        stress = pw.compute_backward_convolution_with_stress(saved)
        if selection == 'none' and corrected:
            stress += odd_fft_stress_correction(pw, theta, adjoint)
        rows.append((spectrum, stress, active_work(pw)))
    # The unused allocation tail in copy_work_array is intentionally excluded.
    for name, actual, expected in zip(('spectrum', 'stress', 'convolution'), rows[1], rows[0]):
        delta = np.abs(actual-expected)
        record_property(name+'_max_abs', float(delta.max()))
        record_property(name+'_max_rel', float((delta/np.maximum(abs(expected), 1e-300)).max()))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-11)
    for first, repeat in zip(rows[1], rows[2]):
        np.testing.assert_array_equal(first, repeat)


@pytest.mark.parametrize('shape', [(8, 10, 12), (7, 9, 11)])
def test_device_strain_fd(cuda, monkeypatch, shape):
    """Six strains and three adjacent steps; energy includes changing dv."""
    from ciderpress.gpaw.tests.test_new_stress import directions
    monkeypatch.setenv('CIDER_GPU_KERNELS', 'nldf_fft')
    cell = np.array([[7., .4, .2], [.2, 8., .1], [.3, .2, 9.]])
    rng = np.random.default_rng(21)
    theta, adjoint = [rng.normal(size=(*shape, 3))*.01 for _ in range(2)]

    def evaluate(H, derivative=False):
        # Use the native test factory then rebuild for the deformed cell.
        from ciderpress.gpaw.nldf_interface import LibCiderPW
        from ciderpress.gpaw.new_backend import _process_comm
        template = make_pw(shape, 3)
        pw = LibCiderPW(shape, H, _process_comm())
        pw.nalpha = pw.nbeta = 3
        pw.norms_ab, pw.expnts_ab = template.norms_ab, template.expnts_ab
        pw.initialize_backend()
        pw.set_work(np.ones(shape), theta)
        saved = pw.copy_work_array()
        pw.compute_forward_convolution()
        energy = float(np.sum(active_work(pw)*adjoint)*np.linalg.det(H)/np.prod(shape))
        if not derivative:
            return energy
        pw.fftv2(saved)
        pw.set_work(np.ones(shape), adjoint)
        pw._stress_full_rfft = True
        return energy, pw.compute_backward_convolution_with_stress(saved)

    energy, stress = evaluate(cell, True)
    stress += energy*np.eye(3)
    for i, j, A in directions():
        for h in (1e-4, 5e-5, 2.5e-5):
            fd = (evaluate(cell@(np.eye(3)+h*A))-
                  evaluate(cell@(np.eye(3)-h*A)))/(2*h)
            assert stress[i, j] == pytest.approx(fd, abs=2e-8, rel=2e-7)


def test_pasdw_raw_strain(cuda, record_property):
    from ciderpress.gpaw.cider_paw import _pasdw_stress_dovlp
    rng = np.random.default_rng(829)
    dr, p, dp, f, df = [rng.normal(size=s) for s in
        [(3, 193), (13, 193), (3, 13, 193), (17, 193), (3, 17, 193)]]
    expected = (np.einsum('ug,pg,vfg->uvpf', dr, p, df) +
                np.einsum('ug,vpg,fg->uvpf', dr, dp, f))
    actual = _pasdw_stress_dovlp(dr, p, dp, f, df)
    delta = abs(actual-expected)
    record_property('pasdw_dovlp_max_abs', float(delta.max()))
    record_property('pasdw_dovlp_max_rel', float((delta/np.maximum(abs(expected), 1e-300)).max()))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-11)
    np.testing.assert_array_equal(actual, _pasdw_stress_dovlp(dr, p, dp, f, df))


def test_pasdw_slice_strain(cuda, record_property):
    from ciderpress.gpaw.cider_paw import _pasdw_stress_project, _pasdw_stress_overlap
    from ciderpress.gpaw.tests.test_radial_cutoff import slice_factory
    _, make = slice_factory()
    sl = make(np.zeros(3))
    field = np.random.default_rng(825).normal(size=(len(sl.rad_g), 7))
    for name, actual, expected in [
        ('project', _pasdw_stress_project(sl, field), np.dot(sl.get_stress_funcs(), field)),
        ('overlap', _pasdw_stress_overlap(sl, sl.get_funcs(), sl.get_grads()),
         sl.get_ovlp_deriv(sl.get_funcs(), sl.get_grads(), stress=True))]:
        delta = abs(actual-expected)
        record_property(name+'_max_abs', float(delta.max()))
        record_property(name+'_max_rel', float((delta/np.maximum(abs(expected), 1e-300)).max()))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-11)

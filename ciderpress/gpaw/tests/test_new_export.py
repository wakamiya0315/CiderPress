"""Training loader and saved-feature XC contracts, including signed PAW weights."""
import numpy as np
import pytest

from ciderpress.dft.plans import SemilocalPlan, get_rho_tuple_with_grad_cross
from ciderpress.gpaw.cider_kernel import CiderMGGAHybridKernel
from ciderpress.gpaw.new_backend import CiderXC
from ciderpress.gpaw.new_export import (
    export_training_data, integrate_xc, write_training_data, xc_energy_density,
)
from ciderpress.models.train import MOLGP


def record(nspin):
    param = CiderXC()
    rng = np.random.default_rng(117)
    raw = rng.uniform(.1, .9, (nspin, 5, 19))
    raw[:, 1:4] *= .1
    raw[:, 4] += 2
    sl = SemilocalPlan(param.mlfunc.settings.sl_settings, nspin).get_feat(raw)
    nl = rng.uniform(.2, 1, (nspin, param.mlfunc.settings.nldf_settings.nfeat, 19))
    rec = dict(desc=np.concatenate((sl, nl), axis=1), rho_data=nspin*raw,
               wt=np.r_[np.full(7, .2), np.ones(6), -np.ones(6)],
               nspin=nspin, nfeat_sl=sl.shape[1], ngrid=7, exc_core=-.7,
               model_sha256=param.model_sha256,
               e_tot_orig=-20., e_free_orig=-20.1, exc_orig=-2.)
    return rec, param


@pytest.mark.parametrize('nspin', [1, 2])
def test_saved_feature_energy_and_blocks(nspin):
    rec, param = record(nspin)
    raw = rec['rho_data']/nspin
    rho = get_rho_tuple_with_grad_cross(raw, is_mgga=True)
    # Independent normal inference path computes its own semilocal features.
    kernel = CiderMGGAHybridKernel(param.mlfunc, 1., None, None)
    e = np.zeros(19)
    kernel.calculate(e, rho[0], np.zeros_like(rho[0]), rho[1],
        np.zeros_like(rho[1]), rho[2], np.zeros_like(rho[2]),
        rec['desc'][:, rec['nfeat_sl']:])
    for blocksize in (1, 7, 100):
        np.testing.assert_allclose(xc_energy_density(rec, param, blocksize=blocksize),
                                   e, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(integrate_xc(rec, param, blocksize=blocksize),
                                   e @ rec['wt']-.7, atol=1e-13)
    changed = dict(rec, desc=rec['desc'].copy())
    changed['desc'][:, 1] *= 2
    assert abs(integrate_xc(changed, param)-integrate_xc(rec, param)) > 1e-8
    with pytest.raises(ValueError, match='SHA256'):
        integrate_xc(dict(rec, model_sha256='wrong'), param)


@pytest.mark.parametrize('nspin', [1, 2])
def test_training_loader_roundtrip(tmp_path, nspin):
    rec, param = record(nspin)
    dirs = write_training_data(rec, tmp_path, 'slab', reference_energy=-3.75)
    loaded = MOLGP.load_data(dirs, 'slab', False)
    for key in ('desc', 'rho_data', 'wt'):
        np.testing.assert_array_equal(loaded[key], rec[key])
    assert loaded['e_tot_orig']-loaded['exc_orig'] == -18.
    np.testing.assert_allclose(integrate_xc(loaded, param), integrate_xc(rec, param),
                               atol=1e-13)
    np.testing.assert_allclose(loaded['val'] @ loaded['wt'], -3.75, atol=1e-14)
    assert np.all(loaded['val'][7:] == 0)
    dirs = write_training_data(rec, tmp_path, 'no_label')
    assert 'val' not in MOLGP.load_data(dirs, 'no_label', False)
    values = np.arange(19, dtype=float)
    dirs = write_training_data(rec, tmp_path, 'local', reference_values=values)
    np.testing.assert_array_equal(MOLGP.load_data(dirs, 'local', False)['val'], values)


def test_invalid_inputs(tmp_path):
    rec, _ = record(1)
    for name in ('../escape', '', '.', '..', '/absolute'):
        with pytest.raises(ValueError, match='filename'):
            write_training_data(rec, tmp_path, name)
    for values in (np.zeros(3), np.full(19, np.nan)):
        with pytest.raises(ValueError, match='reference_values'):
            write_training_data(rec, tmp_path, 'sample', reference_values=values)
    with pytest.raises(ValueError, match='one reference'):
        write_training_data(rec, tmp_path, 'sample', reference_energy=1,
                            reference_values=np.zeros(19))
    with pytest.raises(ValueError, match='converged'):
        export_training_data(object())

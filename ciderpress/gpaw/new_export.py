"""Training records from the New-GPAW CIDER backend (CPU or CUDA).

``record = export_training_data(calc)`` takes an already converged New GPAW
PW calculator with ``xc=CiderXC(...)``. It never runs SCF, rebuilds orbital
tau, or changes the live XC engine. ``export_features(xc, n, tau, density)``
is the lower-level API for identical-input tests: ``xc`` is a contextualized
``CiderNewMGGAFunctional``; n/tau are its fine-grid UGArrays, including pseudo
cores, and density supplies the PAW matrices. Arrays returned are host FP64.
The NLDF/PASDW forward uses the functional's device. The feature-only PAW
quadrature runs on the host by default. ``gpu=True`` opts into FP64 device
quadrature on a CUDA functional, reusing New's density, projection, spherical
and spline kernels. Private forward caches stay on device; live SCF caches
are untouched. Returned arrays and the HDF5 schema are unchanged.
The bounded process-local ``DEFAULT_PAW_CACHE`` reuses density-independent
host setup tables across both private engines and consecutive exports. Pass
``paw_cache=ExportPAWCache(maxsize=16)`` for an explicit owner, or
``paw_cache=None`` for the original rebuild path. ``cache.clear()`` releases
its entries and resets ``cache.info()`` counters. Each engine has fresh plans,
contributions, lazy device mirrors and geometry/density scratch. Live SCF
tables are never borrowed. Clearing does not invalidate an export in progress.
Joint-tau GPW restart clears GPAW's ASE energy cache. For a checkpoint whose
convergence has been independently qualified, the caller can expose its saved
energies with ``calc.dft.calculate_energy()`` before export; this performs no
SCF and grants no fresh strict-force qualification. Checkpoint provenance is
the caller's responsibility, not inferred from the presence of an energy.

``desc`` has Classic layout (nspin, nfeature, npoint), with SL then NLDF;
``rho_data`` is Classic ``get_descriptors(calc, 'l')``: nspin times
[n, dx n, dy n, dz n, tau], in atomic units (Bohr powers -3,-4,-4,-4,-5).
Tau includes the usual factor 1/2. NLDF and SL spin factors are exactly those
of Classic. ``wt`` is in Bohr**3 and MUST retain negative pseudo-sphere
weights. Point order is the flattened fine grid, then AE/pseudo pairs in atom
order (radial-major, Lebedev-minor). Only the uniform grid is screened when
``screen_dens=True`` (sum of physical spin densities > 1e-6). The default is
False so the full XC integral can be checked without a vacuum truncation.

``integrate_xc(record, CiderXC(...))`` evaluates the saved features/raw density
with the model and external X/C baselines, then adds ``exc_core`` (minus the
PAW setup reference core XC energies). This constant is essential to compare
with the calculator's XC; it is not part of the Classic descriptor arrays.
``integrate_xc(..., gpu=True)`` also selects the existing FP64 mapped-XC
device evaluator. Normalization, external LibXC baselines and the final
signed-weight reduction keep their host interfaces. The default is unchanged.
``e_tot_orig`` and ``exc_orig`` are calculator energies in Hartree; the former
is the zero-temperature extrapolated energy (``e_free_orig`` is also saved).
``exc_replayed`` is a fresh XC evaluation on the exported n/tau/D state, to
expose any difference between the stored SCF energy and its final density.

``write_training_data(record, directory, system_id)`` writes PySCF chkfile
compatible ``REF/SL/NLDF/<system_id>.hdf5``, group ``train_data``, and returns
the directory mapping accepted by ``MOLGP.load_data`` / ``MOLGP2.load_data``.
The REF contains rho_data, wt, nspin and energies; SL/NLDF contain desc.
``cider_parameters_json`` records the model digest and interpolation options.
Call the HDF5 writer once, on a designated writer rank in MPI calculations.
Reference EXX/RPA labels cannot be inferred from a CIDER SCF: supply an
explicit Hartree ``reference_energy`` or pointwise energy-density
``reference_values`` to the writer for training's ``save_refs=True`` path.
A scalar label is encoded in ``val`` on grid points only, with its exact
weighted integral; it does not claim a local reference energy density.
Orbital-occupation derivatives are not exported. Domain/band distribution,
noncollinear spin, and nonperiodic FFT grids are unsupported by this backend.
"""

from pathlib import Path
import json

import numpy as np

from ciderpress.dft.plans import SemilocalPlan, get_rho_tuple_with_grad_cross
from ciderpress.gpaw.atom_descriptor_utils import (
    PASDWCiderFeatureKernel, calculate_paw_sl_features,
)
from ciderpress.gpaw.descriptors import get_atom_wt
from ciderpress.gpaw.config import GPAW_DEFAULT_RHO_TOL
from ciderpress.gpaw.new_backend import CiderNewMGGAFunctional, _Engine
from ciderpress.gpaw._export_paw_cache import DEFAULT_PAW_CACHE, ExportPAWCache
from ciderpress.gpu import policy


class _CachedPAWEngine(_Engine):
    def _initialize_export_paw(self):
        if self.export_paw_cache is None:
            self.paw_kernel.initialize_more_things(self.setups)
        else:
            self.export_paw_cache.initialize(self.paw_kernel, self.export_paw_signature)

    def initialize_paw_kernel(self, kernel, nalpha, encut):
        from ciderpress.gpaw.atom_utils import FastPASDWCiderKernel
        self.paw_kernel = FastPASDWCiderKernel(
            kernel, self._plan, self.aux_gd, self.cut_xcgrid)
        self.paw_kernel.initialize(
            self.dens, self.atomdist, self.atom_partition, self.setups)
        self._initialize_export_paw()
        self.paw_kernel.interpolate_dn1 = self.interpolate_dn1
        self.atom_slices_s = None


def _private_engine(private, cls, cache):
    if cache is not None and not isinstance(cache, ExportPAWCache):
        raise TypeError('paw_cache must be an ExportPAWCache or None')
    engine = cls(private.param, private.grid, private.grad_v)
    engine.export_paw_cache = cache
    engine.export_paw_signature = (
        private.param.model_sha256, private.param.options, private.param.nldf,
        private.xp.__name__, private._device, np.dtype(np.float64).str,
        engine.interpolate_dn1, engine.encut_atom_min,
        engine.pasdw_ovlp_fit, engine.pasdw_store_funcs)
    return engine


class _ExportEngine(_CachedPAWEngine):
    """Forward-only New engine with Classic's PAW feature quadrature."""

    def initialize_paw_kernel(self, kernel, nalpha, encut):
        self.paw_kernel = PASDWCiderFeatureKernel(
            kernel, self._plan, self.aux_gd, self.cut_xcgrid)
        self.paw_kernel.initialize(
            self.dens, self.atomdist, self.atom_partition, self.setups)
        self._initialize_export_paw()
        self.paw_kernel.interpolate_dn1 = self.interpolate_dn1
        self.atom_slices_s = None


def export_features(functional, nt_sr, taut_sr, density, *, screen_dens=False,
                    gpu=False, paw_cache=DEFAULT_PAW_CACHE):
    """Export n/tau/D; paw_cache=None disables process-local PAW table reuse."""
    if not isinstance(functional, CiderNewMGGAFunctional):
        raise TypeError('Expected New-GPAW CIDER functional')
    if functional.setups is None:
        raise ValueError('CIDER functional needs PAW context')
    if gpu and functional._device is None:
        raise ValueError('gpu=True export requires a CUDA functional')
    with policy(functional._device):
        # Construction and dispatch both use the calculation's CUDA device,
        # even if the caller selected another device after the SCF.
        # Context creation clones Setup objects before PASDW attaches scratch.
        private = CiderNewMGGAFunctional(functional.param, functional.grid,
                                       functional.xp)
        private.engine = _private_engine(private, _ExportEngine, paw_cache)
        private.set_context(setups=functional.setups,
                            relpos_ac=functional._relpos,
                            atomdist=density.D_asii.layout.atomdist)
        private.set_density(density)
        if nt_sr.dims != (private._nspins,) or taut_sr.dims != nt_sr.dims:
            raise ValueError('Export n/tau/D spin shapes differ')
        for arr in (nt_sr, taut_sr):
            if arr.desc.comm.size != 1 or any(
                not np.array_equal(getattr(arr.desc, key),
                                   getattr(functional.grid, key))
                for key in ('cell_cv', 'size_c', 'pbc_c')):
                raise ValueError('Export requires the functional fine grid')
        n = private._transfer(nt_sr, np, 'export_n').data
        tau = private._transfer(taut_sr, np, 'export_tau').data
        engine = private.engine
        engine.dens.nt_sg = n
        rho = engine._get_cider_inputs(n, tau)[1]
        engine._setup_plan()
        engine.fft_obj.initialize_backend()
        engine.initialize_more_things()
        nspin = len(rho)
        nldf = engine.call_fwd(
            engine._plan.get_rho_tuple(rho, with_spin=True))[0] * nspin
        # Classic's feature quadrature needs NumPy caches. The opt-in path
        # consumes the private forward caches on device instead.
        paw = engine.paw_kernel
        if gpu:
            from ciderpress.gpaw._export_gpu import paw_features
            ae_nl, ps_nl, ae_rho, ps_rho = paw_features(engine)
        else:
            from gpaw.gpu import as_np
            paw.fr_asgLq = {a: as_np(x) for a, x in paw.fr_asgLq.items()}
            paw.df_asgLq = {a: as_np(x) for a, x in paw.df_asgLq.items()}
            coefficients = {a: as_np(x) for a, x in engine.c_asiq.items()}
            with policy(None):
                ae_nl, ps_nl = paw.calculate_paw_cider_features_p2(
                    engine.setups, engine.D_asp, coefficients, None, None)
                ae_rho, ps_rho = calculate_paw_sl_features(
                    engine.setups, engine.D_asp, True)
        plan = SemilocalPlan(functional.param.mlfunc.settings.sl_settings, nspin)
        mask = (n.sum(axis=0).ravel() > 1e-6 if screen_dens else
                np.ones(n[0].size, dtype=bool))
        raw_parts = [rho.reshape(nspin, 5, -1)[..., mask] * nspin]
        sl_parts = [plan.get_feat(rho.reshape(nspin, 5, -1))[..., mask]]
        nl_parts = [nldf.reshape(nspin, nldf.shape[1], -1)[..., mask]]
        weights = [np.full(mask.sum(), engine.gd.dv)]
        for a in range(len(engine.setups)):
            for raw, nl, sign in ((ae_rho[a], ae_nl[a], 1),
                                  (ps_rho[a], ps_nl[a], -1)):
                raw_parts.append(raw * nspin)
                sl_parts.append(plan.get_feat(raw))
                nl_parts.append(nl)
                weights.append(sign * get_atom_wt(engine, a))
        sl = np.concatenate(sl_parts, axis=-1)
        nl = np.concatenate(nl_parts, axis=-1)
        return dict(desc=np.concatenate((sl, nl), axis=1),
                    rho_data=np.concatenate(raw_parts, axis=-1),
                    wt=np.concatenate(weights), nspin=nspin, nfeat_sl=sl.shape[1],
                    ngrid=int(mask.sum()), screen_dens=bool(screen_dens),
                    exc_core=-sum(s.xc_correction.e_xc0 for s in engine.setups),
                    cider_parameters_json=json.dumps(functional.param.todict(),
                                                      sort_keys=True),
                    model_sha256=functional.param.model_sha256)


def export_training_data(calc, *, screen_dens=False, gpu=False,
                         paw_cache=DEFAULT_PAW_CACHE):
    """Export a converged calculator, optionally reusing private PAW tables.

    ``paw_cache`` defaults to DEFAULT_PAW_CACHE; pass None to disable it or
    supply an ExportPAWCache whose lifetime/clear operation the caller owns.
    Uninitialized/unfinished calculator states are refused.
    """
    if getattr(calc, '_dft', None) is None or 'energy' not in calc.dft.results:
        raise ValueError('Export requires a converged New-GPAW calculator')
    dft = calc.dft
    pc = dft.pot_calc
    if not isinstance(pc.xc, CiderNewMGGAFunctional):
        raise TypeError('Export requires xc=CiderXC(...)')
    with policy(pc.xc._device):
        # Use the identical interpolation used for the potential, including cores.
        n = pc._interpolate_density(dft.density.nt_sR)[0]
        tau = pc.interpolate(dft.density.taut_sR)
        record = export_features(pc.xc, n, tau, dft.density,
                                 screen_dens=screen_dens, gpu=gpu,
                                 paw_cache=paw_cache)
        record.update(e_tot_orig=float(dft.energies.total_extrapolated),
                      e_free_orig=float(dft.energies.total_free),
                      exc_orig=float(dft.energies._energies['xc']))
        # Replaying a private functional avoids invalidating live force/PAW caches.
        replay = CiderNewMGGAFunctional(pc.xc.param, pc.fine_grid, pc.xp)
        replay.engine = _private_engine(replay, _CachedPAWEngine, paw_cache)
        replay.set_context(setups=pc.setups, relpos_ac=pc.relpos_ac,
                           atomdist=pc.atomdist, communicators=pc.communicators)
        replay.set_density(dft.density)
        egrid = replay.calculate(n, tau)[0]
        record['exc_replayed'] = egrid + sum(replay.engine.E_a_tmp.values())
    return record


def xc_energy_density(record, param, *, blocksize=10000, gpu=False):
    """Evaluate saved SL+NLDF and physical raw ingredients, in Ha/Bohr**3."""
    from ciderpress.dft.xc_evaluator import MappedXC
    from gpaw.xc.libxc import LibXC

    if blocksize < 1:
        raise ValueError('blocksize must be positive')
    digest = record['model_sha256']
    if isinstance(digest, bytes):
        digest = digest.decode('ascii')
    if digest != param.model_sha256:
        raise ValueError('Record/model SHA256 mismatch')
    model = param.mlfunc
    if gpu:
        from ciderpress.gpu import _real_cupy, use_gpu
        from ciderpress.gpu.semilocal import supported

        cp = _real_cupy()
        if cp is None:
            raise RuntimeError('GPU XC replay requires CuPy with a CUDA device')
        if not supported(model, record['desc'].shape[1]):
            raise NotImplementedError('GPU XC replay requires a supported mapped-XC model')
        with policy(cp.cuda.Device().id):
            if not all(use_gpu(k) for k in ('semilocal', 'rbf', 'libxc')):
                raise ValueError('GPU XC replay requires semilocal, rbf and libxc kernels')
            return xc_energy_density(record, param, blocksize=blocksize)
    energy = np.empty(record['wt'].size)
    for start in range(0, energy.size, blocksize):
        stop = start + blocksize
        raw = record['rho_data'][..., start:stop] / record['nspin']
        rho = get_rho_tuple_with_grad_cross(raw, is_mgga=True)
        x = model.settings.normalizers.get_normalized_feature_vector(
            record['desc'][..., start:stop])
        if isinstance(model, MappedXC):
            e = model(x, rhocut=GPAW_DEFAULT_RHO_TOL)[0]
        else:
            e = model(x, rho, rhocut=GPAW_DEFAULT_RHO_TOL)[0]
        e = e * param.options['xmix']
        for name, factor in ((param.options['xkernel'], 1-param.options['xmix']),
                             (param.options['ckernel'], 1)):
            if name is None or factor == 0:
                continue
            kernel = LibXC(name)
            args = [np.zeros_like(e), rho[0].copy(), np.zeros_like(rho[0])]
            if kernel.type in ('GGA', 'MGGA'):
                args += [rho[1].copy(), np.zeros_like(rho[1])]
            if kernel.type == 'MGGA':
                args += [rho[2].copy(), np.zeros_like(rho[2])]
            kernel.calculate(*args)
            e += factor * args[0]
        energy[start:stop] = e
    return energy


def integrate_xc(record, param, *, blocksize=10000, gpu=False):
    """Total XC in Ha, including the setup reference-core subtraction."""
    return float(np.dot(record['wt'], xc_energy_density(
        record, param, blocksize=blocksize, gpu=gpu)) + record['exc_core'])


def write_training_data(record, directory, system_id, *, reference_energy=None,
                        reference_values=None):
    """Write Classic training HDF5 groups; return the trainer's ddir mapping."""
    from pyscf.lib import chkfile

    if not system_id or Path(system_id).name != system_id or system_id in ('.', '..'):
        raise ValueError('system_id must be a single filename component')
    if reference_energy is not None and reference_values is not None:
        raise ValueError('Supply only one reference label representation')
    ref = {k: v for k, v in record.items() if k not in ('desc', 'nfeat_sl')}
    if reference_values is not None:
        values = np.asarray(reference_values, dtype=float)
        if values.shape != record['wt'].shape or not np.isfinite(values).all():
            raise ValueError('reference_values must match wt and be finite')
        ref['val'] = values
    elif reference_energy is not None:
        if not np.isfinite(reference_energy) or record['ngrid'] == 0:
            raise ValueError('A finite reference energy and nonempty grid are needed')
        ref['val'] = np.zeros_like(record['wt'])
        ng = record['ngrid']
        ref['val'][:ng] = reference_energy / record['wt'][:ng].sum()
        ref['reference_energy'] = float(reference_energy)
    nsl = record['nfeat_sl']
    groups = dict(REF=ref, SL={'desc': record['desc'][:, :nsl]},
                  NLDF={'desc': record['desc'][:, nsl:]})
    ddir = dict.fromkeys(('REF', 'SL', 'NLDF', 'NLOF', 'SDMX', 'HYB'))
    for group, values in groups.items():
        path = Path(directory) / group
        path.mkdir(parents=True, exist_ok=True)
        chkfile.save(str(path / (system_id + '.hdf5')), 'train_data', values)
        ddir[group] = str(path)
    return ddir

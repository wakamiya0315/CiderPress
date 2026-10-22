"""CPU-engine CIDER26XCSURFSCI for New GPAW CPU/GPU PW.

Term                       Counted by
=========================  ================================================
Smooth grid XC             calculate(): integral of the fine-grid energy
PAW AE minus pseudo XC     calculate_paw_correction(): owned atom only
PAW XC Hamiltonian         dH_sp += correction (pack_hermitian convention)
Pseudo core density/tau    supplied by New Density, never added here
Kinetic energy correction  New PotentialCalculator, never calculated here
=========================  ================================================

The Classic PASDW/NLDF numerical routines are composed through a small engine;
no Classic calculator, MGGA calculate_impl, tau builder or ekin is invoked.
Density packing doubles off-diagonal entries; Hamiltonian packing does not.
"""

import hashlib
from copy import copy
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import yaml
from gpaw.dft import XC
from gpaw.mpi import serial_comm
from gpaw.new.xc import MGGAFunctional
from gpaw.utilities import pack_density
from gpaw.utilities.partition import AtomPartition
from gpaw.utilities.timing import Timer
from gpaw.xc.libxc import LibXC
from gpaw.xc.mgga import MGGA

from ciderpress.dft.model_utils import (
    get_builtin_model_name,
    validate_cider_composition,
)
from ciderpress.gpaw.cider_fft import (
    _CiderBase,
    add_gradient_correction as _cpu_gradient_correction,
)
from ciderpress.gpaw.cider_kernel import CiderMGGAHybridKernel
from ciderpress.gpaw.cider_paw import CiderPASDW_MPRoutines
from ciderpress.gpaw.config import GPAW_DEFAULT_RHO_TOL
from ciderpress.gpaw.nldf_interface import pwutil


def add_gradient_correction(grad_v, dedrho_sxg, v_sg):
    from ciderpress.gpu import use_gpu
    if use_gpu('fine_grid'):
        from ciderpress.gpu.fine_grid import add_correction
        return add_correction(grad_v, dedrho_sxg, v_sg)
    return _cpu_gradient_correction(grad_v, dedrho_sxg, v_sg)


@lru_cache(maxsize=1)
def _process_comm():
    """Private size-one communicator shared by FFT and PASDW on this process.

    Serial CiderPress needs no native MPI object. MPI CiderPress needs a real
    one even on one rank: the generic GPAW native API duplicates MPI_COMM_SELF.
    That operation is collective only on SELF, never on GPAW world/k-point
    groups; unequal functional construction counts across ranks are safe.
    """
    if not pwutil.ciderpw_has_mpi():
        return serial_comm
    from gpaw import cgpaw
    if not cgpaw.have_mpi or not hasattr(cgpaw.Communicator, 'self_communicator'):
        raise RuntimeError('MPI CiderPress requires GPAW native self_communicator(); '
                           'rebuild the GPAW extension with the Phase 2 MPI hook')
    comm = cgpaw.Communicator().self_communicator()
    assert comm.size == 1 and comm.rank == 0
    return comm


class CiderXC(XC):
    """Serializable, fail-closed New-GPAW XC parameter (PW only).

    Use ``GPAW(..., xc=CiderXC(), legacy_gpaw=False)`` and restore GPWs with
    ``object_hooks={'xc': restore_cider_xc}``. Both the requested interpolation
    parameters and the derived parameters are saved, to detect incompatible
    numerical settings on restoration. Explicit YAML paths are supported.
    """

    type = 'MGGA'

    def __init__(self, model='CIDER26XCSURFSCI', *, xmix=1.0,
                 xkernel=None, ckernel=None, Nalpha=None, lambd=1.8,
                 qmax=300.0, pasdw_ovlp_fit=True, pasdw_store_funcs=False):
        super().__init__('CIDER26XCSURFSCI')
        builtin = get_builtin_model_name(model)
        self.model = str(model)
        path = (files('ciderpress.data.functionals').joinpath(builtin + '.yaml')
                if builtin else Path(model))
        # Hash the exact bytes parsed, not a later reopening of the file.
        data = path.read_bytes()
        self.model_sha256 = hashlib.sha256(data).hexdigest()
        self.mlfunc = yaml.load(data, Loader=yaml.CLoader)
        validate_cider_composition(self.mlfunc, xmix=xmix, xkernel=xkernel,
                                   ckernel=ckernel, backend='GPAW')
        settings = self.mlfunc.settings
        if (settings.sl_settings.level != 'MGGA' or
                settings.nldf_settings.is_empty or
                settings.nldf_settings.version != 'j' or
                not settings.nlof_settings.is_empty or
                not settings.sdmx_settings.is_empty):
            raise NotImplementedError('New CIDER requires MGGA with version-j NLDF')
        if lambd <= 1 or qmax <= 0 or (Nalpha is not None and Nalpha < 2):
            raise ValueError('Invalid NLDF interpolation parameters')
        self.options = dict(xmix=xmix, xkernel=xkernel, ckernel=ckernel,
                            Nalpha=Nalpha, lambd=lambd, qmax=qmax,
                            pasdw_ovlp_fit=pasdw_ovlp_fit,
                            pasdw_store_funcs=pasdw_store_funcs)
        kernel = CiderMGGAHybridKernel(self.mlfunc, xmix, xkernel, ckernel)
        base = _CiderBase(kernel, Nalpha=Nalpha, lambd=lambd, encut=qmax,
                          world=serial_comm)
        self.nldf = dict(Nalpha=base.Nalpha, lambd=float(base.lambd),
                         qmax=qmax, encut=qmax, rhocut=GPAW_DEFAULT_RHO_TOL,
                         expcut=GPAW_DEFAULT_RHO_TOL,
                         alpha_formula='etb', coef_order='gq',
                         use_smooth_expnt_cutoff=True,
                         encut_atom_min=8000.0, interpolate_dn1=False,
                         cut_xcgrid=False, stencil=2)
        self._cell = None

    def todict(self):
        return dict(name=self.name, cider_new_version=1, model=self.model,
                    model_sha256=self.model_sha256, options=self.options.copy(),
                    nldf=self.nldf.copy(), cell=self._cell)

    def functional(self, *, collinear, atoms=None):
        if not collinear:
            raise NotImplementedError('New CIDER does not support spin polarization')
        if atoms is not None:
            cell = atoms.cell.array.tolist()
            if self._cell is not None and not np.array_equal(cell, self._cell):
                raise NotImplementedError('New CIDER does not support cell changes')
            self._cell = cell
        return self

    def new_functional(self, grid, xp=np):
        return CiderNewMGGAFunctional(self, grid, xp)

    def get_setup_name(self):
        return 'PBE'


def restore_cider_xc(d, *, expected=None):
    """GPW object hook; refuse changed model bytes or NLDF implementation.

    ``expected=CiderXC(...)`` additionally pins the caller's desired settings.
    """
    if d.get('cider_new_version') != 1:
        raise ValueError('Unsupported New CIDER restart schema')
    xc = CiderXC(d['model'], **d['options'])
    if d['model_sha256'] != xc.model_sha256:
        raise ValueError('New CIDER model SHA256 mismatch')
    if d['nldf'] != xc.nldf:
        raise ValueError('New CIDER NLDF settings mismatch')
    xc._cell = d['cell']
    if expected is not None:
        for key in ('model_sha256', 'options', 'nldf'):
            if xc.todict()[key] != expected.todict()[key]:
                raise ValueError('New CIDER requested settings mismatch: ' + key)
    return xc


class _SerialAtomDist:
    """Classic work-distribution protocol: all atoms owned once in serial."""

    @staticmethod
    def to_work(a):
        return a

    @staticmethod
    def from_work(a):
        return a


class _GridHybridKernel(CiderMGGAHybridKernel):
    def call_xc_kernel(self, e_g, n_sg, sigma_xg, feat_sg, v_sg,
                       dedsigma_xg, vfeat_sg, tau_sg=None, dedtau_sg=None):
        args = (e_g, n_sg, sigma_xg, feat_sg, v_sg, dedsigma_xg, vfeat_sg)
        kwargs = dict(tau_sg=tau_sg, dedtau_sg=dedtau_sg)
        from ciderpress.gpu import use_gpu
        if use_gpu('grid_xc') and use_gpu('semilocal'):
            from ciderpress.gpu.grid_xc import evaluate, supported
            if supported(self):
                return evaluate(self, *args, **kwargs)
        return super().call_xc_kernel(*args, **kwargs)


class _Engine(CiderPASDW_MPRoutines, _CiderBase):
    """Compose the unchanged Classic numerical engine, without Classic MGGA."""

    def __init__(self, param, grid, grad_v):
        opts = param.options
        kernel = _GridHybridKernel(param.mlfunc, opts['xmix'],
                                      opts['xkernel'], opts['ckernel'])
        _CiderBase.__init__(self, kernel, Nalpha=opts['Nalpha'],
                            lambd=opts['lambd'], encut=opts['qmax'],
                            world=_process_comm())
        self.pasdw_ovlp_fit = opts['pasdw_ovlp_fit']
        self.pasdw_store_funcs = opts['pasdw_store_funcs']
        self.interpolate_dn1 = False
        self.encut_atom_min = 8000.0
        self.cut_xcgrid = False
        self.k2_k = None
        self.setups = None
        self.atom_slices_s = None
        self.timer = Timer()
        self.grad_v = [g.apply for g in grad_v]
        self.set_grid_descriptor(grid._gd)
        self.aux_gd = self.gd.new_descriptor(comm=self.world)
        self.atomdist = _SerialAtomDist()

    def initialize_more_things(self):
        # Same PAW initialization as Classic, with an explicit context instead
        # of the Classic Hamiltonian/wavefunction objects.
        encut, nalpha = self.encut, self.Nalpha
        while encut < self.encut_atom_min:
            encut *= self.lambd
            nalpha += 1
        self.initialize_paw_kernel(self.cider_kernel, nalpha, encut)

    def _get_cider_inputs(self, n_sg, taut_sg):
        # A content token, rather than array identity, detects in-place changes
        # before a property replay.  This does not change the numerical path.
        self._force_input_digest = _force_input_digest(n_sg, taut_sg)
        from ciderpress.gpu import use_gpu
        if use_gpu('fine_grid'):
            from ciderpress.gpu.fine_grid import get_rho
            rho = get_rho(n_sg, self.grad_v, taut_sg)
            self.nspin = len(rho)
            return np.zeros(n_sg.shape[1:]), rho, np.zeros_like(rho)
        return super()._get_cider_inputs(n_sg, taut_sg)

    def call_fwd(self, rho_tuple, grad_mode=0):
        from ciderpress.gpu.grid_nldf import forward, supported
        if supported(self, grad_mode):
            return forward(self, rho_tuple)
        return super().call_fwd(rho_tuple, grad_mode=grad_mode)

    def call_bwd(self, rho_tuple, vfeat_sig, dfeat_sjg, arg_sg, fun_sg,
                 v_sg, dedsigma_xg, dedtau_sg, p_gq=None, dp_gq=None,
                 grad_mode=0):
        from ciderpress.gpu.grid_nldf import backward, supported
        if supported(self, grad_mode):
            return backward(self, rho_tuple, vfeat_sig, dfeat_sjg, arg_sg,
                            fun_sg, v_sg, dedsigma_xg, dedtau_sg, p_gq, dp_gq)
        return super().call_bwd(rho_tuple, vfeat_sig, dfeat_sjg, arg_sg,
                               fun_sg, v_sg, dedsigma_xg, dedtau_sg,
                               p_gq=p_gq, dp_gq=dp_gq, grad_mode=grad_mode)


class CiderNewMGGAFunctional(MGGAFunctional):
    """Fine-grid CIDER XC. See the module's term-counting table.

    ``calculate`` returns only grid XC and derivatives; the cached PAW XC
    energy is returned once per owned atom by ``calculate_paw_correction``.
    No tauct or kinetic correction is added. Input arrays are converted with
    ``to_xp(np)`` before entering the Classic numpy numerical engine.
    """

    def __init__(self, param, grid, xp=np):
        if xp is not np:
            from gpaw.gpu import cupy
            if xp is not cupy:
                raise NotImplementedError('New CIDER supports numpy or GPAW cupy only')
        # Record ownership without changing dispatch or the current device,
        # even if construction subsequently fails. CPUPY is always CPU.
        self._device = (xp.cuda.Device().id
                        if getattr(xp, '__name__', '') == 'cupy' else None)
        if grid.comm.size != 1:
            raise NotImplementedError('New CIDER does not support domain parallelism')
        old = MGGA(LibXC('PBE'), stencil=2)
        old.type = 'MGGA'
        old.name = 'CIDER26XCSURFSCI'
        # Keep the reference MGGA/Gradient objects on NumPy. Optional device
        # stencils reuse their exact coefficients behind the NumPy boundary.
        super().__init__(old, grid)
        self.xp = xp
        self.param = param
        self.engine = _Engine(param, grid, self.grad_v)
        self._relpos = None
        self.setups = None
        self._density = None
        self._paw_ready = False
        # Bounded diagnostics, including no-op boundaries for profile callers.
        self.transfer_timings = []

    def _transfer(self, array, xp, label):
        source = getattr(array, 'xp', None)
        if source is None:
            source = array.layout.xp  # AtomArrays
        if source is xp:
            result, seconds = array, 0.0
        else:
            from gpaw.gpu import cupy, cupy_is_fake
            if not cupy_is_fake:
                cupy.cuda.get_current_stream().synchronize()
            start = perf_counter()
            result = array.to_xp(xp)
            if not cupy_is_fake:
                cupy.cuda.get_current_stream().synchronize()
            seconds = perf_counter() - start
        self.transfer_timings.append(dict(operation=label, seconds=seconds,
            bytes=int(array.data.size * array.data.dtype.itemsize),
            copied=source is not xp))
        del self.transfer_timings[:-256]
        return result

    def set_context(self, *, setups, relpos_ac, atomdist,
                    communicators=None, mode='pw', distribute_density=False):
        if mode != 'pw':
            raise NotImplementedError('New CIDER supports PW mode only')
        if distribute_density:
            raise NotImplementedError('New CIDER does not support distributed density')
        if communicators is not None:
            for key, label in [('d', 'domain'), ('b', 'band')]:
                if communicators[key].size != 1:
                    raise NotImplementedError('New CIDER does not support ' + label +
                                              ' parallelism')
        if atomdist.comm.size != 1 or np.any(atomdist.rank_a != 0):
            raise NotImplementedError('New CIDER requires serial atom distribution')
        if self._relpos is not None and not np.array_equal(relpos_ac, self._relpos):
            raise NotImplementedError('New CIDER does not support atom moves')
        self._relpos = np.array(relpos_ac, copy=True)
        engine = self.engine
        if self.setups is not None and setups is not self.setups:
            raise NotImplementedError('New CIDER does not support replacing setups')
        if self.setups is None:
            self.setups = setups
            # PASDW writes functional-specific radial plans/contribution caches
            # onto Setup objects. Keep those private to this engine, including
            # when evaluating a second XC on the same New density/setups.
            engine.setups = copy(setups)
            copies = {}
            for a, setup in enumerate(setups):
                if id(setup) not in copies:
                    clone = copy(setup)
                    clone.cider_contribs = None
                    copies[id(setup)] = clone
                engine.setups[a] = copies[id(setup)]
        engine.spos_ac = self._relpos
        engine.atom_partition = AtomPartition(engine.world, atomdist.rank_a.copy())
        if self._density is None:
            engine.dens = SimpleNamespace(setups=engine.setups)

    def set_density(self, density):
        if density.nt_sR.dims != (1,):
            raise NotImplementedError('New CIDER does not support spin polarization')
        if density.nt_sR.desc.comm.size != 1:
            raise NotImplementedError('New CIDER does not support domain parallelism')
        engine = self.engine
        if engine.setups is None:
            raise RuntimeError('New CIDER requires set_context before set_density')
        matrices = self._transfer(density.D_asii, np, 'D_to_host')
        if (matrices.layout.atomdist.comm.size != 1 or
                not np.array_equal(matrices.layout.atomdist.rank_a,
                                   engine.atom_partition.rank_a)):
            raise NotImplementedError('New CIDER density atom distribution mismatch')
        if set(matrices.keys()) != set(range(len(engine.setups))):
            raise ValueError('New CIDER requires each atom exactly once')
        D_asp = engine.setups.empty_atomic_matrix(1, engine.atom_partition)
        for a, D_sii in matrices.items():
            D_asp[a][:] = [pack_density(D_ii.real) for D_ii in D_sii]
        engine.dens.D_asp = D_asp
        self._density = density
        self._paw_ready = False

    def calculate(self, nt_sr, taut_sr=None):
        from ciderpress.gpu import policy
        with policy(self._device):
            for array in (nt_sr, taut_sr):
                device = getattr(getattr(array, 'data', None), 'device', None)
                if self._device is not None and device is not None and device.id != self._device:
                    raise ValueError('New CIDER input differs from its recorded CUDA device')
            return self._calculate(nt_sr, taut_sr)

    def _calculate(self, nt_sr, taut_sr=None):
        self._paw_ready = False
        if nt_sr.dims != (1,):
            raise NotImplementedError('New CIDER does not support spin polarization')
        if self._density is None:
            raise RuntimeError('New CIDER requires set_density before calculate')
        if taut_sr is None:
            raise ValueError('New CIDER requires tau including pseudo core tau')
        for desc in (nt_sr.desc, taut_sr.desc):
            if desc.comm.size != 1 or any(not np.array_equal(getattr(desc, key),
                        getattr(self.grid, key)) for key in ('cell_cv', 'size_c', 'pbc_c')):
                raise NotImplementedError('New CIDER does not support grid/cell changes')
        if taut_sr.dims != (1,):
            raise NotImplementedError('New CIDER does not support spin polarization')
        xp = nt_sr.xp
        n = self._transfer(nt_sr, np, 'n_to_host')
        tau = self._transfer(taut_sr, np, 'tau_to_host')
        engine = self.engine
        engine.dens.nt_sg = n.data
        e, rho, dedrho = engine._get_cider_inputs(n.data, tau.data)
        engine.calc_cider(e, rho, dedrho)
        # In this phase the FFT and GPAW grids have identical serial layouts.
        v = n.new(zeroed=True)
        d = n.new(zeroed=True)
        v.data[:] = dedrho[:, 0]
        add_gradient_correction(engine.grad_v, dedrho, v.data)
        d.data[:] = dedrho[:, 4]
        self._paw_ready = True
        return (float(engine.gd.integrate(e)),
                self._transfer(v, xp, 'v_to_device'),
                self._transfer(d, xp, 'dedtau_to_device'))

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        from ciderpress.gpu import policy
        with policy(self._device):
            return self._calculate_paw_correction(setup, D_sp, dH_sp, a)

    def _calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        if not self._paw_ready:
            raise RuntimeError('New CIDER PAW correction requires current grid XC')
        engine = self.engine
        if a not in engine.dens.D_asp or setup is not self.setups[a]:
            raise ValueError('New CIDER PAW atom/setup mismatch')
        if not np.array_equal(D_sp, engine.dens.D_asp[a]):
            raise ValueError('New CIDER PAW density differs from grid evaluation')
        if dH_sp is not None:
            dH_sp += engine.dH_asp_tmp[a]
        return float(engine.E_a_tmp[a])

    def validate_force_calculation(self):
        if not self._paw_ready or self._density is None:
            raise RuntimeError('New CIDER forces require a fresh grid/PAW evaluation')
        density = self._density
        if density.nt_sR.dims != (1,) or density.taut_sR is None:
            raise NotImplementedError('New CIDER forces require one spin and tau')
        if density.nt_sR.desc.comm.size != 1:
            raise NotImplementedError('New CIDER forces require serial domains')
        if not np.array_equal(self.engine.spos_ac, self._relpos):
            raise RuntimeError('New CIDER force geometry is stale')

    def add_forces(self, *args, **kwargs):
        raise NotImplementedError('Use the New CIDER explicit force_contribution hook')

    def force_contribution(self, density, interpolate):
        """Root-owned explicit PASDW force in Ha/Bohr, with private scratch.

        New owns projector/dH, core n/tau, Hartree and vbar terms.  In
        particular this never calls Classic MGGA.add_forces or _get_taut.
        No communicator other than the engine's private SELF is entered.
        """
        from ciderpress.gpu import policy
        self.validate_force_calculation()
        if density is not self._density:
            raise RuntimeError('New CIDER force density is stale')
        with policy(self._device):
            # Use exactly the SCF interpolation and density packing.  Copies
            # protect the caller even if a numerical routine raises midway.
            fields = [interpolate(density.nt_sR), interpolate(density.taut_sR)]
            for field in fields:
                device = getattr(getattr(field, 'data', None), 'device', None)
                device_id = getattr(device, 'id', None)
                if self._device is not None and device_id is not None and device_id != self._device:
                    raise ValueError('New CIDER input differs from its recorded CUDA device')
            n, tau = [field.to_xp(np) for field in fields]
            for array in (n, tau):
                if array.dims != (1,) or any(
                    not np.array_equal(getattr(array.desc, key), getattr(self.grid, key))
                    for key in ('cell_cv', 'size_c', 'pbc_c')
                ):
                    raise NotImplementedError('New CIDER force grid/spin changed')
                if not np.isfinite(array.data).all():
                    raise ValueError('New CIDER forces require finite inputs')
            if _force_input_digest(n.data, tau.data) != self.engine._force_input_digest:
                raise RuntimeError('New CIDER force n/tau differs from the current XC state')
            matrices = density.D_asii.to_xp(np)
            if (matrices.layout.atomdist.comm.size != 1 or
                    np.any(matrices.layout.atomdist.rank_a != 0) or
                    set(matrices.keys()) != set(range(len(self.setups)))):
                raise NotImplementedError('New CIDER force atom distribution changed')
            packed = self.engine.setups.empty_atomic_matrix(1, self.engine.atom_partition)
            for a, D_sii in matrices.items():
                packed[a][:] = [pack_density(D_ii.real) for D_ii in D_sii]
                if not np.isfinite(packed[a]).all():
                    raise ValueError('New CIDER forces require finite D')
                if not np.array_equal(packed[a], self.engine.dens.D_asp[a]):
                    raise RuntimeError('New CIDER force D differs from the current XC state')
            if not hasattr(self, '_force_engine'):
                self._force_engine = _ForceEngine(self.param, self.grid, self.grad_v,
                                                   self.setups, self._relpos)
            engine = self._force_engine
            try:
                return engine.forces(n.data.copy(), tau.data.copy(), packed)
            except BaseException:
                # Never reuse a partially evaluated private engine.
                del self._force_engine
                raise

    def stress_contribution(self, *args, **kwargs):
        self.validate_stress_calculation()

    def validate_stress_calculation(self):
        raise NotImplementedError('New CIDER stress is not implemented')


def _force_input_digest(n, tau):
    return tuple(hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).digest()
                 for a in (n, tau))


class _ForceEngine(_Engine):
    """Property-only mutable PAW/FFT state; never aliases SCF PAW caches."""

    def __init__(self, param, grid, grad_v, setups, relpos):
        super().__init__(param, grid, grad_v)
        self.setups = copy(setups)
        copies = {}
        for a, setup in enumerate(setups):
            if id(setup) not in copies:
                clone = copy(setup)
                clone.cider_contribs = None
                copies[id(setup)] = clone
            self.setups[a] = copies[id(setup)]
        self.spos_ac = np.array(relpos, copy=True)
        self.atom_partition = AtomPartition(self.world, np.zeros(len(setups), int))
        self.dens = SimpleNamespace(setups=self.setups)

    def forces(self, n, tau, packed):
        self.dens.nt_sg = n
        self.dens.D_asp = packed
        self.force_terms = {}
        e, rho, dedrho = self._get_cider_inputs(n, tau)
        forces = self.calc_cider(e, rho, dedrho, grad_mode=1)
        result = np.zeros((len(self.setups), 3))
        for a, force in forces.items():
            result[a] = force
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite New CIDER explicit force')
        return result

    def _calculate_paw_energy_and_potential(self, grad_mode=0):
        result = super()._calculate_paw_energy_and_potential(grad_mode)
        if grad_mode == 1:
            self.force_terms['Q'] = np.array([
                np.einsum('vsiq,siq->v', self._save_v_avsiq[a], self.c_asiq[a])
                for a in range(len(self.setups))])
        return result

    def get_fft_work(self, s, pot=False, grad_mode=0):
        super().get_fft_work(s, pot=pot, grad_mode=grad_mode)
        if pot and grad_mode == 1 and s == self.nspin - 1:
            self.force_terms['B'] = np.array([
                np.einsum('vsiq,siq->v', self._save_v_avsiq[a], self._save_c_asiq[a])
                for a in range(len(self.setups))])

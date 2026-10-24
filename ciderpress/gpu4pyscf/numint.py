"""Full-grid gpu4pyscf NumInt backed by stage-1 MolecularXC; no CPU XC."""
from time import perf_counter
import numpy as np
from .native import preload_cpu, require_version
preload_cpu()
require_version()
import cupy as cp
from gpu4pyscf.dft import numint as gpu_numint
from ciderpress.gpu import policy
from ciderpress.gpu.lcao_nldf import MolecularXC


def molecule_identity(mol):
    # Mole.dumps also contains mutable logging/cache attributes. Freeze only
    # the physical AO/Hamiltonian definition, excluding libcint scratch slots.
    return (mol.spin, mol.charge, mol.cart, mol._atm.tobytes(), mol._bas.tobytes(),
            mol._env[20:].tobytes(), mol._ecpbas.tobytes())


class CiderNumInt(gpu_numint.NumInt):
    def __init__(self, cpu_shell, *, schedule, tile_size, ao_block=4096):
        super().__init__()
        self.cpu_shell = cpu_shell
        self.schedule, self.tile_size, self.ao_block = schedule, tile_size, ao_block
        self.engine = None
        self.calls = []
        self.capture_density = False
        self.last_rho = None
        self._mol_identity = molecule_identity(cpu_shell.mol)
        self._grid = None

    def _remember(self, mol, grids):
        if cp.cuda.runtime.getDeviceCount() != 1 or gpu_numint.num_devices != 1:
            raise NotImplementedError("One full grid on exactly one GPU is required")
        if molecule_identity(mol) != self._mol_identity:
            raise ValueError("Molecule changed; reconstruct the CIDER calculator")
        if self._grid is None:
            for key in ("coords", "weights"):
                np.testing.assert_array_equal(cp.asnumpy(getattr(grids, key)),
                                              getattr(self.cpu_shell.grids, key))
            self._grid = grids
            self._coords = grids.coords.copy()
            self._weights = grids.weights.copy()
        if (grids is not self._grid or grids.coords is None or grids.weights is None or grids.coords.shape != self._coords.shape or
                grids.weights.shape != self._weights.shape or
                not bool(cp.all(grids.coords == self._coords)) or
                not bool(cp.all(grids.weights == self._weights))):
            raise ValueError("Grid changed; reconstruct the CIDER calculator")

    def nr_rks(self, mol, grids, *args, **kwargs):
        self._remember(mol, grids)
        return super().nr_rks(mol, grids, *args, **kwargs)

    def nr_uks(self, mol, grids, *args, **kwargs):
        self._remember(mol, grids)
        return super().nr_uks(mol, grids, *args, **kwargs)

    def block_loop(self, *args, **kwargs):
        kwargs["blksize"] = min(kwargs.get("blksize") or self.ao_block, self.ao_block)
        yield from super().block_loop(*args, **kwargs)

    def eval_xc_eff(self, xc_code, rho, deriv=1, *, omega=None, xctype=None,
                    spin=None, work=None, inplace=False):
        if deriv != 1 or omega not in (None, 0) or inplace or xctype not in (None, "MGGA"):
            raise NotImplementedError("CIDER GPU supports ground-state MGGA first derivatives only")
        ns = 2 if rho.ndim == 3 else 1
        shape = (2, 5, len(self.cpu_shell.grids.weights)) if ns == 2 else (5, len(self.cpu_shell.grids.weights))
        if self._grid is None or rho.shape != shape or ns != self.cpu_shell._numint.nldfgen.plan.nspin:
            raise ValueError("Expected the complete bound CiderGrids density with matching spin")
        if spin is not None and spin != ns-1:
            raise ValueError("Spin/rho mismatch")
        cp.cuda.get_current_stream().synchronize()
        start = perf_counter()
        setup_s = 0.
        if self.engine is None:
            with policy(cp.cuda.Device().id):
                self.engine = MolecularXC(self.cpu_shell._numint, self.cpu_shell.grids.weights,
                                           schedule=self.schedule, tile_size=self.tile_size)
            cp.cuda.get_current_stream().synchronize()
            setup_s = perf_counter()-start
        if self.capture_density:
            self.last_rho = rho.copy()
        exc, vxc = self.engine.evaluate(rho)
        cp.cuda.get_current_stream().synchronize()
        self.calls.append(dict(setup_s=setup_s, xc_s=perf_counter()-start-setup_s))
        return exc, vxc, None, None

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError("No CIDER GPU response, forces, NLC, or conversion")

    eval_xc = _unsupported
    nr_rks_fxc = nr_uks_fxc = nr_rks_fxc_st = _unsupported
    cache_xc_kernel = cache_xc_kernel1 = _unsupported
    nr_nlc_vxc = nr_rks_fxc_st = _unsupported
    to_gpu = to_cpu = _unsupported

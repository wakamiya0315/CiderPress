# Experimental molecular CIDER GPU SCF

```python
from ciderpress.gpu4pyscf import make_cider_calc_gpu  # before gpu4pyscf
mf = make_cider_calc_gpu(mol, level=3, df="def2-universal-jfit")
from ciderpress.gpu4pyscf.scf import make_pbe_calc
pbe = make_pbe_calc(mf)  # same grid, full AO space and immutable J setup
pbe.kernel()
mf.kernel(dm0=pbe.make_rdm1())
print(mf.cider_memory_policy)
```

The return value is a gpu4pyscf RKS/UKS calculator. Pass
`unrestricted=True` for spin-zero UKS. The only supported model is the named
`CIDER26XCSURFSCI`, with `xmix=1`, `xkernel=None`, `ckernel=None`. One visible
CUDA GPU and gpu4pyscf **exactly 1.9.0** are required. Unsupported options,
models, force/response requests and calculator conversions raise. In
particular, this is a new-molecule factory; it does not convert a CPU CIDER
object through PySCF's generic `to_gpu` method.

`df=None` selects direct J. The only DF option is explicit
`"def2-universal-jfit"` with `only_dfj=True`. A per-instance adapter removes
the extra Mole argument at gpu4pyscf 1.9.0's `df/df_jk.py:226` call to
`df/df.py:118`. Integral and metric arithmetic is unchanged. All positive
overlap eigenvectors are retained to match CPU PySCF's full AO space.

`deterministic_j=True` is the default. DF uses gpu4pyscf's resident CDERI
with auxiliary blocks of 128; direct J uses an immutable four-center ERI
tensor constructed on CPU at setup and contracted on GPU each step. This
direct path is limited to small systems by its O(nao^4) storage and rejects
excessive host/device estimates. No integral quantization is used. cuSOLVER
deterministic mode is enabled and cuBLAS atomics are disabled on the active
handles. `deterministic_j=False` is for stock-J diagnostics; it does not
provide the strict residency/reproducibility protocol.

One-electron hcore (including scalar ECP) and overlap integrals are explicitly
constructed with CPU PySCF during setup and uploaded once. All subsequent
calls return GPU copies of those immutable matrices. This avoids repeated
GPU hcore summation differences that diffuse overlap spaces amplify.
No CPU CIDER evaluation occurs in SCF. `make_pbe_calc` shares immutable
matrices and J data with the CIDER calculator; run the two sequentially.

CPU native libxc consumers are loaded and checked before gpu4pyscf. The
authoritative CiderGrids use alignment 1, no sorting or density pruning.
The pure-MGGA `R2SCAN` string is only gpu4pyscf's dispatch facade; the bound
full-grid NumInt evaluates GPU CIDER, never r2SCAN or CPU XC. Default
convergence is energy 1e-9 Ha / gradient 1e-6. Reconstruct the calculator
after changing molecule, basis, model, numerical settings or grid.

`schedule="auto"` compares a conservative whole-SCF byte estimate with the
available device memory and 90% of capacity. It includes NLDF table/work
storage, fixed 4096-point AO/density/contraction blocks, matrices, DIIS,
J/DF work and a 4 GB reserve. The estimate uses the stage-1 AD7 measurements;
it is not an allocator guarantee. `resident`/`tiled` can be requested
explicitly, subject to that same estimate. `tile_size=8192` controls the
tiled interpolation. Actual whole-SCF device/RSS peaks must be measured
for each workload. Host spilling in DIIS is disabled; automatic CPU
checkpointing is disabled (`chkfile=None`) because it copies MOs each cycle.
Explicit final-state export remains the caller's responsibility.

The stage-1 model/plan reject list remains in effect: no SDMX, fractional
Laplacian, hybrid features, Gaussian plans, i/ij/k NLDF, variable normalizers,
nonzero smooth floor, response, forces, or multi-GPU. One setup stream must
be reused. No stage-1 arithmetic is changed here. Raw features retain the
known CPU ATC rounding floor; energy/Vxc/SCF gates are separate.

`SCFFeatureGenerator` in `descriptors.py` is the **E0 measurement**: raw
features on the exact SCF quadrature with its onsite_direct interpolator.
It is not a GPU train_gen implementation, and does not establish training
acceptability. It supplies the official `ft.from_pyscf(feature_generator=)`
contract, with explicit transfers only at export. E1 remains the official
CPU feature generator and writer. See the bench `docs/results/pyg-scf.md`
for measured gates, differences, memory, costs and open issues.

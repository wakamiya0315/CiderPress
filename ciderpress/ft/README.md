# Fine-tuning a frozen CIDER model

`ciderpress.ft` wraps `MOLGP2`, `DFTKernel2` and their differentiable RBF
kernels. It fits a new GP around the **frozen full-XC mean**. It does not
recover the original model's training data or Bayesian posterior.

```python
from ciderpress.ft import FineTuner, ReactionSet, Constraints

# records = {system_id: export_training_data(converged_new_gpaw_calc), ...}
# Attach compositions: the GPAW exporter does not include atomic identities.
data = ReactionSet.from_records(
    records, compositions={"Ni_O": "Ni3O", "Ni": "Ni3", "O2": "O2"}
)
data.add_reaction(
    "O/Ni",
    {"Ni_O": 1, "Ni": -1, "O2": -0.5},
    reference=0.4675,
    unit="eV",
    noise=0.03,
    group="Ni",
)
protect_set = ReactionSet.from_records(molecular_records)
ft = FineTuner(
    base="CIDER26XCSURFSCI",
    n_control=200,
    constraints=Constraints(molecules=protect_set, max_abs_correction=0.5),
)
ft.fit(data)
report = ft.evaluate(test_data)  # separate, balanced held-out observations
ft.save("CIDER26SS_FT.yaml")
print(report["groups"], ft.fit_info)
```

## Records and units

Records are mappings with `desc=(nspin,nfeature,npoint)`,
`rho_data=(nspin,5,npoint)`, `wt=(npoint,)`, `e_tot_orig`, `exc_orig`, and
optionally `exc_core` (default zero for all-electron molecules).
`rho_data` is **nspin times physical** `[n, dx n, dy n, dz n, tau]`, with
tau including the factor 1/2. Descriptors are unnormalized, with SL then NLDF
blocks in the frozen model's ordering. Densities, gradients, tau and volume
weights use atomic units. Record energies are always Hartree. Explicit
non-Hartree `energy_unit` or `units['energy']` declarations are rejected;
undeclared units follow the existing exporter contract. A wrong numeric unit
without metadata cannot be detected automatically.

The signed PAW weights must be retained, including negative weights. GPAW's
`exc_core` reference constant is added **once** to the full XC, and never to
the residual integral. Do not replace `exc_orig` with a replayed XC while
leaving `e_tot_orig` unchanged: these two scalars must describe the same
fixed density, orbitals and original functional. Use converged, qualified
states and consistent geometry, setups, magnetism and energy conventions.

`ReactionSet.from_hdf5(directories, system_ids, compositions=...)` reads the
REF/SL/NLDF layout written by `ciderpress.gpaw.new_export.write_training_data`
or the GPAW-free `ciderpress.ft.write_records`. An unused pointwise `val`
reference is not required. Every system participating
in a reaction needs `composition` (formula or element/count mapping),
`symbols`, or `numbers`. Elemental imbalance raises an error. For charged
reactions, electronic/charge balance remains the caller's responsibility.

Reaction references and noise standard deviations share the explicitly given
`unit`: `eV`, `kcal/mol` (also `kcal per mol`), or `Hartree`/`Ha`. An explicit
`noise_unit` may override it. All conversions happen before calling the engine:
`0.03 eV` noise is about `0.00110 Ha`, **not** `0.03 Ha`. Groups are arbitrary
strings. `evaluate(unit=...)` returns MAE and ME, with ME defined as prediction
minus reference, globally and per group.

For molecules, a small adapter uses the existing analyzer and descriptor tools:

```python
from ciderpress.ft import from_pyscf

# mf: a converged PySCF RKS/UKS pure KS calculation, for example PBE
record = from_pyscf(mf, base="CIDER26XCSURFSCI", grids_level=3)
protect_set = ReactionSet.from_records({"H2": record})
```

It exports SL+NLDF, raw density, all-electron weights, energies and atomic
numbers. HF, hybrids, NLC and dispersion are refused by this convenience
adapter. It uses the analyzer's separate grid, so grid convergence and
small-energy quadrature agreement still need qualification. This default
behavior, including `grids_level`, is unchanged.

To retain the exact SCF quadrature, including custom pruning and point order:

```python
from ciderpress.ft import write_records

record = from_pyscf(
    mf, grids=mf.grids,
    provenance={
        "df": {"enabled": True, "auxbasis": "def2-universal-jfit"},
        "grid": {"source": "SCF", "level": mf.grids.level},
        "model_sha256": model_sha,
        "versions": versions,
    },
)
directories = write_records({"H2": record}, "training")
data = ReactionSet.from_hdf5(directories, ["H2"])
# Equivalently: directories = data.to_hdf5("training-copy")
```

Use actual DF/grid/version information from the generating calculation in
`provenance`; no provenance is inferred automatically. Its dictionary has no
required keys and does not affect fitting. It is stored as JSON in
`REF/<id>.hdf5:/train_data/provenance_json` and restored to `provenance` by
`ReactionSet.from_hdf5`. JSON values (nested dictionaries/lists, strings,
numbers, booleans and `None`) are supported; NumPy scalars/arrays become JSON
scalars/lists. Python objects such as grid-pruning callables should be named
explicitly instead of serialized. Original atomic numbers/order and
`feature_settings_yaml` from `from_pyscf` are retained. For custom records,
provide `nfeat_sl` to split the descriptor blocks and include feature settings
when available. Missing atomic numbers are derived from composition/symbols;
composition alone does not specify atomic order.

The writer returns all six directory keys expected by `MOLGP2.load_data`,
with unused feature families set to `None`. Both readers retain identical
`desc`, `rho_data` and signed `wt` arrays. Existing files for supplied IDs are
replaced. Reactions/labels in a `ReactionSet` are not serialized. Pointwise
`val` is omitted by default even if present in a record; `include_val=True`
writes an existing finite pointwise array. No EXX computation is performed.
`MOLGP2.load_data(directories, id, False)` can read files without `val`;
the separate legacy `store_mol_covs` reference path still requires it.

Alternatively pass `grid_spec={"level": 3, "prune": None, ...}` to build a
grid, or `analyzer=ana` to reuse a prepared matching-density analyzer. An
analyzer is mutually exclusive with grid options; a grid object is mutually
exclusive with `grid_spec`/`grids_level`. The supplied grids must already be
built. Explicit-grid export compares **both coordinates and weights**, not
just the grid level. Identical grids reuse saved `mf.e_tot` and
`mf.scf_summary["exc"]` without an extra CPU energy/XC evaluation. Different
grids reevaluate only XC using a private copy of the generating integrator
and adjust `e_tot_orig` by the same XC difference, preserving `e_tot-exc`.
The saved SCF scalars must be present. CIDER grid specifications retain the
SCF's `CiderGrids` family; supplied grids on a different quadrature must also
be compatible with the generating XC integrator.

`feature_generator=callable` replaces `get_descriptors`. The callable receives
`(analyzer, settings)` for SL, NLDF and raw density (`settings="l"`), and must
return the same raw arrays/order/spin scaling. For example, a wrapper can
pass an audited inner NLDF grid and convolution settings only on the NLDF
call. This controls descriptor generation separately from the outer sampling
grid. These options support CPU PySCF objects; transferring a GPU SCF state
and qualifying its density, grid and energy remain the caller's responsibility.

General custom exporters may supply the same record schema. Feature count is checked;
`feature_settings_yaml` is compared when supplied by this adapter. Legacy
GPAW records without that field cannot prove transform/settings identity;
the caller must ensure they used the frozen model's settings. The SCF model
digest need not equal the frozen model digest during density/refit cycles.

## Mean, residual and regularisation

For each density the mean is recomputed as

```
E0 = e_tot_orig - exc_orig + integral(frozen_base_xc) + exc_core
residual = reference_reaction - sum(stoichiometry * E0)
```

`integral` above excludes the separately written core constant. The original
base kernels, coefficients, control points, settings and baselines are copied
unchanged. By default, a SEP RBF multiplied by `LDA_X` is added on the base's
exchange transforms (`components="x"`). `components="c"` instead adds an NPOL
RBF multiplied by `GGA_C_PBE` on its correlation transforms; `"xc"` adds both.
These describe the residual, even when labels are total-XC reaction energies.
NPOL averages the two spin feature vectors before transformation, limiting
the correlation spin dependence it can learn. Every residual kernel has
`additive_baseline=None`, so PBE is not added again.

Length scales are inherited from the corresponding frozen RBF. `prior_scale`
(default 0.1) is the dimensionless enhancement standard deviation of the new
GP, providing RKHS regularisation through the existing solver. There is no
hyperparameter optimisation or uncertainty API. `jitter=1e-12` is the numerical
diagonal term used by the engine in control and observation space; do not use
noise scales far below its square root in Hartree without a conditioning audit.
The wrapper integrates the kernel explicitly with the **inference** cutoff
`rhocut=1e-8`, including spin scaling, instead of the training loader's fixed
1e-6 mask. `blocksize` bounds integration memory. Integration is CPU-only.

Control candidates come only from systems used in training observations and
the supplied protection set, plus UEG points. Deterministic, equal-budget
subsampling per system bounds the pool (`max_candidates=1500`); the engine's
dense pivoted Cholesky then selects at most `n_control` points per component.
This is a selection from that bounded pool, not a pivot over every grid point.
Dense pool memory grows quadratically; raise the cap deliberately.

Optional `candidate_groups={record_id: group_name}` and
`group_fractions={group_name: fraction}` stratify that pool. Supply both;
positive fractions must sum to one and every training/protection ID must be
mapped. IDs are the original `ReactionSet` IDs, including those supplied in
`constraints.molecules`. Each group receives `floor(max_candidates * fraction)`
candidate rows, with an equal candidate-row budget of `group_limit // record_count`
per record. The grid budget divides that by the number of descriptor rows
per point (one or two, depending on kernel and spin). Zero budgets skip that
group; unused capacity is not redistributed. UEG records are appended
separately with their legacy per-record sampling budget, so their rows are
additional to `max_candidates` in grouped mode. Sampling remains deterministic
linspace and the kernel's pivoted Cholesky is unchanged on both devices.
Omitting both options preserves the legacy arithmetic exactly.

The configuration, realized record budgets and selected control origins are
saved in `fit_info` and model YAML metadata. Identical descriptors can have
multiple origins: attribution uses the first pool occurrence and records the
alternative `possible_groups` explicitly. Group fractions constrain candidate
capacity, not the fractions surviving Cholesky reduction.

`Constraints` defaults, in eV:

| Setting | Default | Meaning |
| --- | --- | --- |
| `ueg` | `True` | Zero correction per electron at 9 densities from 1e-4 to 10 Bohr^-3, zeta=0, 0.5, 0.99 |
| `ueg_noise` | 0.001 | Soft UEG standard deviation per electron |
| `molecules` | `None` | Supplied molecular protection records or references |
| `molecular_noise` | 0.01 | Zero-correction standard deviation per supplied molecule |
| `absolute_noise` | 0.05 | Zero-centred absolute correction penalty, per atom, on training/protection records |
| `max_abs_correction` | 0.5 | Hard absolute correction cap per atom, on those same records |

`Constraints(unit=...)` changes the units of **all** these energy scales.
Density and polarisation grids can be customised. With an observation-free
protection set, corrections are anchored to zero. With supplied training
observations, their high-level references are fitted instead. For example:

```python
protect_set.add_energy("H2", reference=high_level_H2_Ha, unit="Hartree", noise=0.001)
# Or add balanced molecular atomisation/reaction references with add_reaction.
```

If the fitted absolute correction exceeds the hard cap, all residual
coefficients are uniformly scaled down. `fit_info['absolute_scale'] < 1`
records this, and `evaluate` exposes the resulting loss of reaction accuracy.
This conservative operation preserves the shape of the residual and shrinks
zero anchors; it is not an optimisation of the full constrained loss.
`max_abs_before_ha_per_atom`, `max_abs_after_ha_per_atom` and
`max_ueg_ha_per_electron` are measured training/protection diagnostics.
Use `None` for either absolute setting to disable that safeguard, or `ueg=False`
to disable UEG points, only after checking absolute energies and potentials.

These are **finite-record** safeguards. They cannot bound arbitrary unseen
densities, XC potentials, atomic limits not represented in the protection set,
forces or SCF trajectories. Supply atoms and diverse molecules as protection
where those limits matter. The p11 +513 eV correction demonstrates why a small
reaction MAE alone is insufficient; the new constraints are not an accuracy or
SCF-stability guarantee.

## Splits and density/refit cycles

`add_reaction(..., split="train"|"validation"|"test")` labels observations;
`fit` uses only training observations and their records. `select(split="test")`
returns held-out observations. An explicit `evaluate(data)` evaluates all
observations in the passed set. Unused/held-out-only systems never select
controls or receive absolute penalties. A system shared by a training and a
test reaction necessarily participates in both; split by chemical groups to
control that overlap.

```python
parts = data.split_by_group(validation=["Co"], test=["Ni"])
ft.fit(parts["train"])
report = ft.evaluate(parts["test"])
for group, train, test in data.leave_one_group_out():
    fold = FineTuner(constraints=Constraints(molecules=protect_set))
    fold.fit(train)
    print(group, fold.evaluate(test))
```

Empty split parts are `None`. Do not put held-out molecules into a protection
pool when claiming fully independent validation. Every refit starts from the
same frozen base, resets observations and reselects controls; identical input
records produce identical fits.

```python
def recompute(model, previous_data, step):
    # User code: build calculators with model, converge and qualify each SCF,
    # then export records with SAME-density total and XC energies in Hartree.
    # Return all the original IDs. Raise on failed/unstable SCF.
    return updated_records


history = ft.fit_self_consistent(data, recompute, n_steps=2)
```

The helper first fits the starting records, then calls the hook for steps 1
and 2 and refits after each. References, stoichiometry, groups and splits stay
fixed. The mean always uses the original frozen base, even though the hook's
record energies came from the current corrected model. Protection records
remain fixed unless the hook explicitly updates `ft.constraints.molecules`.
The helper imposes no convergence criterion and starts no calculation itself.

## Saving and using the result

Saved files are ordinary trusted-object `MappedXC2` YAML with small
`ft_metadata` provenance. Load only trusted YAML. They include base and residual
inference kernels, not training records or posterior state. `FineTuner.load`
restores prediction/evaluation; a fresh fit is possible, but molecular records
must be supplied again if the saved fit used them.

```python
from ciderpress.gpaw.new_backend import CiderXC
from ciderpress.gpaw.calculator import get_cider_functional
from ciderpress.pyscf.dft import make_cider_calc

xc_new = CiderXC("CIDER26SS_FT.yaml")  # New GPAW, CPU or CUDA
xc_classic = get_cider_functional(
    "CIDER26SS_FT.yaml", xmix=1.0, xkernel=None, ckernel=None
)
mf = make_cider_calc(pyscf_ks, "CIDER26SS_FT.yaml")
restored = FineTuner.load("CIDER26SS_FT.yaml")
```

Full-XC composition must remain `xmix=1`, with no additional X/C functional.
Saving verifies the GPU semilocal **representation compatibility predicate**;
the tests check it again after YAML loading. This is not a real-GPU numerical
or SCF check. Qualify CPU/GPU energies, potentials, forces and SCF stability
before scientific use of a fitted model.

## Verification

### Optional CUDA fitting and prediction

This path is experimental: the p12-ftgpu real-record audit found FP64
quadrature differences amplified beyond its 1e-10 coefficient/prediction gate.
Use the CPU default for that pilot until the strict fit gate is qualified.

`FineTuner(..., device="gpu")` selects CuPy FP64 grid quadratures. The default
`device="cpu"` retains the original arithmetic, control selection, dense solve
and YAML format. `FineTuner.load(path, device="gpu")` selects the device again;
device choice is an execution option and is not stored in the model. Loading
without that option always uses CPU, including a model fitted on GPU.

The GPU route reuses `ciderpress.gpu.semilocal.evaluate_device` (the mapped
model pipeline used by New GPAW), its feature maps and libxc baselines. FT's
residual covariance uses a blocked ordinary RBF distance matrix on device.
`blocksize` bounds the number of grid points; covariance workspace scales as
`O(blocksize * nspin * n_control)`, independently of total grid size. Input
normalization, deterministic candidate sampling/control selection and the
small dense solve remain on CPU. Signed PAW weights, spin scaling, density
cutoffs and the one-time core constant are retained.

Explicit GPU selection requires real CuPy and a CUDA device, and raises on
unsupported kernels or runtime errors. FT calls the resident pipeline directly,
so `CIDER_GPU_KERNELS` does not disable this explicit request. CPU/GPU reduction
roundoff can differ; validate the application and conditioning of each fit.
GPU tests in `test_gpu.py` skip when CuPy/CUDA is unavailable. The optional
`benchmark_gpu.py` consumes a frozen Stage C catalog using the bench's unchanged
`harness.p12_ftC`; it writes timings and comparison arrays without running SCF.

Run `python -m pytest ciderpress/ft/tests -q` in a built CiderPress environment.
Tests cover known-correction recovery (SEP/NPOL, one/two spins), refit
invariance, units, splits, molecular/UEG/absolute safeguards, SCF label
invariance, HDF5 loading, trusted-YAML round trips, signed exported-record
quadrature, three calculator factories and PySCF RKS/UKS molecular export.
The real-record test uses `P12_FT_EXPORT_DIR` (REF/SL/NLDF with `sample.hdf5`)
or generates a small CPU New-GPAW He calculation when no record is available.
GPAW-dependent checks require GPAW and PAW datasets. The historical smoke
regression additionally uses `P12_FT_SMOKE_DIR` with p11-cmr Pt-O, Pt-clean
and gas-O2 exports; otherwise it skips explicitly. It tests the available
Pt/O subset, not reproduction of the full nine-state +513 eV fit.

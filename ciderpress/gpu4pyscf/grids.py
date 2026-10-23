"""Authoritative unsorted CiderGrids quadrature, and whole-SCF memory policy."""


def attach_grid(mf, cpu_grid):
    import cupy as cp
    from gpu4pyscf.dft.gen_grid import Grids
    grid = Grids(mf.mol)
    grid.level = cpu_grid.level
    grid.alignment = 1
    # Build stock metadata; coordinates/weights are the CIDER indexer's order.
    grid.build(sort_grids=False)
    grid.coords = cp.asarray(cpu_grid.coords)
    grid.weights = cp.asarray(cpu_grid.weights)
    mf.grids = grid
    return grid


def memory_estimate(mol, grid, nspin, *, df=False, ao_block=4096, tile_size=8192):
    """Conservative byte estimate, not an allocator-level guarantee.

    NLDF fit uses the stage-1 AD7 measurements with an all-pairs upper bound.
    Add AO/density/contraction, direct/DF J work, eigensolver/DIIS and 4 GB
    reserve. This is the stock-J estimate; the frontend adds strict-J tensor
    storage separately. AO block size is fixed by the pinned integrator.
    """
    n, g, a = mol.natm, len(grid.weights), mol.nao_nr()
    pair = g * max(0, n-1)
    module_base = 0.45e9 + 107e6*n
    resident = module_base + 1004*pair
    tiled = module_base + 60*pair + 8*121*min(g, tile_size)
    ao = 8*min(g, ao_block)*a*12
    density = 8*g*(60*nspin+34*8)
    matrices = 8*a*a*(40*nspin+24)
    jwork = 8*a*a*(16 if df else 32) + 1e9
    reserve = 4e9
    working = ao+density+matrices+jwork+reserve
    return dict(resident_module_bytes=int(resident), tiled_module_bytes=int(tiled),
                scf_working_bytes=int(working), ao_bytes=int(ao), density_bytes=int(density),
                matrix_bytes=int(matrices), j_bytes=int(jwork), reserve_bytes=int(reserve),
                resident_whole_bytes=int(resident+working), tiled_whole_bytes=int(tiled+working),
                ao_block=ao_block, tile_size=tile_size, pair_upper_bound=pair)


def choose_schedule(estimate, requested="auto", fraction=.90):
    import cupy as cp
    if requested not in ("auto", "resident", "tiled"):
        raise ValueError("schedule must be auto, resident or tiled")
    free, total = cp.cuda.runtime.memGetInfo()
    limit = int(min(free, fraction*total))
    choice = requested
    if choice == "auto":
        choice = "resident" if estimate["resident_whole_bytes"] <= limit else "tiled"
    if estimate[choice+"_whole_bytes"] > limit:
        raise MemoryError(f"Estimated whole SCF {choice}={estimate[choice+'_whole_bytes']} exceeds budget {limit}")
    return choice, dict(estimate, selected=choice, requested=requested,
                        available_bytes=free, total_bytes=total, limit_bytes=limit)

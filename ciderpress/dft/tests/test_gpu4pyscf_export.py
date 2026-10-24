"""Real CUDA molecular E0 export, schema guards and signed FT round trips."""
from copy import deepcopy
import importlib.util
import json
import numpy as np
import pytest


@pytest.fixture(scope="module")
def cuda():
    if any(importlib.util.find_spec(n) is None for n in ("cupy", "gpu4pyscf")):
        pytest.skip("Real CUDA CuPy and gpu4pyscf required")
    from ciderpress.gpu4pyscf.native import preload_cpu
    preload_cpu()
    from ciderpress.gpu import _real_cupy
    cp = _real_cupy()
    if cp is None or cp.cuda.runtime.getDeviceCount() != 1:
        pytest.skip("Exactly one real CUDA GPU required")
    return cp


def context():
    return dict(state=dict(id="test", accepted=True, resolved=True,
                    qualification=dict(open_shell_accepted=True, method="small-basis API test only")),
                source_geometry=dict(database="test", subset="test", system="H/H2",
                    files={"inline":dict(sha256="0"*64)}),
                provenance=dict(ciderpress=dict(commit="test"), bench=dict(commit="test")))


@pytest.fixture(params=[(0, None), (0, "def2-universal-jfit"), (1, None)])
def exported(cuda, request):
    from pyscf import gto
    from ciderpress.gpu4pyscf import make_cider_calc_gpu
    from ciderpress.gpu4pyscf.scf import make_pbe_calc
    from ciderpress.gpu4pyscf.export import from_gpu_scf
    spin, df = request.param
    mol = gto.M(atom="H 0 0 0" if spin else "H 0 0 0; H 0 0 .75",
                spin=spin, basis="sto-3g", verbose=0)
    mf = make_cider_calc_gpu(mol, level=0, df=df)
    pbe = make_pbe_calc(mf)
    pbe.kernel()
    mf.kernel(dm0=pbe.make_rdm1())
    assert mf.converged and pbe.converged
    return mf, from_gpu_scf(mf, **context())


def test_official_roundtrip(cuda, exported, tmp_path):
    from ciderpress.gpu4pyscf.export import write_e0_records, array_sha256
    from ciderpress.ft import ReactionSet
    from ciderpress.models.train import MOLGP2
    from ciderpress.ft.records import integrate, validate_record
    from ciderpress.dft.model_utils import load_cider_model
    from pyscf import lib
    mf, record = exported
    assert record["desc"].shape[:2] == (2 if mf.mol.spin else 1, 6)
    assert record["rho_data"].shape[1] == 5
    assert record["nfeat_sl"] == 3 and record["exc_core"] == 0 and "val" not in record
    np.testing.assert_array_equal(record["wt"], cuda.asnumpy(mf.grids.weights))
    assert abs(integrate(load_cider_model("CIDER26XCSURFSCI"), record, rhocut=1e-9)
               - record["exc_orig"]) < 1e-7
    dirs = write_e0_records({"test":record}, tmp_path)
    loaded = ReactionSet.from_hdf5(dirs,["test"]).records["test"]
    legacy = validate_record(MOLGP2.load_data(dirs,"test",False))
    for value in (loaded,legacy):
        for key in ("desc", "rho_data", "wt", "numbers"):
            np.testing.assert_array_equal(record[key],value[key])
    assert loaded["provenance"] == record["provenance"]
    assert "val" not in lib.chkfile.load(dirs["REF"]+"/test.hdf5","train_data")
    manifest = json.loads((tmp_path/"manifests/test.json").read_text())
    assert manifest["arrays"]["wt"]["sha256"] == array_sha256(record["wt"])
    assert len(manifest["files"]) == 3


def test_fail_closed(cuda, exported, tmp_path):
    from ciderpress.gpu4pyscf.export import from_gpu_scf, cpu_shell_from_gpu, write_e0_records
    mf, record = exported
    opts = context()
    opts["state"]["resolved"] = False
    with pytest.raises(ValueError,match="unresolved"):
        from_gpu_scf(mf,**opts)
    mf.converged = False
    with pytest.raises(ValueError,match="converged"):
        cpu_shell_from_gpu(mf)
    mf.converged = True
    saved = cpu_shell_from_gpu(mf).e0_snapshot
    saved.update(e_tot=float(mf.e_tot), exc_orig=float(mf.scf_summary["exc"]))
    cpu_shell_from_gpu(mf,saved_state=saved)
    saved["weights"][0] *= -1
    with pytest.raises(ValueError,match="saved weights"):
        cpu_shell_from_gpu(mf,saved_state=saved)
    original = mf.grids.coords[0,0].copy()
    mf.grids.coords[0,0] += 1e-5
    with pytest.raises(ValueError,match="coords"):
        cpu_shell_from_gpu(mf)
    mf.grids.coords[0,0] = original
    for key, value in (("desc",record["desc"][:,:5]),("numbers",np.array([])),
                       ("numbers",np.array([np.nan])),
                       ("wt",record["wt"]*0), ("rho_data",record["rho_data"]*np.nan)):
        invalid = dict(record, **{key:value})
        with pytest.raises(ValueError):
            write_e0_records({"invalid":invalid},tmp_path)


def test_signed_weights(cuda, exported, tmp_path):
    # Synthetic signed quadrature exercises writer/loader preservation even
    # when the physical small-molecule CiderGrids happens to be all positive.
    from ciderpress.gpu4pyscf.export import write_e0_records, array_sha256
    from ciderpress.ft import ReactionSet
    _, original = exported
    record = deepcopy(original)
    index = int(np.argmax(record["wt"]))
    record["wt"][index] = -abs(record["wt"][index])
    record["provenance"]["grid"]["weights_sha256"] = array_sha256(record["wt"])
    dirs = write_e0_records({"signed":record},tmp_path)
    loaded = ReactionSet.from_hdf5(dirs,["signed"]).records["signed"]
    assert loaded["wt"][index] < 0
    assert loaded["wt"].tobytes() == record["wt"].tobytes()


def test_ecp_physical_composition(cuda,tmp_path):
    from pyscf import gto
    from ciderpress.gpu4pyscf import make_cider_calc_gpu
    from ciderpress.gpu4pyscf.scf import make_pbe_calc
    from ciderpress.gpu4pyscf.export import from_gpu_scf,write_e0_records
    from ciderpress.ft import ReactionSet
    from ciderpress.models.train import MOLGP2
    mol=gto.M(atom="I 0 0 0; H 0 0 1.6",basis="def2-svp",ecp={"I":"def2-svp"},verbose=0)
    assert mol.atom_charges().tolist()==[25,1]
    mf=make_cider_calc_gpu(mol,level=0,df="def2-universal-jfit")
    pbe=make_pbe_calc(mf);pbe.kernel();mf.kernel(dm0=pbe.make_rdm1())
    assert pbe.converged and mf.converged
    record=from_gpu_scf(mf,**context())
    assert record["numbers"].tolist()==[53,1]
    assert record["provenance"]["composition"]["ecp_core_electrons"]==28
    dirs=write_e0_records({"test":record},tmp_path)
    assert ReactionSet.from_hdf5(dirs,["test"]).records["test"]["numbers"].tolist()==[53,1]
    assert MOLGP2.load_data(dirs,"test",False)["numbers"].tolist()==[53,1]

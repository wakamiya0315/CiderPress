"""Load and check CPU libxc consumers before loading gpu4pyscf."""
import ctypes
import sys

_handles = None


def preload_cpu():
    global _handles
    if _handles is not None:
        return _handles
    from pyscf.dft import libxc
    from ciderpress.lib import load_library
    libraries = {n: load_library(n) for n in (
        "libfft_wrapper", "libmcider", "libnumint", "libpwutil", "libsbt", "libxc_utils")}
    if sys.platform.startswith("linux"):
        class DlInfo(ctypes.Structure):
            _fields_ = [("file", ctypes.c_char_p), ("base", ctypes.c_void_p),
                        ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
        dl = ctypes.CDLL("libdl.so.2")
        dl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(DlInfo))
        dl.dladdr.restype = ctypes.c_int
        for handle in (libxc._itrf, libraries["libxc_utils"]):
            for symbol in ("xc_func_init", "xc_lda_exc_vxc"):
                info = DlInfo()
                if not dl.dladdr(ctypes.cast(getattr(handle, symbol), ctypes.c_void_p), ctypes.byref(info)):
                    raise RuntimeError("Cannot identify native libxc binding")
                if b"gpu4pyscf" in info.file:
                    raise RuntimeError("CPU libxc bound to GPU libxc; import ciderpress.gpu4pyscf first")
    _handles = libxc._itrf, libraries
    return _handles


def require_version():
    try:
        import gpu4pyscf
    except ModuleNotFoundError as exc:
        if exc.name != "gpu4pyscf":
            raise
        raise ImportError(
            "CIDER GPU SCF requires gpu4pyscf 1.9.0; install that optional "
            "dependency in a CUDA environment before calling make_cider_calc_gpu"
        ) from exc
    if gpu4pyscf.__version__ != "1.9.0":
        raise NotImplementedError("CIDER bridge and DF-J adapter require gpu4pyscf exactly 1.9.0")

"""Classic GPAW imports shared by GPAW 25.7 and the gpaw.old layout.

Resolve lazily so importing a grid helper does not load the calculator.
These are the original GPAW objects; the public gpaw.GPAW factory is untouched.
"""

from importlib import import_module

_CLASSIC_MODULES = {
    "GPAW": "calculator",
    "GridDescriptor": "grid_descriptor",
    "PWDescriptor": "pw.descriptor",
    "ReciprocalSpaceDensity": "pw.density",
}


def __getattr__(name):
    if name not in _CLASSIC_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = "gpaw.old." + _CLASSIC_MODULES[name]
    try:
        imported = import_module(module)
    except ModuleNotFoundError as error:
        # Do not hide missing dependencies inside an installed Classic module.
        if error.name != "gpaw.old" and error.name != module:
            raise
        imported = import_module("gpaw." + _CLASSIC_MODULES[name])
    value = getattr(imported, name)
    globals()[name] = value
    return value


def get_gpaw_comm(comm):
    """Return the GPAW MPIObject consumed by CiderPress's C interface.

    GPAW master returns an integer MPI_Comm address from get_c_object().
    The cgpaw communicator itself still has the original MPIObject layout.
    """
    obj = comm.get_c_object()
    if isinstance(obj, int):
        from gpaw import cgpaw

        # Unwrap GPAW's debug communicator, when present.
        while hasattr(comm, "comm"):
            comm = comm.comm
        if not isinstance(comm, cgpaw.Communicator):
            raise TypeError("CiderPress's MPI interface requires GPAW_MPI_BACKEND=cgpaw")
        return comm
    return obj

"""Ordered FP64 SLX, SLT, SLN and V maps used by CIDER26XCSURFSCI.

Each sample has one owner. Reverse maps update raw channels sequentially
in FeatureList order, including updates when maps share a raw channel.
"""
from functools import lru_cache

import numpy as np

from ciderpress.gpu import cupy


def signature(features, nraw=None):
    from ciderpress.dft.transform_data import SLXMap, SLTMap, SLNMap, VMap
    rows = []
    for f in features.feat_list:
        if type(f) is SLXMap:
            c = 2 * (3 * np.pi**2) ** (1.0 / 3)
            rows.append(("SLX", f.i, f.j, float(f.gamma / c**2)))
        elif type(f) is SLTMap:
            rows.append(("SLT", f.i, f.j, float(f.const)))
        elif type(f) is SLNMap:
            rows.append(("SLN", f.i, 0, float(3.0 * f.gamma / (4 * np.pi))))
        elif type(f) is VMap:
            rows.append(("V", f.i, 0, float(f.gamma), float(f.scale), float(f.center)))
        else:
            return None
    if nraw is not None:
        from operator import index
        normalized = []
        for row in rows:
            kind, i, j, *params = row
            if isinstance(i, (bool, np.bool_)) or isinstance(j, (bool, np.bool_)):
                return None
            try:
                i = index(i)
                j = index(j)
            except TypeError:
                return None
            if not (-nraw <= i < nraw and -nraw <= j < nraw):
                return None
            normalized.append((kind, i % nraw, j % nraw, *params))
        rows = normalized
    return tuple(rows)


@lru_cache(maxsize=128)
def _kernel(rows, nraw, backward):
    fragments = []
    for k, row in enumerate(rows):
        kind, i, j, c, *other = row
        x, y = f"x[{i}*n+p]", f"x[{j}*n+p]"
        c = repr(c)
        d = f"df[p*{len(rows)}+{k}]"
        a, b = f"out[{i}*n+p]", f"out[{j}*n+p]"
        if kind == "SLX":
            body = f"double rho = fmax({x}, 1e-10), r83 = pow(rho, 8.0/3), sigma = {c}*{y};"
            if backward:
                body += f"double r53 = (8.0/3)*pow(rho, 5.0/3), fac = r83+sigma; fac=fac*fac; fac={d}/fac; {a} -= fac*sigma*r53; {b} += fac*r83*{c};"
            else:
                body += f"out[p*{len(rows)}+{k}] = sigma/(r83+sigma);"
        elif kind == "SLT":
            body = f"double rho = fmax({x}, 1e-10), t0 = {c}*pow(rho, 5.0/3), tau = {y};"
            if backward:
                body += f"double fac=1.0/(tau+t0), v0=-2*{d}*tau*fac*fac, vt=2*{d}*t0*fac*fac; {a} += v0*{c}*(5.0/3)*pow(rho, 2.0/3); {b} += vt;"
            else:
                body += f"out[p*{len(rows)}+{k}] = (tau-t0)/(tau+t0);"
        elif kind == "SLN":
            body = (f"{a} -= {d}*{c}/3.0*pow(1+{c}*{x}, -4.0/3);" if backward else
                    f"out[p*{len(rows)}+{k}] = pow(1+{c}*{x}, -1.0/3);")
        else:
            scale, center = other
            # Python evaluates scale * gamma before multiplying the array.
            sc = repr(scale * row[3])
            body = (f"double fac=1+{c}*{x}; {a} += {d}*{repr(scale)}*{c}/(fac*fac);" if backward else
                    f"out[p*{len(rows)}+{k}] = -{repr(center)}+{sc}*{x}/(1+{c}*{x});")
        fragments.append("{" + body + "}")
    init = f"for(int j=0;j<{nraw};++j) out[j*n+p]=0;" if backward else ""
    source = r'''extern "C" __global__ void maps(const double* x, const double* df, double* out, long long n) {
        long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(p>=n) return;
    ''' + init + "\n".join(fragments) + "}"
    return cupy().RawKernel(source, "maps", options=("--fmad=false",))


def forward(features, x):
    cp = cupy()
    nraw, n = x.shape
    rows = signature(features, nraw)
    if rows is None:
        raise ValueError('Unsupported descriptor map or feature index')
    out = cp.empty((n, len(rows)))
    if n:
        _kernel(rows, nraw, False)(((n+127)//128,), (128,), (x, x, out, np.int64(n)))
    return out


def backward(features, x, df):
    cp = cupy()
    nraw, n = x.shape
    rows = signature(features, nraw)
    if rows is None:
        raise ValueError('Unsupported descriptor map or feature index')
    out = cp.empty_like(x)
    if n:
        _kernel(rows, nraw, True)(((n+127)//128,), (128,), (x, df, out, np.int64(n)))
    return out

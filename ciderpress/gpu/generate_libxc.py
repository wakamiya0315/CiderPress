"""Extract the six first-derivative routines from a pristine libxc 7.0.0 tree.

Usage: python -m ciderpress.gpu.generate_libxc /path/to/libxc-7.0.0
The generated header retains libxc's MPL-2.0 license and operation order.
No symbolic simplification or handwritten functional formulas are used.
"""
import hashlib
from pathlib import Path
import re
import sys

_SHA256 = {
    "util.h": "7679744982b3b944f3054b7d2b298612d7b4f3a1051f4f48f4fded2ef26f04b5",
    "lda_x": "1102ae1c6e81d29cf45c608664cfc762b7b7231c55c0d78b04d54f4d1d352e47",
    "gga_x_pbe": "d80e94afcee4e0fbf5b64c2bc37a7f80798a2f6d4a2a50a13d06bff8a3e8357c",
    "gga_c_pbe": "0b0045c5027af1e250128a4243285715714d31ebbf6ff83dab52ca03cc17ad70",
}


def generate(root, target):
    rows = []
    manifest = []
    for name, family, params in [
        ("lda_x", "lda", {"alpha": "1.0"}),
        ("gga_x_pbe", "gga", {"kappa": "0.804", "mu": "0.2195149727645171"}),
        ("gga_c_pbe", "gga", {"beta": "0.06672455060314922",
         "gamma": "0.031090690869654895034", "BB": "1.0"}),
    ]:
        path = root / f"src/maple2c/{family}_exc/{name}.c"
        source = path.read_text()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != _SHA256[name]:
            raise ValueError(f"Not the prescribed libxc 7.0.0 generated source: {path}")
        manifest.append(f"// {path.relative_to(root)} SHA256 {sha}")
        for spin in ("unpol", "pol"):
            match = re.search(r"func_vxc_" + spin + r"\([^\n]+\)\n\{(.*?)\n\}", source, re.S)
            if match is None:
                raise ValueError(f"Missing vxc {name} {spin}")
            body = match[1]
            body = re.sub(r"\n  \w+_params \*params;\n", "\n", body)
            body = re.sub(r"\n  assert\(p->params != NULL\);\n  params = .*?;\n", "\n", body)
            for key, value in params.items():
                body = body.replace("params->" + key, "(" + value + ")")
            body = body.replace("p->dens_threshold", "dens_threshold")
            body = body.replace("p->zeta_threshold", "2.2204460492503130808472633361816e-16")
            body = re.sub(r"if\(out->\w+ != NULL && \(p->info->flags & XC_FLAGS_HAVE_\w+\)\)", "", body)
            body = re.sub(r"out->(\w+)\[ip\*p->dim.\w+ \+ (\d+)\]", r"\1[\2]", body)
            assert "p->" not in body and "out->" not in body and "params" not in body
            rows.append(f"__device__ inline void {name}_{spin}(const double* rho, const double* sigma, double* zk, double* vrho, double* vsigma, double dens_threshold) {{\n{body}\n}}\n")
    util_path = root / "src/util.h"
    util = util_path.read_text()
    util_sha = hashlib.sha256(util_path.read_bytes()).hexdigest()
    if util_sha != _SHA256['util.h']:
        raise ValueError(f'Not the prescribed libxc 7.0.0 util.h: {util_path}')
    manifest.append(f'// src/util.h SHA256 {util_sha}')
    util_notice = re.match(r'/\*.*?\*/', util, re.S).group()
    constants = "\n".join(re.findall(r"^#define M_\w+\s+[^\n]+", util, re.M))
    preamble = """/* Derived from libxc 7.0.0 Maple-generated first derivatives.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 * Reproduce with generate_libxc.py. Changes: fixed default parameters,
 * CUDA device qualifiers, local output pointers; arithmetic is unchanged.
 */
#define POW_1_3(x) cbrt(x)
#define POW_3_2(x) ((x)*sqrt(x))
#define M_PI 3.14159265358979323846
#define my_piecewise3(c, x1, x2) ((c) ? (x1) : (x2))
#define my_piecewise5(c1, x1, c2, x2, x3) ((c1) ? (x1) : ((c2) ? (x2) : (x3)))
"""
    output = preamble + util_notice + '\n' + "\n".join(manifest) + "\n" + constants + "\n" + "\n".join(rows)
    target.write_text('\n'.join(line.rstrip() for line in output.splitlines()) + '\n')


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(__file__).with_name("libxc_7.cuh"))

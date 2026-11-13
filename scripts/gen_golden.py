#!/usr/bin/env python3
# Author: Claude (Anthropic)
"""Generates the "golden" values (src/test/resources/golden/*.properties) from
the official NPB 3.4.4 reference (OMP built WITHOUT OpenMP = serial version).

The official NPB 3.4.4 distribution, unpacked as ./NPB3.4.4 at the repository root
(not included in this repository), is required.

What is stored (only what the official verification does not already cover):
  * CG: zeta and ||r|| of EVERY iteration (the verification only checks the last one).
  * EP: gc, sx, sy (17 digits) and the counters q0..q9 (the verification ignores q).
  * MG: final L2 norm with 17 digits (the verification uses epsilon 1e-8).
  * FT: complex checksum of EVERY iteration with 17 digits (the verification compares
    with 1e-12; with 17 digits it showed that the SER inverse FFT traversed the
    dimensions in a different order than the official code).
  * SP, BT and LU: RMS norms of the residual (xcr) and of the solution error (xce), 5
    components each, with 17 digits (the official verification compares with epsilon
    1e-8; with 17 digits the order of the operations shows, e.g. `btuz/ac` versus
    MPI's `btuz*acinv`).
  * IS: digest of the sorted array (number of keys and sums of k, k^2, k^3 in 64 bits).
    The official verification only checks 5 ranges per iteration and the order of the
    array, not that the array is the expected one. SER/IS.java prints the same line
    (" Sorted keys digest = ..."); the script adds it to the official is.c in the
    temporary copy.
The CG ||r|| (rounding floor ~1e-15) is also stored: with the same compiler and flags
it is bit-for-bit reproducible and the Java SER version matches it exactly; if
regenerated with another compiler it may vary in the last bits (it is compared with
tolerance -Dgolden.cg.rnorm.rtol, 0 by default).

Reproducibility: -O2 -ffp-contract=off (no FMA), OMP_NUM_THREADS=1. Only a temporary
COPY of the reference is patched (NPB3.4.4/ is not touched):
  * cg.f90: prints all iterations, with 17 significant digits;
  * mg.f90: prints the norm with 17 digits;
  * SP, BT and LU /verify.f90: print the norms (and LU's surface integral, xci) with 17 digits.
Each run must end in VERIFICATION SUCCESSFUL or the script aborts.

Usage:  python3 -I scripts/gen_golden.py [CLASSES [KERNELS]]    (default SWAB and cg,ep,mg,ft,is,sp,bt,lu)
Requires gfortran, gcc and make. The generated files are committed, so
`mvn test` does not need gfortran.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFF = os.path.join(ROOT, "NPB3.4.4", "NPB3.4-OMP")
OUT = os.path.join(ROOT, "src", "test", "resources", "golden")
CLASSES = sys.argv[1] if len(sys.argv) > 1 else "SWAB"


def sub_once(path, old, new):
    with open(path, encoding="latin-1") as f:
        s = f.read()
    if s.count(old) != 1:
        sys.exit("patch not applicable in %s: %r appears %d times" % (path, old, s.count(old)))
    with open(path, "w", encoding="latin-1") as f:
        f.write(s.replace(old, new))


def patch(ref):
    cg = os.path.join(ref, "CG", "cg.f90")
    sub_once(cg, "if( it.eq.1 .or. it.eq.niter .or. mod(it,5).eq.0 )then", "if( .true. )then")
    sub_once(cg, "9001    format( 4x, i5, 6x, e21.14, f20.13 )",
             "9001    format( 4x, i5, 6x, es26.17, 2x, es26.17 )")
    mg = os.path.join(ref, "MG", "mg.f90")
    sub_once(mg, "201        format(' L2 Norm is ', E20.13)",
             "201        format(' L2 Norm is ', ES26.17)")
    for app in ("SP", "BT"):
        ver = os.path.join(ref, app, "verify.f90")
        sub_once(ver, " 2010   format(' FAILURE: ', i2, E20.13, E20.13, E20.13)",
                 " 2010   format(' FAILURE: ', i2, ES26.17, ES26.17, ES26.17)")
        sub_once(ver, " 2011   format('          ', i2, E20.13, E20.13, E20.13)",
                 " 2011   format('          ', i2, ES26.17, ES26.17, ES26.17)")
    lu = os.path.join(ref, "LU", "verify.f90")
    sub_once(lu, " 2010   format(' FAILURE: ', i2, 2x, E20.13, E20.13, E20.13)",
             " 2010   format(' FAILURE: ', i2, 2x, ES26.17, ES26.17, ES26.17)")
    sub_once(lu, " 2011   format('          ', i2, 2x, E20.13, E20.13, E20.13)",
             " 2011   format('          ', i2, 2x, ES26.17, ES26.17, ES26.17)")
    sub_once(lu, " 2031   format(' FAILURE: ', 4x, E20.13, E20.13, E20.13)",
             " 2031   format(' FAILURE: ', 4x, ES26.17, ES26.17, ES26.17)")
    sub_once(lu, " 2032   format('          ', 4x, E20.13, E20.13, E20.13)",
             " 2032   format('          ', 4x, ES26.17, ES26.17, ES26.17)")
    ft = os.path.join(ref, "FT", "ft.f90")
    sub_once(ft, " 30   format (' T =',I5,5X,'Checksum =',1P2D22.12)",
             " 30   format (' T =',I5,5X,'Checksum =',1P2D28.17)")
    ep = os.path.join(ref, "EP", "ep.f90")
    sub_once(ep, "'Sums = ',1p,2d25.15/", "'Sums = ',1p,2d28.17/")
    isc = os.path.join(ref, "IS", "is.c")
    sub_once(isc, """    if( j != 0 )
        printf( "Full_verify: number of keys out of sort: %ld\\n", (long)j );""",
             """    {
        unsigned long long c=0, s1=0, s2=0, s3=0;
        for( i=0; i<NUM_KEYS; i++ ) {
            unsigned long long kk = (unsigned long long) key_array[i];
            c++; s1+=kk; s2+=kk*kk; s3+=kk*kk*kk;
        }
        printf( " Sorted keys digest = %llu %llu %llu %llu\\n", c, s1, s2, s3 );
    }
    if( j != 0 )
        printf( "Full_verify: number of keys out of sort: %ld\\n", (long)j );""")


def build_and_run(ref, bench, cls):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    subprocess.run(["make", bench, "CLASS=" + cls], cwd=ref, check=True,
                   capture_output=True, env=env)
    exe = os.path.join(ref, "bin", "%s.%s.x" % (bench, cls))
    out = subprocess.run([exe], cwd=ref, check=True, capture_output=True,
                         text=True, env=env).stdout
    if not re.search(r"Verification\s*=\s*SUCCESSFUL", out):
        sys.exit("reference %s %s does not verify:\n%s" % (bench, cls, out))
    return out


def fnum(tok):
    return float(tok.replace("D", "E").replace("d", "e"))


def parse_cg(out):
    rows = re.findall(r"^\s+(\d+)\s+(\S+)\s+(\S+)\s*$", out, re.M)
    res = {"niter": str(len(rows))}
    for it, rnorm, zeta in rows:
        res["zeta.%d" % int(it)] = repr(fnum(zeta))
        res["rnorm.%d" % int(it)] = repr(fnum(rnorm))
    if [int(r[0]) for r in rows] != list(range(1, len(rows) + 1)):
        sys.exit("CG: non-consecutive iterations")
    return res


def parse_ep(out):
    res = {}
    res["gc"] = repr(fnum(re.search(r"No\. Gaussian Pairs =\s*(\S+)", out).group(1)))
    m = re.search(r"Sums =\s*(\S+)\s+(\S+)", out)
    res["sx"], res["sy"] = repr(fnum(m.group(1))), repr(fnum(m.group(2)))
    counts = re.findall(r"^\s*(\d+)\s+(\d+)\.\s*$", out.split("Counts:")[1], re.M)
    if [int(c[0]) for c in counts] != list(range(10)):
        sys.exit("EP: incomplete counters")
    for i, q in counts:
        res["q.%d" % int(i)] = q
    return res


def parse_ft(out):
    rows = re.findall(r"^\s*T =\s*(\d+)\s+Checksum =\s*(\S+)\s+(\S+)\s*$", out, re.M)
    if [int(r[0]) for r in rows] != list(range(1, len(rows) + 1)) or not rows:
        sys.exit("FT: non-consecutive iterations")
    res = {"niter": str(len(rows))}
    for it, re_, im in rows:
        res["chk.%d.re" % int(it)] = repr(fnum(re_))
        res["chk.%d.im" % int(it)] = repr(fnum(im))
    return res


def parse_is(out):
    m = re.search(r"Sorted keys digest =\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", out)
    if not m:
        sys.exit("IS: digest not found")
    return {"digest.%s" % k: v for k, v in zip(("n", "s1", "s2", "s3"), m.groups())}


def parse_mg(out):
    return {"rnm2": repr(fnum(re.search(r"L2 Norm is\s+(\S+)", out).group(1)))}


def parse_sp(out):
    """xcr/xce norms of SP and BT (same verify.f90 format)."""
    res = {}
    for key, title in (("xcr", "Comparison of RMS-norms of residual"),
                       ("xce", "Comparison of RMS-norms of solution error")):
        if title not in out:
            sys.exit("SP: missing section %r" % title)
        sec = out.split(title)[1]
        rows = re.findall(r"^\s+(\d)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", sec, re.M)[:5]
        if [int(r[0]) for r in rows] != [1, 2, 3, 4, 5]:
            sys.exit("SP: incomplete components in %s" % key)
        for m, val, _ref, _dif in rows:
            res["%s.%s" % (key, m)] = repr(fnum(val))
    return res


def parse_lu(out):
    """LU: xcr and xce norms (as SP and BT) and the surface integral xci."""
    res = parse_sp(out)
    m = re.search(r"Comparison of surface integral\s*\n\s+(\S+)\s+(\S+)\s+(\S+)\s*\n", out)
    if not m:
        sys.exit("LU: missing surface integral")
    res["xci"] = repr(fnum(m.group(1)))
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True,
                              text=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "NPB3.4-OMP")
        shutil.copytree(OFF, ref)
        mk = os.path.join(ref, "config", "make.def")
        shutil.copy(mk + ".template", mk)
        with open(mk, encoding="latin-1") as f:
            s = f.read()
        s = re.sub(r"^FFLAGS\s*=.*$", "FFLAGS = -O2 -ffp-contract=off", s, flags=re.M)
        s = re.sub(r"^CFLAGS\s*=.*$", "CFLAGS = -O2 -ffp-contract=off", s, flags=re.M)
        with open(mk, "w", encoding="latin-1") as f:
            f.write(s)
        patch(ref)
        parsers = {"cg": parse_cg, "ep": parse_ep, "mg": parse_mg, "ft": parse_ft, "is": parse_is,
                   "sp": parse_sp, "bt": parse_sp, "lu": parse_lu}
        wanted = sys.argv[2].split(",") if len(sys.argv) > 2 else list(parsers)
        for bench in wanted:
            parser = parsers[bench]
            for cls in CLASSES:
                print("reference %s %s ..." % (bench, cls), flush=True)
                vals = parser(build_and_run(ref, bench, cls))
                path = os.path.join(OUT, "%s.%s.properties" % (bench, cls))
                with open(path, "w") as f:
                    f.write("# Generated by scripts/gen_golden.py; do not edit by hand.\n")
                    f.write("# Reference: NPB 3.4.4 OMP without OpenMP, %s, -O2 -ffp-contract=off\n"
                            % gfortran)
                    for k, v in vals.items():
                        f.write("%s=%s\n" % (k, v))
                print("  -> " + os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()

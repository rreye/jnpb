#!/usr/bin/env python3
# Author: Claude (Anthropic)
"""Generates the "golden" values for the MPJ versions (src/test/resources/golden/mpi/)
from the official NPB 3.4.4 MPI reference (mpif90 + mpirun).

The official NPB 3.4.4 distribution, unpacked as ./NPB3.4.4 at the repository root
(not included in this repository), is required.

Unlike SER (see gen_golden.py), there is one file per kernel and class holding the
values for EACH number of processes: the order of the sums in the reductions depends
on np, and the official code and MPJ follow the same scheme, so with the same np they
must match (for CG, bit for bit: a difference of 0 was measured in zeta and ||r|| for
S, W and A with 1, 2, 4, 8 and 16 processes).

What is stored:
  * CG: zeta of EVERY iteration (17 digits). MPJ only prints iteration 1, every 5th and
    the last, so the test compares those; the rest is kept in case all are printed
    some day.
  * MG: final L2 norm of the residual with 17 digits (the only thing both print).
  * FT: complex checksum of EVERY iteration (17 digits); the official verification
    only compares with 1e-12.
  * EP: gc, sx, sy (17 digits) and the counters q0..q9 (the verification ignores q).
  * IS: digest of the global sorted array (number of keys and sums of k, k^2, k^3 in
    64 bits). The official verification only checks 5 ranges per iteration; the digest
    is the same for any np. MPJ/IS.java prints the same line (" Sorted keys digest = ...").
  * DT: L2 norm of the result (sum of the data stream of the sink node) with 17 digits,
    per graph (BH, WH, SH) and number of processes. NPB-MPI 3.4.4 leaves class B
    unverified (defect: verified stays at -1 although the values exist); for B the
    script checks the norm against the values documented in dt.c.
  * SP, BT and LU: RMS norms of the residual (xcr) and of the solution error (xce), 5
    components each, with 17 digits (the official verification compares with epsilon
    1e-8); LU also the surface integral (xci).

Only a temporary COPY of the reference is patched (NPB3.4.4/ is not touched), built
with -O2 -ffp-contract=off. Each run must end in VERIFICATION SUCCESSFUL or the
script aborts.

Usage:  python3 -I scripts/gen_golden_mpi.py [--kernels cg,mg,ft,ep,is,sp,bt,lu,dt] [--classes SWA] [--nps 1,2,4,8,16]
Requires mpif90, mpicc, mpirun and make. The generated files are committed, so
`mvn test` does not need MPI.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "NPB3.4.4", "NPB3.4-MPI")
OUT = os.path.join(ROOT, "src", "test", "resources", "golden", "mpi")


def sub_once(path, old, new):
    with open(path, encoding="latin-1") as f:
        s = f.read()
    if s.count(old) != 1:
        sys.exit("patch not applicable in %s: %r appears %d times" % (path, old, s.count(old)))
    with open(path, "w", encoding="latin-1") as f:
        f.write(s.replace(old, new))


def patch(ref):
    cg = os.path.join(ref, "CG", "cg.f90")
    sub_once(cg, "(it.eq.1 .or. it.eq.niter .or. mod(it,5).eq.0) )then", "(.true.) )then")
    sub_once(cg, "9001 format( 4x, i5, 6x, e21.14, f20.13 )",
             "9001 format( 4x, i5, 6x, es26.17, 2x, es26.17 )")
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
    isc = os.path.join(ref, "IS", "is.c")
    sub_once(isc, """    if( j != 0 )
    {
        printf( "Processor %d:  Full_verify: number of keys out of sort: %d\\n",""",
             """    {
        unsigned long long hl[4], hg[4];
        hl[0]=hl[1]=hl[2]=hl[3]=0;
        for( i=0; i<total_local_keys; i++ ) {
            unsigned long long kk = (unsigned long long) key_array[i];
            hl[0]++; hl[1]+=kk; hl[2]+=kk*kk; hl[3]+=kk*kk*kk;
        }
        MPI_Reduce( hl, hg, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm_work );
        if( my_rank == 0 )
            printf( " Sorted keys digest = %llu %llu %llu %llu\\n", hg[0], hg[1], hg[2], hg[3] );
    }
    if( j != 0 )
    {
        printf( "Processor %d:  Full_verify: number of keys out of sort: %d\\n",""")
    dt = os.path.join(ref, "DT", "dt.c")
    sub_once(dt, 'fprintf(stderr," %s L2 Norm = %f\\n",bmname,rnm2);',
             'fprintf(stderr," %s L2 Norm = %.17g\\n",bmname,rnm2);')
    ep = os.path.join(ref, "EP", "ep.f90")
    sub_once(ep, "'Sums = ',1p,2d25.15/", "'Sums = ',1p,2d28.17/")
    ft = os.path.join(ref, "FT", "ft.f90")
    sub_once(ft, " 30         format (' T =',I5,5X,'Checksum =',1P2D22.12)",
             " 30         format (' T =',I5,5X,'Checksum =',1P2D28.17)")
    mg = os.path.join(ref, "MG", "mg.f90")
    sub_once(mg, " 201           format(' L2 Norm is ', E20.13)",
             " 201           format(' L2 Norm is ', ES26.17)")


def fnum(tok):
    return float(tok.replace("D", "E").replace("d", "e"))


def parse_cg(out):
    rows = re.findall(r"^\s+(\d+)\s+(\S+)\s+(\S+)\s*$", out, re.M)
    if [int(r[0]) for r in rows] != list(range(1, len(rows) + 1)) or not rows:
        sys.exit("CG: non-consecutive iterations")
    res = {"niter": str(len(rows))}
    for it, _rnorm, zeta in rows:
        res["zeta.%d" % int(it)] = repr(fnum(zeta))
    return res


def parse_mg(out):
    m = re.search(r"L2 Norm is\s+(\S+)", out)
    if not m:
        sys.exit("MG: norm not found")
    return {"rnm2": repr(fnum(m.group(1)))}


def parse_ft(out):
    rows = re.findall(r"^\s*T =\s*(\d+)\s+Checksum =\s*(\S+)\s+(\S+)\s*$", out, re.M)
    if [int(r[0]) for r in rows] != list(range(1, len(rows) + 1)) or not rows:
        sys.exit("FT: non-consecutive iterations")
    res = {"niter": str(len(rows))}
    for it, re_, im in rows:
        res["chk.%d.re" % int(it)] = repr(fnum(re_))
        res["chk.%d.im" % int(it)] = repr(fnum(im))
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


def parse_is(out):
    m = re.search(r"Sorted keys digest =\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", out)
    if not m:
        sys.exit("IS: digest not found")
    return {"digest.%s" % k: v for k, v in zip(("n", "s1", "s2", "s3"), m.groups())}


def parse_sp(out):
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


def parse_dt(out):
    m = re.search(r"L2 Norm =\s*(\S+)", out)
    if not m:
        sys.exit("DT: norm not found")
    return {"norm": repr(float(m.group(1)))}


# DT: graph nodes per class: (BH and WH, SH). Tested with the minimum and with a few more processes
# (the extra ones do not take part). Verification values from dt.c for class B (which the official code does not use).
DT_NODES = {"S": (5, 12), "W": (11, 32), "A": (21, 80), "B": (43, 192)}
DT_EXTRA = {"S": (3, 8), "W": (5, 10), "A": (11, 20), "B": (0, 0)}
DT_B_VALUE = {"BH": 4317114.0, "WH": 7877279917.0, "SH": 1836863082.0}


KERNELS = {"cg": parse_cg, "mg": parse_mg, "ft": parse_ft, "ep": parse_ep, "is": parse_is,
           "sp": parse_sp, "bt": parse_sp, "lu": parse_lu, "dt": parse_dt}
# Default process counts (MG: includes 32 and 64, where give3Ex used to fail)
DEFAULT_NPS = {"cg": "1,2,4,8,16", "mg": "1,2,4,8,16,32,64", "ft": "1,2,4,8,16,32,64",
               # EP accepts any number of processes (not only powers of 2)
               "ep": "1,2,3,4,7,8,16,32",
               "is": "1,2,4,8,16,32,64",
               # SP requires a square number of processes and cells of >= 3 points (S: 12^3 => np <= 16)
               "sp": "1,4,9,16,25,36,64",
               # BT: same as SP (square number of processes)
               "bt": "1,4,9,16,25,36,64",
               # LU: np1 x np2 grid with np2 <= np1 <= 2*np2 (not only powers of 2: 6, 9, 12, ...) and
               # subdomains of >= 3 points (S: 12^3 => np <= 16)
               "lu": "1,2,4,6,8,9,12,16"}


def build(ref, bench, cls):
    subprocess.run(["make", bench, "CLASS=" + cls], cwd=ref, check=True, capture_output=True)
    return os.path.join(ref, "bin", "%s.%s.x" % (bench, cls))


def run(exe, np_, extra=(), merge=False):
    r = subprocess.run(["mpirun", "--oversubscribe", "-np", str(np_), exe] + list(extra), check=True,
                       capture_output=True, text=True)
    out = r.stdout + (r.stderr if merge else "")
    if not re.search(r"Verification\s*=\s*SUCCESSFUL", out):
        sys.exit("reference %s with %d processes does not verify:\n%s" % (exe, np_, out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="SWA")
    ap.add_argument("--nps", default=None, help="default depends on the kernel (DEFAULT_NPS)")
    ap.add_argument("--kernels", default="cg")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True,
                              text=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "NPB3.4-MPI")
        shutil.copytree(REF, ref)
        mk = os.path.join(ref, "config", "make.def")
        shutil.copy(mk + ".template", mk)
        with open(mk, encoding="latin-1") as f:
            s = f.read()
        s = re.sub(r"^FFLAGS\s*=.*$", "FFLAGS = -O2 -ffp-contract=off", s, flags=re.M)
        s = re.sub(r"^CFLAGS\s*=.*$", "CFLAGS = -O2 -ffp-contract=off", s, flags=re.M)
        with open(mk, "w", encoding="latin-1") as f:
            f.write(s)
        patch(ref)
        for bench in a.kernels.split(","):
            for cls in a.classes:
                print("reference %s %s ..." % (bench, cls), flush=True)
                exe = build(ref, bench, cls)
                lines = []
                if bench == "dt":
                    for g in ("BH", "WH", "SH"):
                        n = DT_NODES[cls][1 if g == "SH" else 0]
                        for np_ in sorted({n} | ({n + DT_EXTRA[cls][1 if g == "SH" else 0]}
                                                 if DT_EXTRA[cls][0] else set())):
                            r = subprocess.run(["mpirun", "--oversubscribe", "-np", str(np_), exe, g],
                                               check=True, capture_output=True, text=True)
                            out = r.stdout + r.stderr
                            norm = float(parse_dt(out)["norm"])
                            ok = re.search(r"Verification\s*=\s*SUCCESSFUL", out) is not None
                            if cls == "B":  # the official code does not verify B (defect): compare with dt.c
                                ok = norm == DT_B_VALUE[g]
                            if not ok and np_ > n:
                                # NPB-MPI 3.4.4 DT with extra processes sometimes gives another norm (e.g. SH
                                # class S with >= 16 processes); the documented value is the right one and
                                # the Java version always gives it: that np is not stored.
                                print("  %s np=%d: the official code does NOT verify (norm %r), skipped" % (g, np_, norm),
                                      flush=True)
                                continue
                            if not ok:
                                sys.exit("DT %s %s np=%d does not verify:\n%s" % (cls, g, np_, out))
                            print("  %s np=%d ok" % (g, np_), flush=True)
                            lines.append("%s.np%d.norm=%r" % (g, np_, norm))
                    path = os.path.join(OUT, "dt.%s.properties" % cls)
                    with open(path, "w") as f:
                        f.write("# Generated by scripts/gen_golden_mpi.py; do not edit by hand.\n")
                        f.write("# Reference: NPB 3.4.4 MPI (DT), mpirun --oversubscribe, "
                                "-O2 -ffp-contract=off\n")
                        f.write("\n".join(lines) + "\n")
                    print("  -> " + os.path.relpath(path, ROOT))
                    continue
                for np_ in [int(x) for x in (a.nps or DEFAULT_NPS[bench]).split(",")]:
                    vals = KERNELS[bench](run(exe, np_))
                    print("  np=%d ok" % np_, flush=True)
                    lines += ["np%d.%s=%s" % (np_, k, v) for k, v in vals.items()]
                path = os.path.join(OUT, "%s.%s.properties" % (bench, cls))
                with open(path, "w") as f:
                    f.write("# Generated by scripts/gen_golden_mpi.py; do not edit by hand.\n")
                    f.write("# Reference: NPB 3.4.4 MPI, %s, mpirun --oversubscribe, "
                            "-O2 -ffp-contract=off\n" % gfortran)
                    f.write("\n".join(lines) + "\n")
                print("  -> " + os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()

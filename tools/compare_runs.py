#!/usr/bin/env python3
"""
Compare DEM runs: C++ vs C++ (chaos twin test) and C++ vs Fortran.

The GPU code accumulates forces with atomicAdd, so the summation order changes
from run to run and two identical C++ runs are NOT bit-identical.  That makes a
free twin test: if run A vs run B diverges as much as C++ vs Fortran, the
difference is chaos, not a porting bug.

Usage
-----
    python compare_runs.py runA/ runB/ [runFortran/]

Each directory must contain energy.dat and (optionally) f-d_curve.dat.

energy.dat columns:
    istep engr enva enka enca enfr enxd enxf err enfr1 enfr2 enfr3

Tool penetration is derived from the force energy, since enxf accumulates only
over force-assigned DOFs with non-zero bval, i.e. the tool's vertical load:
    descent [mm] = |enxf| / F_applied,   F_applied = 5.0e8 force units = 500 N
"""

import sys
import os
import numpy as np

F_APPLIED = 5.0e8          # bval(tool,3) in mm-g-s force units  (= 500 N)
COLS = ['istep', 'engr', 'enva', 'enka', 'enca', 'enfr',
        'enxd', 'enxf', 'err', 'enfr1', 'enfr2', 'enfr3']


def load_energy(d):
    p = os.path.join(d, 'energy.dat')
    a = np.loadtxt(p, skiprows=1)
    return {c: a[:, i] for i, c in enumerate(COLS) if i < a.shape[1]}


def load_tool_z(d, grain=None):
    """Tool centroid z from f-d_curve.dat (col 8).  Returns (istep, z) or None."""
    p = os.path.join(d, 'f-d_curve.dat')
    if not os.path.exists(p):
        return None
    a = np.loadtxt(p)
    if a.ndim != 2 or a.shape[1] < 8:
        return None
    if grain is not None:
        a = a[a[:, 1] == grain]
    return a[:, 0], a[:, 7]


def align(x1, y1, x2, y2):
    """Intersect on istep."""
    common = np.intersect1d(x1, x2)
    i1 = np.searchsorted(x1, common)
    i2 = np.searchsorted(x2, common)
    return common, y1[i1], y2[i2]


def report_pair(name, a, b):
    print(f'\n=== {name} ===')
    print(f'{"quantity":10} {"final A":>14} {"final B":>14} {"rel diff":>10}')
    for c in ['enva', 'enka', 'enca', 'enfr', 'enxd', 'enxf', 'engr']:
        if c not in a or c not in b:
            continue
        s, ya, yb = align(a['istep'], a[c], b['istep'], b[c])
        fa, fb = ya[-1], yb[-1]
        scale = max(abs(fa), abs(fb), 1e-30)
        print(f'{c:10} {fa:14.5e} {fb:14.5e} {abs(fa-fb)/scale:9.2%}')

    # energy-balance residual, normalised by total dissipation
    for tag, r in (('A', a), ('B', b)):
        diss = abs(r['enca'][-1]) + abs(r['enfr'][-1])
        print(f'  energy balance {tag}: |err|/dissipation = '
              f'{abs(r["err"][-1])/max(diss,1e-30):.3e}')

    # where the two series first separate by 1%
    s, ya, yb = align(a['istep'], a['enfr'], b['istep'], b['enfr'])
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(ya - yb) / np.maximum(np.abs(ya), 1e-30)
    k = np.argmax(rel > 0.01)
    if rel.max() > 0.01:
        print(f'  friction energy first differs by >1% at step {int(s[k])}')
    else:
        print('  friction energy stays within 1% for the whole run')


def report_penetration(dirs, labels):
    print('\n=== tool descent from enxf (mm) ===')
    for d, lab in zip(dirs, labels):
        e = load_energy(d)
        desc = abs(e['enxf'][-1]) / F_APPLIED
        print(f'  {lab:12} {desc:8.2f} mm   (enxf = {e["enxf"][-1]:.4e})')
    zs = [load_tool_z(d) for d in dirs]
    if all(z is not None for z in zs):
        print('\n=== tool centroid z from f-d_curve.dat (mm) ===')
        for z, lab in zip(zs, labels):
            print(f'  {lab:12} start {z[1][0]:8.2f}  end {z[1][-1]:8.2f}  '
                  f'drop {z[1][0]-z[1][-1]:7.2f}')


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    dirs = sys.argv[1:]
    labels = [os.path.basename(os.path.normpath(d)) or d for d in dirs]
    data = [load_energy(d) for d in dirs]

    report_pair(f'{labels[0]} vs {labels[1]}  (same code: chaos twin test)',
                data[0], data[1])
    if len(dirs) >= 3:
        report_pair(f'{labels[0]} vs {labels[2]}  (C++ vs Fortran)',
                    data[0], data[2])
        print('\nREADING: if the two comparisons show a similar spread, the')
        print('difference between C++ and Fortran is chaotic divergence, not a')
        print('porting bug.  If A-vs-B stays much closer than A-vs-Fortran,')
        print('there is still a code difference to find.')

    report_penetration(dirs, labels)
    return 0


if __name__ == '__main__':
    sys.exit(main())

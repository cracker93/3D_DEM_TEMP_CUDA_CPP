#!/usr/bin/env python3
"""
Verify the REAL tool motion in every simulation case
====================================================

Measures, from solver output rather than from the input file, what the tamping
tool actually did in each case, and compares it against (a) what the input file
asked for and (b) what the paper reports.

Quantities measured per case
----------------------------
    r_c        orbital radius of the tool CENTROID            (circle fit)
    f          excitation frequency                           (phase slope)
    theta      wobble inclination amplitude                   (|omega_h| / 2 pi f)
    sign       whether the wobble ADDS at the tine or at the root   (phase, MEASURED)
    A_tine     orbital radius at the TINE                     |r_c + k*L_half|
    A_root     orbital radius at the ROOT                     |r_c - k*L_half|
    xi_node    axial position of the stationary node, positive DOWNWARD
               from the centroid toward the tine

Geometry convention
-------------------
    xi is measured from the tool centroid, POSITIVE DOWNWARD (toward the tine).
    The tine surface is at xi = +L_half, the root at xi = -L_half.
    Orbit radius along the tool is, to first order in theta,

        R(xi) = | r_c + k * xi |,      k = +theta if the wobble adds at the
                                           tine, -theta if it adds at the root

    Exact would be r_c + L_half*sin(theta); at theta ~ 1.4e-3 the difference is
    ~5e-10 mm, and even at theta ~ 1.4e-2 it is ~2e-4 mm.  Negligible either way.

    L_half = 182.5 mm is the centroid -> tine SURFACE distance from in_gm.dat
    (tool spans z = 510.0 .. 875.0, centroid at z = 692.5).  The lowest
    sub-sphere CENTRE is 174.5 mm below the centroid; use --l-half 174.5 if you
    prefer the contacting-sphere reference.  The difference is ~0.012 mm.

PRIMARY SOURCE: f-d_curve.dat
    Small (~1 MB per segment) and written every nprn2 = 500 steps = 1 kHz, so
    there is no aliasing at any frequency used (max 170 Hz, Nyquist 500 Hz).

    14 columns, format 617 = (' ',i7,1x,i7,12(1x,e12.6)):

        istep grain  Fx x  Fy y  Fz z  Mx wx  My wy  Mz wz
          1     2     3 4   5 6   7 8   9 10  11 12  13 14

    (The 12 reals exist because the commented-out second write's continuation
    lines are NOT commented, so they attach to the live write statement.)

    It has no Euler angles, so the tine is obtained from R(xi) above.

OPTIONAL CHECK: output.dat  (--verify)
    300 MB per segment, written every nprn1 = 10000 steps = 50 Hz, which IS
    aliased -- but a Kasa circle fit is order-independent, so undersampled
    points still lie on the true circle and the radius is still recovered.
    output.dat carries the Euler angles, so the tine is reconstructed
    geometrically with no assumed relation.  Record layout per grain:

        istep k
        vgx vgy vgz
        vgwx vgwy vgwz
        gcx gcy gcz
        gwx gwy gwz          <- Euler angles

Usage
-----
    python verify_tool_motion.py /path/to/New
    python verify_tool_motion.py /path/to/New --segment 15s --csv motion.csv
    python verify_tool_motion.py /path/to/New --verify Diameter_0.1mm/35hz/15s
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

G = 9.81
DT = 2.0e-6              # s, solver time step
L_HALF = 182.500         # mm, centroid -> tine surface, from in_gm.dat
F_APPLIED_N = 500.0      # N, bval(3792,3) = -5.0e8 force units = 500 N
DIAMETERS = ["0.3", "0.5", "1", "1.5", "2", "3", "4", "5"]   # rerun matrix (tine rotation diameter, mm)
FREQS = ["35", "70", "105", "140", "155", "170"]

# tolerances for the enforced diagnostics
TOL_RC_RATIO = 2e-3      # |r_c/(D/4) - 1| beyond this -> flag
TOL_THETA_SD = 1e-3      # theta_sd/theta beyond this -> omega_h not constant
TOL_CIRCLE = 1e-3        # circle_scatter/r_c beyond this -> not a clean circle
TOL_FREQ_HZ = 0.05       # |f_meas - f_nominal| beyond this -> flag
TOL_WZ = 1e-3            # |omega_z| / |omega_h| beyond this -> tool is spinning
TOL_INPUT = 1e-3         # relative mismatch vs in_bc.dat -> flag


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def fit_circle(x, y):
    """Least-squares circle fit (Kasa).

    A circle obeys  x^2 + y^2 = 2a x + 2b y + c   with c = R^2 - a^2 - b^2,
    which is LINEAR in (a, b, c), so one lstsq gives the centre and radius
    without iteration.  More robust than (max-min)/2, which needs the samples
    to land exactly on the extremes.  Order-independent, hence safe on aliased
    data.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    (a, bb, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    R = math.sqrt(max(c + a * a + bb * bb, 0.0))
    resid = np.hypot(x - a, y - bb) - R          # per-sample radial error
    return a, bb, R, float(np.std(resid))


def angular_coverage(x, y, cx, cy, nbin=24):
    """Fraction of phase bins visited.  Guards against a degenerate fit when
    an aliased sample rate happens to lock onto a few phases only."""
    ang = np.arctan2(y - cy, x - cx)
    idx = ((ang + np.pi) / (2 * np.pi) * nbin).astype(int) % nbin
    return len(np.unique(idx)) / nbin


def frequency_from_phase(x, y, cx, cy, t):
    """Driving frequency from the unwrapped orbital phase.  Valid only when the
    sample rate is above Nyquist (true for f-d_curve.dat, NOT for output.dat)."""
    ang = np.unwrap(np.arctan2(y - cy, x - cx))
    slope = np.polyfit(t, ang, 1)[0]
    return abs(slope) / (2 * np.pi)


def mean_abs_linear(r_c, k, xi0, xi1):
    """Mean of |r_c + k*xi| over xi in [xi0, xi1].

    Uses the antiderivative u|u|/2 with u = r_c + k*xi, so a sign change inside
    the interval (a node on the tool) is handled correctly."""
    if abs(k) < 1e-15:
        return abs(r_c)
    u0 = r_c + k * xi0
    u1 = r_c + k * xi1
    integral = (u1 * abs(u1) - u0 * abs(u0)) / (2.0 * k)
    return abs(integral) / (xi1 - xi0)


def euler_R(alp, bet, gam):
    """Rotation matrix exactly as built in dem3d (global = R . local)."""
    ca, sa = math.cos(alp), math.sin(alp)
    cb, sb = math.cos(bet), math.sin(bet)
    cg, sg = math.cos(gam), math.sin(gam)
    return np.array([
        [cb,      -sb * ca,               sb * sa],
        [cg * sb,  cg * cb * ca - sg * sa, -cg * cb * sa - sg * ca],
        [sg * sb,  sg * cb * ca + cg * sa, -sg * cb * sa + cg * ca]])


# ---------------------------------------------------------------------------
# input-file cross-check
# ---------------------------------------------------------------------------

def read_in_bc(case_dir):
    """Last 6-token data line of in_bc.dat: hz Amp Zkak itl1a itl1b pret.

    Closes the loop: without this we only verify that the solver did what the
    input said, not that the input said what we intended."""
    path = os.path.join(case_dir, 'in_bc.dat')
    if not os.path.exists(path):
        return {}
    with open(path, errors='replace') as fh:
        lines = [l for l in fh if l.strip()]
    for line in reversed(lines):
        tok = line.replace('d', 'e').replace('D', 'E').split()
        if len(tok) == 6:
            try:
                return dict(hz_in=float(tok[0]), Amp_in=float(tok[1]),
                            Zkak_in=float(tok[2]), pret_in=float(tok[5]))
            except ValueError:
                continue
    return {}


# ---------------------------------------------------------------------------
# per-case measurement from f-d_curve.dat
# ---------------------------------------------------------------------------

def measure_case(path, l_half=L_HALF):
    d = np.loadtxt(path)
    if d.ndim != 2 or d.shape[1] < 8:
        raise ValueError(f'{path}: expected >=8 columns, got {d.shape}')

    istep = d[:, 0]
    x, y, z = d[:, 3], d[:, 5], d[:, 7]
    t = (istep - istep[0]) * DT

    cx, cy, r_c, scatter = fit_circle(x, y)
    f = frequency_from_phase(x, y, cx, cy, t)
    cover = angular_coverage(x, y, cx, cy)

    if d.shape[1] >= 14:
        wx, wy, wz = d[:, 9], d[:, 11], d[:, 13]
        # The prescribed wobble is wx = -2 pi f theta cos, wy = -2 pi f theta sin,
        # so |omega_h| is CONSTANT at 2 pi f theta.  Take the mean, not the peak:
        # it uses every sample and is insensitive to one noisy point.  The std is
        # then a hard correctness check -- see flag 'WSD' below.
        w_h = np.hypot(wx, wy)
        theta = float(np.mean(w_h) / (2 * np.pi * f))
        theta_sd = float(np.std(w_h) / (2 * np.pi * f))
        wz_ratio = float(np.mean(np.abs(wz)) / max(np.mean(w_h), 1e-30))

        # --- MEASURED sign of the wobble relative to the orbit -------------
        # displacement u = r_c (cos p, sin p);  omega_h = -2 pi f Zkak (cos p, sin p)
        # so sign(u . omega_h) = -sign(Zkak).  Zkak < 0 -> wobble ADDS at the tine.
        # Without this the tine/root assignment is an assumption, not a measurement.
        dot = float(np.mean((x - cx) * wx + (y - cy) * wy))
        tine_adds = dot > 0
    else:
        theta = theta_sd = wz_ratio = float('nan')
        dot = float('nan')
        tine_adds = True

    k = theta if tine_adds else -theta          # signed slope of R(xi)
    off = l_half * theta

    return dict(
        n_samples=len(d), t_span_s=float(t[-1]),
        f_meas_Hz=float(f),
        orbit_centre_x_mm=float(cx), orbit_centre_y_mm=float(cy),
        r_centroid_meas_mm=float(r_c),
        circle_scatter_mm=scatter, circle_scatter_rel=scatter / max(r_c, 1e-30),
        angular_coverage=cover,
        theta_meas_rad=theta, theta_sd_rad=theta_sd,
        theta_sd_rel=theta_sd / max(theta, 1e-30),
        omega_z_rel=wz_ratio,
        wobble_phase_dot=dot, wobble_adds_at_tine=bool(tine_adds),
        wobble_offset_mm=off,
        A_tine_true_mm=float(abs(r_c + k * l_half)),
        A_root_true_mm=float(abs(r_c - k * l_half)),
        A_mean_centroid_to_tine_mm=mean_abs_linear(r_c, k, 0.0, l_half),
        xi_node_mm=float(-r_c / k) if k else float('nan'),
        z_start_mm=float(z[0]), z_end_mm=float(z[-1]),
        z_change_mm=float(z[-1] - z[0]),
        Fz_contact_mean_N=float(d[:, 6].mean()) / 1e6,
    )


# ---------------------------------------------------------------------------
# optional direct check from output.dat
# ---------------------------------------------------------------------------

def verify_with_euler(out_path, tool_grain=3792, l_half=L_HALF):
    """Reconstruct the tine orbit from the Euler angles, assuming nothing."""
    recs = []
    with open(out_path, errors='replace') as fh:
        nptotl, _nstep = (int(v) for v in fh.readline().split()[:2])
        for _ in range(nptotl):
            fh.readline()                              # i, nset(i), gv(i)
        while True:
            line = fh.readline()
            if not line:
                break
            tok = line.split()
            if len(tok) == 2 and tok[1] == str(tool_grain):
                fh.readline()                          # translational velocity
                fh.readline()                          # angular velocity
                gc = [float(v) for v in fh.readline().split()]
                gw = [float(v) for v in fh.readline().split()]
                recs.append(gc + gw)
    if len(recs) < 8:
        raise ValueError(f'{out_path}: only {len(recs)} tool records found')
    a = np.array(recs)
    gc, gw = a[:, 0:3], a[:, 3:6]

    R0 = euler_R(*gw[0])
    axis = int(np.argmax(np.abs(R0[2, :])))            # body axis mapping to vertical
    align = abs(R0[2, axis])
    e = np.zeros(3)
    e[axis] = np.sign(R0[2, axis])

    tip = np.array([gc[i] - l_half * (euler_R(*gw[i]) @ e) for i in range(len(gc))])
    tcx, tcy, R_tine, sc = fit_circle(tip[:, 0], tip[:, 1])
    ccx, ccy, r_c, _ = fit_circle(gc[:, 0], gc[:, 1])
    cov_t = angular_coverage(tip[:, 0], tip[:, 1], tcx, tcy)
    cov_c = angular_coverage(gc[:, 0], gc[:, 1], ccx, ccy)
    return dict(r_c=r_c, R_tine=R_tine, scatter=sc, n=len(gc),
                axis=axis, axis_alignment=align,
                coverage_tine=cov_t, coverage_centroid=cov_c)


# ---------------------------------------------------------------------------

def build_flags(m):
    """Enforce the diagnostics instead of just recording them."""
    f = []
    if abs(m['ratio_r_centroid_vs_D_over_4'] - 1) > TOL_RC_RATIO:
        f.append('RC')      # centroid radius != D/4 input rule
    if m['theta_sd_rel'] > TOL_THETA_SD:
        f.append('WSD')     # |omega_h| not constant -> check icode(3792,4:6)
    if m['circle_scatter_rel'] > TOL_CIRCLE:
        f.append('FIT')     # orbit is not a clean circle
    if m['angular_coverage'] < 0.9:
        f.append('COV')     # poor phase coverage, fit may be ill-conditioned
    if abs(m['f_meas_Hz'] - m['f_nominal_Hz']) > TOL_FREQ_HZ:
        f.append('FREQ')
    if m['omega_z_rel'] > TOL_WZ:
        f.append('WZ')      # tool spinning about its own axis; bval(6) should be 0
    if not m['wobble_adds_at_tine']:
        f.append('ROOT')    # wobble adds at the ROOT, not the tine
    if m['z_change_mm'] > 0.5:
        f.append('WDRW')    # tool rising: this looks like the withdrawal phase
    if m.get('Amp_in') is not None and m.get('Amp_in') == m.get('Amp_in'):
        if abs(m['r_centroid_meas_mm'] - m['Amp_in']) > TOL_INPUT * max(m['Amp_in'], 1e-6):
            f.append('IN_A')    # measured centroid radius != Amp in in_bc.dat
        if abs(m['theta_meas_rad'] - abs(m['Zkak_in'])) > TOL_INPUT * abs(m['Zkak_in']):
            f.append('IN_T')    # measured theta != |Zkak| in in_bc.dat
    return ','.join(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base_dir')
    ap.add_argument('--segment', default='15s', help='which restart segment to read')
    ap.add_argument('--csv', default='tool_motion.csv')
    ap.add_argument('--l-half', type=float, default=L_HALF,
                    help='centroid -> tine distance in mm (182.5 surface, 174.5 sphere centre)')
    ap.add_argument('--verify', metavar='RELPATH',
                    help='also reconstruct the tine from Euler angles in '
                         'output.dat under this case folder')
    args = ap.parse_args()

    rows, problems = [], []
    for Ds in DIAMETERS:
        D = float(Ds)
        for fs in FREQS:
            case_dir = os.path.join(args.base_dir, f'Diameter_{Ds}mm', f'{fs}hz',
                                    args.segment)
            p = os.path.join(case_dir, 'f-d_curve.dat')
            if not os.path.exists(p):
                problems.append(f'missing  Diameter_{Ds}mm/{fs}hz/{args.segment}')
                continue
            try:
                m = measure_case(p, args.l_half)
            except Exception as exc:
                problems.append(f'failed   Diameter_{Ds}mm/{fs}hz: {exc}')
                continue

            fn = float(fs)
            A_nom = D / 2                      # what the paper calls A (Table 3)
            g_pub = 4 * np.pi ** 2 * fn ** 2 * A_nom * 1e-3 / G

            m.update(
                case=f'D{Ds}mm_{fs}Hz',
                D_folder_mm=D,
                f_nominal_Hz=fn,
                A_nominal_mm=A_nom,
                D_tine_nominal_mm=D,
                Amp_expected_D_over_4_mm=D / 4,
                ratio_r_centroid_vs_D_over_4=m['r_centroid_meas_mm'] / (D / 4),
                D_tine_true_mm=2 * m['A_tine_true_mm'],
                ratio_A_tine_true_vs_nominal=m['A_tine_true_mm'] / A_nom,
                Gamma_published=g_pub,
                Gamma_centroid=4 * np.pi ** 2 * fn ** 2 * m['r_centroid_meas_mm'] * 1e-3 / G,
                Gamma_true_tine=4 * np.pi ** 2 * fn ** 2 * m['A_tine_true_mm'] * 1e-3 / G,
                Gamma_true_mean=4 * np.pi ** 2 * fn ** 2 * m['A_mean_centroid_to_tine_mm'] * 1e-3 / G,
            )
            m['Gamma_ratio_true_over_published'] = m['Gamma_true_tine'] / g_pub
            m.update(read_in_bc(case_dir))
            m.setdefault('hz_in', float('nan'))
            m.setdefault('Amp_in', float('nan'))
            m.setdefault('Zkak_in', float('nan'))
            m.setdefault('pret_in', float('nan'))
            m['flags'] = build_flags(m)
            rows.append(m)

    if not rows:
        print('nothing measured', file=sys.stderr)
        for q in problems:
            print('  ' + q, file=sys.stderr)
        return 1

    hdr = (f"{'D_fold':>7}{'f_nom':>6}{'r_c':>9}{'D/4':>8}{'r_c/(D/4)':>10}"
           f"{'theta':>11}{'A_tine':>9}{'A_tine/A_nom':>13}"
           f"{'G_pub':>9}{'G_true':>9}{'xi_node':>9}  flags")
    print(hdr)
    print('-' * len(hdr))
    for m in rows:
        print(f"{m['D_folder_mm']:>7g}{m['f_nominal_Hz']:>6g}"
              f"{m['r_centroid_meas_mm']:>9.5f}{m['Amp_expected_D_over_4_mm']:>8.4f}"
              f"{m['ratio_r_centroid_vs_D_over_4']:>10.5f}{m['theta_meas_rad']:>11.4e}"
              f"{m['A_tine_true_mm']:>9.4f}{m['ratio_A_tine_true_vs_nominal']:>13.3f}"
              f"{m['Gamma_published']:>9.1f}{m['Gamma_true_tine']:>9.1f}"
              f"{m['xi_node_mm']:>9.1f}  {m['flags']}")

    rr = np.array([m['ratio_r_centroid_vs_D_over_4'] for m in rows])
    th = np.array([m['theta_meas_rad'] for m in rows])
    fe = np.array([m['f_meas_Hz'] - m['f_nominal_Hz'] for m in rows])
    gp = np.array([m['Gamma_published'] for m in rows])
    gt = np.array([m['Gamma_true_tine'] for m in rows])
    rt = np.array([m['ratio_A_tine_true_vs_nominal'] for m in rows])
    flagged = [m for m in rows if m['flags']]

    print(f"\ncases measured            : {len(rows)}")
    print(f"r_c / (D/4)               : {rr.min():.5f} .. {rr.max():.5f}")
    print(f"theta                     : {th.min():.6e} .. {th.max():.6e}")
    print(f"|f_measured - f_nominal|  : max {np.abs(fe).max():.4f} Hz")
    print(f"A_tine_true / A_nominal   : {rt.min():.2f} .. {rt.max():.2f}")
    print(f"Gamma published range     : {gp.min():.2f} .. {gp.max():.1f}  "
          f"(factor {gp.max()/gp.min():.0f})")
    print(f"Gamma TRUE range          : {gt.min():.2f} .. {gt.max():.1f}  "
          f"(factor {gt.max()/gt.min():.0f})")
    print(f"cases with Gamma_true < 1 : {(gt < 1).sum()}  "
          f"(published: {(gp < 1).sum()})")
    print(f"cases 20 < Gamma_true<100 : {((gt > 20) & (gt < 100)).sum()}  "
          f"(published: {((gp > 20) & (gp < 100)).sum()})")
    print(f"cases with Gamma_true>100 : {(gt > 100).sum()}  "
          f"(published: {(gp > 100).sum()})")

    if abs(rr - 1).max() < TOL_RC_RATIO:
        print('  => r_c = D/4 in every case: the input rule is uniform.')
    if th.std() / th.mean() < 1e-3:
        print(f'  => theta is CONSTANT at {th.mean():.4e} rad, so the wobble adds a '
              f'FIXED {args.l_half * th.mean():.4f} mm at the tine regardless of D.')
        print('     A_tine = D/4 + that constant, so only the D where the two are')
        print('     equal is correct.  Gamma must be rebuilt from A_tine_true_mm.')
    else:
        print('  => theta VARIES between cases; the tine table must be rebuilt per case.')

    if flagged:
        print(f'\nFLAGGED CASES ({len(flagged)}):')
        for m in flagged:
            print(f"  {m['case']:<16} {m['flags']}")
        print('  RC=centroid radius != D/4   WSD=|omega_h| not constant   FIT=poor circle')
        print('  COV=poor phase coverage     FREQ=frequency mismatch      WZ=axial spin')
        print('  ROOT=wobble adds at root    WDRW=tool rising (withdrawal)')
        print('  IN_A/IN_T=measured motion disagrees with in_bc.dat')
    else:
        print('\nno cases flagged.')

    if args.verify:
        op = os.path.join(args.base_dir, args.verify, 'output.dat')
        if os.path.exists(op):
            print(f'\ndirect Euler reconstruction from {args.verify}/output.dat ...')
            try:
                v = verify_with_euler(op, l_half=args.l_half)
                print(f"  records used     {v['n']}")
                print(f"  body axis        {v['axis']}  (|R0[2,axis]| = {v['axis_alignment']:.6f})")
                print(f"  phase coverage   centroid {v['coverage_centroid']:.2f}, "
                      f"tine {v['coverage_tine']:.2f}  (aliased at 50 Hz; a Kasa fit is")
                print( "                   order-independent, so coverage is what matters)")
                print(f"  centroid radius  {v['r_c']:.6f} mm")
                print(f"  tine radius      {v['R_tine']:.6f} mm   "
                      f"(circle scatter {v['scatter']:.1e})")
                print(f"  relation gives   {v['r_c'] + args.l_half * th.mean():.6f} mm")
                if v['coverage_tine'] < 0.9:
                    print('  WARNING: poor phase coverage, the aliased fit may be '
                          'ill-conditioned')
            except Exception as exc:
                print(f'  failed: {exc}', file=sys.stderr)
        else:
            print(f'\n--verify: no output.dat under {args.verify}', file=sys.stderr)

    if problems:
        print('\nissues:')
        for q in problems:
            print('  ' + q)

    keys = [
        # identification
        'case', 'D_folder_mm', 'f_nominal_Hz',
        # what the paper reports
        'A_nominal_mm', 'D_tine_nominal_mm', 'Gamma_published',
        # what in_bc.dat asked for
        'hz_in', 'Amp_in', 'Zkak_in', 'pret_in', 'Amp_expected_D_over_4_mm',
        # what the solver actually did
        'f_meas_Hz', 'r_centroid_meas_mm', 'ratio_r_centroid_vs_D_over_4',
        'theta_meas_rad', 'theta_sd_rad', 'theta_sd_rel',
        'wobble_adds_at_tine', 'wobble_phase_dot', 'wobble_offset_mm',
        # the true tool motion
        'A_root_true_mm', 'A_tine_true_mm', 'D_tine_true_mm',
        'A_mean_centroid_to_tine_mm', 'ratio_A_tine_true_vs_nominal', 'xi_node_mm',
        # corrected Gamma
        'Gamma_centroid', 'Gamma_true_tine', 'Gamma_true_mean',
        'Gamma_ratio_true_over_published',
        # quality / diagnostics
        'circle_scatter_mm', 'circle_scatter_rel', 'angular_coverage',
        'omega_z_rel', 'z_start_mm', 'z_end_mm', 'z_change_mm',
        'Fz_contact_mean_N', 'n_samples', 't_span_s', 'flags',
    ]
    with open(args.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {args.csv}')
    print('key columns: A_tine_true_mm (what the ballast actually saw) and '
          'Gamma_true_tine (the corrected abscissa for Figs. 13-19, 22, 23)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

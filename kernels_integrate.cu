#include "dem3d.h"

extern __constant__ double d_mat_rho[NMAT];

// ============================================================
// Device: 3x3 matrix-vector multiply
// ============================================================
__device__
void d_matvec3(const double R[3][3], const double v[3], double out[3]) {
    out[0] = R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2];
    out[1] = R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2];
    out[2] = R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2];
}

// ============================================================
// Device: Build rotation matrix from Euler angles (alpha, beta, gamma)
// ============================================================
__device__
void d_euler_matrix(double alp, double bet, double gam, double R[3][3]) {
    double ca = cos(alp), sa = sin(alp);
    double cb = cos(bet), sb = sin(bet);
    double cg = cos(gam), sg = sin(gam);
    R[0][0] = cb;          R[0][1] = -sb*ca;            R[0][2] = sb*sa;
    R[1][0] = cg*sb;       R[1][1] = cg*cb*ca - sg*sa;  R[1][2] = -cg*cb*sa - sg*ca;
    R[2][0] = sg*sb;       R[2][1] = sg*cb*ca + cg*sa;  R[2][2] = -sg*cb*sa + cg*ca;
}

// ============================================================
// Device: Transpose 3x3
// ============================================================
__device__
void d_transp3(const double A[3][3], double B[3][3]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            B[i][j] = A[j][i];
}

// ============================================================
// FIX A1: per-ELEMENT reduction of contact forces/moments into grains.
// One thread per element (57k) instead of one thread per grain: the wall grain
// has 7410 sub-spheres and the tool 3015, so a single thread walking those
// serially was ~40% of the whole step.
// grain_force_out must be zeroed before this kernel runs.
// ============================================================
__global__
void kernel_reduce_grain_forces(
    const double *pforce, const int *np_arr,
    const double *xc, const double *yc, const double *zc,
    const double *gcx, const double *gcy, const double *gcz,
    double *grain_force_out, int nelem)
{
    int ei = blockIdx.x * blockDim.x + threadIdx.x;
    if (ei >= nelem) return;
    int k = np_arr[ei];

    double fx = pforce[ei*6+0], fy = pforce[ei*6+1], fz = pforce[ei*6+2];
    double abx = xc[ei]-gcx[k], aby = yc[ei]-gcy[k], abz = zc[ei]-gcz[k];

    atomicAdd(&grain_force_out[k*6+0], fx);
    atomicAdd(&grain_force_out[k*6+1], fy);
    atomicAdd(&grain_force_out[k*6+2], fz);
    atomicAdd(&grain_force_out[k*6+3], pforce[ei*6+3] - fy*abz + fz*aby);
    atomicAdd(&grain_force_out[k*6+4], pforce[ei*6+4] - fz*abx + fx*abz);
    atomicAdd(&grain_force_out[k*6+5], pforce[ei*6+5] - fx*aby + fy*abx);
}

// ============================================================
// KERNEL: Integrate grain translations
// ============================================================
__global__
void kernel_integrate_translation(
    const double *pforce, const int *np_arr, const int *nm_arr,
    double *gcx, double *gcy, double *gcz,
    double *vgx, double *vgy, double *vgz,
    const double *gv, const int *nset, const int *elemStart,
    const double *xc, const double *yc, const double *zc,
    const int *icode_flat, const double *bval_flat,
    double dtime, double gx, double gy, double gz, double adamp,
    int nptotl,
    double *g_envx, double *g_envy, double *g_envz,
    double *g_engr,
    double *g_encx, double *g_ency, double *g_encz,
    double *g_enxf, double *g_enxd,
    double *grain_force_out,
    double *grain_gcx_prev, double *grain_gcy_prev, double *grain_gcz_prev)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nptotl) return;

    int es = elemStart[k];
    int nsk = nset[k];
    int mat = nm_arr[es];
    double rho_g = d_mat_rho[mat];
    double gmass = rho_g * gv[k];

    // FIX A1: totals already reduced by kernel_reduce_grain_forces.
    double pftx  = grain_force_out[k*6+0];
    double pfty  = grain_force_out[k*6+1];
    double pftz  = grain_force_out[k*6+2];
    double pftwx = grain_force_out[k*6+3];
    double pftwy = grain_force_out[k*6+4];
    double pftwz = grain_force_out[k*6+5];
    (void)nsk;

    // Apply force BCs
    if (icode_flat[k*6+0]==0) pftx += bval_flat[k*6+0];
    if (icode_flat[k*6+1]==0) pfty += bval_flat[k*6+1];
    if (icode_flat[k*6+2]==0) pftz += bval_flat[k*6+2];
    if (icode_flat[k*6+3]==0) pftwx += bval_flat[k*6+3];
    if (icode_flat[k*6+4]==0) pftwy += bval_flat[k*6+4];
    if (icode_flat[k*6+5]==0) pftwz += bval_flat[k*6+5];

    double accx = pftx/gmass + gx;
    double accy = pfty/gmass + gy;
    double accz = pftz/gmass + gz;
    if (icode_flat[k*6+0]==0 && bval_flat[k*6+0]!=0.0) accx -= gx;
    if (icode_flat[k*6+1]==0 && bval_flat[k*6+1]!=0.0) accy -= gy;
    if (icode_flat[k*6+2]==0 && bval_flat[k*6+2]!=0.0) accz -= gz;

    double vxp = vgx[k], vyp = vgy[k], vzp = vgz[k];
    vgx[k] += accx*dtime; vgy[k] += accy*dtime; vgz[k] += accz*dtime;
    vgx[k] *= adamp; vgy[k] *= adamp; vgz[k] *= adamp;

    if (icode_flat[k*6+0]==1) vgx[k] = bval_flat[k*6+0];
    if (icode_flat[k*6+1]==1) vgy[k] = bval_flat[k*6+1];
    if (icode_flat[k*6+2]==1) vgz[k] = bval_flat[k*6+2];

    // Kinetic energy
    double vax=(vxp+vgx[k])/2, vay=(vyp+vgy[k])/2, vaz=(vzp+vgz[k])/2;
    atomicAdd(g_envx, 0.5*gmass*vax*vax);
    atomicAdd(g_envy, 0.5*gmass*vay*vay);
    atomicAdd(g_envz, 0.5*gmass*vaz*vaz);
    atomicAdd(g_engr, -gmass*(gx*vgx[k]+gy*vgy[k]+gz*vgz[k])*dtime);

    // Artificial damping energy
    double dp = adamp;
    atomicAdd(g_encx, 0.5*gmass*(vxp+(1.0/dp+1.0)*vgx[k]/2.0)*((1.0/dp-1.0)*vgx[k]/2.0));
    atomicAdd(g_ency, 0.5*gmass*(vyp+(1.0/dp+1.0)*vgy[k]/2.0)*((1.0/dp-1.0)*vgy[k]/2.0));
    atomicAdd(g_encz, 0.5*gmass*(vzp+(1.0/dp+1.0)*vgz[k]/2.0)*((1.0/dp-1.0)*vgz[k]/2.0));

    // External energy
    if (icode_flat[k*6+0]==0) atomicAdd(g_enxf, -bval_flat[k*6+0]*vgx[k]*dtime);
    if (icode_flat[k*6+1]==0) atomicAdd(g_enxf, -bval_flat[k*6+1]*vgy[k]*dtime);
    if (icode_flat[k*6+2]==0) atomicAdd(g_enxf, -bval_flat[k*6+2]*vgz[k]*dtime);

    double pftx0 = grain_force_out[k*6+0], pfty0 = grain_force_out[k*6+1], pftz0 = grain_force_out[k*6+2];
    if (icode_flat[k*6+0]==1) atomicAdd(g_enxd, bval_flat[k*6+0]*pftx0*dtime);
    if (icode_flat[k*6+1]==1) atomicAdd(g_enxd, bval_flat[k*6+1]*pfty0*dtime);
    if (icode_flat[k*6+2]==1) atomicAdd(g_enxd, bval_flat[k*6+2]*pftz0*dtime);

    // Store previous position
    grain_gcx_prev[k] = gcx[k];
    grain_gcy_prev[k] = gcy[k];
    grain_gcz_prev[k] = gcz[k];

    // Update position
    gcx[k] += vgx[k]*dtime;
    gcy[k] += vgy[k]*dtime;
    gcz[k] += vgz[k]*dtime;
}

// ============================================================
// KERNEL: Integrate grain rotations (Euler equations)
// + update element positions
// ============================================================
__global__
void kernel_integrate_rotation(
    double *gcx, double *gcy, double *gcz,
    double *gwx, double *gwy, double *gwz,
    double *vgwx, double *vgwy, double *vgwz,
    const double *gv, const double *gi_flat, // gi_flat[k*3+{0,1,2}]
    const int *nset, const int *elemStart,
    const int *nm_arr,
    double *xc, double *yc, double *zc,
    double *vxc_e, double *vyc_e, double *vzc_e,
    double *vwx_e, double *vwy_e, double *vwz_e,
    const double *vgx, const double *vgy, const double *vgz,
    const int *icode_flat, const double *bval_flat,
    const double *grain_force_out,
    const double *grain_gcx_prev, const double *grain_gcy_prev, const double *grain_gcz_prev,
    double dtime, int nptotl,
    double *g_envw, double *g_enxf, double *g_enxd,
    double *rot_rrt, double *rot_rr2, double *rot_rr3)   // FIX A2
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nptotl) return;

    int es = elemStart[k];
    int mat = nm_arr[es];
    double rho_g = d_mat_rho[mat];

    double alp = gwx[k], bet = gwy[k], gam = gwz[k];

    // Build R_t
    double rr[3][3], rrt[3][3];
    d_euler_matrix(alp, bet, gam, rr);
    d_transp3(rr, rrt);

    // Apply angular velocity BCs
    if (icode_flat[k*6+3]==1) vgwx[k] = bval_flat[k*6+3];
    if (icode_flat[k*6+4]==1) vgwy[k] = bval_flat[k*6+4];
    if (icode_flat[k*6+5]==1) vgwz[k] = bval_flat[k*6+5];

    double vgwG[3] = {vgwx[k], vgwy[k], vgwz[k]};
    double vgw[3];
    d_matvec3(rrt, vgwG, vgw);

    double pftwx = grain_force_out[k*6+3];
    double pftwy = grain_force_out[k*6+4];
    double pftwz = grain_force_out[k*6+5];
    // Note: force BCs for rotation were already applied in translation kernel to grain_force_out
    // Actually we need the raw + BC version:
    if (icode_flat[k*6+3]==0) pftwx += bval_flat[k*6+3];
    if (icode_flat[k*6+4]==0) pftwy += bval_flat[k*6+4];
    if (icode_flat[k*6+5]==0) pftwz += bval_flat[k*6+5];

    double fmG[3] = {pftwx, pftwy, pftwz};
    double fm[3];
    d_matvec3(rrt, fmG, fm);

    double gi1 = gi_flat[k*3+0]*rho_g;
    double gi2 = gi_flat[k*3+1]*rho_g;
    double gi3 = gi_flat[k*3+2]*rho_g;

    double vgwl[3] = {vgw[0], vgw[1], vgw[2]};
    double alpm = alp, betm = bet, gamm = gam;

    // Iterative Euler equation solver
    double vgwc[3], alpl, betl, gaml;
    double vgwln[3] = {vgw[0], vgw[1], vgw[2]}; // midpoint for energy (like Fortran vgwln)
    for (int it = 0; it < 5; it++) {
        double aw0 = (fm[0] + (gi2-gi3)*vgwl[1]*vgwl[2]) / gi1;
        double aw1 = (fm[1] + (gi3-gi1)*vgwl[2]*vgwl[0]) / gi2;
        double aw2 = (fm[2] + (gi1-gi2)*vgwl[0]*vgwl[1]) / gi3;

        vgwc[0] = vgw[0] + aw0*dtime;
        vgwc[1] = vgw[1] + aw1*dtime;
        vgwc[2] = vgw[2] + aw2*dtime;

        if (icode_flat[k*6+3]==1) vgwc[0] = vgw[0];
        if (icode_flat[k*6+4]==1) vgwc[1] = vgw[1];
        if (icode_flat[k*6+5]==1) vgwc[2] = vgw[2];

        if (sin(betm) == 0.0) betm += 1.0e-6;

        double tmp = vgwc[2]*sin(alpm) - vgwc[1]*cos(alpm);
        double dgw0 = vgwc[0] - tmp/tan(betm);
        double dgw1 = vgwc[1]*sin(alpm) + vgwc[2]*cos(alpm);
        double dgw2 = tmp/sin(betm);

        alpl = alp + dgw0*dtime;
        betl = bet + dgw1*dtime;
        gaml = gam + dgw2*dtime;

        double vln0 = (vgw[0]+vgwc[0])*0.5;
        double vln1 = (vgw[1]+vgwc[1])*0.5;
        double vln2 = (vgw[2]+vgwc[2])*0.5;
        double aln = (alp+alpl)*0.5, bln = (bet+betl)*0.5, gln = (gam+gaml)*0.5;

        // Always update vgwln (midpoint for energy), matching Fortran's vgwln
        vgwln[0] = vln0; vgwln[1] = vln1; vgwln[2] = vln2;

        double err = fabs(vgwl[0]-vln0) + fabs(vgwl[1]-vln1) + fabs(vgwl[2]-vln2)
                   + fabs(alpm-aln) + fabs(betm-bln) + fabs(gamm-gln);
        if (err < 1.0e-10) break;

        vgwl[0] = vln0; vgwl[1] = vln1; vgwl[2] = vln2;
        alpm = aln; betm = bln; gamm = gln;
    }

    // R_t+dt/2
    double rr2[3][3];
    d_euler_matrix(alpm, betm, gamm, rr2);

    double vgwG2[3];
    d_matvec3(rr2, vgwc, vgwG2);

    double vgwxp = vgwx[k], vgwyp = vgwy[k], vgwzp = vgwz[k];
    vgwx[k] = vgwG2[0]; vgwy[k] = vgwG2[1]; vgwz[k] = vgwG2[2];

    // Update Euler angles
    gwx[k] = alpl; gwy[k] = betl; gwz[k] = gaml;

    // Rotational kinetic energy (using midpoint vgwln, matching Fortran)
    atomicAdd(g_envw, 0.5*(gi1*vgwln[0]*vgwln[0] + gi2*vgwln[1]*vgwln[1] + gi3*vgwln[2]*vgwln[2]));

    // External energy from rotation
    if (icode_flat[k*6+3]==0) atomicAdd(g_enxf, -bval_flat[k*6+3]*vgwx[k]*dtime);
    if (icode_flat[k*6+4]==0) atomicAdd(g_enxf, -bval_flat[k*6+4]*vgwy[k]*dtime);
    if (icode_flat[k*6+5]==0) atomicAdd(g_enxf, -bval_flat[k*6+5]*vgwz[k]*dtime);

    double pftwx0 = grain_force_out[k*6+3], pftwy0 = grain_force_out[k*6+4], pftwz0 = grain_force_out[k*6+5];
    if (icode_flat[k*6+3]==1) atomicAdd(g_enxd, bval_flat[k*6+3]*pftwx0*dtime);
    if (icode_flat[k*6+4]==1) atomicAdd(g_enxd, bval_flat[k*6+4]*pftwy0*dtime);
    if (icode_flat[k*6+5]==1) atomicAdd(g_enxd, bval_flat[k*6+5]*pftwz0*dtime);

    // FIX A2: store this grain's three rotation matrices; the element update is
    // done by kernel_update_elements, one thread per element.
    double rr3[3][3];
    d_euler_matrix(alpl, betl, gaml, rr3);   // R at t+dt
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
        rot_rrt[k*9 + a*3 + b] = rrt[a][b];  // R(t)^T
        rot_rr2[k*9 + a*3 + b] = rr2[a][b];  // R(t+dt/2)
        rot_rr3[k*9 + a*3 + b] = rr3[a][b];  // R(t+dt)
    }
}

// ============================================================
// FIX A2: per-ELEMENT position/velocity update, one thread per element.
// Grains whose six DOFs are all velocity-prescribed at zero (the container
// walls) never move, so their elements are skipped: exact, and it removes the
// slowest thread (7410 sub-spheres).
// ============================================================
__global__
void kernel_update_elements(
    const int *np_arr, const int *grain_fixed,
    const double *rot_rrt, const double *rot_rr2, const double *rot_rr3,
    const double *gcx, const double *gcy, const double *gcz,
    const double *grain_gcx_prev, const double *grain_gcy_prev, const double *grain_gcz_prev,
    const double *vgx, const double *vgy, const double *vgz,
    const double *vgwx, const double *vgwy, const double *vgwz,
    double *xc, double *yc, double *zc,
    double *vxc_e, double *vyc_e, double *vzc_e,
    double *vwx_e, double *vwy_e, double *vwz_e,
    int nelem)
{
    int ei = blockIdx.x * blockDim.x + threadIdx.x;
    if (ei >= nelem) return;
    int k = np_arr[ei];
    if (grain_fixed[k]) return;

    double rrt[3][3], rr2[3][3], rr3[3][3];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
        rrt[a][b] = rot_rrt[k*9 + a*3 + b];
        rr2[a][b] = rot_rr2[k*9 + a*3 + b];
        rr3[a][b] = rot_rr3[k*9 + a*3 + b];
    }

    double al[3] = {xc[ei]-grain_gcx_prev[k], yc[ei]-grain_gcy_prev[k], zc[ei]-grain_gcz_prev[k]};
    double tmp1[3];
    d_matvec3(rrt, al, tmp1);            // to body frame

    double tmp2[3];
    d_matvec3(rr2, tmp1, tmp2);          // branch at t+dt/2, for velocity
    double wx = vgwx[k], wy = vgwy[k], wz = vgwz[k];
    vxc_e[ei] = vgx[k] + wy*tmp2[2] - wz*tmp2[1];
    vyc_e[ei] = vgy[k] + wz*tmp2[0] - wx*tmp2[2];
    vzc_e[ei] = vgz[k] + wx*tmp2[1] - wy*tmp2[0];
    vwx_e[ei] = wx; vwy_e[ei] = wy; vwz_e[ei] = wz;

    double tmp3[3];
    d_matvec3(rr3, tmp1, tmp3);          // branch at t+dt, for position
    xc[ei] = gcx[k] + tmp3[0];
    yc[ei] = gcy[k] + tmp3[1];
    zc[ei] = gcz[k] + tmp3[2];
}

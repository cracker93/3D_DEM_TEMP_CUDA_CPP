#include "dem3d.h"

// Declared in kernels_helpers.cu
extern __constant__ double d_mat_rho[NMAT];
extern __constant__ double d_mat_akn[NMAT];
extern __constant__ double d_mat_aks[NMAT];
extern __constant__ double d_mat_cn[NMAT];
extern __constant__ double d_mat_cs[NMAT];
extern __constant__ double d_mat_phimu[NMAT];

__device__ double periodic_wrap(double x, double center, double length, int isPeriodic);
__device__ void d_c_judge(double,double,double,double, double,double,double,double,
                          double&,double&,double&, double&, double&,double&,double&);
__device__ void d_cfij(double,double,double, double,double,double, double,double,double,
                       double,double,double, double,double,double, double,double,double,
                       double,int, double,double,double, double,double,double, double,
                       double,double,double,double,double,
                       double&,double&,double&,
                       double&,double&,double&,
                       double&,double&,double&, double&,double&,double&,
                       double&,double&,double&,double&,
                       double&,double&,double&,
                       double&,double&,double&,
                       int,int,int);

// ============================================================
// KERNEL: Initialize forces to zero
// ============================================================
__global__
void kernel_init_forces(double *pforce, int *icount, int nelem) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelem) return;
    for (int d = 0; d < 6; d++) pforce[i*6+d] = 0.0;
    icount[i] = 0;
}

// ============================================================
// KERNEL: Main contact detection and force computation
// One thread per element i, checks all 27 neighbor cells
// Only processes pairs where j > i to avoid double counting
// ============================================================
// PERF: warp-level sum (all 32 lanes must call it)
__device__ __forceinline__ double wsum(double v){
    for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffff,v,o);
    return v;
}

__global__
void kernel_contact_forces(
    const double *xc, const double *yc, const double *zc, const double *rc,
    const double *vxc, const double *vyc, const double *vzc,
    const double *vwx, const double *vwy, const double *vwz,
    const int *np, const int *nm,
    const double *gcx, const double *gcy, const double *gcz,
    const int *sortedIndex, const int *cellStart, const int *cellEnd,
    int ndivx, int ndivy, int ndivz,
    double dmxmin, double dmymin, double dmzmin,
    double xdiv, double ydiv, double zdiv,
    int ipbx, int ipby, int ipbz, int ipb3,
    double pblx, double pbly, double pblz,
    double pbcx, double pbcy, double pbcz,
    const int *old_neib, const double *old_fcont, const double *old_alpha,
    const int *old_icount,
    int *new_neib, double *new_fcont, double *new_alpha, int *new_icount,
    double *pforce,
    double *sig_out,
    const int *ipboun, int npboun,
    double dtime, double vr, int mode1, int istep, int nstep,
    int nkk1, int nkk2, int nelem,
    double *g_enkn, double *g_enks, double *g_enkt,
    double *g_encn, double *g_encs, double *g_enct,
    double *g_enfr, double *g_enfr1, double *g_enfr2, double *g_enfr3,
    int *g_ncont, int *g_diag)
{
    // PERF: threads walk the spatially sorted order (neighbouring threads ->
    // neighbouring elements).  No early return: every lane must reach wsum().
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = tid < nelem;
    int i = active ? sortedIndex[tid] : 0;
    double ls[9] = {0,0,0,0,0,0,0,0,0}; double lnc = 0.0;

    double xci = periodic_wrap(xc[i], pbcx, pblx, ipbx);
    double yci = periodic_wrap(yc[i], pbcy, pbly, ipby);
    double zci = periodic_wrap(zc[i], pbcz, pblz, ipbz);

    int cx = max(0, min((int)((xci - dmxmin) / xdiv), ndivx-1));
    int cy = max(0, min((int)((yci - dmymin) / ydiv), ndivy-1));
    int cz = max(0, min((int)((zci - dmzmin) / zdiv), ndivz-1));

    double le_enkn=0, le_enks=0, le_enkt=0;
    double le_encn=0, le_encs=0, le_enct=0;
    double le_enfr=0, le_enfr1=0, le_enfr2=0, le_enfr3=0;

    if (active)
    for (int di = -1; di <= 1; di++)
    for (int dj = -1; dj <= 1; dj++)
    for (int dk = -1; dk <= 1; dk++) {
        int ci = cx+di, cj = cy+dj, ck = cz+dk;
        double px=0, py=0, pz=0;

        if (ci<0)       { if (!ipbx) continue; ci+=ndivx; px=-pblx; }
        if (ci>=ndivx)  { if (!ipbx) continue; ci-=ndivx; px=pblx; }
        if (cj<0)       { if (!ipby) continue; cj+=ndivy; py=-pbly; }
        if (cj>=ndivy)  { if (!ipby) continue; cj-=ndivy; py=pbly; }
        if (ck<0)       { if (!ipbz) continue; ck+=ndivz; pz=-pblz; }
        if (ck>=ndivz)  { if (!ipbz) continue; ck-=ndivz; pz=pblz; }

        int cell = ci*ndivy*ndivz + cj*ndivz + ck;
        int s0 = cellStart[cell], s1 = cellEnd[cell];

        for (int s = s0; s < s1; s++) {
            int j = sortedIndex[s];
            if (j <= i) continue;
            if (np[j] == np[i]) continue;

            double xcj = periodic_wrap(xc[j], pbcx, pblx, ipbx) + px;
            double ycj = periodic_wrap(yc[j], pbcy, pbly, ipby) + py;
            double zcj = periodic_wrap(zc[j], pbcz, pblz, ipbz) + pz;

            // PERF: cheap reject without sqrt; identical to d_c_judge's test (dist < ri+rj)
            { double ddx=xcj-xci, ddy=ycj-yci, ddz=zcj-zci, rr=rc[i]+rc[j];
              if (ddx*ddx+ddy*ddy+ddz*ddz >= rr*rr) continue; }
            double anx, any, anz, ovrlap, cpx, cpy, cpz;
            d_c_judge(xci,yci,zci,rc[i], xcj,ycj,zcj,rc[j],
                      anx,any,anz,ovrlap,cpx,cpy,cpz);

            if (ovrlap >= 0.0) continue;
            // FIX 2a: do NOT skip the contact when the list is full.  The old
            // "if (my_cnt >= NEIMAX) continue;" silently dropped the contact
            // FORCE, which the Fortran never does (it stops instead).

            // Old contact history lookup
            int jcont = 0;
            double ofc1=0, ofc2=0, oal=0;
            // FIX 2b: clamp.  new_icount can exceed NEIMAX (see below), and an
            // unclamped bound reads into the NEXT element's slots, which can
            // produce a spurious history match.
            int oic = min(old_icount[i], NEIMAX);
            for (int nn = 0; nn < oic; nn++) {
                if (old_neib[i*NEIMAX+nn] == j) {
                    jcont = 1;
                    ofc1 = old_fcont[(i*NEIMAX+nn)*2+0];
                    ofc2 = old_fcont[(i*NEIMAX+nn)*2+1];
                    oal  = old_alpha[i*NEIMAX+nn];
                    break;
                }
            }

            double fc1=ofc1, fc2=ofc2, al=oal;

            double aknt = sqrt(d_mat_akn[nm[i]]*d_mat_akn[nm[j]]);
            double akst = sqrt(d_mat_aks[nm[i]]*d_mat_aks[nm[j]]);
            double cnt  = sqrt(d_mat_cn[nm[i]]*d_mat_cn[nm[j]]);
            double cst  = sqrt(d_mat_cs[nm[i]]*d_mat_cs[nm[j]]);
            double phimut = sqrt(d_mat_phimu[nm[i]]*d_mat_phimu[nm[j]]);

            if (mode1 == 1) {
                double fac = (double)istep/(double)nstep;
                aknt*=fac; akst*=fac; cnt*=fac; cst*=fac;
            }

            double fx,fy,fz, fw1x,fw1y,fw1z, fw2x,fw2y,fw2z;
            d_cfij(xci,yci,zci, vxc[i],vyc[i],vzc[i], vwx[i],vwy[i],vwz[i],
                   xcj,ycj,zcj, vxc[j],vyc[j],vzc[j], vwx[j],vwy[j],vwz[j],
                   dtime,jcont, anx,any,anz, cpx,cpy,cpz, ovrlap,
                   aknt,akst,cnt,cst,phimut,
                   fc1,fc2,al, fx,fy,fz, fw1x,fw1y,fw1z, fw2x,fw2y,fw2z,
                   le_enfr,le_enfr1,le_enfr2,le_enfr3,
                   le_enkn,le_enks,le_enkt, le_encn,le_encs,le_enct,
                   nkk1,nkk2,j);

            // Store contact for i (use atomicAdd to avoid race with j-threads).
            // NOTE: new_icount[i] counts FORWARD contacts of thread i plus
            // REVERSE contacts written by threads k < i, so it can overflow even
            // when this thread has found fewer than NEIMAX partners.
            int ic = atomicAdd(&new_icount[i], 1);
            if (ic < NEIMAX) {
                new_neib[i*NEIMAX + ic] = j;
                new_fcont[(i*NEIMAX + ic)*2+0] = fc1;
                new_fcont[(i*NEIMAX + ic)*2+1] = fc2;
                new_alpha[i*NEIMAX + ic] = al;
            } else {
                // FIX 2c: history lost -> the shear spring of this contact would
                // silently reset every step.  Count it so the run can be stopped.
                atomicAdd(&g_diag[0], 1);
            }
            atomicMax(&g_diag[1], ic+1);

            // Forces on i (atomic)
            atomicAdd(&pforce[i*6+0], fx);
            atomicAdd(&pforce[i*6+1], fy);
            atomicAdd(&pforce[i*6+2], fz);
            atomicAdd(&pforce[i*6+3], fw1x);
            atomicAdd(&pforce[i*6+4], fw1y);
            atomicAdd(&pforce[i*6+5], fw1z);

            // Forces on j (Newton 3rd)
            atomicAdd(&pforce[j*6+0], -fx);
            atomicAdd(&pforce[j*6+1], -fy);
            atomicAdd(&pforce[j*6+2], -fz);
            atomicAdd(&pforce[j*6+3], fw2x);
            atomicAdd(&pforce[j*6+4], fw2y);
            atomicAdd(&pforce[j*6+5], fw2z);

            // Store reverse contact for j
            int jc = atomicAdd(&new_icount[j], 1);
            if (jc < NEIMAX) {
                new_neib[j*NEIMAX+jc] = i;
                new_fcont[(j*NEIMAX+jc)*2+0] = -fc1;
                new_fcont[(j*NEIMAX+jc)*2+1] = -fc2;
                new_alpha[j*NEIMAX+jc] = -al;
            } else {
                atomicAdd(&g_diag[0], 1);
            }
            atomicMax(&g_diag[1], jc+1);

            // Stress
            if (vr > 0.0) {
                // PERF: ipboun is now a per-grain flag array (see gpu_memory.cu)
                bool skip = ipboun[np[i]] || ipboun[np[j]];
                if (!skip) {
                    double gxi = gcx[np[i]]+xci-xc[i], gxj = gcx[np[j]]+xcj-xc[j];
                    double gyi = gcy[np[i]]+yci-yc[i], gyj = gcy[np[j]]+ycj-yc[j];
                    double gzi = gcz[np[i]]+zci-zc[i], gzj = gcz[np[j]]+zcj-zc[j];
                    double lx=gxj-gxi, ly=gyj-gyi, lz=gzj-gzi;
                    ls[0]+=lx*fx/vr; ls[1]+=lx*fy/vr; ls[2]+=lx*fz/vr;
                    ls[3]+=ly*fx/vr; ls[4]+=ly*fy/vr; ls[5]+=ly*fz/vr;
                    ls[6]+=lz*fx/vr; ls[7]+=lz*fy/vr; ls[8]+=lz*fz/vr;
                }
            }
            lnc += 1.0;
        }
    }

    // i's contact count already incremented per-contact above via atomicAdd

    // PERF: reduce energies, stress and contact count per warp, then one
    // atomic per warp instead of one per thread / per contact.
    double e[10] = {le_enkn,le_enks,le_enkt,le_encn,le_encs,le_enct,
                    le_enfr,le_enfr1,le_enfr2,le_enfr3};
    double *ge[10] = {g_enkn,g_enks,g_enkt,g_encn,g_encs,g_enct,
                      g_enfr,g_enfr1,g_enfr2,g_enfr3};
    #pragma unroll
    for (int k=0;k<10;k++) e[k] = wsum(e[k]);
    #pragma unroll
    for (int k=0;k<9;k++)  ls[k] = wsum(ls[k]);
    lnc = wsum(lnc);
    if ((threadIdx.x & 31) == 0) {
        for (int k=0;k<10;k++) atomicAdd(ge[k], e[k]);
        if (vr > 0.0) for (int k=0;k<9;k++) atomicAdd(&sig_out[k], ls[k]);
        atomicAdd(g_ncont, (int)(lnc + 0.5));
    }
}

// ============================================================
// KERNEL: Wall contact forces
// ============================================================
__global__
void kernel_wall_forces(
    const double *xc, const double *yc, const double *zc, const double *rc,
    const double *vxc, const double *vyc, const double *vzc,
    const int *nm, double *pforce,
    int ipbx, int ipby, int ipbz,
    double pbcx, double pbcy, double pbcz,
    double pblx, double pbly, double pblz,
    double *g_enkn, double *g_encn,
    double dtime, int nelem)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelem) return;

    double pos[3], vel[3];
    pos[0] = periodic_wrap(xc[i], pbcx, pblx, ipbx);
    pos[1] = periodic_wrap(yc[i], pbcy, pbly, ipby);
    pos[2] = periodic_wrap(zc[i], pbcz, pblz, ipbz);
    vel[0] = vxc[i]; vel[1] = vyc[i]; vel[2] = vzc[i];

    double bdp[6] = {pbcx-pblx/2, pbcx+pblx/2,
                     pbcy-pbly/2, pbcy+pbly/2,
                     pbcz-pblz/2, pbcz+pblz/2};
    int ipb[3] = {ipbx, ipby, ipbz};
    int mat = nm[i];

    for (int ib = 0; ib < 3; ib++) {
        if (ipb[ib] == 1) continue;

        // Bottom
        double dd = pos[ib] - bdp[ib*2];
        double ov = dd - rc[i];
        if (ov < 0.0) {
            double fb = -d_mat_akn[mat]*ov - vel[ib]*d_mat_cn[mat];
            atomicAdd(&pforce[i*6+ib], fb);
            atomicAdd(g_enkn, d_mat_akn[mat]*ov*ov*0.5);
            atomicAdd(g_encn, d_mat_cn[mat]*vel[ib]*vel[ib]*dtime);
        }
        // Top
        dd = bdp[ib*2+1] - pos[ib];
        ov = dd - rc[i];
        if (ov < 0.0) {
            double fb = d_mat_akn[mat]*ov - vel[ib]*d_mat_cn[mat];
            atomicAdd(&pforce[i*6+ib], fb);
            atomicAdd(g_enkn, d_mat_akn[mat]*ov*ov*0.5);
            atomicAdd(g_encn, d_mat_cn[mat]*vel[ib]*vel[ib]*dtime);
        }
    }
}

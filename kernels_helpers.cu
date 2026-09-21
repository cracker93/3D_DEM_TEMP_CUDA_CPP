#include "dem3d.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

// ============================================================
// Device constant memory for materials
// ============================================================
__constant__ double d_mat_rho[NMAT];
__constant__ double d_mat_akn[NMAT];
__constant__ double d_mat_aks[NMAT];
__constant__ double d_mat_cn[NMAT];
__constant__ double d_mat_cs[NMAT];
__constant__ double d_mat_phimu[NMAT];

void upload_material_constants(MaterialProps &mat) {
    cudaMemcpyToSymbol(d_mat_rho, mat.rho, sizeof(double)*NMAT);
    cudaMemcpyToSymbol(d_mat_akn, mat.akn, sizeof(double)*NMAT);
    cudaMemcpyToSymbol(d_mat_aks, mat.aks, sizeof(double)*NMAT);
    cudaMemcpyToSymbol(d_mat_cn, mat.cn, sizeof(double)*NMAT);
    cudaMemcpyToSymbol(d_mat_cs, mat.cs, sizeof(double)*NMAT);
    cudaMemcpyToSymbol(d_mat_phimu, mat.phimu, sizeof(double)*NMAT);
}

// ============================================================
// Device helper: periodic wrap
// ============================================================
__device__
double periodic_wrap(double x, double center, double length, int isPeriodic) {
    if (isPeriodic)
        return x - round((x - center) / length) * length;
    return x;
}

// ============================================================
// KERNEL: Compute cell index for spatial hashing
// ============================================================
__global__
void kernel_compute_cell_index(
    const double *xc, const double *yc, const double *zc,
    int *cellIndex, int *particleIndex, int nelem,
    int ipbx, int ipby, int ipbz,
    double pbcx, double pbcy, double pbcz,
    double pblx, double pbly, double pblz,
    double dmxmin, double dmymin, double dmzmin,
    double xdiv, double ydiv, double zdiv,
    int ndivx, int ndivy, int ndivz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelem) return;

    double xct = periodic_wrap(xc[i], pbcx, pblx, ipbx);
    double yct = periodic_wrap(yc[i], pbcy, pbly, ipby);
    double zct = periodic_wrap(zc[i], pbcz, pblz, ipbz);

    int ix = max(0, min((int)((xct - dmxmin) / xdiv), ndivx - 1));
    int iy = max(0, min((int)((yct - dmymin) / ydiv), ndivy - 1));
    int iz = max(0, min((int)((zct - dmzmin) / zdiv), ndivz - 1));

    cellIndex[i] = ix * ndivy * ndivz + iy * ndivz + iz;
    particleIndex[i] = i;
}

// ============================================================
// KERNEL: Find cell start/end boundaries after sort
// ============================================================
__global__
void kernel_find_cell_bounds(
    const int *sortedCellIndex, int *cellStart, int *cellEnd, int nelem)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelem) return;

    int cell = sortedCellIndex[i];
    if (i == 0 || sortedCellIndex[i - 1] != cell)
        cellStart[cell] = i;
    if (i == nelem - 1 || sortedCellIndex[i + 1] != cell)
        cellEnd[cell] = i + 1;
}

// ============================================================
// Device: Contact judgment between two spheres
// ============================================================
__device__
void d_c_judge(double xc1, double yc1, double zc1, double rc1,
               double xc2, double yc2, double zc2, double rc2,
               double &anx, double &any, double &anz,
               double &ovrlap, double &cpx, double &cpy, double &cpz)
{
    double dx = xc2 - xc1, dy = yc2 - yc1, dz = zc2 - zc1;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    if (dist < 1.0e-30) { ovrlap = 1.0; return; }
    anx = dx/dist; any = dy/dist; anz = dz/dist;
    double rr = rc1 + rc2;
    ovrlap = -(rr - dist);
    cpx = xc1 + dx*rc1/rr;
    cpy = yc1 + dy*rc1/rr;
    cpz = zc1 + dz*rc1/rr;
}

// ============================================================
// Device: Contact force computation (cfij)
// ============================================================
__device__
void d_cfij(
    double xci, double yci, double zci,
    double vxci, double vyci, double vzci,
    double vwxi, double vwyi, double vwzi,
    double xcj, double ycj, double zcj,
    double vxcj, double vycj, double vzcj,
    double vwxj, double vwyj, double vwzj,
    double dtime, int jcont,
    double anx, double any, double anz,
    double cpx, double cpy, double cpz, double ovrlap,
    double akn, double aks, double cn, double cs, double phimu_deg,
    double &fcont1, double &fcont2, double &alpha_val,
    double &fx, double &fy, double &fz,
    double &fwx1, double &fwy1, double &fwz1,
    double &fwx2, double &fwy2, double &fwz2,
    double &l_enfr, double &l_enfr1, double &l_enfr2, double &l_enfr3,
    double &l_enkn, double &l_enks, double &l_enkt,
    double &l_encn, double &l_encs, double &l_enct,
    int nkk1, int nkk2, int j_idx)
{
    const double PI = 3.14159265358979323846;
    const double RAD = PI / 180.0;
    double akt = aks, ct = cs;

    double alx = cpx-xci, aly = cpy-yci, alz = cpz-zci;
    double blx = cpx-xcj, bly = cpy-ycj, blz = cpz-zcj;

    // Relative velocity at contact point
    double relvx = (vxcj-vxci) - (vwzj*bly-vwzi*aly) + (vwyj*blz-vwyi*alz);
    double relvy = (vycj-vyci) - (vwxj*blz-vwxi*alz) + (vwzj*blx-vwzi*alx);
    double relvz = (vzcj-vzci) - (vwyj*blx-vwyi*alx) + (vwxj*bly-vwxi*aly);

    double vwn1 = anx*vwxi + any*vwyi + anz*vwzi;
    double vwn2 = anx*vwxj + any*vwyj + anz*vwzj;
    alpha_val += (vwn1 + vwn2) * 0.5 * dtime;

    // Contact orientation vectors
    double sx, sy, sz, tx, ty, tz;
    double anyz = sqrt(any*any + anz*anz);

    if (anyz < 1.0e-30) {
        double sgn = (anx >= 0.0) ? 1.0 : -1.0;
        double ca = cos(alpha_val), sa = sin(alpha_val);
        sx = 0.0; sy = sgn*ca; sz = sa;
        tx = 0.0; ty = -sgn*sa; tz = ca;
    } else {
        double sgn = 1.0;
        double sgam = sgn*anyz, cgam = anx;
        double sbet = sgn*anz/anyz, cbet = sgn*any/anyz;
        double ca = cos(alpha_val), sa = sin(alpha_val);
        sx = -ca*sgam; sy = ca*cgam*cbet - sa*sbet; sz = ca*cgam*sbet + sa*cbet;
        tx = sa*sgam;  ty = -sa*cgam*cbet - ca*sbet; tz = -sa*cgam*sbet + ca*cbet;
    }

    double relvn = anx*relvx + any*relvy + anz*relvz;
    double relvs = sx*relvx + sy*relvy + sz*relvz;
    double relvt = tx*relvx + ty*relvy + tz*relvz;

    double relds = relvs * dtime;
    double reldt = relvt * dtime;

    if (jcont == 0) {
        if (-ovrlap < -relvn*dtime) {
            double fac = ovrlap / (relvn*dtime);
            relds *= fac; reldt *= fac;
        }
    }

    double fn = ovrlap * akn;

    if (fn < 0.0) {
        double fs = fcont1 + relds*aks;
        double ft = fcont2 + reldt*akt;
        double totds = fs/aks, totdt = ft/akt;

        double fnt = fn + cn*relvn;
        double fst = fs + cs*relvs;
        double ftt = ft + ct*relvt;

        double frict = fabs(fnt * tan(phimu_deg*RAD));
        double fstt = sqrt(fst*fst + ftt*ftt);

        // Sliding
        if (fabs(fstt) > frict) {
            if (fstt != 0.0) {
                fst = frict*fst/fstt;
                ftt = frict*ftt/fstt;
            } else { fst = 0.0; ftt = 0.0; }

            double dds1 = (fst - fcont1) / (aks + cs/dtime);
            double dds2 = relds - dds1;
            relvs = dds1/dtime;
            fs = fcont1 + aks*dds1;
            totds = fs/aks;

            double ddt1 = (ftt - fcont2) / (akt + ct/dtime);
            double ddt2 = reldt - ddt1;
            relvt = ddt1/dtime;
            ft = fcont2 + akt*ddt1;
            totdt = ft/akt;

            double fr_ene = fabs(frict * sqrt(dds2*dds2 + ddt2*ddt2));
            l_enfr += fr_ene;
            int j1 = j_idx + 1; // 1-based for comparison
            if (j1 > nkk2) l_enfr2 += fr_ene;
            else if (j1 > nkk1) l_enfr3 += fr_ene;
            else l_enfr1 += fr_ene;
        }

        fcont1 = fs; fcont2 = ft;
        l_enkn += akn*ovrlap*ovrlap*0.5;
        l_enks += aks*totds*totds*0.5;
        l_enkt += akt*totdt*totdt*0.5;

        if (jcont == 0) {
            l_encn += fabs(cn*relvn*ovrlap);
            l_encs += fabs(cs*relvs*relds);
            l_enct += fabs(ct*relvt*reldt);
        } else {
            l_encn += cn*relvn*relvn*dtime;
            l_encs += cs*relvs*relvs*dtime;
            l_enct += ct*relvt*relvt*dtime;
        }

        fx = anx*fnt + sx*fst + tx*ftt;
        fy = any*fnt + sy*fst + ty*ftt;
        fz = anz*fnt + sz*fst + tz*ftt;

        fwx1 = aly*fz - alz*fy; fwy1 = alz*fx - alx*fz; fwz1 = alx*fy - aly*fx;
        fwx2 = -bly*fz + blz*fy; fwy2 = -blz*fx + blx*fz; fwz2 = -blx*fy + bly*fx;
    } else {
        fcont1 = 0.0; fcont2 = 0.0; alpha_val = 0.0;
        fx = fy = fz = 0.0;
        fwx1 = fwy1 = fwz1 = 0.0;
        fwx2 = fwy2 = fwz2 = 0.0;
    }
}

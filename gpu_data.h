#ifndef GPU_DATA_H
#define GPU_DATA_H

#include "dem3d.h"

struct GPUData {
    double *d_xc, *d_yc, *d_zc, *d_rc;
    double *d_vxc, *d_vyc, *d_vzc;
    double *d_vwx, *d_vwy, *d_vwz;
    int *d_np, *d_nm;

    double *d_gcx, *d_gcy, *d_gcz;
    double *d_gwx, *d_gwy, *d_gwz;
    double *d_vgx, *d_vgy, *d_vgz;
    double *d_vgwx, *d_vgwy, *d_vgwz;
    double *d_gv, *d_gi;
    int *d_nset, *d_elemStart;

    int *d_neib_cur, *d_neib_old;
    double *d_fcont_cur, *d_fcont_old;
    double *d_alpha_cur, *d_alpha_old;
    int *d_icount_cur, *d_icount_old;
    double *d_pforce;

    int *d_cellIndex, *d_particleIndex, *d_sortedIndex;
    int *d_cellStart, *d_cellEnd;

    int *d_icode;
    double *d_bval;
    int *d_ipboun;

    double *d_enkn, *d_enks, *d_enkt;
    double *d_encn, *d_encs, *d_enct;
    double *d_enfr, *d_enfr1, *d_enfr2, *d_enfr3;
    double *d_envx, *d_envy, *d_envz, *d_envw;
    double *d_engr;
    double *d_encx, *d_ency, *d_encz;
    double *d_enxf, *d_enxd;
    double *d_sig;
    int *d_ncont;
    double *d_grain_force;
    int *d_diag;              // [0]=NEIMAX overflow count, [1]=max contacts/element
    double *d_gcx_prev, *d_gcy_prev, *d_gcz_prev;
};

void allocate_gpu(GPUData &g, int nelem, int nptotl);
void upload_to_gpu(GPUData &g, SimState &ss);
void download_from_gpu(GPUData &g, SimState &ss);
void upload_material_constants(MaterialProps &mat);

// Kernel declarations
extern "C" {
void launch_compute_cell_index(GPUData &g, int nelem,
    int ipbx, int ipby, int ipbz,
    double pbcx, double pbcy, double pbcz,
    double pblx, double pbly, double pblz,
    double dmxmin, double dmymin, double dmzmin,
    double xdiv, double ydiv, double zdiv,
    int ndivx, int ndivy, int ndivz);

void launch_sort_and_find_bounds(GPUData &g, int nelem, int totalCells);

void launch_contact_forces(GPUData &g, SimState &ss,
    int ndivx, int ndivy, int ndivz,
    double dmxmin, double dmymin, double dmzmin,
    double xdiv, double ydiv, double zdiv,
    double vr, int istep);

// Per-step contact diagnostics: NEIMAX overflow count and max contacts/element.
void fetch_contact_diag(GPUData &g, int &overflow, int &maxcnt);

void launch_wall_forces(GPUData &g, SimState &ss);

void launch_integrate(GPUData &g, SimState &ss, int istep);
}

#endif

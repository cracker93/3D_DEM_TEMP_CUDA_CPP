#include "dem3d.h"
#include "gpu_data.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

// External kernel declarations
extern __global__ void kernel_compute_cell_index(
    const double*, const double*, const double*,
    int*, int*, int,
    int,int,int, double,double,double, double,double,double,
    double,double,double, double,double,double, int,int,int);
extern __global__ void kernel_find_cell_bounds(const int*, int*, int*, int);
extern __global__ void kernel_init_forces(double*, int*, int);
extern __global__ void kernel_contact_forces(
    const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,
    const double*,const double*,const double*,
    const int*,const int*,
    const double*,const double*,const double*,
    const int*,const int*,const int*,
    int,int,int, double,double,double, double,double,double,
    int,int,int,int, double,double,double, double,double,double,
    const int*,const double*,const double*,const int*,
    int*,double*,double*,int*,
    double*,double*,const int*,int,
    double,double,int,int,int,int,int,int,
    double*,double*,double*,double*,double*,double*,
    double*,double*,double*,double*,int*,int*);
extern __global__ void kernel_wall_forces(
    const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,
    const int*,double*,
    int,int,int, double,double,double, double,double,double,
    double*,double*,double,int);
extern __global__ void kernel_integrate_translation(
    const double*,const int*,const int*,
    double*,double*,double*, double*,double*,double*,
    const double*,const int*,const int*,
    const double*,const double*,const double*,
    const int*,const double*,
    double,double,double,double,double,int,
    double*,double*,double*,double*,
    double*,double*,double*,double*,double*,
    double*,double*,double*,double*);
extern __global__ void kernel_integrate_rotation(
    double*,double*,double*, double*,double*,double*,
    double*,double*,double*,
    const double*,const double*,
    const int*,const int*,const int*,
    double*,double*,double*,
    double*,double*,double*, double*,double*,double*,
    const double*,const double*,const double*,
    const int*,const double*,
    const double*,const double*,const double*,const double*,
    double,int,double*,double*,double*);

static const int BLOCK = 256;

extern "C" void launch_compute_cell_index(GPUData &g, int nelem,
    int ipbx, int ipby, int ipbz,
    double pbcx, double pbcy, double pbcz,
    double pblx, double pbly, double pblz,
    double dmxmin, double dmymin, double dmzmin,
    double xdiv, double ydiv, double zdiv,
    int ndivx, int ndivy, int ndivz)
{
    int grid = (nelem+BLOCK-1)/BLOCK;
    kernel_compute_cell_index<<<grid,BLOCK>>>(
        g.d_xc, g.d_yc, g.d_zc,
        g.d_cellIndex, g.d_particleIndex, nelem,
        ipbx,ipby,ipbz, pbcx,pbcy,pbcz, pblx,pbly,pblz,
        dmxmin,dmymin,dmzmin, xdiv,ydiv,zdiv, ndivx,ndivy,ndivz);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_sort_and_find_bounds(GPUData &g, int nelem, int totalCells) {
    thrust::device_ptr<int> keys(g.d_cellIndex);
    thrust::device_ptr<int> vals(g.d_particleIndex);
    thrust::sort_by_key(keys, keys+nelem, vals);
    CUDA_CHECK(cudaMemcpy(g.d_sortedIndex, g.d_particleIndex, sizeof(int)*nelem, cudaMemcpyDeviceToDevice));

    thrust::device_ptr<int> cs(g.d_cellStart);
    thrust::device_ptr<int> ce(g.d_cellEnd);
    thrust::fill(cs, cs+totalCells, 0);
    thrust::fill(ce, ce+totalCells, 0);

    int grid = (nelem+BLOCK-1)/BLOCK;
    kernel_find_cell_bounds<<<grid,BLOCK>>>(g.d_cellIndex, g.d_cellStart, g.d_cellEnd, nelem);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_contact_forces(GPUData &g, SimState &ss,
    int ndivx, int ndivy, int ndivz,
    double dmxmin, double dmymin, double dmzmin,
    double xdiv, double ydiv, double zdiv,
    double vr, int istep)
{
    int ne = ss.elem.nelem;
    int grid = (ne+BLOCK-1)/BLOCK;

    // Initialize forces and new contact counts
    kernel_init_forces<<<grid,BLOCK>>>(g.d_pforce, g.d_icount_cur, ne);
    CUDA_CHECK(cudaGetLastError());

    // Zero energy accumulators for this step
    double zero = 0.0; int izero = 0;
    CUDA_CHECK(cudaMemcpy(g.d_enkn,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_enks,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_enkt,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_encn,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_encs,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_enct,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_enfr,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_enfr1,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_enfr2,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_enfr3,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_ncont,&izero,4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g.d_diag,0,sizeof(int)*2));   // overflow count, max count
    double sig_zero[9] = {0};
    CUDA_CHECK(cudaMemcpy(g.d_sig,sig_zero,72,cudaMemcpyHostToDevice));

    kernel_contact_forces<<<grid,BLOCK>>>(
        g.d_xc,g.d_yc,g.d_zc,g.d_rc,
        g.d_vxc,g.d_vyc,g.d_vzc,
        g.d_vwx,g.d_vwy,g.d_vwz,
        g.d_np,g.d_nm,
        g.d_gcx,g.d_gcy,g.d_gcz,
        g.d_sortedIndex,g.d_cellStart,g.d_cellEnd,
        ndivx,ndivy,ndivz,
        dmxmin,dmymin,dmzmin, xdiv,ydiv,zdiv,
        ss.pb.ipbx,ss.pb.ipby,ss.pb.ipbz,ss.pb.ipb3,
        ss.pb.pblx,ss.pb.pbly,ss.pb.pblz,
        ss.pb.pbcx,ss.pb.pbcy,ss.pb.pbcz,
        g.d_neib_old,g.d_fcont_old,g.d_alpha_old,g.d_icount_old,
        g.d_neib_cur,g.d_fcont_cur,g.d_alpha_cur,g.d_icount_cur,
        g.d_pforce,g.d_sig,
        g.d_ipboun,ss.bc.npboun,
        ss.dtime,vr,ss.mode1,istep,ss.bc.nstep,
        ss.nkk1,ss.nkk2,ne,
        g.d_enkn,g.d_enks,g.d_enkt,
        g.d_encn,g.d_encs,g.d_enct,
        g.d_enfr,g.d_enfr1,g.d_enfr2,g.d_enfr3,
        g.d_ncont,g.d_diag);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void fetch_contact_diag(GPUData &g, int &overflow, int &maxcnt) {
    int h[2] = {0,0};
    CUDA_CHECK(cudaMemcpy(h, g.d_diag, sizeof(int)*2, cudaMemcpyDeviceToHost));
    overflow = h[0];
    maxcnt   = h[1];
}

extern "C" void launch_wall_forces(GPUData &g, SimState &ss) {
    int ne = ss.elem.nelem;
    int grid = (ne+BLOCK-1)/BLOCK;
    kernel_wall_forces<<<grid,BLOCK>>>(
        g.d_xc,g.d_yc,g.d_zc,g.d_rc,
        g.d_vxc,g.d_vyc,g.d_vzc,
        g.d_nm,g.d_pforce,
        ss.pb.ipbx,ss.pb.ipby,ss.pb.ipbz,
        ss.pb.pbcx,ss.pb.pbcy,ss.pb.pbcz,
        ss.pb.pblx,ss.pb.pbly,ss.pb.pblz,
        g.d_enkn,g.d_encn,
        ss.dtime,ne);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_integrate(GPUData &g, SimState &ss, int istep) {
    int ng = ss.grain.nptotl;
    int grid = (ng+BLOCK-1)/BLOCK;

    double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(g.d_envx,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_envy,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_envz,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_envw,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_engr,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_encx,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_ency,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_encz,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_enxf,&zero,8,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_enxd,&zero,8,cudaMemcpyHostToDevice));

    kernel_integrate_translation<<<grid,BLOCK>>>(
        g.d_pforce,g.d_np,g.d_nm,
        g.d_gcx,g.d_gcy,g.d_gcz,
        g.d_vgx,g.d_vgy,g.d_vgz,
        g.d_gv,g.d_nset,g.d_elemStart,
        g.d_xc,g.d_yc,g.d_zc,
        g.d_icode,g.d_bval,
        ss.dtime,ss.gx,ss.gy,ss.gz,ss.adamp,ng,
        g.d_envx,g.d_envy,g.d_envz,g.d_engr,
        g.d_encx,g.d_ency,g.d_encz,
        g.d_enxf,g.d_enxd,
        g.d_grain_force,
        g.d_gcx_prev,g.d_gcy_prev,g.d_gcz_prev);
    CUDA_CHECK(cudaGetLastError());

    kernel_integrate_rotation<<<grid,BLOCK>>>(
        g.d_gcx,g.d_gcy,g.d_gcz,
        g.d_gwx,g.d_gwy,g.d_gwz,
        g.d_vgwx,g.d_vgwy,g.d_vgwz,
        g.d_gv,g.d_gi,
        g.d_nset,g.d_elemStart,g.d_nm,
        g.d_xc,g.d_yc,g.d_zc,
        g.d_vxc,g.d_vyc,g.d_vzc,
        g.d_vwx,g.d_vwy,g.d_vwz,
        g.d_vgx,g.d_vgy,g.d_vgz,
        g.d_icode,g.d_bval,
        g.d_grain_force,
        g.d_gcx_prev,g.d_gcy_prev,g.d_gcz_prev,
        ss.dtime,ng,
        g.d_envw,g.d_enxf,g.d_enxd);
    CUDA_CHECK(cudaGetLastError());
}

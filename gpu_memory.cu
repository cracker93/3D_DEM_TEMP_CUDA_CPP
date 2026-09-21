#include "dem3d.h"
#include "gpu_data.h"

void allocate_gpu(GPUData &g, int nelem, int nptotl) {
    #define ALLOC_D(ptr, type, n) CUDA_CHECK(cudaMalloc(&ptr, sizeof(type)*(n)))
    ALLOC_D(g.d_xc,double,nelem); ALLOC_D(g.d_yc,double,nelem); ALLOC_D(g.d_zc,double,nelem);
    ALLOC_D(g.d_rc,double,nelem);
    ALLOC_D(g.d_vxc,double,nelem); ALLOC_D(g.d_vyc,double,nelem); ALLOC_D(g.d_vzc,double,nelem);
    ALLOC_D(g.d_vwx,double,nelem); ALLOC_D(g.d_vwy,double,nelem); ALLOC_D(g.d_vwz,double,nelem);
    ALLOC_D(g.d_np,int,nelem); ALLOC_D(g.d_nm,int,nelem);

    ALLOC_D(g.d_gcx,double,nptotl); ALLOC_D(g.d_gcy,double,nptotl); ALLOC_D(g.d_gcz,double,nptotl);
    ALLOC_D(g.d_gwx,double,nptotl); ALLOC_D(g.d_gwy,double,nptotl); ALLOC_D(g.d_gwz,double,nptotl);
    ALLOC_D(g.d_vgx,double,nptotl); ALLOC_D(g.d_vgy,double,nptotl); ALLOC_D(g.d_vgz,double,nptotl);
    ALLOC_D(g.d_vgwx,double,nptotl); ALLOC_D(g.d_vgwy,double,nptotl); ALLOC_D(g.d_vgwz,double,nptotl);
    ALLOC_D(g.d_gv,double,nptotl); ALLOC_D(g.d_gi,double,nptotl*3);
    ALLOC_D(g.d_nset,int,nptotl); ALLOC_D(g.d_elemStart,int,nptotl);

    int cs = nelem*NEIMAX;
    ALLOC_D(g.d_neib_cur,int,cs); ALLOC_D(g.d_neib_old,int,cs);
    ALLOC_D(g.d_fcont_cur,double,cs*2); ALLOC_D(g.d_fcont_old,double,cs*2);
    ALLOC_D(g.d_alpha_cur,double,cs); ALLOC_D(g.d_alpha_old,double,cs);
    ALLOC_D(g.d_icount_cur,int,nelem); ALLOC_D(g.d_icount_old,int,nelem);
    ALLOC_D(g.d_pforce,double,nelem*6);
    ALLOC_D(g.d_diag,int,2);

    ALLOC_D(g.d_cellIndex,int,nelem);
    ALLOC_D(g.d_particleIndex,int,nelem);
    ALLOC_D(g.d_sortedIndex,int,nelem);
    int mc = (NBX+1)*(NBX+1)*(NBX+1);
    ALLOC_D(g.d_cellStart,int,mc);
    ALLOC_D(g.d_cellEnd,int,mc);

    ALLOC_D(g.d_icode,int,NGR*6);
    ALLOC_D(g.d_bval,double,NGR*6);
    ALLOC_D(g.d_ipboun,int,NGR);

    ALLOC_D(g.d_enkn,double,1); ALLOC_D(g.d_enks,double,1); ALLOC_D(g.d_enkt,double,1);
    ALLOC_D(g.d_encn,double,1); ALLOC_D(g.d_encs,double,1); ALLOC_D(g.d_enct,double,1);
    ALLOC_D(g.d_enfr,double,1); ALLOC_D(g.d_enfr1,double,1);
    ALLOC_D(g.d_enfr2,double,1); ALLOC_D(g.d_enfr3,double,1);
    ALLOC_D(g.d_envx,double,1); ALLOC_D(g.d_envy,double,1);
    ALLOC_D(g.d_envz,double,1); ALLOC_D(g.d_envw,double,1);
    ALLOC_D(g.d_engr,double,1);
    ALLOC_D(g.d_encx,double,1); ALLOC_D(g.d_ency,double,1); ALLOC_D(g.d_encz,double,1);
    ALLOC_D(g.d_enxf,double,1); ALLOC_D(g.d_enxd,double,1);
    ALLOC_D(g.d_sig,double,9);
    ALLOC_D(g.d_ncont,int,1);
    ALLOC_D(g.d_grain_force,double,nptotl*6);
    ALLOC_D(g.d_gcx_prev,double,nptotl);
    ALLOC_D(g.d_gcy_prev,double,nptotl);
    ALLOC_D(g.d_gcz_prev,double,nptotl);
    #undef ALLOC_D
}

void upload_to_gpu(GPUData &g, SimState &ss) {
    int ne = ss.elem.nelem, ng = ss.grain.nptotl;
    #define UP(d,s,t,n) CUDA_CHECK(cudaMemcpy(d,s,sizeof(t)*(n),cudaMemcpyHostToDevice))
    UP(g.d_xc,ss.elem.xc,double,ne); UP(g.d_yc,ss.elem.yc,double,ne);
    UP(g.d_zc,ss.elem.zc,double,ne); UP(g.d_rc,ss.elem.rc,double,ne);
    UP(g.d_vxc,ss.elem.vxc,double,ne); UP(g.d_vyc,ss.elem.vyc,double,ne);
    UP(g.d_vzc,ss.elem.vzc,double,ne);
    UP(g.d_vwx,ss.elem.vwx,double,ne); UP(g.d_vwy,ss.elem.vwy,double,ne);
    UP(g.d_vwz,ss.elem.vwz,double,ne);
    UP(g.d_np,ss.elem.np,int,ne); UP(g.d_nm,ss.elem.nm,int,ne);

    UP(g.d_gcx,ss.grain.gcx,double,ng); UP(g.d_gcy,ss.grain.gcy,double,ng);
    UP(g.d_gcz,ss.grain.gcz,double,ng);
    UP(g.d_gwx,ss.grain.gwx,double,ng); UP(g.d_gwy,ss.grain.gwy,double,ng);
    UP(g.d_gwz,ss.grain.gwz,double,ng);
    UP(g.d_vgx,ss.grain.vgx,double,ng); UP(g.d_vgy,ss.grain.vgy,double,ng);
    UP(g.d_vgz,ss.grain.vgz,double,ng);
    UP(g.d_vgwx,ss.grain.vgwx,double,ng); UP(g.d_vgwy,ss.grain.vgwy,double,ng);
    UP(g.d_vgwz,ss.grain.vgwz,double,ng);
    UP(g.d_gv,ss.grain.gv,double,ng);

    double *gi_flat = new double[ng*3];
    for (int k=0;k<ng;k++) { gi_flat[k*3]=ss.grain.gi[k][0]; gi_flat[k*3+1]=ss.grain.gi[k][1]; gi_flat[k*3+2]=ss.grain.gi[k][2]; }
    UP(g.d_gi,gi_flat,double,ng*3); delete[] gi_flat;

    UP(g.d_nset,ss.grain.nset,int,ng);
    int *es = new int[ng]; es[0]=0;
    for (int k=1;k<ng;k++) es[k]=es[k-1]+ss.grain.nset[k-1];
    UP(g.d_elemStart,es,int,ng); delete[] es;

    int *ic_flat = new int[NGR*6]; double *bv_flat = new double[NGR*6];
    memset(ic_flat,0,sizeof(int)*NGR*6); memset(bv_flat,0,sizeof(double)*NGR*6);
    for (int k=0;k<ng;k++) for (int d=0;d<6;d++) { ic_flat[k*6+d]=ss.bc.icode[k][d]; bv_flat[k*6+d]=ss.bc.bval[k][d]; }
    UP(g.d_icode,ic_flat,int,NGR*6); UP(g.d_bval,bv_flat,double,NGR*6);
    delete[] ic_flat; delete[] bv_flat;

    UP(g.d_ipboun,ss.bc.ipboun,int,ss.bc.npboun);

    // FIX 1: cudaMalloc does NOT zero memory.  The Fortran sets ngbold = 0 and
    // cFs = 0 when icld = 0 at istep = 1, so the "old" contact buffers must be
    // cleared explicitly here, or the history lookup can match garbage.
    CUDA_CHECK(cudaMemset(g.d_icount_old, 0, sizeof(int)*ne));
    CUDA_CHECK(cudaMemset(g.d_neib_old, 0xff, sizeof(int)*ne*NEIMAX));   // = -1
    CUDA_CHECK(cudaMemset(g.d_fcont_old, 0, sizeof(double)*ne*NEIMAX*2));
    CUDA_CHECK(cudaMemset(g.d_alpha_old, 0, sizeof(double)*ne*NEIMAX));
    CUDA_CHECK(cudaMemset(g.d_icount_cur, 0, sizeof(int)*ne));
    CUDA_CHECK(cudaMemset(g.d_neib_cur, 0xff, sizeof(int)*ne*NEIMAX));
    CUDA_CHECK(cudaMemset(g.d_fcont_cur, 0, sizeof(double)*ne*NEIMAX*2));
    CUDA_CHECK(cudaMemset(g.d_alpha_cur, 0, sizeof(double)*ne*NEIMAX));
    CUDA_CHECK(cudaMemset(g.d_diag, 0, sizeof(int)*2));

    if (ss.bc.icld != 0) {
        int *nf = new int[ne*NEIMAX]; double *ff = new double[ne*NEIMAX*2]; double *af = new double[ne*NEIMAX];
        int *icf = new int[ne];
        memset(nf,0,sizeof(int)*ne*NEIMAX); memset(ff,0,sizeof(double)*ne*NEIMAX*2); memset(af,0,sizeof(double)*ne*NEIMAX);
        for (int i=0;i<ne;i++) { icf[i]=ss.cont.icount[i]; for (int j=0;j<ss.cont.icount[i];j++) {
            nf[i*NEIMAX+j]=ss.cont.neib[i][j]; ff[(i*NEIMAX+j)*2]=ss.cont.fcont[i][j][0];
            ff[(i*NEIMAX+j)*2+1]=ss.cont.fcont[i][j][1]; af[i*NEIMAX+j]=ss.cont.alpha[i][j]; }}
        UP(g.d_neib_old,nf,int,ne*NEIMAX); UP(g.d_fcont_old,ff,double,ne*NEIMAX*2);
        UP(g.d_alpha_old,af,double,ne*NEIMAX); UP(g.d_icount_old,icf,int,ne);
        delete[] nf; delete[] ff; delete[] af; delete[] icf;
    }
    #undef UP
}

void download_from_gpu(GPUData &g, SimState &ss) {
    int ne = ss.elem.nelem, ng = ss.grain.nptotl;
    #define DN(d,s,t,n) CUDA_CHECK(cudaMemcpy(d,s,sizeof(t)*(n),cudaMemcpyDeviceToHost))
    DN(ss.elem.xc,g.d_xc,double,ne); DN(ss.elem.yc,g.d_yc,double,ne); DN(ss.elem.zc,g.d_zc,double,ne);
    DN(ss.elem.vxc,g.d_vxc,double,ne); DN(ss.elem.vyc,g.d_vyc,double,ne); DN(ss.elem.vzc,g.d_vzc,double,ne);
    DN(ss.elem.vwx,g.d_vwx,double,ne); DN(ss.elem.vwy,g.d_vwy,double,ne); DN(ss.elem.vwz,g.d_vwz,double,ne);
    DN(ss.grain.gcx,g.d_gcx,double,ng); DN(ss.grain.gcy,g.d_gcy,double,ng); DN(ss.grain.gcz,g.d_gcz,double,ng);
    DN(ss.grain.gwx,g.d_gwx,double,ng); DN(ss.grain.gwy,g.d_gwy,double,ng); DN(ss.grain.gwz,g.d_gwz,double,ng);
    DN(ss.grain.vgx,g.d_vgx,double,ng); DN(ss.grain.vgy,g.d_vgy,double,ng); DN(ss.grain.vgz,g.d_vgz,double,ng);
    DN(ss.grain.vgwx,g.d_vgwx,double,ng); DN(ss.grain.vgwy,g.d_vgwy,double,ng); DN(ss.grain.vgwz,g.d_vgwz,double,ng);

    int *nf = new int[ne*NEIMAX]; double *ff = new double[ne*NEIMAX*2]; double *af = new double[ne*NEIMAX];
    // FIX 3: main.cu now swaps the contact buffers at the END of the step, so
    // *_cur holds the list just computed.  (Previously the swap happened before
    // the output block and new_in_cf.dat was written one step stale.)
    DN(nf,g.d_neib_cur,int,ne*NEIMAX); DN(ff,g.d_fcont_cur,double,ne*NEIMAX*2);
    DN(af,g.d_alpha_cur,double,ne*NEIMAX); DN(ss.cont.icount,g.d_icount_cur,int,ne);
    for (int i=0;i<ne;i++) if (ss.cont.icount[i] > NEIMAX) ss.cont.icount[i] = NEIMAX;
    for (int i=0;i<ne;i++) for (int j=0;j<ss.cont.icount[i]&&j<NEIMAX;j++) {
        ss.cont.neib[i][j]=nf[i*NEIMAX+j]; ss.cont.fcont[i][j][0]=ff[(i*NEIMAX+j)*2];
        ss.cont.fcont[i][j][1]=ff[(i*NEIMAX+j)*2+1]; ss.cont.alpha[i][j]=af[i*NEIMAX+j]; }
    delete[] nf; delete[] ff; delete[] af;
    #undef DN
}

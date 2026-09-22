// dem3d_cuda - Main program
// C++/CUDA rewrite of dem3d_v24.f
// GPU-accelerated DEM for connected spheres
// Reads/writes same file formats as original Fortran code

#include "dem3d.h"
#include "gpu_data.h"

// FIX 4: SimState contains fixed-size arrays (NCC = 420000, NEIMAX = 30) and is
// roughly 400 MB.  As a local variable it overflows the 8 MB default stack on
// Linux and segfaults immediately.  'static' puts it in .bss instead.
static SimState ss;

int main(int argc, char **argv) {
    printf("=== dem3d_cuda: GPU-accelerated DEM for connected spheres ===\n");

    memset(&ss, 0, sizeof(ss));
    ss.adamp = 1.0;
    printf("+++++++artificial damping: %f+++++++++++\n", ss.adamp);

    FILE *f_info = fopen("info.dat", "w");
    fprintf(f_info, "dem3d_cuda GPU version\n");

    // [1] Read input files (same format as Fortran)
    read_in_gm(ss);
    read_in_bc(ss);
    read_in_cf(ss);
    modify_positions_periodic(ss);

    // [2] Open output files
    FILE *f_output = fopen("output.dat", "w");
    FILE *f_energy = fopen("energy.dat", "w");
    FILE *f_out2   = fopen("out2.dat", "w");
    FILE *f_cont   = fopen("cont.dat", "w");
    FILE *f_cont2  = fopen("cont2.dat", "w");
    FILE *f_fd     = fopen("f-d_curve.dat", "w");
    FILE *f_cont3  = fopen("cont3.dat", "w");

    // [3] Initialize GPU
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) { fprintf(stderr, "No CUDA devices!\n"); return 1; }
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %zu MB)\n", prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem/(1024*1024));

    // PRE-SIMULATION DIAGNOSTIC: Check initial angular velocities
    {
        double sum_envw_init = 0;
        double max_vgw = 0;
        int max_vgw_grain = 0;
        int ng0 = ss.grain.nptotl, ne0 = ss.elem.nelem;
        for (int k = 0; k < ng0; k++) {
            double rho = ss.mat.rho[0];
            double gi1 = ss.grain.gi[k][0]*rho;
            double gi2 = ss.grain.gi[k][1]*rho;
            double gi3 = ss.grain.gi[k][2]*rho;
            double vw2 = ss.grain.vgwx[k]*ss.grain.vgwx[k] 
                       + ss.grain.vgwy[k]*ss.grain.vgwy[k]
                       + ss.grain.vgwz[k]*ss.grain.vgwz[k];
            sum_envw_init += 0.5*(gi1*ss.grain.vgwx[k]*ss.grain.vgwx[k]
                                 +gi2*ss.grain.vgwy[k]*ss.grain.vgwy[k]
                                 +gi3*ss.grain.vgwz[k]*ss.grain.vgwz[k]);
            if (vw2 > max_vgw) { max_vgw = vw2; max_vgw_grain = k; }
        }
        printf("INIT DIAG: sum envw (global frame, approx) = %e\n", sum_envw_init);
        printf("INIT DIAG: max |vgw| at grain %d = %e\n", max_vgw_grain, sqrt(max_vgw));
        printf("INIT DIAG: vgwx/y/z[%d] = %e %e %e\n", max_vgw_grain,
               ss.grain.vgwx[max_vgw_grain], ss.grain.vgwy[max_vgw_grain], 
               ss.grain.vgwz[max_vgw_grain]);
        printf("INIT DIAG: icld=%d\n", ss.bc.icld);
        
        // Also check element angular velocities
        double sum_elem_vw = 0;
        for (int i = 0; i < ne0; i++) {
            sum_elem_vw += ss.elem.vwx[i]*ss.elem.vwx[i] 
                         + ss.elem.vwy[i]*ss.elem.vwy[i]
                         + ss.elem.vwz[i]*ss.elem.vwz[i];
        }
        printf("INIT DIAG: sum |elem vw|^2 = %e, mean |elem vw| = %e\n", 
               sum_elem_vw, sqrt(sum_elem_vw/ne0));
    }

    GPUData gpu;
    memset(&gpu, 0, sizeof(gpu));
    allocate_gpu(gpu, ss.elem.nelem, ss.grain.nptotl);
    upload_material_constants(ss.mat);
    upload_to_gpu(gpu, ss);

    int ne = ss.elem.nelem, ng = ss.grain.nptotl;
    double pi = 4.0 * atan(1.0);

    // Contact-list health counters (FIX 2)
    int maxcnt_run = 0, overflow_steps = 0;
    long long overflow_total = 0;

    // Persistent energy accumulators
    double engr_acc = 0, enfr_acc = 0, enfr1_acc = 0, enfr2_acc = 0, enfr3_acc = 0;
    double encn_acc = 0, encs_acc = 0, enct_acc = 0;
    double encx_acc = 0, ency_acc = 0, encz_acc = 0;
    double enxf_acc = 0, enxd_acc = 0;

    // Evolving boundary positions
    double pbx1 = ss.pb.pbcx - ss.pb.pblx/2.0;
    double pbx2 = ss.pb.pbcx + ss.pb.pblx/2.0;
    double pby1 = ss.pb.pbcy - ss.pb.pbly/2.0;
    double pby2 = ss.pb.pbcy + ss.pb.pbly/2.0;
    double pbz1 = ss.pb.pbcz - ss.pb.pblz/2.0;
    double pbz2 = ss.pb.pbcz + ss.pb.pblz/2.0;

    fprintf(f_energy, "   istep   engr      enva       enka"
            "        enca       enfr       enxd       enxf       err"
            "      enfr1     enfr2     enfr3\n");

    // Write initial output headers
    fprintf(f_out2, " %d %d\n", ne, ss.bc.nstep);
    for (int i = 0; i < ne; i++) fprintf(f_out2, " %d %e\n", i+1, ss.elem.rc[i]);
    fprintf(f_output, " %d %d\n", ng, ss.bc.nstep);
    for (int i = 0; i < ng; i++) fprintf(f_output, " %d %d %e\n", i+1, ss.grain.nset[i], ss.grain.gv[i]);

    // ============================================================
    // [4] TIME STEPPING LOOP
    // ============================================================
    for (int istep = 1; istep <= ss.bc.nstep; istep++) {
        ss.istep_cur = istep;
        if (istep % ss.bc.nprn1 == 0 || istep == 1)
            printf("%d/%d\n", istep + ss.bc.iloop, ss.bc.nstep + ss.bc.iloop);

        // --- Tamping tool BCs ---
        double ddtm = ss.tamp.pret + ss.dtime * (double)istep;
        for (int it = ss.tamp.itl1a; it <= ss.tamp.itl1b; it++) {
            double hz = ss.tamp.hz, A = ss.tamp.Amp, Z = ss.tamp.Zkak;
            ss.bc.bval[it][0] = -2*pi*hz*A*sin(2*pi*hz*ddtm);
            ss.bc.bval[it][1] =  2*pi*hz*A*cos(2*pi*hz*ddtm);
            ss.bc.bval[it][3] =  2*pi*hz*(-Z)*cos(2*pi*hz*ddtm);
            ss.bc.bval[it][4] = -2*pi*hz*Z*sin(2*pi*hz*ddtm);
            ss.bc.bval[it][5] = 0.0;
            for (int d = 0; d < 6; d++) ss.bc.bval0[it][d] = ss.bc.bval[it][d];
        }

        // --- Update periodic boundaries ---
        pbx2 += ss.pb.vpbx*ss.dtime; pby2 += ss.pb.vpby*ss.dtime; pbz2 += ss.pb.vpbz*ss.dtime;
        ss.pb.pblx = pbx2-pbx1; ss.pb.pbly = pby2-pby1; ss.pb.pblz = pbz2-pbz1;
        ss.pb.pbcx = (pbx2+pbx1)/2; ss.pb.pbcy = (pby2+pby1)/2; ss.pb.pbcz = (pbz2+pbz1)/2;

        // --- Gradually-increased loading ---
        if (ss.bc.icld == 2) {
            int nstep2 = (int)(0.4*(double)ss.bc.nstep);
            for (int kk = 0; kk < ss.bc.npboun; kk++) {
                int ik = ss.bc.ipboun[kk];
                double tmp = (istep <= nstep2) ? (double)istep/(double)nstep2 : 1.0;
                for (int d = 0; d < 6; d++)
                    if (ss.bc.icode[ik][d] == 1) ss.bc.bval[ik][d] = ss.bc.bval0[ik][d]*tmp;
            }
        }

        // Upload bval to GPU
        {
            double *bf = new double[NGR*6];
            memset(bf, 0, sizeof(double)*NGR*6);
            for (int k = 0; k < ng; k++) for (int d = 0; d < 6; d++) bf[k*6+d] = ss.bc.bval[k][d];
            CUDA_CHECK(cudaMemcpy(gpu.d_bval, bf, sizeof(double)*NGR*6, cudaMemcpyHostToDevice));
            delete[] bf;
        }

        // --- Compute domain bounds (CPU, fast) ---
        CUDA_CHECK(cudaMemcpy(ss.elem.xc, gpu.d_xc, sizeof(double)*ne, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ss.elem.yc, gpu.d_yc, sizeof(double)*ne, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ss.elem.zc, gpu.d_zc, sizeof(double)*ne, cudaMemcpyDeviceToHost));

        double dxn,dxx,dyn,dyx,dzn,dzx;
        for (int i = 0; i < ne; i++) {
            double x=ss.elem.xc[i],y=ss.elem.yc[i],z=ss.elem.zc[i],r=ss.elem.rc[i];
            if (ss.pb.ipbx) x -= round((x-ss.pb.pbcx)/ss.pb.pblx)*ss.pb.pblx;
            if (ss.pb.ipby) y -= round((y-ss.pb.pbcy)/ss.pb.pbly)*ss.pb.pbly;
            if (ss.pb.ipbz) z -= round((z-ss.pb.pbcz)/ss.pb.pblz)*ss.pb.pblz;
            if (i==0) { dxn=x-r;dxx=x+r;dyn=y-r;dyx=y+r;dzn=z-r;dzx=z+r; }
            else { dxn=fmin(dxn,x-r);dxx=fmax(dxx,x+r);dyn=fmin(dyn,y-r);dyx=fmax(dyx,y+r);dzn=fmin(dzn,z-r);dzx=fmax(dzx,z+r); }
        }
        if (ss.pb.ipbx) { dxn=ss.pb.pbcx-ss.pb.pblx/2; dxx=ss.pb.pbcx+ss.pb.pblx/2; }
        if (ss.pb.ipby) { dyn=ss.pb.pbcy-ss.pb.pbly/2; dyx=ss.pb.pbcy+ss.pb.pbly/2; }
        if (ss.pb.ipbz) { dzn=ss.pb.pbcz-ss.pb.pblz/2; dzx=ss.pb.pbcz+ss.pb.pblz/2; }

        double bw=dxx-dxn, bh=dyx-dyn, bd=dzx-dzn;
        double em = ss.elemax;
        int ndx=(int)(bw/em+1e-8), ndy=(int)(bh/em+1e-8), ndz=(int)(bd/em+1e-8);
        if (ndx>NBX||ndy>NBX||ndz>NBX) { em=fmax(fmax(bw,bh),bd)/(NBX-1); ndx=(int)(bw/em+1e-8); ndy=(int)(bh/em+1e-8); ndz=(int)(bd/em+1e-8); }
        ndx=fmax(1,ndx); ndy=fmax(1,ndy); ndz=fmax(1,ndz);
        double xd=bw/ndx, yd=bh/ndy, zd=bd/ndz;
        int tc = ndx*ndy*ndz;

        // --- GPU: Spatial hash ---
        launch_compute_cell_index(gpu, ne,
            ss.pb.ipbx,ss.pb.ipby,ss.pb.ipbz,
            ss.pb.pbcx,ss.pb.pbcy,ss.pb.pbcz,
            ss.pb.pblx,ss.pb.pbly,ss.pb.pblz,
            dxn,dyn,dzn, xd,yd,zd, ndx,ndy,ndz);
        launch_sort_and_find_bounds(gpu, ne, tc);

        // --- GPU: Contact forces ---
        double vr = (ss.pb.ipb3==1) ? ss.pb.pblx*ss.pb.pbly*ss.pb.pblz : ss.stress.vr_ini;
        launch_contact_forces(gpu, ss, ndx,ndy,ndz, dxn,dyn,dzn, xd,yd,zd, vr, istep);

        // --- GPU: Wall forces ---
        launch_wall_forces(gpu, ss);

        // --- GPU: Time integration ---
        launch_integrate(gpu, ss, istep);
        CUDA_CHECK(cudaDeviceSynchronize());

        // FIX 3: the contact-buffer swap used to happen HERE, before the output
        // block, so download_from_gpu() read *_cur = the PREVIOUS step's list and
        // new_in_cf.dat was written one step stale relative to new_in_gm.dat.
        // The swap now happens at the very end of the step (see below).

        // --- Contact-list diagnostics (FIX 2) ---
        int cdiag_overflow = 0, cdiag_maxcnt = 0;
        fetch_contact_diag(gpu, cdiag_overflow, cdiag_maxcnt);
        if (cdiag_maxcnt > maxcnt_run) maxcnt_run = cdiag_maxcnt;
        if (cdiag_overflow > 0) {
            overflow_steps++;
            overflow_total += cdiag_overflow;
            if (overflow_steps == 1) {
                fprintf(stderr,
                  "\n*** NEIMAX OVERFLOW at step %d: %d contact(s) could not be\n"
                  "    stored (max contacts on one element = %d, NEIMAX = %d).\n"
                  "    Their shear-force history is lost, so those contacts reset\n"
                  "    their tangential spring every step.  The Fortran stops in\n"
                  "    this situation.  Increase NEIMAX in dem3d.h and rebuild.\n\n",
                  istep, cdiag_overflow, cdiag_maxcnt, NEIMAX);
                fprintf(f_info, "NEIMAX overflow first seen at step %d\n", istep);
            }
#if ABORT_ON_NEIMAX_OVERFLOW
            fprintf(stderr, "Aborting (set ABORT_ON_NEIMAX_OVERFLOW=0 to continue).\n");
            return 2;
#endif
        }

        // --- Retrieve step energies from GPU ---
        double enkn,enks,enkt, step_encn,step_encs,step_enct;
        double step_enfr,step_enfr1,step_enfr2,step_enfr3;
        double envx,envy,envz,envw, step_engr;
        double step_encx,step_ency,step_encz, step_enxf,step_enxd;
        int ncont;

        // DIAGNOSTIC: Print detailed energy breakdown at early steps
        if (istep <= 3) {
            // Download grain velocities to check
            double *tvgx = new double[ng], *tvgy = new double[ng], *tvgz = new double[ng];
            CUDA_CHECK(cudaMemcpy(tvgx, gpu.d_vgx, sizeof(double)*ng, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(tvgy, gpu.d_vgy, sizeof(double)*ng, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(tvgz, gpu.d_vgz, sizeof(double)*ng, cudaMemcpyDeviceToHost));
            double sum_ke = 0;
            for (int k = 0; k < ng; k++) {
                double m = ss.mat.rho[ss.elem.nm[0]] * ss.grain.gv[k]; // approximate
                sum_ke += 0.5 * m * (tvgx[k]*tvgx[k] + tvgy[k]*tvgy[k] + tvgz[k]*tvgz[k]);
            }
            printf("DIAG step %d: CPU-computed KE from downloaded vg = %e\n", istep, sum_ke);
            printf("DIAG step %d: vgx[0]=%e vgy[0]=%e vgz[0]=%e\n", istep, tvgx[0], tvgy[0], tvgz[0]);
            printf("DIAG step %d: vgx[100]=%e vgy[100]=%e vgz[100]=%e\n", istep, tvgx[100], tvgy[100], tvgz[100]);
            printf("DIAG step %d: grain gv[0]=%e nset[0]=%d\n", istep, ss.grain.gv[0], ss.grain.nset[0]);
            // Also download angular velocities and Euler angles
            double *tvgwx = new double[ng], *tvgwy = new double[ng], *tvgwz = new double[ng];
            double *tgwx = new double[ng], *tgwy = new double[ng], *tgwz = new double[ng];
            CUDA_CHECK(cudaMemcpy(tvgwx, gpu.d_vgwx, sizeof(double)*ng, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(tvgwy, gpu.d_vgwy, sizeof(double)*ng, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(tvgwz, gpu.d_vgwz, sizeof(double)*ng, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(tgwx, gpu.d_gwx, sizeof(double)*ng, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(tgwy, gpu.d_gwy, sizeof(double)*ng, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(tgwz, gpu.d_gwz, sizeof(double)*ng, cudaMemcpyDeviceToHost));
            printf("DIAG step %d: vgwx[0]=%e vgwy[0]=%e vgwz[0]=%e (global)\n", istep, tvgwx[0], tvgwy[0], tvgwz[0]);
            printf("DIAG step %d: gwx[0]=%e gwy[0]=%e gwz[0]=%e (Euler angles)\n", istep, tgwx[0], tgwy[0], tgwz[0]);
            printf("DIAG step %d: gi[0]=%e %e %e rho=%e\n", istep,
                   ss.grain.gi[0][0], ss.grain.gi[0][1], ss.grain.gi[0][2],
                   ss.mat.rho[ss.elem.nm[0]]);
            // Compute envw manually on CPU using BODY FRAME (correct)
            double sum_envw_global = 0;
            double sum_envw_body = 0;
            for (int k = 0; k < ng; k++) {
                int es = 0;
                for (int kk = 0; kk < k; kk++) es += ss.grain.nset[kk];
                double rho = ss.mat.rho[ss.elem.nm[es]];
                double gi1 = ss.grain.gi[k][0]*rho;
                double gi2 = ss.grain.gi[k][1]*rho;
                double gi3 = ss.grain.gi[k][2]*rho;
                // Global-frame (wrong physics but diagnostic)
                sum_envw_global += 0.5*(gi1*tvgwx[k]*tvgwx[k] + gi2*tvgwy[k]*tvgwy[k] + gi3*tvgwz[k]*tvgwz[k]);
                // Body-frame via R^T * vgwG
                double a=tgwx[k], b=tgwy[k], g=tgwz[k];
                double ca=cos(a),sa=sin(a),cb=cos(b),sb=sin(b),cg=cos(g),sg=sin(g);
                // R matrix (same as Fortran and CUDA)
                double R00=cb,      R01=-sb*ca,          R02=sb*sa;
                double R10=cg*sb,   R11=cg*cb*ca-sg*sa,  R12=-cg*cb*sa-sg*ca;
                double R20=sg*sb,   R21=sg*cb*ca+cg*sa,  R22=-sg*cb*sa+cg*ca;
                // R^T * vgwG = body frame
                double bw0 = R00*tvgwx[k] + R10*tvgwy[k] + R20*tvgwz[k];
                double bw1 = R01*tvgwx[k] + R11*tvgwy[k] + R21*tvgwz[k];
                double bw2 = R02*tvgwx[k] + R12*tvgwy[k] + R22*tvgwz[k];
                sum_envw_body += 0.5*(gi1*bw0*bw0 + gi2*bw1*bw1 + gi3*bw2*bw2);
                if (k < 3 || k == 252) {
                    printf("DIAG step %d: grain %d body-frame vgw = %e %e %e\n", istep, k, bw0, bw1, bw2);
                    printf("DIAG step %d: grain %d envw_contrib = %e (body) vs %e (global)\n", istep, k,
                           0.5*(gi1*bw0*bw0 + gi2*bw1*bw1 + gi3*bw2*bw2),
                           0.5*(gi1*tvgwx[k]*tvgwx[k] + gi2*tvgwy[k]*tvgwy[k] + gi3*tvgwz[k]*tvgwz[k]));
                }
            }
            printf("DIAG step %d: CPU envw (global frame) = %e\n", istep, sum_envw_global);
            printf("DIAG step %d: CPU envw (body frame)   = %e\n", istep, sum_envw_body);
            delete[] tvgwx; delete[] tvgwy; delete[] tvgwz;
            delete[] tgwx; delete[] tgwy; delete[] tgwz;
            printf("DIAG step %d: ncont will be fetched...\n", istep);
            delete[] tvgx; delete[] tvgy; delete[] tvgz;
        }
        CUDA_CHECK(cudaMemcpy(&enkn,gpu.d_enkn,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&enks,gpu.d_enks,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&enkt,gpu.d_enkt,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_encn,gpu.d_encn,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_encs,gpu.d_encs,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_enct,gpu.d_enct,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_enfr,gpu.d_enfr,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_enfr1,gpu.d_enfr1,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_enfr2,gpu.d_enfr2,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_enfr3,gpu.d_enfr3,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&envx,gpu.d_envx,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&envy,gpu.d_envy,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&envz,gpu.d_envz,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&envw,gpu.d_envw,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_engr,gpu.d_engr,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_encx,gpu.d_encx,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_ency,gpu.d_ency,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_encz,gpu.d_encz,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_enxf,gpu.d_enxf,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&step_enxd,gpu.d_enxd,8,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&ncont,gpu.d_ncont,4,cudaMemcpyDeviceToHost));

        // Accumulate persistent energies
        engr_acc += step_engr;
        enfr_acc += step_enfr; enfr1_acc += step_enfr1; enfr2_acc += step_enfr2; enfr3_acc += step_enfr3;
        encn_acc += step_encn; encs_acc += step_encs; enct_acc += step_enct;
        encx_acc += step_encx; ency_acc += step_ency; encz_acc += step_encz;
        enxf_acc += step_enxf; enxd_acc += step_enxd;

        double enva = envx+envy+envz+envw;
        double enka = enkn+enks+enkt;
        double enca = encn_acc+encs_acc+enct_acc+encx_acc+ency_acc+encz_acc;

        if (istep == 1) engr_acc = -(enva + engr_acc + enka + enca + enfr_acc);

        double err = enva + engr_acc + enka + enca + enfr_acc + enxf_acc + enxd_acc;

        // DIAGNOSTIC at early steps
        if (istep <= 3) {
            printf("DIAG step %d: envx=%e envy=%e envz=%e envw=%e\n", istep, envx, envy, envz, envw);
            printf("DIAG step %d: enva=%e enka=%e (enkn=%e enks=%e enkt=%e)\n", istep, enva, enka, enkn, enks, enkt);
            printf("DIAG step %d: enca=%e (encn_acc=%e encs_acc=%e enct_acc=%e encx=%e ency=%e encz=%e)\n",
                   istep, enca, encn_acc, encs_acc, enct_acc, encx_acc, ency_acc, encz_acc);
            printf("DIAG step %d: engr_acc=%e enfr_acc=%e enxf_acc=%e enxd_acc=%e\n", istep, engr_acc, enfr_acc, enxf_acc, enxd_acc);
            printf("DIAG step %d: err=%e ncont=%d\n", istep, err, ncont);
        }

        // --- Stress-controlled periodic boundary feedback ---
        double sig_h[9];
        CUDA_CHECK(cudaMemcpy(sig_h, gpu.d_sig, 72, cudaMemcpyDeviceToHost));
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) ss.stress.sig[i][j]=sig_h[i*3+j];

        double av_r = pow(ss.avvol*0.75/pi, 0.3333);
        double pb_fac = 0.1, amass = ss.avrho*ss.avvol;
        double pb_m = amass/((2*av_r)*(2*av_r))*pb_fac;
        if (ss.pb.ic_pbx==0) { ss.pb.vpbx -= (sig_h[0]-ss.pb.sig_pbx)/pb_m*ss.dtime; ss.pb.vpbx *= ss.adamp; }
        else ss.pb.sig_pbx = sig_h[0];
        if (ss.pb.ic_pby==0) { ss.pb.vpby -= (sig_h[4]-ss.pb.sig_pby)/pb_m*ss.dtime; ss.pb.vpby *= ss.adamp; }
        else ss.pb.sig_pby = sig_h[4];
        if (ss.pb.ic_pbz==0) { ss.pb.vpbz -= (sig_h[8]-ss.pb.sig_pbz)/pb_m*ss.dtime; ss.pb.vpbz *= ss.adamp; }
        else ss.pb.sig_pbz = sig_h[8];

        // --- Output ---
        if (istep % ss.bc.nprn2 == 0 || istep == 1) {
            fprintf(f_energy, " %7d %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e\n",
                    istep, engr_acc, enva, enka, enca, enfr_acc, enxd_acc, enxf_acc, err,
                    enfr1_acc, enfr2_acc, enfr3_acc);
            printf("mean stress: %d %e\n", istep, sig_h[0]+sig_h[4]+sig_h[8]);
            fprintf(f_cont2, " %d %d\n", istep, ncont);
            fflush(f_energy); fflush(f_cont2);
        }

        // Full output at nprn1 intervals
        if (istep % ss.bc.nprn1 == 0 || istep == 1 || istep == ss.bc.nstep) {
            download_from_gpu(gpu, ss);
            write_new_in_gm(ss);
            write_new_in_cf(ss);

            // output.dat
            int icr = 0;
            for (int k = 0; k < ng; k++) {
                fprintf(f_output, " %d %d\n", istep, k+1);
                fprintf(f_output, " %e %e %e\n", ss.grain.vgx[k], ss.grain.vgy[k], ss.grain.vgz[k]);
                fprintf(f_output, " %e %e %e\n", ss.grain.vgwx[k], ss.grain.vgwy[k], ss.grain.vgwz[k]);
                fprintf(f_output, " %e %e %e\n", ss.grain.gcx[k], ss.grain.gcy[k], ss.grain.gcz[k]);
                fprintf(f_output, " %e %e %e\n", ss.grain.gwx[k], ss.grain.gwy[k], ss.grain.gwz[k]);
            }

            // out2.dat
            fprintf(f_out2, " %d\n", istep);
            icr = 0;
            for (int k = 0; k < ng; k++) {
                for (int j = 0; j < ss.grain.nset[k]; j++) {
                    fprintf(f_out2, " %6d %19.11e %19.11e %19.11e\n",
                            icr+1, ss.elem.xc[icr], ss.elem.yc[icr], ss.elem.zc[icr]);
                    fprintf(f_out2, " %19.11e %19.11e %19.11e\n",
                            ss.grain.gwx[k], ss.grain.gwy[k], ss.grain.gwz[k]);
                    icr++;
                }
            }
            fflush(f_output); fflush(f_out2);
        }

        // f-d_curve: download grain forces and write for load particles
        if (istep % ss.bc.nprn2 == 0 || istep == 1) {
            double *gf = new double[ng*6];
            CUDA_CHECK(cudaMemcpy(gf, gpu.d_grain_force, sizeof(double)*ng*6, cudaMemcpyDeviceToHost));
            // Also need positions
            if (istep % ss.bc.nprn1 != 0 && istep != 1) {
                CUDA_CHECK(cudaMemcpy(ss.grain.gcx, gpu.d_gcx, sizeof(double)*ng, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(ss.grain.gcy, gpu.d_gcy, sizeof(double)*ng, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(ss.grain.gcz, gpu.d_gcz, sizeof(double)*ng, cudaMemcpyDeviceToHost));
            }
            for (int lp = 0; lp < ss.bc.nloadp; lp++) {
                int k = ss.bc.loadp[lp] - 1; // convert to 0-based
                fprintf(f_fd, " %7d %7d %12.6e %12.6e %12.6e %12.6e %12.6e %12.6e\n",
                        istep, k+1, gf[k*6], ss.grain.gcx[k],
                        gf[k*6+1], ss.grain.gcy[k], gf[k*6+2], ss.grain.gcz[k]);
            }
            delete[] gf;
            fflush(f_fd);
        }

        // Write new_in_bc.dat at output intervals or last step
        if (istep % ss.bc.nprn1 == 0 || istep == ss.bc.nstep) {
            write_new_in_bc(ss);
        }

        // --- Swap contact buffers (FIX 3: after all output) ---
        std::swap(gpu.d_neib_cur, gpu.d_neib_old);
        std::swap(gpu.d_fcont_cur, gpu.d_fcont_old);
        std::swap(gpu.d_alpha_cur, gpu.d_alpha_old);
        std::swap(gpu.d_icount_cur, gpu.d_icount_old);
    }

    // Cleanup
    fclose(f_output); fclose(f_energy); fclose(f_out2);
    fclose(f_cont); fclose(f_cont2); fclose(f_fd); fclose(f_cont3);
    fclose(f_info);

    printf("=== Simulation complete ===\n");
    printf("contact-list health: max contacts on one element = %d (NEIMAX = %d)\n",
           maxcnt_run, NEIMAX);
    if (overflow_total > 0)
        printf("WARNING: NEIMAX overflow on %d step(s), %lld contact(s) lost history.\n",
               overflow_steps, overflow_total);
    else
        printf("no NEIMAX overflow: all contact histories were preserved.\n");
    return 0;
}

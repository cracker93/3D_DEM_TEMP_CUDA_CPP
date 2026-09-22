#include "dem3d.h"
#include <string>

// ============================================================
// Helper: Read file, replace Fortran 'D' exponents with 'E'
// so that C's fscanf can parse them correctly.
// ============================================================
static FILE* fopen_fortran(const char *filename) {
    FILE *fin = fopen(filename, "r");
    if (!fin) return nullptr;
    // Read entire file
    fseek(fin, 0, SEEK_END);
    long sz = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    char *buf = new char[sz+1];
    size_t nr = fread(buf, 1, sz, fin);
    buf[nr] = '\0';
    fclose(fin);
    // Replace D/d exponents with E/e
    for (long i = 0; i < (long)nr; i++) {
        if ((buf[i] == 'D' || buf[i] == 'd') && i > 0 && i < (long)nr-1) {
            // Check if it's a Fortran exponent: digit before, +/- or digit after
            char prev = buf[i-1];
            char next = buf[i+1];
            if ((prev >= '0' && prev <= '9') && (next == '+' || next == '-' || (next >= '0' && next <= '9'))) {
                buf[i] = 'E';
            }
        }
    }
    // Write to temp file and reopen
    FILE *tmp = tmpfile();
    fwrite(buf, 1, nr, tmp);
    fseek(tmp, 0, SEEK_SET);
    delete[] buf;
    return tmp;
}

// ============================================================
// Read in_gm.dat — grain geometry and material data
// Matches Fortran format exactly
// ============================================================
void read_in_gm(SimState &ss) {
    FILE *fp = fopen_fortran("in_gm.dat");
    if (!fp) {
        fprintf(stderr, "Error: cannot open in_gm.dat\n");
        exit(1);
    }

    // Line 1: dtime, gx, gy, gz
    fscanf(fp, "%lf %lf %lf %lf", &ss.dtime, &ss.gx, &ss.gy, &ss.gz);

    // Line 2: nmater
    fscanf(fp, "%d", &ss.mat.nmater);

    double cmin = 1e30, akmax = 0.0;
    for (int i = 0; i < ss.mat.nmater; i++) {
        fscanf(fp, "%lf %lf %lf %lf %lf %lf",
               &ss.mat.rho[i], &ss.mat.akn[i], &ss.mat.aks[i],
               &ss.mat.cn[i], &ss.mat.cs[i], &ss.mat.phimu[i]);
        // Skip rest of line (Fortran files may have trailing comments like "*density of tool")
        { int c; while ((c = fgetc(fp)) != '\n' && c != EOF); }
        double cmin_i = fmin(ss.mat.cn[i], ss.mat.cs[i]);
        double akmax_i = fmax(ss.mat.akn[i], ss.mat.aks[i]);
        if (i == 0) {
            cmin = cmin_i;
            akmax = akmax_i;
        } else {
            if (cmin_i < cmin) cmin = cmin_i;
            if (akmax_i > akmax) akmax = akmax_i;
        }
    }

    printf("in_gm.dat header has read.\n");

    // Number of grains and friction energy grouping
    fscanf(fp, "%d %d %d", &ss.grain.nptotl, &ss.nkk1, &ss.nkk2);
    printf("nptotl %d\n", ss.grain.nptotl);

    int icr = 0;
    ss.elemax = 0.0;
    double tvol = 0.0;
    ss.avrho = 0.0;
    double vs_r = 0.0;
    double pi = 4.0 * atan(1.0);
    ss.gmmin = 1e30;

    for (int k = 0; k < ss.grain.nptotl; k++) {
        int idum, imater, nsetk;
        fscanf(fp, "%d %d %d", &idum, &imater, &nsetk);
        imater--; // Convert to 0-based
        ss.grain.nset[k] = nsetk;

        fscanf(fp, "%lf %lf %lf %lf",
               &ss.grain.gv[k], &ss.grain.gi[k][0],
               &ss.grain.gi[k][1], &ss.grain.gi[k][2]);
        fscanf(fp, "%lf %lf %lf",
               &ss.grain.vgx[k], &ss.grain.vgy[k], &ss.grain.vgz[k]);
        fscanf(fp, "%lf %lf %lf",
               &ss.grain.vgwx[k], &ss.grain.vgwy[k], &ss.grain.vgwz[k]);
        fscanf(fp, "%lf %lf %lf",
               &ss.grain.gcx[k], &ss.grain.gcy[k], &ss.grain.gcz[k]);
        fscanf(fp, "%lf %lf %lf",
               &ss.grain.gwx[k], &ss.grain.gwy[k], &ss.grain.gwz[k]);

        tvol += ss.grain.gv[k];

        double gmass_k = ss.grain.gv[k] * ss.mat.rho[imater];
        if (gmass_k < ss.gmmin) ss.gmmin = gmass_k;

        // Read elements for this grain
        for (int i = 0; i < nsetk; i++) {
            int idum2;
            fscanf(fp, "%d %lf", &idum2, &ss.elem.rc[icr]);
            fscanf(fp, "%lf %lf %lf",
                   &ss.elem.xc[icr], &ss.elem.yc[icr], &ss.elem.zc[icr]);

            if (ss.elem.rc[icr] * 2.0 > ss.elemax)
                ss.elemax = ss.elem.rc[icr] * 2.0;

            ss.elem.nm[icr] = imater;
            ss.elem.np[icr] = k; // 0-based grain index

            ss.avrho += ss.mat.rho[imater];
            vs_r += 4.0 / 3.0 * pi * pow(ss.elem.rc[icr], 3);
            icr++;
        }
    }

    ss.elem.nelem = icr;
    ss.avvol = tvol / (double)ss.grain.nptotl;
    ss.avrho /= (double)ss.elem.nelem;

    printf("elemax %f\n", ss.elemax);
    printf("Number of spheres: %d\n", ss.elem.nelem);
    printf("Total volume of grains Vs: %e\n", tvol);
    printf("check for spherical grains Vs_r: %e\n", vs_r);

    fclose(fp);

    // Set element velocities from grain velocities
    icr = 0;
    for (int k = 0; k < ss.grain.nptotl; k++) {
        for (int j = 0; j < ss.grain.nset[k]; j++) {
            double alx = ss.elem.xc[icr] - ss.grain.gcx[k];
            double aly = ss.elem.yc[icr] - ss.grain.gcy[k];
            double alz = ss.elem.zc[icr] - ss.grain.gcz[k];
            ss.elem.vxc[icr] = ss.grain.vgx[k] + ss.grain.vgwy[k]*alz - ss.grain.vgwz[k]*aly;
            ss.elem.vyc[icr] = ss.grain.vgy[k] + ss.grain.vgwz[k]*alx - ss.grain.vgwx[k]*alz;
            ss.elem.vzc[icr] = ss.grain.vgz[k] + ss.grain.vgwx[k]*aly - ss.grain.vgwy[k]*alx;
            ss.elem.vwx[icr] = ss.grain.vgwx[k];
            ss.elem.vwy[icr] = ss.grain.vgwy[k];
            ss.elem.vwz[icr] = ss.grain.vgwz[k];
            icr++;
        }
    }

    // Initialize icode to 0
    for (int i = 0; i < ss.grain.nptotl; i++) {
        for (int j = 0; j < 6; j++) {
            ss.bc.icode[i][j] = 0;
        }
    }

    // Check dtime
    double amass = ss.avrho * ss.avvol;
    double ccrit = 2.0 * sqrt(amass * akmax);
    double aksi = cmin / ccrit;
    double dtimec = (sqrt(aksi*aksi + 1.0) - aksi) * sqrt(amass / akmax);

    double ccrit2 = 2.0 * sqrt(ss.gmmin * akmax);
    double aksi2 = cmin / ccrit2;
    double dtimec2 = (sqrt(aksi2*aksi2 + 1.0) - aksi2) * sqrt(ss.gmmin / akmax);

    printf("amass,avvol,akmax: %e %e %e\n", amass, ss.avvol, akmax);
    printf("avrho, elemax: %e %e\n", ss.avrho, ss.elemax);
    printf("delta t vs. critical value: %e %e\n", ss.dtime, dtimec);
    printf("cf. for lightest grain: %e\n", dtimec2);
    if (ss.dtime > dtimec) {
        printf("!!!! TIME INCREMENT IS BIGGER THAN CRITICAL VALUE !!!!\n");
    }
}

// ============================================================
// Read in_bc.dat — boundary conditions
// ============================================================
void read_in_bc(SimState &ss) {
    FILE *fp = fopen_fortran("in_bc.dat");
    if (!fp) {
        fprintf(stderr, "Error: cannot open in_bc.dat\n");
        exit(1);
    }

    fscanf(fp, "%d %d %d %d %d",
           &ss.bc.icld, &ss.bc.iloop, &ss.bc.nstep,
           &ss.bc.nprn1, &ss.bc.nprn2);

    // Stabilization mode
    if (ss.bc.icld == -1) {
        ss.mode1 = 1;
        ss.bc.icld = 0;
        printf("Initial stabilization. k is gradually increased.\n");
        printf("Initial velocity = 0.\n");
        // Set all velocities to zero
        for (int i = 0; i < ss.elem.nelem; i++) {
            ss.elem.vxc[i] = 0.0;
            ss.elem.vyc[i] = 0.0;
            ss.elem.vzc[i] = 0.0;
            ss.elem.vwx[i] = 0.0;
            ss.elem.vwy[i] = 0.0;
            ss.elem.vwz[i] = 0.0;
        }
    } else {
        ss.mode1 = 0;
    }

    // Gradually-increased loading
    int nstep2 = 0;
    if (ss.bc.icld == 2) {
        double fdfac = 0.8;
        nstep2 = (int)(2.0 * (1.0 - fdfac) * (double)ss.bc.nstep);
        printf("Gradually-increased loading mode\n");
        printf("Final displacement is %f%% of the setting.\n", fdfac * 100);
        printf("the step to reach constant speed is %d\n", nstep2);
    }

    // Loading particles
    fscanf(fp, "%d", &ss.bc.nloadp);
    for (int i = 0; i < ss.bc.nloadp; i++) {
        fscanf(fp, "%d", &ss.bc.loadp[i]);
    }

    // Boundary particles
    fscanf(fp, "%d", &ss.bc.npboun);
    for (int k = 0; k < ss.bc.npboun; k++) {
        int grainIdx;
        fscanf(fp, "%d", &grainIdx);
        grainIdx--; // Convert to 0-based
        ss.bc.ipboun[k] = grainIdx;

        int ic1, ic2, ic3;
        double bv1, bv2, bv3;
        fscanf(fp, "%d %lf %d %lf %d %lf",
               &ic1, &bv1, &ic2, &bv2, &ic3, &bv3);
        // Note: Fortran reads: i, icode(i,1),bval(i,1), icode(i,2),bval(i,2), icode(i,3),bval(i,3)
        // But the grain index was already read as the first field on this line
        // Re-read: the Fortran format is:
        //   read(20,*) i, icode(i,1),bval(i,1), icode(i,2),bval(i,2), icode(i,3),bval(i,3)
        // So the first int on the line is the grain number, already read above
        // Wait — looking more carefully at the Fortran:
        //   read(20,*) i, icode(i,1),bval(i,1), ...
        // 'i' is the grain number. Let me re-parse.
        // Actually grainIdx was from the first field. The rest follows.

        ss.bc.icode[grainIdx][0] = ic1;
        ss.bc.bval[grainIdx][0] = bv1;
        ss.bc.icode[grainIdx][1] = ic2;
        ss.bc.bval[grainIdx][1] = bv2;
        ss.bc.icode[grainIdx][2] = ic3;
        ss.bc.bval[grainIdx][2] = bv3;

        int ic4, ic5, ic6;
        double bv4, bv5, bv6;
        fscanf(fp, "%d %lf %d %lf %d %lf",
               &ic4, &bv4, &ic5, &bv5, &ic6, &bv6);
        ss.bc.icode[grainIdx][3] = ic4;
        ss.bc.bval[grainIdx][3] = bv4;
        ss.bc.icode[grainIdx][4] = ic5;
        ss.bc.bval[grainIdx][4] = bv5;
        ss.bc.icode[grainIdx][5] = ic6;
        ss.bc.bval[grainIdx][5] = bv6;

        printf("controlled grain no. %d (grain %d)\n", k+1, grainIdx+1);

        // For displacement control: convert displacement to velocity
        if (ss.bc.icode[grainIdx][0] == 1)
            ss.bc.bval[grainIdx][0] /= ss.dtime * (double)ss.bc.nstep;
        if (ss.bc.icode[grainIdx][1] == 1)
            ss.bc.bval[grainIdx][1] /= ss.dtime * (double)ss.bc.nstep;
        if (ss.bc.icode[grainIdx][2] == 1)
            ss.bc.bval[grainIdx][2] /= ss.dtime * (double)ss.bc.nstep;

        for (int d = 0; d < 6; d++)
            ss.bc.bval0[grainIdx][d] = ss.bc.bval[grainIdx][d];
    }

    // Periodic boundaries
    double pbx1, pbx2, pby1, pby2, pbz1, pbz2;
    double vpbx0, vpby0, vpbz0;
    fscanf(fp, "%d %lf %lf %d %lf %lf",
           &ss.pb.ipbx, &pbx1, &pbx2, &ss.pb.ic_pbx, &vpbx0, &ss.pb.sig_pbx);
    fscanf(fp, "%d %lf %lf %d %lf %lf",
           &ss.pb.ipby, &pby1, &pby2, &ss.pb.ic_pby, &vpby0, &ss.pb.sig_pby);
    fscanf(fp, "%d %lf %lf %d %lf %lf",
           &ss.pb.ipbz, &pbz1, &pbz2, &ss.pb.ic_pbz, &vpbz0, &ss.pb.sig_pbz);

    ss.pb.ipb3 = (ss.pb.ipbx + ss.pb.ipby + ss.pb.ipbz == 3) ? 1 : 0;
    if (ss.pb.ipb3) printf("periodic in three directions.\n");

    // Tamping tool parameters
    fscanf(fp, "%lf %lf %lf %d %d %lf",
           &ss.tamp.hz, &ss.tamp.Amp, &ss.tamp.Zkak,
           &ss.tamp.itl1a, &ss.tamp.itl1b, &ss.tamp.pret);
    // Convert to 0-based
    ss.tamp.itl1a--;
    ss.tamp.itl1b--;

    // Compute boundary velocities
    ss.pb.vpbx = vpbx0;
    ss.pb.vpby = vpby0;
    ss.pb.vpbz = vpbz0;
    if (ss.pb.ic_pbx == 1) ss.pb.vpbx = vpbx0 / (double)ss.bc.nstep / ss.dtime;
    if (ss.pb.ic_pby == 1) ss.pb.vpby = vpby0 / (double)ss.bc.nstep / ss.dtime;
    if (ss.pb.ic_pbz == 1) ss.pb.vpbz = vpbz0 / (double)ss.bc.nstep / ss.dtime;

    // Period lengths and centers
    ss.pb.pblx = pbx2 - pbx1;
    ss.pb.pbly = pby2 - pby1;
    ss.pb.pblz = pbz2 - pbz1;
    ss.pb.pbcx = (pbx2 + pbx1) / 2.0;
    ss.pb.pbcy = (pby2 + pby1) / 2.0;
    ss.pb.pbcz = (pbz2 + pbz1) / 2.0;

    // Print boundary info
    auto printBound = [](const char *axis, int ipb, double p1, double p2) {
        if (ipb == 1)
            printf("periodic boundary %f < %s < %f\n", p1, axis, p2);
        else
            printf("smooth wall boundary %f < %s < %f\n", p1, axis, p2);
    };
    printBound("x", ss.pb.ipbx, pbx1, pbx2);
    printBound("y", ss.pb.ipby, pby1, pby2);
    printBound("z", ss.pb.ipbz, pbz1, pbz2);

    auto printControl = [](const char *axis, int ic, double v, double sig) {
        if (ic == 1)
            printf("%s- boundary is displ. control %e\n", axis, v);
        else
            printf("%s- boundary is stress control %e\n", axis, sig);
    };
    printControl("x", ss.pb.ic_pbx, ss.pb.vpbx, ss.pb.sig_pbx);
    printControl("y", ss.pb.ic_pby, ss.pb.vpby, ss.pb.sig_pby);
    printControl("z", ss.pb.ic_pbz, ss.pb.vpbz, ss.pb.sig_pbz);

    fclose(fp);
}

// ============================================================
// Read in_cf.dat — contact force history
// ============================================================
void read_in_cf(SimState &ss) {
    if (ss.bc.icld == 0) return;

    printf("icld=1: Reading shear force data from in_cf.dat\n");
    FILE *fp = fopen_fortran("in_cf.dat");
    if (!fp) {
        fprintf(stderr, "Error: cannot open in_cf.dat\n");
        exit(1);
    }

    for (int i = 0; i < ss.elem.nelem; i++) {
        int idum, ic;
        fscanf(fp, "%d %d", &idum, &ic);
        idum--; // Convert to 0-based
        if (idum != i) {
            fprintf(stderr, "Inconsistency in reading in_cf.dat at element %d\n", i);
            exit(1);
        }
        ss.cont.icount[i] = ic;
        for (int j = 0; j < ic; j++) {
            int neib_j;
            double fc1, fc2, al;
            fscanf(fp, "%d %lf %lf %lf", &neib_j, &fc1, &fc2, &al);
            neib_j--; // Convert to 0-based
            ss.cont.neib[i][j] = neib_j;
            ss.cont.fcont[i][j][0] = fc1;
            ss.cont.fcont[i][j][1] = fc2;
            ss.cont.alpha[i][j] = al;
        }
    }
    fclose(fp);
}

// ============================================================
// Modify particle positions into periodic domain
// ============================================================
void modify_positions_periodic(SimState &ss) {
    double pi = 4.0 * atan(1.0);

    int idum = 0;
    int ii = 0;
    for (int k = 0; k < ss.grain.nptotl; k++) {
        double gcx0 = ss.grain.gcx[k];
        double gcy0 = ss.grain.gcy[k];
        double gcz0 = ss.grain.gcz[k];

        if (ss.pb.ipbx == 1)
            ss.grain.gcx[k] -= round((ss.grain.gcx[k] - ss.pb.pbcx) / ss.pb.pblx) * ss.pb.pblx;
        if (ss.pb.ipby == 1)
            ss.grain.gcy[k] -= round((ss.grain.gcy[k] - ss.pb.pbcy) / ss.pb.pbly) * ss.pb.pbly;
        if (ss.pb.ipbz == 1)
            ss.grain.gcz[k] -= round((ss.grain.gcz[k] - ss.pb.pbcz) / ss.pb.pblz) * ss.pb.pblz;

        double difx = ss.grain.gcx[k] - gcx0;
        double dify = ss.grain.gcy[k] - gcy0;
        double difz = ss.grain.gcz[k] - gcz0;

        for (int j = 0; j < ss.grain.nset[k]; j++) {
            ss.elem.xc[ii] += difx;
            ss.elem.yc[ii] += dify;
            ss.elem.zc[ii] += difz;

            // Check if this is a boundary grain — skip for domain extent
            bool isBoundary = false;
            for (int ipb = 0; ipb < ss.bc.npboun; ipb++) {
                if (k == ss.bc.ipboun[ipb]) { isBoundary = true; break; }
            }

            if (!isBoundary) {
                if (idum == 0) {
                    idum = 1;
                    ss.xcmin = ss.elem.xc[ii]; ss.xcmax = ss.elem.xc[ii];
                    ss.ycmin = ss.elem.yc[ii]; ss.ycmax = ss.elem.yc[ii];
                    ss.zcmin = ss.elem.zc[ii]; ss.zcmax = ss.elem.zc[ii];
                } else {
                    if (ss.elem.xc[ii] < ss.xcmin) ss.xcmin = ss.elem.xc[ii];
                    if (ss.elem.xc[ii] > ss.xcmax) ss.xcmax = ss.elem.xc[ii];
                    if (ss.elem.yc[ii] < ss.ycmin) ss.ycmin = ss.elem.yc[ii];
                    if (ss.elem.yc[ii] > ss.ycmax) ss.ycmax = ss.elem.yc[ii];
                    if (ss.elem.zc[ii] < ss.zcmin) ss.zcmin = ss.elem.zc[ii];
                    if (ss.elem.zc[ii] > ss.zcmax) ss.zcmax = ss.elem.zc[ii];
                }
            }
            ii++;
        }
    }

    double pi_val = 4.0 * atan(1.0);
    double av_r = pow(ss.avvol * 0.75 / pi_val, 1.0/3.0);
    if (ss.pb.ipbx == 0) { ss.xcmin += av_r; ss.xcmax -= av_r; }
    if (ss.pb.ipby == 0) { ss.ycmin += av_r; ss.ycmax -= av_r; }
    if (ss.pb.ipbz == 0) { ss.zcmin += av_r; ss.zcmax -= av_r; }

    ss.stress.vr_ini = (ss.xcmax - ss.xcmin) * (ss.ycmax - ss.ycmin) * (ss.zcmax - ss.zcmin);

    printf("domain of the grains:\n");
    printf("x: %e %e\n", ss.xcmin, ss.xcmax);
    printf("y: %e %e\n", ss.ycmin, ss.ycmax);
    printf("z: %e %e\n", ss.zcmin, ss.zcmax);

    if (ss.pb.ipb3 == 0) {
        printf("Vr_ini for stress calculation: %e\n", ss.stress.vr_ini);
    }
}

// ============================================================
// Write new_in_gm.dat — matches Fortran output subroutine
// ============================================================
// FIX 5: restart files are written with full double precision (%.17g).
// Previously positions used %e (7 significant digits): a coordinate of
// ~338 mm was rounded to 1e-4 mm, i.e. 3-6% of a typical 1.6e-3 mm contact
// overlap, so every restart jolted all contact forces.  The Fortran writes
// full precision (list-directed), so this was a port-only error.
void write_new_in_gm(SimState &ss) {
    FILE *fp = fopen("new_in_gm.dat", "w");
    if (!fp) return;

    fprintf(fp, " %.17g %.17g %.17g %.17g\n",
            ss.dtime, ss.gx, ss.gy, ss.gz);
    fprintf(fp, " %d\n", ss.mat.nmater);
    for (int i = 0; i < ss.mat.nmater; i++) {
        fprintf(fp, " %.17g %.17g %.17g %.17g %.17g %.17g\n",
                ss.mat.rho[i], ss.mat.akn[i], ss.mat.aks[i],
                ss.mat.cn[i], ss.mat.cs[i], ss.mat.phimu[i]);
    }

    fprintf(fp, " %10d%10d%10d\n", ss.grain.nptotl, ss.nkk1, ss.nkk2);

    int icr = 0;
    for (int k = 0; k < ss.grain.nptotl; k++) {
        fprintf(fp, " %d %d %d\n", k+1, ss.elem.nm[icr]+1, ss.grain.nset[k]);
        fprintf(fp, " %.17g %.17g %.17g %.17g\n",
                ss.grain.gv[k], ss.grain.gi[k][0], ss.grain.gi[k][1], ss.grain.gi[k][2]);
        fprintf(fp, " %.17g %.17g %.17g\n", ss.grain.vgx[k], ss.grain.vgy[k], ss.grain.vgz[k]);
        fprintf(fp, " %.17g %.17g %.17g\n", ss.grain.vgwx[k], ss.grain.vgwy[k], ss.grain.vgwz[k]);
        fprintf(fp, " %.17g %.17g %.17g\n", ss.grain.gcx[k], ss.grain.gcy[k], ss.grain.gcz[k]);
        fprintf(fp, " %.17g %.17g %.17g\n", ss.grain.gwx[k], ss.grain.gwy[k], ss.grain.gwz[k]);
        for (int i = 0; i < ss.grain.nset[k]; i++) {
            fprintf(fp, " %d %.17g\n", i+1, ss.elem.rc[icr]);
            fprintf(fp, " %.17g %.17g %.17g\n", ss.elem.xc[icr], ss.elem.yc[icr], ss.elem.zc[icr]);
            icr++;
        }
    }
    fclose(fp);
}

// ============================================================
// Write new_in_bc.dat — matches Fortran output subroutine
// ============================================================
void write_new_in_bc(SimState &ss) {
    FILE *fp = fopen("new_in_bc.dat", "w");
    if (!fp) return;

    int icld00 = ss.bc.icld;
    if (icld00 != 2) icld00 = 1;
    else icld00 = 1;

    int iloopt = ss.bc.iloop + ss.bc.nstep;
    fprintf(fp, " %10d %10d %10d %10d %10d\n",
            icld00, iloopt, ss.bc.nstep, ss.bc.nprn1, ss.bc.nprn2);
    fprintf(fp, " %d\n", ss.bc.nloadp);
    for (int i = 0; i < ss.bc.nloadp; i++) {
        fprintf(fp, " %d\n", ss.bc.loadp[i]);
    }
    fprintf(fp, " %d\n", ss.bc.npboun);
    for (int k = 0; k < ss.bc.npboun; k++) {
        int gi = ss.bc.ipboun[k]; // 0-based
        double bv[6];
        for (int d = 0; d < 6; d++) bv[d] = ss.bc.bval[gi][d];
        if (ss.bc.icode[gi][0] == 1) bv[0] *= ss.dtime * (double)ss.bc.nstep;
        if (ss.bc.icode[gi][1] == 1) bv[1] *= ss.dtime * (double)ss.bc.nstep;
        if (ss.bc.icode[gi][2] == 1) bv[2] *= ss.dtime * (double)ss.bc.nstep;
        if (ss.bc.icode[gi][3] == 1) bv[3] *= ss.dtime * (double)ss.bc.nstep;
        if (ss.bc.icode[gi][4] == 1) bv[4] *= ss.dtime * (double)ss.bc.nstep;
        if (ss.bc.icode[gi][5] == 1) bv[5] *= ss.dtime * (double)ss.bc.nstep;

        fprintf(fp, " %7d %2d %.17g %2d %.17g %2d %.17g\n",
                gi+1, ss.bc.icode[gi][0], bv[0],
                ss.bc.icode[gi][1], bv[1],
                ss.bc.icode[gi][2], bv[2]);
        fprintf(fp, "   %2d %.17g %2d %.17g %2d %.17g\n",
                ss.bc.icode[gi][3], bv[3],
                ss.bc.icode[gi][4], bv[4],
                ss.bc.icode[gi][5], bv[5]);
    }

    // Periodic boundaries
    double vpbx0 = ss.pb.vpbx, vpby0 = ss.pb.vpby, vpbz0 = ss.pb.vpbz;
    if (ss.pb.ic_pbx == 1) vpbx0 *= (double)ss.bc.nstep * ss.dtime;
    if (ss.pb.ic_pby == 1) vpby0 *= (double)ss.bc.nstep * ss.dtime;
    if (ss.pb.ic_pbz == 1) vpbz0 *= (double)ss.bc.nstep * ss.dtime;

    fprintf(fp, " %6d %.17g %.17g %3d %.17g %.17g\n",
            ss.pb.ipbx, ss.pb.pbcx - ss.pb.pblx/2.0, ss.pb.pbcx + ss.pb.pblx/2.0,
            ss.pb.ic_pbx, vpbx0, ss.pb.sig_pbx);
    fprintf(fp, " %6d %.17g %.17g %3d %.17g %.17g\n",
            ss.pb.ipby, ss.pb.pbcy - ss.pb.pbly/2.0, ss.pb.pbcy + ss.pb.pbly/2.0,
            ss.pb.ic_pby, vpby0, ss.pb.sig_pby);
    fprintf(fp, " %6d %.17g %.17g %3d %.17g %.17g\n",
            ss.pb.ipbz, ss.pb.pbcz - ss.pb.pblz/2.0, ss.pb.pbcz + ss.pb.pblz/2.0,
            ss.pb.ic_pbz, vpbz0, ss.pb.sig_pbz);

    // Tamping
    // FIX 6: advance pret by the time simulated in this segment, so that a
    // restart continues the tool's orbit and wobble at the correct phase.
    // Previously the old pret was written back unchanged; after a segment that
    // is not a whole number of vibration cycles, the restarted tool jumped in
    // phase and its orbit centre shifted.  (The Fortran has the same issue.)
    // Uses the step at which the file is written, so an intermediate snapshot
    // (written every nprn1 steps) also restarts at the right phase.
    double pret_next = ss.tamp.pret + ss.dtime * (double)ss.istep_cur;
    fprintf(fp, " %.17g %.17g %.17g %8d %8d %.17g\n",
            ss.tamp.hz, ss.tamp.Amp, ss.tamp.Zkak,
            ss.tamp.itl1a+1, ss.tamp.itl1b+1, pret_next);

    fclose(fp);
}

// ============================================================
// Write new_in_cf.dat
// ============================================================
void write_new_in_cf(SimState &ss) {
    FILE *fp = fopen("new_in_cf.dat", "w");
    if (!fp) return;
    for (int i = 0; i < ss.elem.nelem; i++) {
        fprintf(fp, " %d %d\n", i+1, ss.cont.icount[i]);
        for (int j = 0; j < ss.cont.icount[i]; j++) {
            fprintf(fp, "    %7d %.17g %.17g %.17g\n",
                    ss.cont.neib[i][j]+1,
                    ss.cont.fcont[i][j][0], ss.cont.fcont[i][j][1],
                    ss.cont.alpha[i][j]);
        }
    }
    fclose(fp);
}

#ifndef DEM3D_H
#define DEM3D_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============================================================
// Constants matching Fortran parameters
// ============================================================
#define NCC    420000   // Max number of elements (spheres)
#define NMAT   10       // Max number of materials
#define NGR    40000    // Max number of grains
#define NBX    500      // Max box divisions per axis
#define NEIMAX 30       // Max neighbors per element

// Stop the run when an element exceeds NEIMAX contacts (the Fortran stops too).
// Set to 0 to continue with a warning instead.
#ifndef ABORT_ON_NEIMAX_OVERFLOW
#define ABORT_ON_NEIMAX_OVERFLOW 1
#endif

// ============================================================
// CUDA error checking macro
// ============================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// Material properties (per material type)
// ============================================================
struct MaterialProps {
    double rho[NMAT];
    double akn[NMAT];
    double aks[NMAT];
    double cn[NMAT];
    double cs[NMAT];
    double phimu[NMAT];
    int nmater;
};

// ============================================================
// Element (sphere) data — stored as Structure of Arrays for GPU
// ============================================================
struct ElementData {
    double rc[NCC];             // radius
    double xc[NCC], yc[NCC], zc[NCC];  // position
    double vxc[NCC], vyc[NCC], vzc[NCC]; // translational velocity
    double vwx[NCC], vwy[NCC], vwz[NCC]; // angular velocity
    int    np[NCC];             // parent grain index (1-based in Fortran, 0-based here)
    int    nm[NCC];             // material index (0-based)
    int    nelem;               // total number of elements
};

// ============================================================
// Grain data
// ============================================================
struct GrainData {
    double gcx[NGR], gcy[NGR], gcz[NGR];   // center position
    double gwx[NGR], gwy[NGR], gwz[NGR];   // Euler angles
    double vgx[NGR], vgy[NGR], vgz[NGR];   // translational velocity
    double vgwx[NGR], vgwy[NGR], vgwz[NGR]; // angular velocity (global)
    double gv[NGR];                          // volume
    double gi[NGR][3];                       // principal moments of inertia (divided by rho)
    int    nset[NGR];                        // number of elements per grain
    int    nptotl;                            // total number of grains
};

// ============================================================
// Boundary condition data
// ============================================================
struct BCData {
    int    icld;
    int    iloop;
    int    nstep;
    int    nprn1, nprn2;
    int    nloadp;
    int    npboun;
    int    loadp[NGR];
    int    ipboun[NGR];
    int    icode[NGR][6];     // 0=force, 1=displacement/velocity
    double bval[NGR][6];      // boundary values
    double bval0[NGR][6];     // initial boundary values
};

// ============================================================
// Periodic boundary data
// ============================================================
struct PBData {
    int    ipbx, ipby, ipbz, ipb3;
    double pblx, pbly, pblz;         // period lengths
    double pbcx, pbcy, pbcz;         // period centers
    double vpbx, vpby, vpbz;         // boundary velocities
    double sig_pbx, sig_pby, sig_pbz; // target stresses
    int    ic_pbx, ic_pby, ic_pbz;   // control mode (0=stress, 1=displ)
    double bdp[6];                    // absolute boundary positions
};

// ============================================================
// Tamping tool data
// ============================================================
struct TampData {
    double hz, Amp, Zkak;
    int    itl1a, itl1b;
    double pret;
};

// ============================================================
// Energy data
// ============================================================
struct EnergyData {
    double envx, envy, envz, envw;   // kinetic energy components
    double engr;                      // gravitational potential energy
    double enfr;                      // friction energy dissipation
    double enfr1, enfr2, enfr3;       // friction energy by group
    double enkn, enks, enkt;          // strain energy
    double encn, encs, enct;          // damping energy
    double encx, ency, encz;          // artificial damping energy
    double enxf;                      // external energy (force assigned)
    double enxd;                      // external energy (displ assigned)
};

// ============================================================
// Stress tensor
// ============================================================
struct StressData {
    double sig[3][3];
    double vr_ini;
};

// ============================================================
// Contact data (per element)
// ============================================================
struct ContactData {
    int    icount[NCC];              // number of contacts per element
    int    neib[NCC][NEIMAX];        // neighbor indices
    double fcont[NCC][NEIMAX][2];    // stored shear forces (s and t)
    double alpha[NCC][NEIMAX];       // accumulated rotation angle
    double pforce[NCC][6];           // total force on element
};

// ============================================================
// GPU device data structures (flattened for CUDA)
// ============================================================
struct DeviceElementData {
    double *rc;
    double *xc, *yc, *zc;
    double *vxc, *vyc, *vzc;
    double *vwx, *vwy, *vwz;
    int    *np, *nm;
    int    nelem;
};

struct DeviceGrainData {
    double *gcx, *gcy, *gcz;
    double *gwx, *gwy, *gwz;
    double *vgx, *vgy, *vgz;
    double *vgwx, *vgwy, *vgwz;
    double *gv;
    double *gi;      // flattened [NGR*3]
    int    *nset;
    int    *elemStart; // prefix sum: starting element index for each grain
    int    nptotl;
};

struct DeviceMaterialData {
    double *rho, *akn, *aks, *cn, *cs, *phimu;
};

struct DeviceContactData {
    int    *icount;
    int    *neib;           // flattened [NCC * NEIMAX]
    double *fcont;          // flattened [NCC * NEIMAX * 2]
    double *alpha;          // flattened [NCC * NEIMAX]
    double *pforce;         // flattened [NCC * 6]
};

// Spatial hashing on GPU
struct DeviceSpatialHash {
    int    *cellIndex;      // cell index per element
    int    *sortedIndex;    // sorted element indices
    int    *cellStart;      // start index in sorted array per cell
    int    *cellEnd;        // end index in sorted array per cell
    int    totalCells;
    int    ndivx, ndivy, ndivz;
    double dmxmin, dmymin, dmzmin;
    double xdiv, ydiv, zdiv;
};

// Contact pair for history tracking
struct ContactPair {
    int i, j;             // element indices (i < j)
    double fcont1, fcont2; // stored shear forces
    double alpha;          // accumulated rotation
};

// ============================================================
// GPU-side stress accumulation
// ============================================================
struct DeviceStressData {
    double *sig;   // [9] flattened 3x3
};

// ============================================================
// GPU-side energy accumulation
// ============================================================
struct DeviceEnergyData {
    double *enkn, *enks, *enkt;
    double *encn, *encs, *enct;
    double *enfr, *enfr1, *enfr2, *enfr3;
};

// ============================================================
// Global simulation state
// ============================================================
struct SimState {
    MaterialProps mat;
    ElementData   elem;
    GrainData     grain;
    BCData        bc;
    PBData        pb;
    TampData      tamp;
    EnergyData    energy;
    StressData    stress;
    ContactData   cont;

    double dtime;
    double gx, gy, gz;       // gravity
    double elemax;            // max element diameter
    double adamp;             // artificial damping
    int    mode1;             // stabilization mode
    int    nkk1, nkk2;       // friction energy grouping
    int    ncont;             // contact count

    // Derived quantities
    double avvol, avrho, gmmin;
    double xcmin, xcmax, ycmin, ycmax, zcmin, zcmax;
};

// ============================================================
// Function declarations — I/O
// ============================================================
void read_in_gm(SimState &ss);
void read_in_bc(SimState &ss);
void read_in_cf(SimState &ss);
void modify_positions_periodic(SimState &ss);
void write_new_in_gm(SimState &ss);
void write_new_in_bc(SimState &ss);
void write_new_in_cf(SimState &ss);
void write_output(SimState &ss, FILE *f1, FILE *f8, FILE *f11, int istep);
void write_new_files(SimState &ss, int istep);
void write_energy(SimState &ss, FILE *f4, int istep);
void write_cont2(SimState &ss, FILE *f9, int istep);

// ============================================================
// Function declarations — GPU kernels (in kernels.cu)
// ============================================================

// Spatial hashing
void gpu_compute_cell_indices(DeviceElementData &delem, DeviceSpatialHash &hash,
                              PBData &pb, int nelem);
void gpu_sort_by_cell(DeviceSpatialHash &hash, int nelem);
void gpu_find_cell_bounds(DeviceSpatialHash &hash, int nelem);

// Contact detection and force computation
void gpu_compute_contact_forces(DeviceElementData &delem,
                                DeviceGrainData &dgrain,
                                DeviceMaterialData &dmat,
                                DeviceContactData &dcont,
                                DeviceContactData &dcont_old,
                                DeviceSpatialHash &hash,
                                PBData &pb,
                                DeviceStressData &dstress,
                                DeviceEnergyData &denergy,
                                int nelem, int mode1, int istep, int nstep,
                                int nkk1, int nkk2,
                                int *d_ipboun, int npboun, int ipb3,
                                double dtime, double vr);

// Boundary contact forces (walls)
void gpu_wall_contact_forces(DeviceElementData &delem,
                             DeviceMaterialData &dmat,
                             DeviceContactData &dcont,
                             DeviceEnergyData &denergy,
                             PBData &pb, int nelem, int mode1,
                             int istep, int nstep, double dtime);

// Grain motion integration
void gpu_integrate_grains(DeviceElementData &delem,
                          DeviceGrainData &dgrain,
                          DeviceMaterialData &dmat,
                          DeviceContactData &dcont,
                          int nptotl, int nelem,
                          double dtime, double gx, double gy, double gz,
                          double adamp, int *d_icode, double *d_bval,
                          double *d_envx, double *d_envy, double *d_envz,
                          double *d_envw, double *d_engr,
                          double *d_encxyz,
                          double *d_enxf, double *d_enxd);

// ============================================================
// Utility
// ============================================================
static inline double deg2rad(double d) { return d * M_PI / 180.0; }
static inline double rad2deg(double r) { return r * 180.0 / M_PI; }

#endif // DEM3D_H

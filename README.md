# dem3d_cuda — GPU-Accelerated 3D DEM for Connected Spheres

CUDA/C++ rewrite of `dem3d_v24.f` (Matsushima, University of Tsukuba).  
Targets NVIDIA RTX 3070 (Compute Capability 8.6).

## Requirements

- NVIDIA GPU (RTX 3070 or compatible)
- CUDA Toolkit 11.0+ (tested with 11.x / 12.x)
- g++ 9+ or compatible C++ compiler
- NVIDIA driver 450+

## Build

```bash
cd dem3d_cuda
make
```

For debugging:
```bash
# Edit Makefile, uncomment debug flags, then:
make clean && make
```

For maximum numerical compatibility with Fortran:
```bash
# Edit Makefile, uncomment --fmad=false line, then:
make clean && make
```

## Usage

Place input files in the same directory as the executable, then run:

```bash
./dem3d_cuda
```

### Input files (identical format to Fortran version):
- `in_gm.dat` — Grain geometry and material properties
- `in_bc.dat` — Boundary conditions, periodic boundaries, tamping parameters
- `in_cf.dat` — Contact force history (for restart, required if icld≠0)

### Output files (identical format to Fortran version):
- `output.dat` — Grain positions/velocities per output step
- `out2.dat` — Element positions per output step
- `energy.dat` — Energy balance
- `f-d_curve.dat` — Force-displacement curves for load particles
- `cont.dat` — Contact information
- `cont2.dat` — Contact count per step
- `cont3.dat` — Large-force contacts
- `new_in_gm.dat` — Updated geometry (for restart)
- `new_in_bc.dat` — Updated BCs (for restart)
- `new_in_cf.dat` — Updated contact history (for restart)
- `info.dat` — Run information

## Architecture

```
main.cu              — Time stepping loop, I/O orchestration
io.cpp               — File reading/writing (in_gm.dat, in_bc.dat, etc.)
kernels_helpers.cu   — Device functions: contact judgment, force computation
kernels_contact.cu   — Contact detection kernel, wall forces
kernels_integrate.cu — Euler equation integration, element position update
gpu_memory.cu        — GPU allocation, host↔device transfers
launch_kernels.cu    — Kernel launch wrappers
dem3d.h              — All data structures and constants
gpu_data.h           — GPU data struct and launch declarations
```

## GPU Parallelization Strategy

| Stage | Fortran (CPU) | CUDA (GPU) |
|-------|--------------|------------|
| Spatial hashing | Linked-list ibox/link | Sort-based cell hashing via Thrust |
| Contact detection | Sequential element loop | One thread per element, 27-cell scan |
| Force computation | Sequential in unbforce | Per-contact in detection kernel |
| Grain integration | Sequential grain loop | One thread per grain |
| Element update | Sequential | One thread per grain (loop over elements) |


## Recent fixes (validation branch)

Four bugs were found by comparing this code line-by-line against the Fortran
original (`TampRot-Jfx6.f`).  The contact law, the integrator and the boundary
conditions were verified term by term and match; these are the differences that
were found.

| # | File | Problem | Effect |
|---|------|---------|--------|
| 1 | `gpu_memory.cu` | `d_icount_old` / `d_neib_old` / `d_fcont_old` / `d_alpha_old` were never cleared when `icld = 0`. `cudaMalloc` does not zero memory. | At step 1 the history lookup could match garbage and load undefined shear forces. Fortran clears `ngbold`/`cFs` at `istep = 1`. |
| 2 | `kernels_contact.cu` | NEIMAX overflow handled silently: (a) `if (my_cnt >= NEIMAX) continue;` dropped the contact **force**; (b) `new_icount` counts forward *and* reverse contacts, so a slot could overflow while `my_cnt < 30`, storing the force but losing the history; (c) `old_icount[i]` was used unclamped and could read into the next element's slots. | Affected contacts reset their tangential spring every step: more sliding, more friction/damping energy, deeper tool penetration. Fortran stops instead. |
| 3 | `main.cu` | The contact buffers were swapped *before* the output block, so `download_from_gpu()` read the previous step's list. | `new_in_cf.dat` was one step stale relative to `new_in_gm.dat`; every restart lost the histories of contacts formed in the last step. |
| 4 | `main.cu` | `SimState ss;` was a ~400 MB local variable. | Immediate segfault on Linux (8 MB default stack). Now `static`. |

Fix 2 adds a diagnostic: the run reports the maximum number of contacts on any
element and aborts if NEIMAX is exceeded (set `ABORT_ON_NEIMAX_OVERFLOW 0` in
`dem3d.h` to continue with a warning instead).  If overflow does occur, raise
`NEIMAX` in `dem3d.h` and rebuild.

### Known remaining differences from the Fortran

- **Element velocity branch vector.** In the Fortran element loop `rr` is
  overwritten with R(t+dt) inside the loop, so every element after the first in
  each grain uses R(t+dt) rather than R(t+dt/2) for its velocity. This code uses
  R(t+dt/2) for all elements (arguably more correct). Difference is O(omega*dt).
- **sin(beta) = 0 guard** does not also reset `bet` on the first iteration as the
  Fortran does. Irrelevant unless a grain sits exactly at beta = 0 or pi.
- **Non-determinism.** Forces and energies are accumulated with `atomicAdd`, so
  two identical runs of this code are not bit-identical. See the twin test below.

## Build

```bash
make                    # normal build, -arch=native
make FMAD=false         # validation build: disables fused multiply-add
make ARCH=sm_80         # pin the architecture if -arch=native is unavailable
make clean
```

`-arch=native` needs CUDA >= 11.5.  Otherwise: A100 `sm_80`, RTX 3090/A40/A6000
`sm_86`, RTX 4090/L40S/RTX 6000 Ada `sm_89`, H100 `sm_90`.

## Validating against the Fortran

Because the GPU summation order varies, two identical C++ runs diverge from each
other exactly as chaos would.  Use that as the control:

```bash
mkdir -p runA runB && cp in_gm.dat in_bc.dat runA/ && cp in_gm.dat in_bc.dat runB/
cd runA && ../dem3d_cuda | tee run.log && cd ..
cd runB && ../dem3d_cuda | tee run.log && cd ..
python tools/compare_runs.py runA runB fortran_run
```

If A-vs-B shows a spread comparable to A-vs-Fortran, the difference is chaotic
divergence and the port reproduces the physics.  If A-vs-B stays much closer,
there is still a code difference to find.

## Notes on Numerical Differences

Results will be functionally identical but not bit-for-bit due to:
- Floating-point summation order (parallel vs sequential)
- FMA instructions (disable with `--fmad=false` for closer match)
- Math library differences (`sin`, `cos`, `sqrt`)

For validation, compare macroscopic quantities (stress, energy, force-displacement curves)
rather than individual particle positions over long runs.

## Performance

Expected 10–50× speedup over single-threaded Fortran on typical problem sizes
(1,000–40,000 grains). Actual speedup depends on:
- Number of elements and contacts
- Contact density
- GPU occupancy (best with >10,000 elements)

## Index Convention

The Fortran code uses 1-based indexing. This C++ code uses 0-based indexing internally.
Input/output files maintain 1-based indexing for compatibility.

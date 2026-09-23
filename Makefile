# ============================================================
# dem3d_cuda — build
#
#   make                 normal build (fast)
#   make FMAD=false      validation build: disables fused multiply-add so the
#                        arithmetic matches the Fortran more closely
#   make ARCH=sm_80      pin the GPU architecture (default: native)
#   make DEBUG=1         -g -G, device debug, very slow
#   make clean
#
# ARCH=native needs CUDA >= 11.5.  If nvcc complains, set it explicitly:
#   A100 = sm_80   RTX 4090 = sm_89   RTX 3090/A40/A6000 = sm_86
#   L40/L40S/RTX 6000 Ada = sm_89     H100 = sm_90
# ============================================================

NVCC      ?= nvcc
ARCH      ?= native
STD       ?= c++17
TARGET    ?= dem3d_cuda

NVCCFLAGS  = -std=$(STD) -arch=$(ARCH)

ifeq ($(DEBUG),1)
  NVCCFLAGS += -g -G -O0
else
  NVCCFLAGS += -O3 -lineinfo
endif

# Bit-for-bit friendliness with the Fortran reference.
ifeq ($(FMAD),false)
  NVCCFLAGS += --fmad=false
endif

# Phase timing report (adds device syncs; diagnostic builds only).
ifeq ($(TIMING),1)
  NVCCFLAGS += -DDEM_TIMING=1
endif

SRCS = main.cu io.cpp gpu_memory.cu kernels_helpers.cu kernels_contact.cu \
       kernels_integrate.cu launch_kernels.cu
OBJS = $(addsuffix .o,$(basename $(SRCS)))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# io.cpp uses CUDA headers, so compile it with nvcc as CUDA too.
%.o: %.cu dem3d.h gpu_data.h
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

%.o: %.cpp dem3d.h gpu_data.h
	$(NVCC) $(NVCCFLAGS) -x cu -dc -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean

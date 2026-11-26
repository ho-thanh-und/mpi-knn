CPP=mpicxx
# No -march: the AVX / AVX-512 kernels carry per-function target attributes and are
# chosen at run time, so one binary runs on every lab machine. -ffp-contract=off keeps
# multiply and add separate (no FMA) so every distance matches the reference bit for bit.
# -std=c++14 is accepted by every GCC since 5; the *_SKIP_MPICXX macros keep mpi.h from
# pulling in the legacy MPI C++ bindings that some distro builds still ship.
CPPFLAGS= -std=c++14 -O2 -fno-tree-vectorize -fno-tree-slp-vectorize -ffp-contract=off -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX

.PHONY: all clean

all: engine engine.debug

clean:
	rm -rf engine engine.debug

engine: engine.cpp common.cpp
	${CPP} ${CPPFLAGS} $^ -o $@

engine.debug: engine.cpp common.cpp
	${CPP} ${CPPFLAGS} -DDEBUG $^ -o $@

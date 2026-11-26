# MPI KNN

A distributed **k-nearest neighbors classifier in C++ and MPI**, focused on balancing query work, reducing communication overhead, and improving data locality.

This project explores the systems side of similarity search: distributing independent queries across processes while keeping nearest-neighbor selection and classification deterministic.

## How it works

- **Replicated data, distributed queries.** Rank 0 broadcasts the dataset and labels to all processes. Processes work on dynamically assigned query batches; rank 0 also performs computation.
- **Cache-conscious scanning.** Features are arranged in blocks using a structure-of-arrays layout. Each scan accumulates squared distances for groups of four points and can stop early when every partial distance in the group exceeds the current cutoff.
- **Bounded top-k selection.** A max-heap retains the best candidates without sorting the full dataset.
- **Overlapped communication.** Nonblocking sends and receives, double buffers, and progress checks keep query requests and results moving during computation.
- **Optional query compression.** Decimal packing is used only when values round-trip exactly; other values fall back to raw doubles. Set `KNN_COMPRESS=0` to disable it.

Equal distances favor the larger point index; tied classification votes favor the larger label. Results are reported in query order.

## Build and run

Use a Linux environment with **GNU Make**, a **C++14 compiler**, and **Open MPI** providing `mpicxx` and `mpirun`. On Windows, use WSL.

```bash
make
python3 generate_input.py --num_data 1000 --num_queries 20 --num_attrs 8 \
  --min 0 --max 1 --minK 5 --maxK 5 --num_labels 3 \
  --seed 42 --output sample.in
mpirun -np 4 ./engine < sample.in
```

The generator requires Python 3. The regular executable prints a checksum per query and reports elapsed time to standard error. To inspect predicted labels, neighbor indices, and squared distances:

```bash
mpirun -np 4 ./engine.debug < sample.in
```

## Input format

The first line contains `num_data num_queries num_attributes`. It is followed by one line per labeled point, then one line per query:

```text
3 1 2
0 0.0 0.0
1 1.0 1.0
1 2.0 2.0
Q 2 0.9 0.9
```

Data rows contain a label and feature values. Query rows start with `Q`, followed by `k` and the feature values. Use positive counts and `1 <= k <= num_data`; inputs are expected to follow this format.

Additional datasets are bundled in `inputs.zip`.

## Compare process counts

Use the same generated input for each run:

```bash
mpirun -np 1 ./engine < sample.in > single.out
mpirun -np 4 ./engine < sample.in > parallel.out
diff single.out parallel.out
```

Matching checksums provide a consistency check across process counts. For performance measurements, use a larger input, repeat runs, and record hardware, process placement, MPI version, and dataset dimensions.

Each process keeps its own dataset copy, so memory use grows with process count. Speedup also depends on query count, dimensionality, batch balance, and communication costs.

## Project layout

| File | Purpose |
| :--- | :--- |
| `engine.cpp` | Query scheduling, distance scans, top-k selection, and MPI communication |
| `engine.h`, `common.h` | Engine interface and shared data types |
| `common.cpp` | Input parsing, result reporting, and timing |
| `generate_input.py` | Reproducible random input generation |
| `inputs.zip` | Additional input datasets |
| `Makefile` | Regular and debug builds |

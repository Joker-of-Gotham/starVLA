# StarVLA H200 Performance Tools

Packaged binaries:

- `*_perf.real`: non-MPI NCCL test binaries found on the CPU node under `/usr/local/bin`.
- `lib/verifiable/libverifiable.so.0`: runtime dependency for the packaged NCCL tests.
- `src/nccl-tests/`: nccl-tests source without `.git` or build outputs, for rebuilding on a GPU node if the packaged binaries are incompatible.
- `cuda_samples/bin/p2pBandwidthLatencyTest`: built from NVIDIA CUDA Samples v12.4 for GPU P2P validation.

Wrappers named `all_reduce_perf`, `reduce_scatter_perf`, etc. set `LD_LIBRARY_PATH`
to the packaged `libverifiable` and the StarVLA env's CUDA/NCCL Python wheel
libraries before executing the real binary.

The official `nvbandwidth` v0.9 source is included under `src/nvbandwidth`.
This CPU zone is missing the Boost program_options development dependency
required to build the binary. On a GPU node with that dependency available, run
`scripts/build_optional_nvbandwidth.sh` from the envSet root, then
`scripts/install_perf_tools.sh`.

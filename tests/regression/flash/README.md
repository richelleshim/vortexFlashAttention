# FlashAttention Regression Test

This directory builds the host test harness and Vortex kernel for FlashAttention. Use the Makefile targets to compile and run either the SIMT baseline or the tensor-core (TCU) variant.

## Build the test

```sh
make -C tests/regression/flash
```

This produces the host binary `flash` and the device binary `kernel.vxbin`.

## Run the standard (SIMT) kernel

Pass `--kernel=simt` (default) to select the baseline implementation. Choose the appropriate driver target for your environment.

- Software simulator (simx):
  ```sh
  make -C tests/regression/flash run-simx OPTS="--kernel=simt"
  ```
- RTL simulator:
  ```sh
  make -C tests/regression/flash run-rtlsim OPTS="--kernel=simt"
  ```
- OPAE/FPGA boards:
  ```sh
  make -C tests/regression/flash run-opae OPTS="--kernel=simt"
  ```
- XRT flow (set `TARGET=hw` or `TARGET=hw_emu` as needed):
  ```sh
  make -C tests/regression/flash run-xrt OPTS="--kernel=simt"
  ```

The program prints occupancy, tile sizes, and verification status; a zero exit code indicates the GPU result matches the CPU reference.

## Benchmarking notes

To compare against the TCU path, run again with `--kernel=tcu` and identical `OPTS`. Timing and instruction counts depend on the selected driver; for cycle-level measurements use the RTL/XRT flows that capture trace data. Keep the tile size at the SIMT default (block_size=4) when benchmarking the standard kernel so the code paths remain identical to the baseline implementation.

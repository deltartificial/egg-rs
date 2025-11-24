# EGGROLL in Rust

A minimalist implementation of the EGGROLL (Evolution Guided General Optimization via Low-rank Learning) algorithm in Rust.

This project demonstrates integer-only training of a language model directly on the CPU (optimized for Apple Silicon/M-series chips), completely bypassing the need for GPUs, floating-point arithmetic, or heavy ML frameworks like PyTorch or JAX.

## Key Features

*   **Pure Rust**: Minimal dependencies (rayon for threading, memmap2 for memory mapping).
*   **Apple Silicon Optimized**: Vectorized operations using ARM NEON intrinsics and parallelized via rayon.
*   **Integer Only**: Operates entirely on `i8` weights/activations with `i32` accumulation. No float math in the training loop.
*   **Gradient Free**: Uses Evolution Strategies (ES) with low-rank perturbations instead of backpropagation.

## Quick Start

### 1. Prepare Data
Ensure you have a text dataset named `input.txt` in the current directory.

### 2. Compile
```bash
cargo build --release
```

### 3. Run
```bash
./target/release/egg-rs
```

## Configuration

Configuration constants are defined at the top of `src/main.rs`:

```rust
const VOCAB_SIZE: usize = 256;
const HIDDEN_DIM: usize = 128;
const N_LAYERS: usize = 2;
const SEQ_LEN: usize = 512;
const POPULATION_SIZE: usize = 32;
```

## References

*   **C Implementation**: [d0rc/egg.c](https://github.com/d0rc/egg.c)
*   **Original JAX Implementation**: [ESHyperscale/nano-egg](https://github.com/ESHyperscale/nano-egg)
*   **Original Paper & Project**: [EGGROLL Website](https://eshyperscale.github.io/)

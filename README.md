# Game of Life on the GPU with Rust

An implementation of Game of Life using the amdgpu Rust target to let it run on the GPU.

No shortage of hacks and interesting low-level details.

## Build & Run

How to build and run:
1. Install **nightly** Rust in a way of your choice (rustup is recommended)
1. Setup **`ROCM_PATH`** as described in [amdgpu-rs](https://github.com/Flakebi/amdgpu-rs)
    - Alternatively, use the nix flake with `nix develop`
1. Compile the cpu program: `cd cpu && cargo build --release`
1. Run `rocminfo` to find your GPU’s gfx version (look for the ISA name)
    - Alternatively, run `target/release/cpu` once. It will fail but print the gfx versions at the top
    - See the Rust docs for more info: https://doc.rust-lang.org/nightly/rustc/platform-support/amdgcn-amd-amdhsa.html
1. Compile the gpu program: `cd gpu && CARGO_BUILD_RUSTFLAGS='-Ctarget-cpu=gfx<my version>' cargo build --release`
1. Run the program: `cd gpu && ../cpu/target/release/cpu`

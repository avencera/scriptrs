fmt:
    cargo fmt
    cargo fmt --manifest-path crates/scriptrs-python/Cargo.toml

clippy:
    cargo clippy --all --all-targets --all-features -- -D warnings
    cargo clippy --manifest-path crates/scriptrs-python/Cargo.toml --all-targets -- -D warnings

test *args:
    cargo test {{args}}

# scriptrs runs on native backends only; fail if ONNX Runtime comes back
# through any feature or dependency
no-ort:
    #!/usr/bin/env bash
    set -euo pipefail
    # capture the tree first, so a cargo failure fails the gate instead of
    # reading as "no ort", and an early grep exit can't SIGPIPE cargo
    deps=$(cargo tree --all-features --target all -e normal,build --prefix none)
    if grep -qE '^ort(-sys)? ' <<<"$deps"; then echo "ort is in the dependency tree" >&2; exit 1; fi

# parity tests against NeMo reference tensors (dev-tools/nemo/dump_ref.py);
# set SCRIPTRS_NEMO_REF to use a copy other than /shared/datasets/parakeet-ref
parity *args:
    cargo test --all-features --lib nemo_parity -- --ignored {{args}}

check:
    just fmt
    just no-ort
    just clippy
    just test

# GPU tests (`gpu_` prefix): NeMo parity and the scheduler; they need the GPU, so
# submit them through the homebased gpu0 queue instead of running them directly
gpu-parity *args:
    cargo test --release --features cuda,long-form,vad,low-precision --lib --test scheduled gpu_ -- --ignored --nocapture --test-threads 1 {{args}}

# write every precision's weight file from the exported FP32 weights (~3 GB under
# /shared/models/scriptrs-native) and check each decodes as load-time quantization
weight-files:
    cargo test --release --features cuda,low-precision --lib written_weights_match_load_time_quantization -- --ignored --nocapture

# rebuild src/cuda/ptx/<area>.<tier>.ptx from crates/scriptrs-cuda-kernels with the
# cuda-oxide nightly; a hand-built stand-in until cudakit-build owns the toolchain.
# `attention` needs mma.sync and cp.async, so it builds for sm_80 and the rest for sm_75
cuda-kernels area="encoder":
    #!/usr/bin/env bash
    set -euo pipefail
    arch=sm_75
    if [ "{{area}}" = attention ]; then arch=sm_80; fi
    out=$(mktemp -d)
    trap 'rm -rf "$out"' EXIT
    cd crates/scriptrs-cuda-kernels
    # cuda-oxide writes PTX only while rustc compiles the crate
    env -u RUSTUP_TOOLCHAIN -u CARGO -u RUSTC CARGO_TARGET_DIR="$PWD/target" \
        cargo +nightly-2026-08-28 clean -p scriptrs-cuda-kernels
    env -u RUSTUP_TOOLCHAIN -u CARGO -u RUSTC CARGO_TARGET_DIR="$PWD/target" \
        CUDA_OXIDE_PTX_DIR="$out" CUDA_OXIDE_LLC="${CUDA_OXIDE_LLC:-/usr/bin/llc-21}" \
        LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-21/lib}" \
        cargo +nightly-2026-08-28 oxide build --arch "$arch" --features {{area}}
    cp "$out/scriptrs_cuda_kernels.ptx" "../../src/cuda/ptx/{{area}}.${arch/_/}.ptx"

# build the Python wheel for this platform into crates/scriptrs-python/target/wheels,
# without the manylinux check that CI runs in its container
python-build:
    uvx --from 'maturin>=1.9,<2' maturin build --release --locked --auditwheel skip -m crates/scriptrs-python/Cargo.toml

# install the wheel into a fresh environment and run its tests; on Linux the model
# tests need the GPU, so submit this through the homebased gpu0 queue there
python-test *args: python-build
    #!/usr/bin/env bash
    set -euo pipefail
    cd crates/scriptrs-python
    venv=target/python-test-venv
    rm -rf "$venv"
    uv venv -q --python 3.12 "$venv"
    version=$(cargo pkgid | sed 's/.*[#@]//')
    uv pip install -q --python "$venv" "target/wheels/scriptrs-${version}"-*.whl pytest numpy
    "$venv"/bin/python -m pytest -rA {{args}}

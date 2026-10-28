fmt:
    cargo fmt --all

clippy:
    cargo clippy --all --all-targets --all-features -- -D warnings

test *args:
    cargo test {{args}}

# scriptrs runs on native backends only; fail if ONNX Runtime comes back
# through any feature or dependency
no-ort:
    if cargo tree --all-features --target all -e normal,build --prefix none | grep -qE '^ort(-sys)? '; then echo "ort is in the dependency tree" >&2; exit 1; fi

# parity tests against NeMo reference tensors (dev-tools/nemo/dump_ref.py);
# set SCRIPTRS_NEMO_REF to use a copy other than /shared/datasets/parakeet-ref
parity *args:
    cargo test --all-features --lib nemo_parity -- --ignored {{args}}

check:
    just fmt
    just no-ort
    just clippy
    just test

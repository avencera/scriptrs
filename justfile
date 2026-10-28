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

check:
    just fmt
    just no-ort
    just clippy
    just test

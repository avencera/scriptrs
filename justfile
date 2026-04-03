fmt:
    cargo fmt --all

clippy:
    cargo clippy --all --all-targets --all-features -- -D warnings

test *args:
    cargo test {{args}}

check:
    just fmt
    just clippy
    just test

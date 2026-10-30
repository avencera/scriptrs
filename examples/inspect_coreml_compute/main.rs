//! Prints which compute devices Core ML plans for each model in a bundle
//!
//! Core ML and its Objective-C bindings exist only on macOS, so elsewhere the example
//! builds to a stub that says so

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    macos::main()
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("CoreML compute inspection is only supported on macOS");
    std::process::ExitCode::FAILURE
}

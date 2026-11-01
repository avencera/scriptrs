use std::process::ExitCode;

// Core ML and its bindings exist only on macOS, so the inspector lives in a module
// that other platforms skip; `--all-targets` then still builds this example there
#[cfg(target_os = "macos")]
mod coreml;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:?}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "macos")]
fn run() -> eyre::Result<()> {
    coreml::run()
}

#[cfg(not(target_os = "macos"))]
fn run() -> eyre::Result<()> {
    eyre::bail!("CoreML compute inspection is only supported on macOS")
}

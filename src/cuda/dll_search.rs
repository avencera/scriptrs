//! Keeps Windows from loading the CUDA libraries out of the current directory
//!
//! cudarc opens the driver and cuBLAS by bare name with `LoadLibraryExW(name, NULL, 0)`,
//! Windows' standard search, which looks in the current directory before `PATH`. It
//! also tries names that no NVIDIA install uses before the real ones, such as
//! `cuda.dll` before `nvcuda.dll` and `cublas.dll` before `cublas64_12.dll`, so a DLL
//! planted under one of those names in the directory scriptrs runs from would load and
//! run its code. `SetDllDirectoryW("")` takes the current directory out of that search.
//! The executable's directory, System32, the Windows directory and `PATH` stay in it,
//! so a CUDA toolkit on `PATH` is still found
//!
//! The setting is process-wide, so after the first CUDA model loads, the host's own
//! bare-name loads skip the current directory too. A host that already set a DLL
//! directory has removed the current directory itself, and is left alone

use std::sync::Once;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetDllDirectoryW(buffer_length: u32, buffer: *mut u16) -> u32;
    fn SetDllDirectoryW(path: *const u16) -> i32;
}

/// Takes the current directory out of the process's DLL search, once
pub(crate) fn exclude_current_directory() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: a zero-length buffer is never written; the call only reports the
        // length of the directory the host set, 0 when there is none
        if unsafe { GetDllDirectoryW(0, std::ptr::null_mut()) } > 0 {
            return;
        }

        let empty = [0u16];
        // SAFETY: `empty` is a NUL-terminated UTF-16 string that outlives the call
        if unsafe { SetDllDirectoryW(empty.as_ptr()) } == 0 {
            tracing::warn!(
                error = %std::io::Error::last_os_error(),
                "could not take the current directory out of the DLL search"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::ffi::{OsStr, c_void};
    use std::fs;
    use std::os::windows::ffi::OsStrExt;
    use std::path::PathBuf;
    use std::process::Command;

    use super::exclude_current_directory;

    /// Set only in the child run, to the probe DLL's file name
    const PROBE: &str = "SCRIPTRS_DLL_SEARCH_PROBE";
    const TEST: &str = "cuda::dll_search::tests::current_directory_is_not_searched";

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    /// Loads and unloads a DLL with the flags cudarc's loader passes
    fn loads(name: &OsStr) -> bool {
        let wide: Vec<u16> = name.encode_wide().chain([0]).collect();
        // SAFETY: `wide` is NUL-terminated, and the probe is a copy of a system DLL
        // whose initializer has no preconditions
        let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), 0) };
        if module.is_null() {
            return false;
        }

        // SAFETY: the handle came from the load above and is released once
        unsafe { FreeLibrary(module) };
        true
    }

    /// A DLL in the current directory loads by bare name until the fix and not after
    /// it, and still loads by absolute path
    #[test]
    fn current_directory_is_not_searched() {
        let Some(name) = env::var_os(PROBE) else {
            // the parent reruns this test in a child whose current directory holds the
            // probe, since moving this process's current directory would move it
            // under every other test running alongside
            let dir = tempfile::tempdir().expect("temp dir");
            // a copy of a system DLL under a name no Windows directory has
            let name = format!("scriptrs_dll_probe_{}.dll", std::process::id());
            let system = PathBuf::from(env::var_os("SystemRoot").expect("SystemRoot"));
            fs::copy(
                system.join("System32").join("version.dll"),
                dir.path().join(&name),
            )
            .expect("copy the probe DLL");

            let status = Command::new(env::current_exe().expect("test binary"))
                .args(["--exact", TEST, "--nocapture", "--test-threads", "1"])
                .env(PROBE, &name)
                .current_dir(dir.path())
                .status()
                .expect("run the child test");
            assert!(status.success(), "the child test failed");
            return;
        };

        assert!(
            loads(&name),
            "the probe should load from the current directory at first"
        );
        exclude_current_directory();
        assert!(!loads(&name), "the probe loaded from the current directory");
        let absolute = env::current_dir().expect("current dir").join(&name);
        assert!(
            loads(absolute.as_os_str()),
            "the probe should load by absolute path"
        );
    }
}

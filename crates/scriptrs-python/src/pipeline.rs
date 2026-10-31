//! The native pipeline behind `scriptrs.Pipeline`

use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::prelude::*;
use scriptrs::{
    LanguageSelection, LongFormConfig, LongFormMode, LongFormTranscriptionPipeline, ModelBundle,
    ParakeetDeployment, ParakeetLanguage, ParakeetVersion, PendingTranscription, SchedulerConfig,
    TranscriptionError, TranscriptionResult,
};

use crate::results::PyTranscriptionResult;
use crate::{deployment, lock, samples, scriptrs_err, transcription_code};

/// Native pipeline; `scriptrs.Pipeline` wraps it
#[pyclass(name = "_Pipeline", module = "scriptrs._native", frozen)]
pub(crate) struct PyPipeline {
    // the Core ML backend is `Send` but not `Sync`, and the lock also runs one
    // call at a time, which Core ML needs for predictions on one model
    inner: Mutex<LongFormTranscriptionPipeline>,
    deployment: ParakeetDeployment,
    config: LongFormConfig,
}

impl PyPipeline {
    fn new(pipeline: LongFormTranscriptionPipeline, vad: bool) -> Self {
        let mut config = pipeline.config().clone();
        config.mode = match vad {
            true => LongFormMode::Vad,
            false => LongFormMode::Fast,
        };

        Self {
            deployment: pipeline.base_pipeline().deployment().clone(),
            inner: Mutex::new(pipeline),
            config,
        }
    }

    fn model(&self) -> &'static str {
        match self.deployment.version() {
            ParakeetVersion::V2 => "parakeet/v2",
            ParakeetVersion::V3 => "parakeet/v3",
        }
    }

    /// The default config with `language` selected, rejecting a language the
    /// model does not transcribe before any work starts
    fn config_for(&self, language: Option<&str>) -> PyResult<LongFormConfig> {
        let mut config = self.config.clone();
        let Some(code) = language else {
            return Ok(config);
        };

        let language = ParakeetLanguage::from_code(&code.trim().to_ascii_lowercase())
            .filter(|language| self.deployment.supports_language(*language))
            .ok_or_else(|| {
                let supported: Vec<_> = self
                    .deployment
                    .languages()
                    .iter()
                    .map(|language| language.code())
                    .collect();
                scriptrs_err(
                    "unsupported_language",
                    format!(
                        "model {} does not transcribe language `{code}`; it supports {}",
                        self.model(),
                        supported.join(", ")
                    ),
                )
            })?;
        config.transcription.language = LanguageSelection::Language(language);
        Ok(config)
    }

    fn run(
        &self,
        py: Python<'_>,
        audio: &[f32],
        config: &LongFormConfig,
    ) -> PyResult<PyTranscriptionResult> {
        py.detach(|| lock(&self.inner).run_with_config(audio, config))
            .map(Into::into)
            .map_err(crate::transcription_err)
    }
}

#[pymethods]
impl PyPipeline {
    #[getter(model)]
    fn model_name(&self) -> &'static str {
        self.model()
    }

    #[getter]
    fn vad(&self) -> bool {
        self.config.mode == LongFormMode::Vad
    }

    #[getter]
    fn languages(&self) -> Vec<&'static str> {
        self.deployment
            .languages()
            .iter()
            .map(|language| language.code())
            .collect()
    }

    #[getter]
    fn max_chunk_seconds(&self) -> f64 {
        self.config.single_window_samples() as f64 / self.deployment.frontend().sample_rate() as f64
    }

    #[pyo3(signature = (samples, language=None))]
    fn transcribe(
        &self,
        py: Python<'_>,
        samples: &Bound<'_, PyAny>,
        language: Option<&str>,
    ) -> PyResult<PyTranscriptionResult> {
        let config = self.config_for(language)?;
        let audio = samples::extract(samples)?;
        self.run(py, &audio, &config)
    }

    #[pyo3(signature = (path, language=None))]
    fn transcribe_file(
        &self,
        py: Python<'_>,
        path: PathBuf,
        language: Option<&str>,
    ) -> PyResult<PyTranscriptionResult> {
        let config = self.config_for(language)?;
        let audio = py
            .detach(|| crate::audio::decode_file_to_mono_16khz(&path))
            .map_err(|message| scriptrs_err("audio_decode", message))?;
        self.run(py, &audio, &config)
    }

    #[pyo3(signature = (inputs, language=None, max_batch=SchedulerConfig::DEFAULT_MAX_BATCH))]
    fn transcribe_many(
        &self,
        py: Python<'_>,
        inputs: Vec<Bound<'_, PyAny>>,
        language: Option<&str>,
        max_batch: usize,
    ) -> PyResult<Vec<PyTranscriptionResult>> {
        // an encoder batch can never fill past the decoder's rows
        let max_rows = SchedulerConfig::DEFAULT_DECODER_ROWS;
        if !(1..=max_rows).contains(&max_batch) {
            return Err(scriptrs_err(
                "configuration",
                format!("max_batch must be between 1 and {max_rows}, got {max_batch}"),
            ));
        }

        let config = self.config_for(language)?;
        let audios = inputs
            .iter()
            .map(samples::extract)
            .collect::<PyResult<Vec<_>>>()?;
        let scheduler = SchedulerConfig {
            max_batch,
            min_batch: SchedulerConfig::DEFAULT_MIN_BATCH.min(max_batch),
            ..SchedulerConfig::default()
        };

        let results = py.detach(|| {
            // held for the whole call, so the scheduler's copy of the model is the
            // only one running, as Core ML needs
            let pipeline = lock(&self.inner);
            run_many(&pipeline, audios, config, scheduler)
        })?;
        results
            .into_iter()
            .enumerate()
            .map(|(index, result)| {
                result.map(Into::into).map_err(|error| {
                    scriptrs_err(
                        transcription_code(&error),
                        format!("input {index}: {error}"),
                    )
                })
            })
            .collect()
    }
}

/// Transcribe every input through a scheduler on a clone of the pipeline, which
/// batches the windows of every input, short or long, together
///
/// The cutoff is the one long-form uses for a single window, so each input runs
/// as the same windows `transcribe` would run it as
///
/// Results keep input order. A failure stops nothing: each input gets its own
/// result, and the caller decides what to raise
fn run_many(
    pipeline: &LongFormTranscriptionPipeline,
    audios: Vec<Vec<f32>>,
    config: LongFormConfig,
    scheduler: SchedulerConfig,
) -> PyResult<Vec<Result<TranscriptionResult, TranscriptionError>>> {
    let scheduler = pipeline
        .clone()
        .into_scheduler_with_config(config, scheduler)
        .map_err(crate::transcription_err)?;
    // every input is submitted before the first wait, so the scheduler sees them
    // all and fills its batches
    let pending: Vec<Result<PendingTranscription, TranscriptionError>> = audios
        .into_iter()
        .map(|audio| scheduler.submit(audio))
        .collect();
    Ok(pending
        .into_iter()
        .map(|pending| pending.and_then(PendingTranscription::wait))
        .collect())
}

/// Download a model for this platform's backend, then load it
#[pyfunction]
#[pyo3(signature = (model, vad, precision, cache_dir=None))]
pub(crate) fn _from_pretrained(
    py: Python<'_>,
    model: &str,
    vad: bool,
    precision: &str,
    cache_dir: Option<PathBuf>,
) -> PyResult<PyPipeline> {
    let deployment = deployment(model)?;
    let precision = parse_precision(precision)?;
    let bundle = py.detach(|| backend::download(deployment, vad, cache_dir))?;
    load(py, backend::with_precision(bundle, precision), vad)
}

/// Load a model from a local directory in this platform's backend layout
#[pyfunction]
pub(crate) fn _from_dir(
    py: Python<'_>,
    path: PathBuf,
    model: &str,
    vad: bool,
    precision: &str,
) -> PyResult<PyPipeline> {
    let deployment = deployment(model)?;
    let precision = parse_precision(precision)?;
    let bundle = backend::from_dir(path, deployment);
    load(py, backend::with_precision(bundle, precision), vad)
}

fn load(py: Python<'_>, bundle: ModelBundle, vad: bool) -> PyResult<PyPipeline> {
    py.detach(|| {
        // a VAD pipeline would otherwise only report its missing model on the first run
        if vad {
            bundle.validate_long_form()?;
        }
        LongFormTranscriptionPipeline::from_bundle(bundle)
    })
    .map(|pipeline| PyPipeline::new(pipeline, vad))
    .map_err(crate::transcription_err)
}

#[cfg(any(target_os = "macos", all(target_os = "linux", target_arch = "x86_64")))]
fn download_err(error: impl std::fmt::Display) -> PyErr {
    scriptrs_err("model_download", format!("model download failed: {error}"))
}

/// Core ML models on macOS
#[cfg(target_os = "macos")]
mod backend {
    use std::path::PathBuf;

    use pyo3::PyResult;
    use scriptrs::{ModelBundle, ModelManager, ParakeetDeployment};

    use super::{PrecisionName, download_err};

    /// Core ML runs its own precision, so the setting is only validated
    pub(super) fn with_precision(bundle: ModelBundle, _precision: PrecisionName) -> ModelBundle {
        bundle
    }

    pub(super) fn download(
        deployment: ParakeetDeployment,
        vad: bool,
        cache_dir: Option<PathBuf>,
    ) -> PyResult<ModelBundle> {
        match cache_dir {
            Some(dir) => ModelManager::with_cache_dir(dir)
                .and_then(|manager| manager.ensure_with_deployment(deployment, vad)),
            None if vad => ModelBundle::from_pretrained_long_form_with_deployment(deployment),
            None => ModelBundle::from_pretrained_with_deployment(deployment),
        }
        .map_err(download_err)
    }

    pub(super) fn from_dir(path: PathBuf, deployment: ParakeetDeployment) -> ModelBundle {
        ModelBundle::from_dir_with_deployment(path, deployment)
    }
}

/// Native CUDA weights on Linux x86_64
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod backend {
    use std::path::PathBuf;

    use pyo3::PyResult;
    use scriptrs::{ModelBundle, ModelManager, ParakeetDeployment, Precision};

    use super::{PrecisionName, download_err};

    pub(super) fn with_precision(bundle: ModelBundle, precision: PrecisionName) -> ModelBundle {
        bundle.with_precision(match precision {
            PrecisionName::Fp16 => Precision::Fp16,
            PrecisionName::Tf32 => Precision::Tf32,
            PrecisionName::Fp32 => Precision::Fp32,
        })
    }

    pub(super) fn download(
        deployment: ParakeetDeployment,
        vad: bool,
        cache_dir: Option<PathBuf>,
    ) -> PyResult<ModelBundle> {
        match cache_dir {
            Some(dir) => ModelManager::with_cache_dir(dir)
                .and_then(|manager| manager.ensure_native(deployment, vad)),
            None if vad => ModelBundle::from_pretrained_native_long_form(deployment),
            None => ModelBundle::from_pretrained_native(deployment),
        }
        .map_err(download_err)
    }

    pub(super) fn from_dir(path: PathBuf, deployment: ParakeetDeployment) -> ModelBundle {
        ModelBundle::from_native_dir(path, deployment)
    }
}

/// No backend: the package imports, and loading a model reports the platform
#[cfg(not(any(target_os = "macos", all(target_os = "linux", target_arch = "x86_64"))))]
mod backend {
    use std::path::PathBuf;

    use pyo3::PyResult;
    use scriptrs::{ModelBundle, ParakeetDeployment};

    use super::PrecisionName;
    use crate::scriptrs_err;

    pub(super) fn with_precision(bundle: ModelBundle, _precision: PrecisionName) -> ModelBundle {
        bundle
    }

    pub(super) fn download(
        _deployment: ParakeetDeployment,
        _vad: bool,
        _cache_dir: Option<PathBuf>,
    ) -> PyResult<ModelBundle> {
        Err(scriptrs_err(
            "unsupported_platform",
            "scriptrs runs on macOS (Core ML) and Linux x86_64 (CUDA) only",
        ))
    }

    pub(super) fn from_dir(path: PathBuf, deployment: ParakeetDeployment) -> ModelBundle {
        ModelBundle::from_dir_with_deployment(path, deployment)
    }
}

enum PrecisionName {
    Fp16,
    Tf32,
    Fp32,
}

fn parse_precision(name: &str) -> PyResult<PrecisionName> {
    match name.to_ascii_lowercase().as_str() {
        "fp16" => Ok(PrecisionName::Fp16),
        "tf32" => Ok(PrecisionName::Tf32),
        "fp32" => Ok(PrecisionName::Fp32),
        _ => Err(scriptrs_err(
            "configuration",
            format!("unknown precision `{name}`; expected `fp16`, `tf32` or `fp32`"),
        )),
    }
}

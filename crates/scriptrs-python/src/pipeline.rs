//! The native pipeline behind `scriptrs.Pipeline`

use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::prelude::*;
use scriptrs::{
    LanguageSelection, LongFormConfig, LongFormMode, LongFormTranscriptionPipeline, ModelBundle,
    ParakeetDeployment, ParakeetLanguage, ParakeetVersion, QueueConfig, QueueError,
    QueuedTranscriptionRequest, QueuedTranscriptionResult, TranscriptionError,
    TranscriptionPipeline, TranscriptionResult,
};

use crate::results::PyTranscriptionResult;
use crate::{deployment, lock, queue_err, samples, scriptrs_err, transcription_code};

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
        self.deployment.hard_sample_max() as f64 / self.deployment.frontend().sample_rate() as f64
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

    #[pyo3(signature = (inputs, language=None, max_batch=QueueConfig::DEFAULT_MAX_BATCH))]
    fn transcribe_many(
        &self,
        py: Python<'_>,
        inputs: Vec<Bound<'_, PyAny>>,
        language: Option<&str>,
        max_batch: usize,
    ) -> PyResult<Vec<PyTranscriptionResult>> {
        // a batch can never fill past the queue's capacity, and a larger value
        // would only size the queue's buffers
        if !(1..=QueueConfig::DEFAULT_CAPACITY).contains(&max_batch) {
            return Err(scriptrs_err(
                "configuration",
                format!(
                    "max_batch must be between 1 and {}, got {max_batch}",
                    QueueConfig::DEFAULT_CAPACITY
                ),
            ));
        }

        let config = self.config_for(language)?;
        let audios = inputs
            .iter()
            .map(samples::extract)
            .collect::<PyResult<Vec<_>>>()?;
        let queue = QueueConfig {
            capacity: QueueConfig::DEFAULT_CAPACITY,
            max_batch,
        };

        let results = py.detach(|| {
            let pipeline = lock(&self.inner);
            run_many(&pipeline, audios, &config, queue)
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

/// Transcribe every input, short ones through a batching queue on a clone of the
/// pipeline's model, and inputs longer than one window through long-form
///
/// Results keep input order. A failure stops nothing: each input gets its own
/// result, and the caller decides what to raise
fn run_many(
    pipeline: &LongFormTranscriptionPipeline,
    audios: Vec<Vec<f32>>,
    config: &LongFormConfig,
    queue: QueueConfig,
) -> PyResult<Vec<Result<TranscriptionResult, TranscriptionError>>> {
    let max_samples = pipeline.base_pipeline().deployment().hard_sample_max();
    let mut results: Vec<Option<Result<TranscriptionResult, TranscriptionError>>> =
        audios.iter().map(|_| None).collect();
    let mut short = Vec::new();
    let mut long = Vec::new();
    for (index, audio) in audios.into_iter().enumerate() {
        // the queue only rejects empty and oversized input, so non-finite samples
        // are caught here, as the single-input path catches them
        if let Some(sample) = audio.iter().position(|sample| !sample.is_finite()) {
            results[index] = Some(Err(TranscriptionError::InvalidAudioSample {
                index: sample,
            }));
        } else if audio.len() <= max_samples {
            short.push((index, audio));
        } else {
            long.push((index, audio));
        }
    }

    if !short.is_empty() {
        let base: TranscriptionPipeline = pipeline.base_pipeline().clone();
        for queued in run_queue(base, short, config, queue).map_err(queue_err)? {
            let index: usize = queued.id.parse().expect("queue ids are input indexes");
            results[index] = Some(queued.result);
        }
    }

    for (index, audio) in long {
        results[index] = Some(pipeline.run_with_config(&audio, config));
    }

    Ok(results
        .into_iter()
        .map(|result| result.expect("every input gets a result"))
        .collect())
}

fn run_queue(
    pipeline: TranscriptionPipeline,
    inputs: Vec<(usize, Vec<f32>)>,
    config: &LongFormConfig,
    queue: QueueConfig,
) -> Result<Vec<QueuedTranscriptionResult>, QueueError> {
    let (sender, mut receiver) = pipeline.into_queued_with_config(config.transcription, queue)?;
    let mut finished = Vec::with_capacity(inputs.len());

    for (index, audio) in inputs {
        let mut request = QueuedTranscriptionRequest::new(index.to_string(), audio);
        loop {
            match sender.try_push(request) {
                Ok(_) => break,
                // the queue is at capacity; taking a result makes room
                Err(QueueError::Full(rejected)) => {
                    request = rejected;
                    finished.push(receiver.recv()?);
                }
                Err(error) => return Err(error),
            }
        }
    }

    // dropping the sender closes the queue, so the receiver ends after the last result
    drop(sender);
    for queued in receiver {
        finished.push(queued?);
    }
    Ok(finished)
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

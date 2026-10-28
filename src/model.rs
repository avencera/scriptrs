#[cfg(target_os = "macos")]
use crate::backend::HostLoop;
use crate::backend::{ChunkDecode, DecodePolicy, EncoderChunk};
use crate::config::TranscriptionConfig;
use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::language::LanguageSelection;
use crate::models::{ModelBundle, RuntimeBundle};
use crate::vocab::Vocabulary;

/// Parakeet runtime plus the deployment and vocabulary its token rule needs
#[derive(Debug, Clone)]
pub(crate) struct ParakeetModel {
    inner: ParakeetModelInner,
    deployment: ParakeetDeployment,
    vocabulary: Vocabulary,
}

impl ParakeetModel {
    pub(crate) fn from_bundle(
        bundle: &ModelBundle,
        vocab: &Vocabulary,
        config: &TranscriptionConfig,
    ) -> Result<Self, TranscriptionError> {
        let deployment = bundle.deployment().clone();
        config.validate_for(&deployment)?;
        Ok(Self {
            inner: ParakeetModelInner::from_bundle(bundle, config)?,
            vocabulary: vocab.clone(),
            deployment,
        })
    }

    /// Encode and greedily decode each chunk, in order
    pub(crate) fn transcribe_batch(
        &self,
        chunks: &[EncoderChunk<'_>],
        language: LanguageSelection,
    ) -> Result<Vec<ChunkDecode>, TranscriptionError> {
        let policy = DecodePolicy::new(&self.deployment, &self.vocabulary, language)?;
        let decoded = self.inner.transcribe_chunks(chunks, &policy)?;
        if decoded.len() != chunks.len() {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "backend decoded {} chunks for a batch of {}",
                decoded.len(),
                chunks.len()
            )));
        }
        Ok(decoded)
    }

    /// Whether one call runs a batch of chunks together on the device, rather than
    /// one chunk after another on the host
    #[cfg(feature = "long-form")]
    pub(crate) fn batches_on_device(&self) -> bool {
        self.inner.batches_on_device()
    }
}

/// Runtime behind [`ParakeetModel`]
///
/// Every variant takes a batch of encoder chunks and returns greedy TDT output
/// under one [`DecodePolicy`]. Step runtimes reach that contract through the
/// shared [`HostLoop`].
#[derive(Debug, Clone)]
enum ParakeetModelInner {
    #[cfg(target_os = "macos")]
    SplitCoreMl {
        model: Box<crate::coreml::ParakeetSplitCoreMlModel>,
        host: HostLoop,
    },
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::model::CudaModel),
    #[cfg(not(target_os = "macos"))]
    Unsupported,
}

impl ParakeetModelInner {
    fn from_bundle(
        bundle: &ModelBundle,
        config: &TranscriptionConfig,
    ) -> Result<Self, TranscriptionError> {
        match bundle.runtime() {
            RuntimeBundle::CoreMl(layout) => {
                #[cfg(target_os = "macos")]
                {
                    let dims = bundle.deployment().decoder_dimensions();
                    Ok(Self::SplitCoreMl {
                        model: Box::new(crate::coreml::ParakeetSplitCoreMlModel::new(
                            layout.encoder_dir(),
                            layout.decoder_dir(),
                            layout.joint_decision_dir(),
                            bundle.deployment().model_io(),
                            dims,
                        )?),
                        // the Core ML encoder takes a fixed-width input sized
                        // for the longest single chunk
                        host: HostLoop::new(dims, config.feature_size, config.max_feature_frames()),
                    })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = (layout, config);
                    Err(TranscriptionError::UnsupportedPlatform)
                }
            }
            #[cfg(feature = "cuda")]
            RuntimeBundle::Native(layout) => Ok(Self::Cuda(crate::cuda::model::CudaModel::load(
                layout.weights_path(),
                bundle.deployment(),
            )?)),
        }
    }

    #[cfg(feature = "long-form")]
    fn batches_on_device(&self) -> bool {
        match self {
            #[cfg(target_os = "macos")]
            Self::SplitCoreMl { .. } => false,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => true,
            #[cfg(not(target_os = "macos"))]
            Self::Unsupported => false,
        }
    }

    fn transcribe_chunks(
        &self,
        chunks: &[EncoderChunk<'_>],
        policy: &DecodePolicy<'_>,
    ) -> Result<Vec<ChunkDecode>, TranscriptionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::SplitCoreMl { model, host } => {
                host.transcribe_chunks(model.as_ref(), chunks, policy)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(model) => model.transcribe_chunks(chunks, policy),
            #[cfg(not(target_os = "macos"))]
            Self::Unsupported => {
                let _ = (chunks, policy);
                Err(TranscriptionError::UnsupportedPlatform)
            }
        }
    }
}

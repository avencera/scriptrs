#[cfg(target_os = "macos")]
use crate::backend::HostLoop;
use crate::backend::{ChunkDecode, DecodePolicy, EncoderChunk};
use crate::decode::RawTranscription;
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
    ) -> Result<Self, TranscriptionError> {
        let deployment = bundle.deployment().clone();
        Ok(Self {
            inner: ParakeetModelInner::from_bundle(bundle)?,
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

    /// A batcher of `shape` that decodes every window it starts in `language`
    pub(crate) fn batcher(
        &self,
        language: LanguageSelection,
        shape: BatcherShape,
    ) -> Result<ModelBatcher, TranscriptionError> {
        match &self.inner {
            #[cfg(feature = "cuda")]
            ParakeetModelInner::Cuda(model) => {
                let policy = DecodePolicy::new(&self.deployment, &self.vocabulary, language)?;
                Ok(ModelBatcher::Cuda {
                    batcher: Box::new(crate::cuda::batcher::CudaBatcher::new(
                        model,
                        shape.rows,
                        shape.max_mel_frames,
                        shape.max_batch,
                        policy.language_mask().as_ref(),
                    )?),
                    shape,
                })
            }
            #[cfg_attr(feature = "cuda", allow(unreachable_patterns))]
            _ => Ok(ModelBatcher::Serial {
                model: Box::new(self.clone()),
                language,
                shape,
                finished: Vec::new(),
                next_row: 0,
            }),
        }
    }

    /// Whether one call runs a batch of chunks together on the device, rather than
    /// one chunk after another on the host
    #[cfg(feature = "long-form")]
    pub(crate) fn batches_on_device(&self) -> bool {
        self.inner.batches_on_device()
    }
}

/// Rows that finished decoding, each with its tokens or why it failed
pub(crate) type FinishedRows = Vec<(usize, Result<RawTranscription, TranscriptionError>)>;

/// Sizes of a [`ModelBatcher`]
#[derive(Debug, Clone, Copy)]
pub(crate) struct BatcherShape {
    /// Windows the decoder holds in flight
    pub rows: usize,
    /// Mel frames one full encoder batch covers, which sets each bucket's largest
    /// batch
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "a host loop pads nothing")
    )]
    pub frames_per_batch: usize,
    /// Floor of each bucket's largest batch
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "a host loop pads nothing")
    )]
    pub min_batch: usize,
    /// Most windows in any encoder batch
    pub max_batch: usize,
    /// Mel frames of the longest window
    pub max_mel_frames: usize,
}

/// A length bucket: the mel frames its windows are padded to, and the most
/// windows one encoder batch of it holds
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Bucket {
    pub mel_frames: usize,
    pub batch: usize,
}

/// Windows from many requests, started in rows as rows free up and returned as
/// each finishes
///
/// A device backend encodes a bucket's windows together and decodes them in a
/// lane whose rows refill as windows finish. A host-loop backend runs each
/// window alone as it is admitted
pub(crate) enum ModelBatcher {
    #[cfg(feature = "cuda")]
    Cuda {
        batcher: Box<crate::cuda::batcher::CudaBatcher>,
        shape: BatcherShape,
    },
    #[cfg_attr(
        feature = "cuda",
        allow(dead_code, reason = "Core ML bundles on macOS")
    )]
    Serial {
        model: Box<ParakeetModel>,
        language: LanguageSelection,
        shape: BatcherShape,
        /// Outputs of admitted windows, by the row id they were given
        finished: FinishedRows,
        next_row: usize,
    },
}

impl ModelBatcher {
    /// The bucket a window of a given number of mel frames runs in, as a function
    /// other threads can call
    pub(crate) fn bucket_fn(&self) -> Box<dyn Fn(usize) -> Bucket + Send + Sync> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { shape, .. } => {
                let shape = *shape;
                Box::new(move |mel_frames| {
                    let bucket = crate::cuda::batcher::bucket_frames(mel_frames);
                    Bucket {
                        mel_frames: bucket,
                        batch: (shape.frames_per_batch / bucket)
                            .clamp(shape.min_batch, shape.max_batch),
                    }
                })
            }
            Self::Serial { shape, .. } => {
                let bucket = Bucket {
                    mel_frames: shape.max_mel_frames,
                    batch: shape.max_batch,
                };
                Box::new(move |_| bucket)
            }
        }
    }

    /// Windows [`Self::admit`] can start now
    pub(crate) fn free_rows(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { batcher, .. } => batcher.free_rows(),
            Self::Serial {
                shape, finished, ..
            } => shape.rows.saturating_sub(finished.len()),
        }
    }

    /// Windows started and not yet returned by [`Self::advance`]
    pub(crate) fn active_rows(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { batcher, .. } => batcher.active_rows(),
            Self::Serial { finished, .. } => finished.len(),
        }
    }

    /// Starts `chunks`, all in `bucket`, as one batch, returning a row id for each
    pub(crate) fn admit(
        &mut self,
        bucket: Bucket,
        chunks: &[EncoderChunk<'_>],
    ) -> Result<Vec<usize>, TranscriptionError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { batcher, .. } => batcher.admit(bucket.mel_frames, bucket.batch, chunks),
            Self::Serial {
                model,
                language,
                finished,
                next_row,
                ..
            } => {
                let _ = bucket;
                Ok(chunks
                    .iter()
                    .map(|chunk| {
                        let row = *next_row;
                        *next_row += 1;
                        let raw = model
                            .transcribe_batch(std::slice::from_ref(chunk), *language)
                            .map(|mut decoded| decoded.remove(0).raw);
                        finished.push((row, raw));
                        row
                    })
                    .collect())
            }
        }
    }

    /// Runs the model until some started window may have finished, and returns
    /// each that has with its tokens or why it failed
    ///
    /// An error here leaves the rows in flight unknown: the caller fails their
    /// windows and calls [`Self::reset`]
    pub(crate) fn advance(&mut self) -> Result<FinishedRows, TranscriptionError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { batcher, .. } => batcher.advance(),
            Self::Serial { finished, .. } => Ok(std::mem::take(finished)),
        }
    }

    /// Forgets every window in flight
    pub(crate) fn reset(&mut self) {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda { batcher, .. } => batcher.reset(),
            Self::Serial { finished, .. } => finished.clear(),
        }
    }
}

/// Runtime behind [`ParakeetModel`]
///
/// Every variant takes a batch of encoder chunks and returns greedy TDT output
/// under one [`DecodePolicy`]. Step runtimes reach that contract through the
/// shared [`HostLoop`]
#[derive(Debug, Clone)]
enum ParakeetModelInner {
    #[cfg(target_os = "macos")]
    SplitCoreMl {
        model: Box<crate::coreml::ParakeetSplitCoreMlModel>,
        host: HostLoop,
    },
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::model::CudaModel),
    // keeps the matches below compiling where no runtime exists
    #[cfg(not(any(target_os = "macos", feature = "cuda")))]
    #[expect(dead_code, reason = "from_bundle fails with UnsupportedPlatform first")]
    Unsupported,
}

impl ParakeetModelInner {
    fn from_bundle(bundle: &ModelBundle) -> Result<Self, TranscriptionError> {
        match bundle.runtime() {
            RuntimeBundle::CoreMl(layout) => {
                #[cfg(target_os = "macos")]
                {
                    let deployment = bundle.deployment();
                    let dims = deployment.decoder_dimensions();
                    let features = deployment.frontend().feature_size();
                    // the Core ML encoder takes a fixed-width input sized for the
                    // longest single chunk, which the host loop pads every chunk to
                    let target_frames =
                        deployment.hard_sample_max() / deployment.frontend().hop_length() + 1;
                    Ok(Self::SplitCoreMl {
                        model: Box::new(crate::coreml::ParakeetSplitCoreMlModel::new(
                            layout.encoder_dir(),
                            layout.decoder_dir(),
                            layout.joint_decision_dir(),
                            deployment.model_io(),
                            dims,
                            features,
                            target_frames,
                        )?),
                        host: HostLoop::new(dims, features, target_frames),
                    })
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = layout;
                    Err(TranscriptionError::UnsupportedPlatform)
                }
            }
            #[cfg(feature = "cuda")]
            RuntimeBundle::Native(layout) => Ok(Self::Cuda(crate::cuda::model::CudaModel::load(
                &layout.weights_path(),
                bundle.deployment(),
                layout.precision(),
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
            #[cfg(not(any(target_os = "macos", feature = "cuda")))]
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
            #[cfg(not(any(target_os = "macos", feature = "cuda")))]
            Self::Unsupported => {
                let _ = (chunks, policy);
                Err(TranscriptionError::UnsupportedPlatform)
            }
        }
    }
}

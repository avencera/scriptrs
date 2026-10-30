mod array;
mod runtime;

use std::cell::RefCell;
use std::env;
use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, PoisonError};

use block2::RcBlock;
use ndarray::{Array1, Array2, Array3};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{MLComputeUnits, MLFeatureProvider, MLModel};
use objc2_foundation::{NSArray, NSMutableDictionary, NSNumber, NSString};

use crate::backend::StepRuntime;
#[cfg(feature = "vad")]
use crate::constants::{VAD_CONTEXT_SAMPLES, VAD_STATE_SIZE, VAD_WINDOW_SAMPLES};
use crate::coreml::array::{contiguous_strides, extract_output, ns_number_array};
use crate::deployment::{DecoderDimensions, ModelIo};
use crate::error::TranscriptionError;
use crate::model_io::{DecoderStepOutputs, JointOutputs};

#[derive(Debug, Clone)]
pub(crate) struct ParakeetSplitCoreMlModel {
    encoder: EncoderCoreMlModel,
    decoder: DecoderCoreMlModel,
    joint_decision: JointDecisionCoreMlModel,
}

#[cfg(feature = "vad")]
#[derive(Debug, Clone)]
pub(crate) struct SileroVadCoreMlModel {
    model: CoreMlModel,
    audio: CachedInputShape,
    hidden_state: CachedInputShape,
    cell_state: CachedInputShape,
    probability_output: CachedOutputKey,
    hidden_output: CachedOutputKey,
    cell_output: CachedOutputKey,
}

impl ParakeetSplitCoreMlModel {
    /// Loads the three split models, with the encoder input fixed at
    /// `[1, feature_size, target_frames]`
    pub(crate) fn new(
        encoder_path: &Path,
        decoder_path: &Path,
        joint_decision_path: &Path,
        io: ModelIo,
        dims: DecoderDimensions,
        feature_size: usize,
        target_frames: usize,
    ) -> Result<Self, TranscriptionError> {
        Ok(Self {
            encoder: EncoderCoreMlModel::new(encoder_path, io, dims, feature_size, target_frames)?,
            decoder: DecoderCoreMlModel::new(decoder_path, io, dims)?,
            joint_decision: JointDecisionCoreMlModel::new(joint_decision_path, io, dims)?,
        })
    }

    pub(crate) fn run_encoder(
        &self,
        mel: &[f32],
        feature_size: usize,
        target_frames: usize,
        lengths: &[i32],
    ) -> Result<(Array3<f32>, usize), TranscriptionError> {
        self.encoder.run(mel, feature_size, target_frames, lengths)
    }

    pub(crate) fn run_decoder(
        &self,
        targets: &Array2<i32>,
        target_length: &Array1<i32>,
        hidden_state: &Array3<f32>,
        cell_state: &Array3<f32>,
    ) -> Result<SplitDecoderCoreMlOutput, TranscriptionError> {
        self.decoder.run(
            targets.as_slice().ok_or_else(|| {
                TranscriptionError::InvalidModelOutput(
                    "decoder targets were not contiguous".to_owned(),
                )
            })?,
            target_length.as_slice().ok_or_else(|| {
                TranscriptionError::InvalidModelOutput(
                    "decoder target lengths were not contiguous".to_owned(),
                )
            })?,
            hidden_state.as_slice().ok_or_else(|| {
                TranscriptionError::InvalidModelOutput(
                    "decoder hidden state was not contiguous".to_owned(),
                )
            })?,
            cell_state.as_slice().ok_or_else(|| {
                TranscriptionError::InvalidModelOutput(
                    "decoder cell state was not contiguous".to_owned(),
                )
            })?,
        )
    }

    pub(crate) fn run_joint(
        &self,
        encoder_step: &Array3<f32>,
        decoder_step: &Array3<f32>,
    ) -> Result<JointOutputs, TranscriptionError> {
        self.joint_decision.run(
            encoder_step.as_slice().ok_or_else(|| {
                TranscriptionError::InvalidModelOutput(
                    "joint encoder step was not contiguous".to_owned(),
                )
            })?,
            decoder_step.as_slice().ok_or_else(|| {
                TranscriptionError::InvalidModelOutput(
                    "joint decoder step was not contiguous".to_owned(),
                )
            })?,
        )
    }
}

impl StepRuntime for ParakeetSplitCoreMlModel {
    fn run_encoder(
        &self,
        input: &[f32],
        feature_size: usize,
        target_frames: usize,
        feature_frames: usize,
    ) -> Result<(Array3<f32>, usize), TranscriptionError> {
        ParakeetSplitCoreMlModel::run_encoder(
            self,
            input,
            feature_size,
            target_frames,
            &[feature_frames as i32],
        )
    }

    fn run_decoder(
        &self,
        targets: &Array2<i32>,
        target_length: &Array1<i32>,
        hidden_state: &Array3<f32>,
        cell_state: &Array3<f32>,
    ) -> Result<DecoderStepOutputs, TranscriptionError> {
        let output = ParakeetSplitCoreMlModel::run_decoder(
            self,
            targets,
            target_length,
            hidden_state,
            cell_state,
        )?;
        Ok(DecoderStepOutputs {
            decoder_step: output.decoder_step,
            hidden_state: output.hidden_state,
            cell_state: output.cell_state,
        })
    }

    fn run_joint(
        &self,
        encoder_step: &Array3<f32>,
        decoder_step: &Array3<f32>,
    ) -> Result<JointOutputs, TranscriptionError> {
        ParakeetSplitCoreMlModel::run_joint(self, encoder_step, decoder_step)
    }
}

#[cfg(feature = "vad")]
impl SileroVadCoreMlModel {
    pub(crate) fn new(path: &Path) -> Result<Self, TranscriptionError> {
        Ok(Self {
            model: CoreMlModel::new(path)?,
            audio: CachedInputShape::new(
                "audio",
                &[1, 1, VAD_WINDOW_SAMPLES + VAD_CONTEXT_SAMPLES],
            ),
            hidden_state: CachedInputShape::new("h", &[1, 1, VAD_STATE_SIZE]),
            cell_state: CachedInputShape::new("c", &[1, 1, VAD_STATE_SIZE]),
            probability_output: CachedOutputKey::new("probability"),
            hidden_output: CachedOutputKey::new("h_out"),
            cell_output: CachedOutputKey::new("c_out"),
        })
    }

    pub(crate) fn run(
        &self,
        audio: &[f32],
        hidden_state: &[f32],
        cell_state: &[f32],
    ) -> Result<SileroVadOutput, TranscriptionError> {
        self.model.predict(
            &[
                CachedCoreMlInput::F32 {
                    cached: &self.audio,
                    values: audio,
                },
                CachedCoreMlInput::F32 {
                    cached: &self.hidden_state,
                    values: hidden_state,
                },
                CachedCoreMlInput::F32 {
                    cached: &self.cell_state,
                    values: cell_state,
                },
            ],
            |output| {
                Ok(SileroVadOutput {
                    probability: vad_probability(scalar_f32(output, &self.probability_output)?)?,
                    hidden_state: vad_state_vector(
                        vector_f32(output, &self.hidden_output)?,
                        "h_out",
                    )?,
                    cell_state: vad_state_vector(vector_f32(output, &self.cell_output)?, "c_out")?,
                })
            },
        )
    }
}

#[derive(Debug, Clone)]
struct EncoderCoreMlModel {
    model: CoreMlModel,
    feature_size: usize,
    target_frames: usize,
    encoder_hidden: usize,
    mel: CachedInputShape,
    length: CachedInputShape,
    encoder_output: CachedOutputKey,
    length_output: CachedOutputKey,
}

impl EncoderCoreMlModel {
    fn new(
        path: &Path,
        io: ModelIo,
        dims: DecoderDimensions,
        feature_size: usize,
        target_frames: usize,
    ) -> Result<Self, TranscriptionError> {
        let encoder_io = io.encoder();
        Ok(Self {
            model: CoreMlModel::new(path)?,
            feature_size,
            target_frames,
            encoder_hidden: dims.encoder_hidden(),
            mel: CachedInputShape::new(encoder_io.input_mel, &[1, feature_size, target_frames]),
            length: CachedInputShape::new(encoder_io.input_length, &[1]),
            encoder_output: CachedOutputKey::new(encoder_io.output_encoder),
            length_output: CachedOutputKey::new(encoder_io.output_length),
        })
    }

    /// Encodes `[1, feature_size, target_frames]` mel input into
    /// `[1, hidden, time]` and its valid time steps
    fn run(
        &self,
        mel: &[f32],
        feature_size: usize,
        target_frames: usize,
        lengths: &[i32],
    ) -> Result<(Array3<f32>, usize), TranscriptionError> {
        // the input shape is cached at load time, so a caller padding to another
        // width would hand Core ML a buffer that does not match it
        if (feature_size, target_frames) != (self.feature_size, self.target_frames) {
            return Err(TranscriptionError::CoreMl(format!(
                "encoder input [1, {feature_size}, {target_frames}] did not match the loaded shape [1, {}, {}]",
                self.feature_size, self.target_frames
            )));
        }

        let ((data, shape), output_lengths) = self.model.predict(
            &[
                CachedCoreMlInput::F32 {
                    cached: &self.mel,
                    values: mel,
                },
                CachedCoreMlInput::I32 {
                    cached: &self.length,
                    values: lengths,
                },
            ],
            |output| {
                let encoder = runtime::output_multi_array(
                    output,
                    &self.encoder_output.key,
                    self.encoder_output.name,
                )?;
                Ok((
                    extract_output(&encoder)?,
                    vector_f32(output, &self.length_output)?,
                ))
            },
        )?;
        if shape.len() != 3 {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "encoder output shape was not 3D: {shape:?}"
            )));
        }

        let encoder = match shape.as_slice() {
            [1, hidden, _] if *hidden == self.encoder_hidden => {
                array3_from_parts(data, &shape, "encoder output")?
            }
            [1, time, hidden] if *hidden == self.encoder_hidden => {
                Array3::from_shape_vec((1, *time, *hidden), data)
                    .map_err(|error| {
                        TranscriptionError::InvalidModelOutput(format!(
                            "failed to shape encoder output: {error}"
                        ))
                    })?
                    .permuted_axes([0, 2, 1])
            }
            _ => {
                return Err(TranscriptionError::InvalidModelOutput(format!(
                    "unexpected encoder output shape: {shape:?}"
                )));
            }
        };
        let time_steps = match output_lengths.first().copied() {
            Some(value) => {
                let value = f32_to_usize(value, self.length_output.name)?;
                if value > 0 { value } else { encoder.shape()[2] }
            }
            None => encoder.shape()[2],
        };
        Ok((encoder, time_steps))
    }
}

#[derive(Debug, Clone)]
struct DecoderCoreMlModel {
    model: CoreMlModel,
    targets: CachedInputShape,
    target_length: CachedInputShape,
    hidden_state: CachedInputShape,
    cell_state: CachedInputShape,
    decoder_output: CachedOutputKey,
    hidden_output: CachedOutputKey,
    cell_output: CachedOutputKey,
}

impl DecoderCoreMlModel {
    fn new(path: &Path, io: ModelIo, dims: DecoderDimensions) -> Result<Self, TranscriptionError> {
        let decoder_io = io.decoder();
        Ok(Self {
            model: CoreMlModel::new(path)?,
            targets: CachedInputShape::new(decoder_io.targets, &[1, 1]),
            target_length: CachedInputShape::new(decoder_io.target_length, &[1]),
            hidden_state: CachedInputShape::new(
                decoder_io.hidden_in,
                &[dims.decoder_layers(), 1, dims.decoder_hidden()],
            ),
            cell_state: CachedInputShape::new(
                decoder_io.cell_in,
                &[dims.decoder_layers(), 1, dims.decoder_hidden()],
            ),
            decoder_output: CachedOutputKey::new(decoder_io.decoder),
            hidden_output: CachedOutputKey::new(decoder_io.hidden_out),
            cell_output: CachedOutputKey::new(decoder_io.cell_out),
        })
    }

    fn run(
        &self,
        targets: &[i32],
        target_length: &[i32],
        hidden_state: &[f32],
        cell_state: &[f32],
    ) -> Result<SplitDecoderCoreMlOutput, TranscriptionError> {
        self.model.predict(
            &[
                CachedCoreMlInput::I32 {
                    cached: &self.targets,
                    values: targets,
                },
                CachedCoreMlInput::I32 {
                    cached: &self.target_length,
                    values: target_length,
                },
                CachedCoreMlInput::F32 {
                    cached: &self.hidden_state,
                    values: hidden_state,
                },
                CachedCoreMlInput::F32 {
                    cached: &self.cell_state,
                    values: cell_state,
                },
            ],
            |output| {
                Ok(SplitDecoderCoreMlOutput {
                    decoder_step: array3_from_output(output, &self.decoder_output, "decoder step")?,
                    hidden_state: array3_from_output(
                        output,
                        &self.hidden_output,
                        "decoder hidden state",
                    )?,
                    cell_state: array3_from_output(
                        output,
                        &self.cell_output,
                        "decoder cell state",
                    )?,
                })
            },
        )
    }
}

#[derive(Debug, Clone)]
struct JointDecisionCoreMlModel {
    model: CoreMlModel,
    encoder_step: CachedInputShape,
    decoder_step: CachedInputShape,
    token_id_output: CachedOutputKey,
    token_prob_output: CachedOutputKey,
    duration_output: CachedOutputKey,
    top_k_ids_output: Option<CachedOutputKey>,
    top_k_logits_output: Option<CachedOutputKey>,
}

impl JointDecisionCoreMlModel {
    fn new(path: &Path, io: ModelIo, dims: DecoderDimensions) -> Result<Self, TranscriptionError> {
        let joint_io = io.joint();
        Ok(Self {
            model: CoreMlModel::new(path)?,
            encoder_step: CachedInputShape::new(
                joint_io.encoder_step,
                &[1, dims.encoder_hidden(), 1],
            ),
            decoder_step: CachedInputShape::new(
                joint_io.decoder_step,
                &[1, dims.decoder_hidden(), 1],
            ),
            token_id_output: CachedOutputKey::new(joint_io.token_id),
            token_prob_output: CachedOutputKey::new(joint_io.token_prob),
            duration_output: CachedOutputKey::new(joint_io.duration),
            top_k_ids_output: joint_io.top_k_ids.map(CachedOutputKey::new),
            top_k_logits_output: joint_io.top_k_logits.map(CachedOutputKey::new),
        })
    }

    fn run(
        &self,
        encoder_step: &[f32],
        decoder_step: &[f32],
    ) -> Result<JointOutputs, TranscriptionError> {
        self.model.predict(
            &[
                CachedCoreMlInput::F32 {
                    cached: &self.encoder_step,
                    values: encoder_step,
                },
                CachedCoreMlInput::F32 {
                    cached: &self.decoder_step,
                    values: decoder_step,
                },
            ],
            |output| {
                let top_k_ids = self
                    .top_k_ids_output
                    .as_ref()
                    .map(|key| vector_usize(output, key))
                    .transpose()?;
                let top_k_logits = self
                    .top_k_logits_output
                    .as_ref()
                    .map(|key| vector_f32(output, key))
                    .transpose()?;
                Ok(JointOutputs {
                    token_id: scalar_usize(output, &self.token_id_output)?,
                    token_prob: scalar_f32(output, &self.token_prob_output)?,
                    duration_bin: scalar_usize(output, &self.duration_output)?,
                    top_k: JointOutputs::bind_top_k(top_k_ids, top_k_logits)?,
                })
            },
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SplitDecoderCoreMlOutput {
    pub decoder_step: Array3<f32>,
    pub hidden_state: Array3<f32>,
    pub cell_state: Array3<f32>,
}

#[cfg(feature = "vad")]
#[derive(Debug, Clone)]
pub(crate) struct SileroVadOutput {
    pub probability: f32,
    pub hidden_state: Vec<f32>,
    pub cell_state: Vec<f32>,
}

pub(crate) struct CoreMlModel {
    model: Retained<MLModel>,
    /// Held across every prediction, and shared by clones, which share `model`
    prediction: Arc<Mutex<()>>,
    noop_deallocator: RcBlock<dyn Fn(NonNull<c_void>)>,
    input_dict: RefCell<Retained<NSMutableDictionary<NSString, AnyObject>>>,
}

// SAFETY: each wrapper is used from one thread at a time: it is not `Sync`, and
// the queue moves its pipeline to the worker thread whole. Clones share the
// `MLModel`, and Apple requires synchronous predictions on one model to run one
// at a time, so every prediction holds the clones' shared `prediction` lock. The
// noop deallocator block captures nothing, and each clone has its own input
// dictionary, only touched inside a prediction, through the `RefCell`. speakrs
// moves its Core ML models to worker threads the same way
unsafe impl Send for CoreMlModel {}

impl std::fmt::Debug for CoreMlModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMlModel").finish_non_exhaustive()
    }
}

impl Clone for CoreMlModel {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            prediction: Arc::clone(&self.prediction),
            noop_deallocator: RcBlock::new(|_: NonNull<c_void>| {}),
            input_dict: RefCell::new(NSMutableDictionary::new()),
        }
    }
}

impl CoreMlModel {
    pub(crate) fn new(path: &Path) -> Result<Self, TranscriptionError> {
        Ok(Self {
            model: runtime::load_model(path, configured_compute_units()?)?,
            prediction: Arc::new(Mutex::new(())),
            noop_deallocator: RcBlock::new(|_: NonNull<c_void>| {}),
            input_dict: RefCell::new(NSMutableDictionary::new()),
        })
    }

    fn predict<T>(
        &self,
        inputs: &[CachedCoreMlInput<'_>],
        output_handler: impl FnOnce(
            &ProtocolObject<dyn MLFeatureProvider>,
        ) -> Result<T, TranscriptionError>,
    ) -> Result<T, TranscriptionError> {
        // the lock guards no data, so a panic in another prediction leaves nothing
        // inconsistent
        let _predicting = self
            .prediction
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        autoreleasepool(|_| {
            let input_dict = self.input_dict.borrow_mut();
            input_dict.removeAllObjects();

            let mut arrays = Vec::with_capacity(inputs.len());
            for input in inputs {
                arrays.push(runtime::insert_cached_input(
                    &input_dict,
                    &self.noop_deallocator,
                    *input,
                )?);
            }

            let input_provider = runtime::build_feature_provider(&input_dict)?;
            let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*input_provider);
            let output_provider = runtime::predict_features(&self.model, input_ref)?;
            output_handler(&output_provider)
        })
    }
}

fn configured_compute_units() -> Result<MLComputeUnits, TranscriptionError> {
    let Some(value) = env::var_os("SCRIPTRS_COREML_COMPUTE_UNITS") else {
        return Ok(MLComputeUnits::CPUAndNeuralEngine);
    };

    // allow benchmark tooling to sweep CoreML backends without changing the default runtime
    let value = value.to_string_lossy();
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "cpu_only" => Ok(MLComputeUnits::CPUOnly),
        "cpu_and_gpu" => Ok(MLComputeUnits::CPUAndGPU),
        "all" => Ok(MLComputeUnits::All),
        "cpu_and_neural_engine" | "cpu_and_ane" | "default" => {
            Ok(MLComputeUnits::CPUAndNeuralEngine)
        }
        _ => Err(TranscriptionError::CoreMl(format!(
            "unsupported SCRIPTRS_COREML_COMPUTE_UNITS value `{value}` expected cpu_only, cpu_and_gpu, all, or cpu_and_neural_engine"
        ))),
    }
}

#[derive(Clone)]
pub(crate) struct CachedInputShape {
    name: Retained<NSString>,
    ns_shape: Retained<NSArray<NSNumber>>,
    ns_strides: Retained<NSArray<NSNumber>>,
    total_elements: usize,
}

// SAFETY: the NSString and NSArrays are immutable after construction and only read
unsafe impl Send for CachedInputShape {}

impl std::fmt::Debug for CachedInputShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedInputShape")
            .field("name", &self.name)
            .field("total_elements", &self.total_elements)
            .finish()
    }
}

impl CachedInputShape {
    fn new(name: &str, shape: &[usize]) -> Self {
        Self {
            name: NSString::from_str(name),
            ns_shape: ns_number_array(shape),
            ns_strides: ns_number_array(&contiguous_strides(shape)),
            total_elements: shape.iter().product(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum CachedCoreMlInput<'a> {
    F32 {
        cached: &'a CachedInputShape,
        values: &'a [f32],
    },
    I32 {
        cached: &'a CachedInputShape,
        values: &'a [i32],
    },
}

#[derive(Clone)]
struct CachedOutputKey {
    name: &'static str,
    key: Retained<NSString>,
}

// SAFETY: the NSString is immutable after construction and only read
unsafe impl Send for CachedOutputKey {}

impl std::fmt::Debug for CachedOutputKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedOutputKey")
            .field("name", &self.name)
            .finish()
    }
}

impl CachedOutputKey {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            key: NSString::from_str(name),
        }
    }
}

fn array3_from_output(
    output: &ProtocolObject<dyn MLFeatureProvider>,
    key: &CachedOutputKey,
    context: &str,
) -> Result<Array3<f32>, TranscriptionError> {
    let array = runtime::output_multi_array(output, &key.key, key.name)?;
    let (data, shape) = extract_output(&array)?;
    array3_from_parts(data, &shape, context)
}

fn array3_from_parts(
    data: Vec<f32>,
    shape: &[usize],
    context: &str,
) -> Result<Array3<f32>, TranscriptionError> {
    if shape.len() != 3 {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "{context} shape was not 3D: {shape:?}"
        )));
    }
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), data).map_err(|error| {
        TranscriptionError::InvalidModelOutput(format!("failed to shape {context}: {error}"))
    })
}

fn scalar_f32(
    output: &ProtocolObject<dyn MLFeatureProvider>,
    key: &CachedOutputKey,
) -> Result<f32, TranscriptionError> {
    let array = runtime::output_multi_array(output, &key.key, key.name)?;
    let (data, _) = extract_output(&array)?;
    data.first().copied().ok_or_else(|| {
        TranscriptionError::InvalidModelOutput(format!("missing scalar `{}`", key.name))
    })
}

fn scalar_usize(
    output: &ProtocolObject<dyn MLFeatureProvider>,
    key: &CachedOutputKey,
) -> Result<usize, TranscriptionError> {
    let value = scalar_f32(output, key)?;
    f32_to_usize(value, key.name)
}

fn vector_f32(
    output: &ProtocolObject<dyn MLFeatureProvider>,
    key: &CachedOutputKey,
) -> Result<Vec<f32>, TranscriptionError> {
    let array = runtime::output_multi_array(output, &key.key, key.name)?;
    let (data, _) = extract_output(&array)?;
    Ok(data)
}

fn vector_usize(
    output: &ProtocolObject<dyn MLFeatureProvider>,
    key: &CachedOutputKey,
) -> Result<Vec<usize>, TranscriptionError> {
    vector_f32(output, key)?
        .into_iter()
        .map(|value| f32_to_usize(value, key.name))
        .collect()
}

#[cfg(feature = "vad")]
fn vad_probability(value: f32) -> Result<f32, TranscriptionError> {
    if !value.is_finite() {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "VAD probability was not finite: {value}"
        )));
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "VAD probability {value} is outside [0, 1]"
        )));
    }

    Ok(value)
}

#[cfg(feature = "vad")]
fn vad_state_vector(values: Vec<f32>, name: &str) -> Result<Vec<f32>, TranscriptionError> {
    if values.len() != VAD_STATE_SIZE {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "VAD state `{name}` length {} is not {VAD_STATE_SIZE}",
            values.len()
        )));
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "VAD state `{name}`[{index}] was not finite: {value}"
            )));
        }
    }

    Ok(values)
}

fn f32_to_usize(value: f32, name: &str) -> Result<usize, TranscriptionError> {
    if !value.is_finite() {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "CoreML value `{name}` was not finite: {value}"
        )));
    }
    if value < 0.0 {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "CoreML value `{name}` was negative: {value}"
        )));
    }
    if value.fract() != 0.0 {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "CoreML value `{name}` was not integral: {value}"
        )));
    }
    if value >= usize::MAX as f32 {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "CoreML value `{name}` was outside the usize range: {value}"
        )));
    }

    Ok(value as usize)
}

#[cfg(test)]
mod tests {
    use super::f32_to_usize;

    #[test]
    fn float_to_usize_rejects_non_finite_non_integral_and_out_of_range_values() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 1.5] {
            assert!(f32_to_usize(value, "test").is_err(), "value={value}");
        }
        assert!(f32_to_usize(usize::MAX as f32, "test").is_err());
        assert_eq!(f32_to_usize(42.0, "test").unwrap(), 42);
    }
}

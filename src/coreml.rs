#[cfg(target_os = "macos")]
mod array;
#[cfg(target_os = "macos")]
mod runtime;

#[cfg(target_os = "macos")]
use std::cell::RefCell;
#[cfg(target_os = "macos")]
use std::collections::HashMap;
#[cfg(target_os = "macos")]
use std::ffi::c_void;
#[cfg(target_os = "macos")]
use std::path::Path;
#[cfg(target_os = "macos")]
use std::ptr::NonNull;

#[cfg(target_os = "macos")]
use block2::RcBlock;
#[cfg(target_os = "macos")]
use ndarray::{Array1, Array2, Array3};
#[cfg(target_os = "macos")]
use objc2::rc::{Retained, autoreleasepool};
#[cfg(target_os = "macos")]
use objc2::runtime::{AnyObject, ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_core_ml::{MLComputeUnits, MLFeatureProvider, MLModel};
#[cfg(target_os = "macos")]
use objc2_foundation::{NSArray, NSMutableDictionary, NSNumber, NSString};

#[cfg(target_os = "macos")]
use crate::constants::{DECODER_HIDDEN_SIZE, DECODER_LAYERS, ENCODER_HIDDEN_SIZE};
#[cfg(all(target_os = "macos", feature = "long-form"))]
use crate::constants::{VAD_CONTEXT_SAMPLES, VAD_STATE_SIZE, VAD_WINDOW_SAMPLES};
#[cfg(target_os = "macos")]
use crate::coreml::array::{contiguous_strides, extract_output, ns_number_array};
#[cfg(target_os = "macos")]
use crate::error::TranscriptionError;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub(crate) struct ParakeetSplitCoreMlModel {
    encoder: CoreMlModel,
    decoder: DecoderCoreMlModel,
    joint_decision: JointDecisionCoreMlModel,
}

#[cfg(all(target_os = "macos", feature = "long-form"))]
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

#[cfg(target_os = "macos")]
impl ParakeetSplitCoreMlModel {
    pub(crate) fn new(
        encoder_path: &Path,
        decoder_path: &Path,
        joint_decision_path: &Path,
    ) -> Result<Self, TranscriptionError> {
        Ok(Self {
            encoder: CoreMlModel::new(encoder_path)?,
            decoder: DecoderCoreMlModel::new(decoder_path)?,
            joint_decision: JointDecisionCoreMlModel::new(joint_decision_path)?,
        })
    }

    pub(crate) fn run_encoder(
        &self,
        mel: Array3<f32>,
        lengths: Vec<i32>,
    ) -> Result<(Array3<f32>, usize), TranscriptionError> {
        let inputs = [
            CoreMlInput::F32 {
                name: "mel",
                values: mel.as_slice().ok_or_else(|| {
                    TranscriptionError::InvalidModelOutput(
                        "encoder input was not contiguous".to_owned(),
                    )
                })?,
                shape: &[1, mel.shape()[1], mel.shape()[2]],
            },
            CoreMlInput::I32 {
                name: "mel_length",
                values: &lengths,
                shape: &[1],
            },
        ];
        let outputs = self
            .encoder
            .predict(&inputs, &["encoder", "encoder_length"])?;
        let tensor = outputs.get("encoder").ok_or_else(|| {
            TranscriptionError::InvalidModelOutput("encoder output `encoder` missing".to_owned())
        })?;
        let shape = tensor.shape.as_slice();
        if shape.len() != 3 {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "encoder output shape was not 3D: {shape:?}"
            )));
        }

        let encoder = match shape {
            [1, hidden, time] if *hidden == ENCODER_HIDDEN_SIZE => {
                array3_from_parts(tensor.data.clone(), shape, "encoder output")?
            }
            [1, time, hidden] if *hidden == ENCODER_HIDDEN_SIZE => {
                Array3::from_shape_vec((1, *time, *hidden), tensor.data.clone())
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
        let time_steps = outputs
            .get("encoder_length")
            .and_then(|tensor| tensor.data.first())
            .copied()
            .map(|value| value as usize)
            .filter(|value| *value > 0)
            .unwrap_or_else(|| encoder.shape()[2]);
        Ok((encoder, time_steps))
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
    ) -> Result<JointDecision, TranscriptionError> {
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

#[cfg(all(target_os = "macos", feature = "long-form"))]
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
        self.model.predict_cached(
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
                    probability: scalar_f32(output, &self.probability_output)?,
                    hidden_state: vector_f32(output, &self.hidden_output)?,
                    cell_state: vector_f32(output, &self.cell_output)?,
                })
            },
        )
    }
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
impl DecoderCoreMlModel {
    fn new(path: &Path) -> Result<Self, TranscriptionError> {
        Ok(Self {
            model: CoreMlModel::new(path)?,
            targets: CachedInputShape::new("targets", &[1, 1]),
            target_length: CachedInputShape::new("target_length", &[1]),
            hidden_state: CachedInputShape::new("h_in", &[DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE]),
            cell_state: CachedInputShape::new("c_in", &[DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE]),
            decoder_output: CachedOutputKey::new("decoder"),
            hidden_output: CachedOutputKey::new("h_out"),
            cell_output: CachedOutputKey::new("c_out"),
        })
    }

    fn run(
        &self,
        targets: &[i32],
        target_length: &[i32],
        hidden_state: &[f32],
        cell_state: &[f32],
    ) -> Result<SplitDecoderCoreMlOutput, TranscriptionError> {
        self.model.predict_cached(
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

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct JointDecisionCoreMlModel {
    model: CoreMlModel,
    encoder_step: CachedInputShape,
    decoder_step: CachedInputShape,
    token_id_output: CachedOutputKey,
    token_prob_output: CachedOutputKey,
    duration_output: CachedOutputKey,
}

#[cfg(target_os = "macos")]
impl JointDecisionCoreMlModel {
    fn new(path: &Path) -> Result<Self, TranscriptionError> {
        Ok(Self {
            model: CoreMlModel::new(path)?,
            encoder_step: CachedInputShape::new("encoder_step", &[1, ENCODER_HIDDEN_SIZE, 1]),
            decoder_step: CachedInputShape::new("decoder_step", &[1, DECODER_HIDDEN_SIZE, 1]),
            token_id_output: CachedOutputKey::new("token_id"),
            token_prob_output: CachedOutputKey::new("token_prob"),
            duration_output: CachedOutputKey::new("duration"),
        })
    }

    fn run(
        &self,
        encoder_step: &[f32],
        decoder_step: &[f32],
    ) -> Result<JointDecision, TranscriptionError> {
        self.model.predict_cached(
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
                Ok(JointDecision {
                    token_id: scalar_usize(output, &self.token_id_output)?,
                    token_prob: scalar_f32(output, &self.token_prob_output)?,
                    duration_step: scalar_usize(output, &self.duration_output)?,
                })
            },
        )
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub(crate) struct SplitDecoderCoreMlOutput {
    pub decoder_step: Array3<f32>,
    pub hidden_state: Array3<f32>,
    pub cell_state: Array3<f32>,
}

#[cfg(all(target_os = "macos", feature = "long-form"))]
#[derive(Debug, Clone)]
pub(crate) struct SileroVadOutput {
    pub probability: f32,
    pub hidden_state: Vec<f32>,
    pub cell_state: Vec<f32>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct JointDecision {
    pub token_id: usize,
    pub token_prob: f32,
    pub duration_step: usize,
}

#[cfg(target_os = "macos")]
pub(crate) struct CoreMlModel {
    model: Retained<MLModel>,
    noop_deallocator: RcBlock<dyn Fn(NonNull<c_void>)>,
    input_dict: RefCell<Retained<NSMutableDictionary<NSString, AnyObject>>>,
}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for CoreMlModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMlModel").finish_non_exhaustive()
    }
}

#[cfg(target_os = "macos")]
impl Clone for CoreMlModel {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            noop_deallocator: RcBlock::new(|_: NonNull<c_void>| {}),
            input_dict: RefCell::new(NSMutableDictionary::new()),
        }
    }
}

#[cfg(target_os = "macos")]
impl CoreMlModel {
    pub(crate) fn new(path: &Path) -> Result<Self, TranscriptionError> {
        Ok(Self {
            model: runtime::load_model(path, MLComputeUnits::CPUAndNeuralEngine)?,
            noop_deallocator: RcBlock::new(|_: NonNull<c_void>| {}),
            input_dict: RefCell::new(NSMutableDictionary::new()),
        })
    }

    pub(crate) fn predict(
        &self,
        inputs: &[CoreMlInput<'_>],
        output_names: &[&str],
    ) -> Result<HashMap<String, CoreMlTensor>, TranscriptionError> {
        self.with_cleared_input_dict(|input_dict, deallocator| {
            runtime::predict(&self.model, input_dict, deallocator, inputs, output_names)
        })
    }

    fn predict_cached<T>(
        &self,
        inputs: &[CachedCoreMlInput<'_>],
        output_handler: impl FnOnce(
            &ProtocolObject<dyn MLFeatureProvider>,
        ) -> Result<T, TranscriptionError>,
    ) -> Result<T, TranscriptionError> {
        self.with_cleared_input_dict(|input_dict, deallocator| {
            let mut arrays = Vec::with_capacity(inputs.len());
            for input in inputs {
                arrays.push(runtime::insert_cached_input(
                    input_dict,
                    deallocator,
                    *input,
                )?);
            }

            let input_provider = runtime::build_feature_provider(input_dict)?;
            let input_ref: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*input_provider);
            let output_provider = runtime::predict_features(&self.model, input_ref)?;
            output_handler(&output_provider)
        })
    }

    fn with_cleared_input_dict<T>(
        &self,
        f: impl FnOnce(
            &NSMutableDictionary<NSString, AnyObject>,
            &RcBlock<dyn Fn(NonNull<c_void>)>,
        ) -> Result<T, TranscriptionError>,
    ) -> Result<T, TranscriptionError> {
        autoreleasepool(|_| {
            let input_dict = self.input_dict.borrow_mut();
            input_dict.removeAllObjects();
            f(&input_dict, &self.noop_deallocator)
        })
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub(crate) struct CoreMlTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
pub(crate) enum CoreMlInput<'a> {
    F32 {
        name: &'a str,
        values: &'a [f32],
        shape: &'a [usize],
    },
    I32 {
        name: &'a str,
        values: &'a [i32],
        shape: &'a [usize],
    },
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub(crate) struct CachedInputShape {
    name: Retained<NSString>,
    ns_shape: Retained<NSArray<NSNumber>>,
    ns_strides: Retained<NSArray<NSNumber>>,
    total_elements: usize,
}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for CachedInputShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedInputShape")
            .field("name", &self.name)
            .field("total_elements", &self.total_elements)
            .finish()
    }
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct CachedOutputKey {
    name: &'static str,
    key: Retained<NSString>,
}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for CachedOutputKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedOutputKey")
            .field("name", &self.name)
            .finish()
    }
}

#[cfg(target_os = "macos")]
impl CachedOutputKey {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            key: NSString::from_str(name),
        }
    }
}

#[cfg(target_os = "macos")]
fn array3_from_output(
    output: &ProtocolObject<dyn MLFeatureProvider>,
    key: &CachedOutputKey,
    context: &str,
) -> Result<Array3<f32>, TranscriptionError> {
    let array = runtime::output_multi_array(output, &key.key, key.name)?;
    let (data, shape) = extract_output(&array)?;
    array3_from_parts(data, &shape, context)
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn scalar_usize(
    output: &ProtocolObject<dyn MLFeatureProvider>,
    key: &CachedOutputKey,
) -> Result<usize, TranscriptionError> {
    let value = scalar_f32(output, key)?;
    if value < 0.0 {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "scalar `{}` was negative: {value}",
            key.name
        )));
    }
    Ok(value as usize)
}

#[cfg(all(target_os = "macos", feature = "long-form"))]
fn vector_f32(
    output: &ProtocolObject<dyn MLFeatureProvider>,
    key: &CachedOutputKey,
) -> Result<Vec<f32>, TranscriptionError> {
    let array = runtime::output_multi_array(output, &key.key, key.name)?;
    let (data, _) = extract_output(&array)?;
    Ok(data)
}

#[cfg(target_os = "macos")]
mod array;
#[cfg(target_os = "macos")]
mod runtime;

#[cfg(target_os = "macos")]
use std::collections::HashMap;
#[cfg(target_os = "macos")]
use std::path::Path;

#[cfg(target_os = "macos")]
use ndarray::{Array1, Array2, Array3};
#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2_core_ml::{MLComputeUnits, MLModel};

#[cfg(target_os = "macos")]
use crate::constants::{DECODER_HIDDEN_SIZE, DECODER_LAYERS, ENCODER_HIDDEN_SIZE};
#[cfg(target_os = "macos")]
use crate::error::TranscriptionError;

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub(crate) struct ParakeetFusedCoreMlModel {
    encoder: CoreMlModel,
    decoder_joint: CoreMlModel,
}

#[cfg(target_os = "macos")]
impl ParakeetFusedCoreMlModel {
    pub(crate) fn new(
        encoder_path: &Path,
        decoder_joint_path: &Path,
    ) -> Result<Self, TranscriptionError> {
        Ok(Self {
            encoder: CoreMlModel::new(encoder_path)?,
            decoder_joint: CoreMlModel::new(decoder_joint_path)?,
        })
    }

    pub(crate) fn run_encoder(
        &self,
        audio_signal: Array3<f32>,
        lengths: Vec<i32>,
    ) -> Result<(Array3<f32>, usize), TranscriptionError> {
        let inputs = [
            CoreMlInput::F32 {
                name: "audio_signal",
                values: audio_signal.as_slice().ok_or_else(|| {
                    TranscriptionError::InvalidModelOutput(
                        "encoder input was not contiguous".to_owned(),
                    )
                })?,
                shape: &[1, audio_signal.shape()[1], audio_signal.shape()[2]],
            },
            CoreMlInput::I32 {
                name: "length",
                values: &lengths,
                shape: &[1],
            },
        ];
        let outputs = self
            .encoder
            .predict(&inputs, &["outputs", "encoded_lengths"])?;
        let tensor = outputs.get("outputs").ok_or_else(|| {
            TranscriptionError::InvalidModelOutput("encoder output `outputs` missing".to_owned())
        })?;
        let shape = tensor.shape.as_slice();
        if shape.len() != 3 {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "encoder output shape was not 3D: {shape:?}"
            )));
        }

        let encoder = match shape {
            [1, hidden, time] if *hidden == ENCODER_HIDDEN_SIZE => {
                Array3::from_shape_vec((1, *hidden, *time), tensor.data.clone()).map_err(
                    |error| {
                        TranscriptionError::InvalidModelOutput(format!(
                            "failed to shape encoder output: {error}"
                        ))
                    },
                )?
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
            .get("encoded_lengths")
            .and_then(|tensor| tensor.data.first())
            .copied()
            .map(|value| value as usize)
            .unwrap_or_else(|| encoder.shape()[2]);
        Ok((encoder, time_steps))
    }

    pub(crate) fn run_decoder(
        &self,
        encoder_outputs: Array3<f32>,
        targets: Array2<i32>,
        target_length: Array1<i32>,
        hidden_state: Array3<f32>,
        cell_state: Array3<f32>,
    ) -> Result<FusedDecoderCoreMlOutput, TranscriptionError> {
        let inputs = [
            CoreMlInput::F32 {
                name: "encoder_outputs",
                values: encoder_outputs.as_slice().ok_or_else(|| {
                    TranscriptionError::InvalidModelOutput(
                        "decoder encoder frame was not contiguous".to_owned(),
                    )
                })?,
                shape: &[1, encoder_outputs.shape()[1], encoder_outputs.shape()[2]],
            },
            CoreMlInput::I32 {
                name: "targets",
                values: targets.as_slice().ok_or_else(|| {
                    TranscriptionError::InvalidModelOutput(
                        "decoder targets were not contiguous".to_owned(),
                    )
                })?,
                shape: &[1, 1],
            },
            CoreMlInput::I32 {
                name: "target_length",
                values: target_length.as_slice().ok_or_else(|| {
                    TranscriptionError::InvalidModelOutput(
                        "decoder target lengths were not contiguous".to_owned(),
                    )
                })?,
                shape: &[1],
            },
            CoreMlInput::F32 {
                name: "input_states_1",
                values: hidden_state.as_slice().ok_or_else(|| {
                    TranscriptionError::InvalidModelOutput(
                        "decoder hidden state was not contiguous".to_owned(),
                    )
                })?,
                shape: &[DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE],
            },
            CoreMlInput::F32 {
                name: "input_states_2",
                values: cell_state.as_slice().ok_or_else(|| {
                    TranscriptionError::InvalidModelOutput(
                        "decoder cell state was not contiguous".to_owned(),
                    )
                })?,
                shape: &[DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE],
            },
        ];
        let outputs = self
            .decoder_joint
            .predict(&inputs, &["outputs", "output_states_1", "output_states_2"])?;
        let logits = outputs.get("outputs").ok_or_else(|| {
            TranscriptionError::InvalidModelOutput("decoder output `outputs` missing".to_owned())
        })?;
        let hidden = outputs.get("output_states_1").ok_or_else(|| {
            TranscriptionError::InvalidModelOutput(
                "decoder output `output_states_1` missing".to_owned(),
            )
        })?;
        let cell = outputs.get("output_states_2").ok_or_else(|| {
            TranscriptionError::InvalidModelOutput(
                "decoder output `output_states_2` missing".to_owned(),
            )
        })?;

        let hidden_state = array3_from_tensor(hidden, "decoder hidden state")?;
        let cell_state = array3_from_tensor(cell, "decoder cell state")?;
        Ok(FusedDecoderCoreMlOutput {
            logits: logits.data.clone(),
            vocab_size: logits.data.len().saturating_sub(5),
            hidden_state,
            cell_state,
        })
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub(crate) struct FusedDecoderCoreMlOutput {
    pub logits: Vec<f32>,
    pub vocab_size: usize,
    pub hidden_state: Array3<f32>,
    pub cell_state: Array3<f32>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub(crate) struct ParakeetSplitCoreMlModel {
    encoder: CoreMlModel,
    decoder: CoreMlModel,
    joint_decision: CoreMlModel,
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
            decoder: CoreMlModel::new(decoder_path)?,
            joint_decision: CoreMlModel::new(joint_decision_path)?,
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
                        "encoder mel input was not contiguous".to_owned(),
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
        let encoder = array3_from_tensor(tensor, "encoder output")?;
        let time_steps = outputs
            .get("encoder_length")
            .and_then(|value| value.data.first())
            .copied()
            .map(|value| value as usize)
            .unwrap_or_else(|| encoder.shape()[2]);
        Ok((encoder, time_steps))
    }

    pub(crate) fn run_decoder_step(
        &self,
        targets: Array2<i32>,
        target_length: Array1<i32>,
        hidden_state: Array3<f32>,
        cell_state: Array3<f32>,
        encoder_step: Array3<f32>,
    ) -> Result<SplitDecoderCoreMlOutput, TranscriptionError> {
        let decoder_outputs = self.decoder.predict(
            &[
                CoreMlInput::I32 {
                    name: "targets",
                    values: targets.as_slice().ok_or_else(|| {
                        TranscriptionError::InvalidModelOutput(
                            "decoder targets were not contiguous".to_owned(),
                        )
                    })?,
                    shape: &[1, 1],
                },
                CoreMlInput::I32 {
                    name: "target_length",
                    values: target_length.as_slice().ok_or_else(|| {
                        TranscriptionError::InvalidModelOutput(
                            "decoder target lengths were not contiguous".to_owned(),
                        )
                    })?,
                    shape: &[1],
                },
                CoreMlInput::F32 {
                    name: "h_in",
                    values: hidden_state.as_slice().ok_or_else(|| {
                        TranscriptionError::InvalidModelOutput(
                            "decoder hidden state was not contiguous".to_owned(),
                        )
                    })?,
                    shape: &[DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE],
                },
                CoreMlInput::F32 {
                    name: "c_in",
                    values: cell_state.as_slice().ok_or_else(|| {
                        TranscriptionError::InvalidModelOutput(
                            "decoder cell state was not contiguous".to_owned(),
                        )
                    })?,
                    shape: &[DECODER_LAYERS, 1, DECODER_HIDDEN_SIZE],
                },
            ],
            &["decoder", "h_out", "c_out"],
        )?;

        let decoder_step = array3_from_output(&decoder_outputs, "decoder", "decoder step")?;
        let hidden_state = array3_from_output(&decoder_outputs, "h_out", "decoder hidden state")?;
        let cell_state = array3_from_output(&decoder_outputs, "c_out", "decoder cell state")?;

        let joint_outputs = self.joint_decision.predict(
            &[
                CoreMlInput::F32 {
                    name: "encoder_step",
                    values: encoder_step.as_slice().ok_or_else(|| {
                        TranscriptionError::InvalidModelOutput(
                            "joint encoder step was not contiguous".to_owned(),
                        )
                    })?,
                    shape: &[1, encoder_step.shape()[1], encoder_step.shape()[2]],
                },
                CoreMlInput::F32 {
                    name: "decoder_step",
                    values: decoder_step.as_slice().ok_or_else(|| {
                        TranscriptionError::InvalidModelOutput(
                            "joint decoder step was not contiguous".to_owned(),
                        )
                    })?,
                    shape: &[1, decoder_step.shape()[1], decoder_step.shape()[2]],
                },
            ],
            &["token_id", "token_prob", "duration"],
        )?;

        let token_id = scalar_usize(&joint_outputs, "token_id")?;
        let token_prob = scalar_f32(&joint_outputs, "token_prob")?;
        let duration = scalar_usize(&joint_outputs, "duration")?;

        Ok(SplitDecoderCoreMlOutput {
            token_id,
            token_prob,
            duration,
            hidden_state,
            cell_state,
        })
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub(crate) struct SplitDecoderCoreMlOutput {
    pub token_id: usize,
    pub token_prob: f32,
    pub duration: usize,
    pub hidden_state: Array3<f32>,
    pub cell_state: Array3<f32>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub(crate) struct CoreMlModel {
    model: Retained<MLModel>,
}

#[cfg(target_os = "macos")]
impl CoreMlModel {
    pub(crate) fn new(path: &Path) -> Result<Self, TranscriptionError> {
        Ok(Self {
            model: runtime::load_model(path, MLComputeUnits::CPUAndNeuralEngine)?,
        })
    }

    pub(crate) fn predict(
        &self,
        inputs: &[CoreMlInput<'_>],
        output_names: &[&str],
    ) -> Result<HashMap<String, CoreMlTensor>, TranscriptionError> {
        runtime::predict(&self.model, inputs, output_names)
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
fn array3_from_tensor(
    tensor: &CoreMlTensor,
    context: &str,
) -> Result<Array3<f32>, TranscriptionError> {
    if tensor.shape.len() != 3 {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "{context} shape was not 3D: {:?}",
            tensor.shape
        )));
    }
    Array3::from_shape_vec(
        (tensor.shape[0], tensor.shape[1], tensor.shape[2]),
        tensor.data.clone(),
    )
    .map_err(|error| {
        TranscriptionError::InvalidModelOutput(format!("failed to shape {context}: {error}"))
    })
}

#[cfg(target_os = "macos")]
fn array3_from_output(
    outputs: &HashMap<String, CoreMlTensor>,
    name: &str,
    context: &str,
) -> Result<Array3<f32>, TranscriptionError> {
    let tensor = outputs.get(name).ok_or_else(|| {
        TranscriptionError::InvalidModelOutput(format!("missing CoreML output `{name}`"))
    })?;
    array3_from_tensor(tensor, context)
}

#[cfg(target_os = "macos")]
fn scalar_f32(
    outputs: &HashMap<String, CoreMlTensor>,
    name: &str,
) -> Result<f32, TranscriptionError> {
    outputs
        .get(name)
        .and_then(|tensor| tensor.data.first())
        .copied()
        .ok_or_else(|| TranscriptionError::InvalidModelOutput(format!("missing scalar `{name}`")))
}

#[cfg(target_os = "macos")]
fn scalar_usize(
    outputs: &HashMap<String, CoreMlTensor>,
    name: &str,
) -> Result<usize, TranscriptionError> {
    let value = scalar_f32(outputs, name)?;
    if value < 0.0 {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "scalar `{name}` was negative: {value}"
        )));
    }
    Ok(value as usize)
}

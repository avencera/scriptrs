use std::path::Path;
use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use crate::config::TranscriptionConfig;
use crate::deployment::{DecoderDimensions, ModelIo, ParakeetDeployment};
use crate::error::TranscriptionError;
use crate::model_io::{DecoderStepOutputs, JointOutputs};
use crate::onnx::contract::GraphContract;
use crate::onnx::error::{GraphKind, OnnxBundleError};
use crate::onnx::session::{
    GraphSession, OutputReader, OutputTensor, SessionProvider, f32_input, i32_input,
};

/// Absolute paths of the three split graphs
pub(crate) struct ParakeetGraphPaths<'a> {
    pub(crate) encoder: &'a Path,
    pub(crate) decoder: &'a Path,
    pub(crate) joint: &'a Path,
}

/// Parakeet split graphs running through ONNX Runtime
///
/// Sessions are shared between clones so one opened bundle owns exactly one
/// set of device sessions.
#[derive(Debug, Clone)]
pub(crate) struct ParakeetOnnxModel(Arc<ParakeetSessions>);

#[derive(Debug)]
struct ParakeetSessions {
    encoder: GraphSession,
    decoder: GraphSession,
    joint: GraphSession,
    io: ModelIo,
    dims: DecoderDimensions,
}

impl ParakeetOnnxModel {
    pub(crate) fn load(
        paths: ParakeetGraphPaths<'_>,
        provider: SessionProvider,
        deployment: &ParakeetDeployment,
    ) -> Result<Self, OnnxBundleError> {
        let io = deployment.model_io();
        let dims = deployment.decoder_dimensions();
        let config = TranscriptionConfig::from_deployment(deployment);

        let encoder = GraphSession::load(
            GraphKind::Encoder,
            paths.encoder,
            provider,
            GraphContract::encoder(io, dims, config.feature_size, config.max_feature_frames()),
        )?;
        let decoder = GraphSession::load(
            GraphKind::Decoder,
            paths.decoder,
            provider,
            GraphContract::decoder(io, dims),
        )?;
        let joint = GraphSession::load(
            GraphKind::Joint,
            paths.joint,
            provider,
            GraphContract::joint(io, dims),
        )?;

        Ok(Self(Arc::new(ParakeetSessions {
            encoder,
            decoder,
            joint,
            io,
            dims,
        })))
    }

    /// Encode padded mel features and return `[1, hidden, time]` plus valid steps
    pub(crate) fn run_encoder(
        &self,
        mel: &[f32],
        feature_size: usize,
        target_frames: usize,
        feature_frames: usize,
    ) -> Result<(Array3<f32>, usize), TranscriptionError> {
        if mel.len() != feature_size * target_frames {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "encoder input holds {} values for shape [1, {feature_size}, {target_frames}]",
                mel.len()
            )));
        }
        if feature_frames > target_frames {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "feature frame count {feature_frames} exceeded encoder target {target_frames}"
            )));
        }
        let mel_length = [i32::try_from(feature_frames).map_err(|_| {
            TranscriptionError::InvalidModelOutput(format!(
                "feature frame count {feature_frames} does not fit the int32 length input"
            ))
        })?];

        let encoder_io = self.0.io.encoder();
        let inputs = vec![
            f32_input(encoder_io.input_mel, &[1, feature_size, target_frames], mel)?,
            i32_input(encoder_io.input_length, &[1], &mel_length)?,
        ];

        self.0.encoder.run(inputs, |outputs| {
            let encoder = outputs.f32(encoder_io.output_encoder)?;
            let time = encoder.shape[2];
            let length = scalar_i32(outputs, encoder_io.output_length)?;
            let time_steps = usize::try_from(length).map_err(|_| {
                TranscriptionError::InvalidModelOutput(format!(
                    "encoder length {length} was negative"
                ))
            })?;
            if time_steps > time {
                return Err(TranscriptionError::InvalidModelOutput(format!(
                    "encoder length {time_steps} exceeded encoder time steps {time}"
                )));
            }

            Ok((array3(encoder, "encoder output")?, time_steps))
        })
    }

    /// Advance the prediction network by one token
    pub(crate) fn run_decoder(
        &self,
        targets: &Array2<i32>,
        target_length: &Array1<i32>,
        hidden_state: &Array3<f32>,
        cell_state: &Array3<f32>,
    ) -> Result<DecoderStepOutputs, TranscriptionError> {
        let decoder_io = self.0.io.decoder();
        let state_shape = [
            self.0.dims.decoder_layers(),
            1,
            self.0.dims.decoder_hidden(),
        ];
        let inputs = vec![
            i32_input(
                decoder_io.targets,
                &[1, 1],
                contiguous(targets.as_slice(), "decoder targets")?,
            )?,
            i32_input(
                decoder_io.target_length,
                &[1],
                contiguous(target_length.as_slice(), "decoder target lengths")?,
            )?,
            f32_input(
                decoder_io.hidden_in,
                &state_shape,
                contiguous(hidden_state.as_slice(), "decoder hidden state")?,
            )?,
            f32_input(
                decoder_io.cell_in,
                &state_shape,
                contiguous(cell_state.as_slice(), "decoder cell state")?,
            )?,
        ];

        self.0.decoder.run(inputs, |outputs| {
            Ok(DecoderStepOutputs {
                decoder_step: array3(outputs.f32(decoder_io.decoder)?, "decoder step")?,
                hidden_state: array3(outputs.f32(decoder_io.hidden_out)?, "decoder hidden state")?,
                cell_state: array3(outputs.f32(decoder_io.cell_out)?, "decoder cell state")?,
            })
        })
    }

    /// Score one encoder frame against the current decoder output
    pub(crate) fn run_joint(
        &self,
        encoder_step: &Array3<f32>,
        decoder_step: &Array3<f32>,
    ) -> Result<JointOutputs, TranscriptionError> {
        let joint_io = self.0.io.joint();
        let inputs = vec![
            f32_input(
                joint_io.encoder_step,
                &[1, self.0.dims.encoder_hidden(), 1],
                contiguous(encoder_step.as_slice(), "joint encoder step")?,
            )?,
            f32_input(
                joint_io.decoder_step,
                &[1, self.0.dims.decoder_hidden(), 1],
                contiguous(decoder_step.as_slice(), "joint decoder step")?,
            )?,
        ];

        self.0.joint.run(inputs, |outputs| {
            let token_id =
                non_negative(scalar_i32(outputs, joint_io.token_id)?, joint_io.token_id)?;
            let token_prob = finite(
                scalar_f32(outputs, joint_io.token_prob)?,
                joint_io.token_prob,
            )?;
            let duration_bin =
                non_negative(scalar_i32(outputs, joint_io.duration)?, joint_io.duration)?;
            let top_k_ids = joint_io
                .top_k_ids
                .map(|name| {
                    outputs
                        .i32(name)?
                        .data
                        .into_iter()
                        .map(|id| non_negative(id, name))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?;
            let top_k_logits = joint_io
                .top_k_logits
                .map(|name| outputs.f32(name).map(|tensor| tensor.data))
                .transpose()?;

            Ok(JointOutputs {
                token_id,
                token_prob,
                duration_bin,
                top_k: JointOutputs::bind_top_k(top_k_ids, top_k_logits)?,
            })
        })
    }
}

fn contiguous<'a, T>(
    values: Option<&'a [T]>,
    context: &str,
) -> Result<&'a [T], TranscriptionError> {
    values.ok_or_else(|| {
        TranscriptionError::InvalidModelOutput(format!("{context} was not contiguous"))
    })
}

fn array3(tensor: OutputTensor<f32>, context: &str) -> Result<Array3<f32>, TranscriptionError> {
    let [batch, rows, columns] = tensor.shape[..] else {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "{context} shape was not 3D: {:?}",
            tensor.shape
        )));
    };
    Array3::from_shape_vec((batch, rows, columns), tensor.data).map_err(|error| {
        TranscriptionError::InvalidModelOutput(format!("failed to shape {context}: {error}"))
    })
}

fn scalar_i32(outputs: &OutputReader<'_, '_>, name: &str) -> Result<i32, TranscriptionError> {
    first(outputs.i32(name)?, name)
}

fn scalar_f32(outputs: &OutputReader<'_, '_>, name: &str) -> Result<f32, TranscriptionError> {
    first(outputs.f32(name)?, name)
}

fn first<T: Copy>(tensor: OutputTensor<T>, name: &str) -> Result<T, TranscriptionError> {
    tensor
        .data
        .first()
        .copied()
        .ok_or_else(|| TranscriptionError::InvalidModelOutput(format!("output `{name}` was empty")))
}

fn non_negative(value: i32, name: &str) -> Result<usize, TranscriptionError> {
    usize::try_from(value).map_err(|_| {
        TranscriptionError::InvalidModelOutput(format!("output `{name}` was negative: {value}"))
    })
}

fn finite(value: f32, name: &str) -> Result<f32, TranscriptionError> {
    if !value.is_finite() {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "output `{name}` was not finite: {value}"
        )));
    }
    Ok(value)
}

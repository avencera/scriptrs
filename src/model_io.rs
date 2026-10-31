#[cfg(any(test, target_os = "macos"))]
use ndarray::Array3;

use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::language::{
    LanguageSelection, TopKCandidates, filter_top_k_language, validate_candidate_logits,
};
use crate::vocab::Vocabulary;

/// Prediction-network outputs for one token step, shared by every step runtime
#[cfg(any(test, target_os = "macos"))]
#[derive(Debug, Clone)]
pub(crate) struct DecoderStepOutputs {
    /// Projected decoder output in `[1, hidden, 1]` order
    pub decoder_step: Array3<f32>,
    /// Next LSTM hidden state in `[layers, 1, hidden]` order
    pub hidden_state: Array3<f32>,
    /// Next LSTM cell state in `[layers, 1, hidden]` order
    pub cell_state: Array3<f32>,
}

/// Encoder hidden states that tests can inject without Core ML
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderOutputs {
    /// Flattened encoder hidden states in `[batch, hidden, time]` order
    pub hidden: Vec<f32>,
    /// Batch size
    pub batch: usize,
    /// Encoder hidden size
    pub hidden_size: usize,
    /// Valid encoder time steps
    pub time_steps: usize,
}

impl EncoderOutputs {
    /// Build encoder outputs with an explicit `[1, hidden, time]` layout
    pub fn from_hidden_steps(hidden_size: usize, time_steps: usize, hidden: Vec<f32>) -> Self {
        Self {
            hidden,
            batch: 1,
            hidden_size,
            time_steps,
        }
    }
}

/// Joint-decision tensors that tests can inject without Core ML
#[derive(Debug, Clone, PartialEq)]
pub struct JointOutputs {
    /// Argmax token id
    pub token_id: usize,
    /// Token probability
    pub token_prob: f32,
    /// Duration bin from the joint head
    pub duration_bin: usize,
    /// Optional nonempty top-K id/logit pairs
    pub top_k: Option<TopKCandidates>,
}

impl JointOutputs {
    /// Bind optional parallel top-K tensors into one typed candidate list
    #[cfg(target_os = "macos")]
    pub(crate) fn bind_top_k(
        ids: Option<Vec<usize>>,
        logits: Option<Vec<f32>>,
    ) -> Result<Option<TopKCandidates>, TranscriptionError> {
        match (ids, logits) {
            (None, None) => Ok(None),
            (Some(ids), Some(logits)) => TopKCandidates::from_parallel(ids, logits).map(Some),
            _ => Err(TranscriptionError::InvalidModelOutput(
                "joint top-k ids and logits must both be present".to_owned(),
            )),
        }
    }

    /// Reject tensors that cannot come from a well-formed joint head
    ///
    /// Every token id, including top-K candidates, must be at most the RNNT
    /// blank id, which is the last id the joint head can emit
    fn validate(&self, blank_id: usize) -> Result<(), TranscriptionError> {
        if self.token_id > blank_id {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "joint token id {} exceeded blank id {blank_id}",
                self.token_id
            )));
        }

        if let Some(top_k) = &self.top_k
            && let Some(bad) = top_k.pairs().iter().find(|pair| pair.token_id > blank_id)
        {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "joint top-k id {} exceeded blank id {blank_id}",
                bad.token_id
            )));
        }

        Ok(())
    }
}

/// Joint step after duration mapping and optional language filtering
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodedJointStep {
    /// Token id after optional language filtering
    pub token_id: usize,
    /// Encoder frames to advance
    pub duration_frames: usize,
    /// Confidence for the selected token
    pub confidence: f32,
}

/// Map fixture joint tensors through a deployment without loading Core ML
pub fn decode_joint_outputs(
    outputs: &JointOutputs,
    deployment: &ParakeetDeployment,
    vocabulary: &Vocabulary,
    language: LanguageSelection,
) -> Result<DecodedJointStep, TranscriptionError> {
    let blank_id = vocabulary.blank_id();
    outputs.validate(blank_id)?;

    let duration_frames = deployment.map_duration_bin(outputs.duration_bin)?;
    let mut token_id = outputs.token_id;
    let mut confidence = outputs.token_prob.clamp(0.0, 1.0);

    if let Some(top_k) = &outputs.top_k {
        validate_candidate_logits(top_k.pairs().iter().map(|pair| pair.logit))?;
    }

    if token_id != blank_id
        && let Some(selected) = language.explicit()
        && let Some(top_k) = &outputs.top_k
        && let Some(filtered) = filter_top_k_language(top_k, vocabulary, selected, blank_id)?
    {
        token_id = filtered.token_id;
        confidence = filtered.probability;
    }

    Ok(DecodedJointStep {
        token_id,
        duration_frames,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::{JointOutputs, decode_joint_outputs};
    use crate::deployment::ParakeetDeployment;
    use crate::error::TranscriptionError;
    use crate::language::{LanguageSelection, ParakeetLanguage, TopKCandidates};
    use crate::vocab::Vocabulary;

    fn vocab() -> Vocabulary {
        Vocabulary::from_tokens(&["<unk>", "cat", "кот", "<pl>", "<blk>"])
    }

    fn top_k(ids: &[usize], logits: &[f32]) -> TopKCandidates {
        TopKCandidates::from_parallel(ids.to_vec(), logits.to_vec()).unwrap()
    }

    #[test]
    fn fixture_duration_bins_map_through_the_deployment() {
        let outputs = JointOutputs {
            token_id: 1,
            token_prob: 0.9,
            duration_bin: 3,
            top_k: None,
        };
        let decoded = decode_joint_outputs(
            &outputs,
            &ParakeetDeployment::v3(),
            &vocab(),
            LanguageSelection::Auto,
        )
        .unwrap();
        assert_eq!(decoded.duration_frames, 3);
        assert_eq!(decoded.token_id, 1);
    }

    #[test]
    fn fixture_top_k_language_filter_selects_known_ids() {
        let outputs = JointOutputs {
            token_id: 2,
            token_prob: 0.95,
            duration_bin: 1,
            top_k: Some(top_k(&[2, 1, 3], &[4.0, 3.0, 0.5])),
        };
        let decoded = decode_joint_outputs(
            &outputs,
            &ParakeetDeployment::v3(),
            &vocab(),
            LanguageSelection::Language(ParakeetLanguage::Polish),
        )
        .unwrap();
        assert_eq!(decoded.token_id, 1);
        assert_eq!(decoded.duration_frames, 1);
    }

    #[test]
    fn joint_decode_rejects_non_finite_language_logits() {
        let outputs = JointOutputs {
            token_id: 2,
            token_prob: 0.95,
            duration_bin: 1,
            top_k: Some(top_k(&[2, 1], &[f32::NAN, 3.0])),
        };
        let error = decode_joint_outputs(
            &outputs,
            &ParakeetDeployment::v3(),
            &vocab(),
            LanguageSelection::Language(ParakeetLanguage::Polish),
        )
        .expect_err("non-finite top-k logits must not reach softmax");
        assert!(matches!(
            error,
            TranscriptionError::InvalidLanguageLogit { index: 0, value }
                if value.is_nan()
        ));

        let outputs = JointOutputs {
            token_id: 2,
            token_prob: 0.95,
            duration_bin: 1,
            top_k: Some(top_k(&[2, 1], &[3.0, f32::INFINITY])),
        };
        assert!(matches!(
            decode_joint_outputs(
                &outputs,
                &ParakeetDeployment::v3(),
                &vocab(),
                LanguageSelection::Auto,
            ),
            Err(TranscriptionError::InvalidLanguageLogit { index: 1, value })
                if value.is_infinite()
        ));
    }

    #[test]
    fn joint_decode_rejects_token_ids_past_the_blank() {
        let outputs = JointOutputs {
            token_id: 5,
            token_prob: 0.9,
            duration_bin: 1,
            top_k: None,
        };
        let error = decode_joint_outputs(
            &outputs,
            &ParakeetDeployment::v3(),
            &vocab(),
            LanguageSelection::Auto,
        )
        .expect_err("token id past the blank cannot come from the joint head");
        assert!(matches!(
            error,
            TranscriptionError::InvalidModelOutput(message)
                if message.contains("token id 5 exceeded blank id 4")
        ));

        let outputs = JointOutputs {
            token_id: 2,
            token_prob: 0.9,
            duration_bin: 1,
            top_k: Some(top_k(&[2, 9], &[1.0, 0.5])),
        };
        assert!(matches!(
            decode_joint_outputs(
                &outputs,
                &ParakeetDeployment::v3(),
                &vocab(),
                LanguageSelection::Language(ParakeetLanguage::Polish),
            ),
            Err(TranscriptionError::InvalidModelOutput(message))
                if message.contains("top-k id 9 exceeded blank id 4")
        ));
    }

    #[test]
    fn joint_decode_rejects_mismatched_top_k_lengths() {
        let error = TopKCandidates::from_parallel(vec![2, 1], vec![3.0])
            .expect_err("parallel top-k tensors must have the same length");

        assert!(matches!(
            error,
            TranscriptionError::InvalidModelOutput(message)
                if message.contains("top-k id/logit length mismatch")
        ));
        assert!(TopKCandidates::from_parallel(vec![], vec![]).is_err());
    }
}

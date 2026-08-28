use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::language::{LanguageSelection, filter_top_k_language, validate_candidate_logits};
use crate::vocab::Vocabulary;

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
    /// Optional top-K token ids
    pub top_k_ids: Option<Vec<usize>>,
    /// Optional top-K logits, parallel to `top_k_ids`
    pub top_k_logits: Option<Vec<f32>>,
}

impl JointOutputs {
    fn validate(&self) -> Result<(), TranscriptionError> {
        if let (Some(top_k_ids), Some(top_k_logits)) =
            (self.top_k_ids.as_deref(), self.top_k_logits.as_deref())
            && top_k_ids.len() != top_k_logits.len()
        {
            return Err(TranscriptionError::InvalidModelOutput(format!(
                "joint top-k id/logit length mismatch: ids={} logits={}",
                top_k_ids.len(),
                top_k_logits.len()
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
    outputs.validate()?;

    let duration_frames = deployment.map_duration_bin(outputs.duration_bin)?;
    let mut token_id = outputs.token_id;
    let mut confidence = outputs.token_prob.clamp(0.0, 1.0);
    let blank_id = vocabulary.blank_id();

    if let Some(top_k_logits) = outputs.top_k_logits.as_deref() {
        validate_candidate_logits(top_k_logits)?;
    }

    if token_id != blank_id
        && let Some(selected) = language.explicit()
        && let (Some(top_k_ids), Some(top_k_logits)) = (
            outputs.top_k_ids.as_deref(),
            outputs.top_k_logits.as_deref(),
        )
        && let Some(filtered) =
            filter_top_k_language(top_k_ids, top_k_logits, vocabulary, selected, blank_id)?
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
    use crate::language::{LanguageSelection, ParakeetLanguage};
    use crate::vocab::Vocabulary;

    fn vocab() -> Vocabulary {
        Vocabulary::from_tokens(&["<unk>", "cat", "кот", "<pl>", "<blk>"])
    }

    #[test]
    fn fixture_duration_bins_map_through_the_deployment() {
        let outputs = JointOutputs {
            token_id: 1,
            token_prob: 0.9,
            duration_bin: 3,
            top_k_ids: None,
            top_k_logits: None,
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
            top_k_ids: Some(vec![2, 1, 3]),
            top_k_logits: Some(vec![4.0, 3.0, 0.5]),
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
            top_k_ids: Some(vec![2, 1]),
            top_k_logits: Some(vec![f32::NAN, 3.0]),
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
            top_k_ids: Some(vec![2, 1]),
            top_k_logits: Some(vec![3.0, f32::INFINITY]),
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
    fn joint_decode_rejects_mismatched_top_k_lengths() {
        let outputs = JointOutputs {
            token_id: 2,
            token_prob: 0.95,
            duration_bin: 1,
            top_k_ids: Some(vec![2, 1]),
            top_k_logits: Some(vec![3.0]),
        };

        let error = decode_joint_outputs(
            &outputs,
            &ParakeetDeployment::v3(),
            &vocab(),
            LanguageSelection::Language(ParakeetLanguage::Polish),
        )
        .expect_err("parallel top-k tensors must have the same length");

        assert!(matches!(
            error,
            TranscriptionError::InvalidModelOutput(message)
                if message.contains("top-k id/logit length mismatch")
        ));
    }
}

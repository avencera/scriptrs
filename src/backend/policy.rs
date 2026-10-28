#[cfg(any(test, target_os = "macos"))]
use crate::constants::MAX_TOKENS_PER_STEP;
use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;
use crate::language::{LanguageSelection, ParakeetLanguage, Script, token_matches_script};
use crate::model_io::DecodedJointStep;
#[cfg(any(test, target_os = "macos"))]
use crate::model_io::{JointOutputs, decode_joint_outputs};
use crate::vocab::Vocabulary;

/// Greedy TDT token rule shared by every backend
///
/// Step backends hand each joint output to [`Self::decide`]. Device backends
/// run the same rule over full joint logits with [`Self::language_mask`];
/// [`decide_from_logits`] is the host reference for that device rule, and the
/// tests hold the two forms equal.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodePolicy<'a> {
    deployment: &'a ParakeetDeployment,
    vocabulary: &'a Vocabulary,
    language: LanguageSelection,
    keep_top_k: bool,
}

impl<'a> DecodePolicy<'a> {
    /// Resolve `language` against the deployment allowlist
    pub(crate) fn new(
        deployment: &'a ParakeetDeployment,
        vocabulary: &'a Vocabulary,
        language: LanguageSelection,
    ) -> Result<Self, TranscriptionError> {
        Ok(Self {
            deployment,
            vocabulary,
            language: deployment.select_language(language)?,
            keep_top_k: false,
        })
    }

    /// Also return the joint top-K candidates for every emitted token
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_top_k(mut self) -> Self {
        self.keep_top_k = true;
        self
    }

    pub(crate) fn blank_id(&self) -> usize {
        self.vocabulary.blank_id()
    }

    /// Non-blank tokens one encoder frame may emit before time is forced on
    ///
    /// Matches NeMo's label-looping `max_symbols`: the cap counts tokens
    /// emitted with duration 0 at one frame, and a blank always advances.
    #[cfg(any(test, target_os = "macos"))]
    pub(crate) const fn max_symbols_per_frame(&self) -> usize {
        MAX_TOKENS_PER_STEP
    }

    pub(crate) fn keep_top_k(&self) -> bool {
        self.keep_top_k
    }

    /// Apply the token rule to one step backend joint output
    #[cfg(any(test, target_os = "macos"))]
    pub(crate) fn decide(
        &self,
        outputs: &JointOutputs,
    ) -> Result<DecodedJointStep, TranscriptionError> {
        decode_joint_outputs(outputs, self.deployment, self.vocabulary, self.language)
    }

    /// Script filter for device backends, or `None` when no filter applies
    ///
    /// A filter applies when the deployment has a top-K language filter and the
    /// run selected one explicit language.
    #[cfg_attr(not(any(test, feature = "cuda")), allow(dead_code))]
    pub(crate) fn language_mask(&self) -> Option<LanguageMask> {
        let top_k = self.deployment.language_filter_top_k()?;
        let language = self.language.explicit()?;
        Some(LanguageMask::new(self.vocabulary, language, top_k))
    }
}

/// Per-token allow list for one explicit language, in vocabulary id order
///
/// A token is allowed when its text matches the language's script, the same
/// test the host top-K filter applies. The blank and ids past the tokenizer
/// table are never allowed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LanguageMask {
    script: Script,
    top_k: usize,
    allowed: Box<[bool]>,
}

impl LanguageMask {
    fn new(vocabulary: &Vocabulary, language: ParakeetLanguage, top_k: usize) -> Self {
        let script = language.script();
        let blank_id = vocabulary.blank_id();
        let allowed = (0..=blank_id)
            .map(|id| {
                id != blank_id
                    && vocabulary
                        .token(id)
                        .is_some_and(|token| token_matches_script(token, script))
            })
            .collect();
        Self {
            script,
            top_k,
            allowed,
        }
    }

    /// Candidate count the filter searches, ranked by logit
    pub(crate) fn top_k(&self) -> usize {
        self.top_k
    }

    /// Whether each token id may be chosen, indexed by id, with `blank_id + 1` entries
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "only the CUDA backend uploads the mask")
    )]
    pub(crate) fn allowed(&self) -> &[bool] {
        &self.allowed
    }
}

/// Host reference for the device token rule over one joint step's logits
///
/// `logits` holds the token logits for ids `0..=blank_id` followed by the
/// duration logits, the NeMo joint layout. Token and duration are first-max
/// argmaxes; the confidence is the token softmax at the chosen id. With a
/// language mask and a non-blank argmax, the highest-logit allowed token
/// among the top-K replaces it, and its confidence becomes the softmax over
/// the top-K logits only, which is what the published Core ML joint plus
/// [`decode_joint_outputs`] compute.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn decide_from_logits(
    policy: &DecodePolicy<'_>,
    logits: &[f32],
    mask: Option<&LanguageMask>,
) -> Result<DecodedJointStep, TranscriptionError> {
    let blank_id = policy.blank_id();
    let bins = policy.deployment.duration_bins();
    if logits.len() != blank_id + 1 + bins.len() {
        return Err(TranscriptionError::InvalidModelOutput(format!(
            "joint logits had {} values, expected {}",
            logits.len(),
            blank_id + 1 + bins.len()
        )));
    }
    if let Some(index) = logits.iter().position(|value| !value.is_finite()) {
        return Err(TranscriptionError::InvalidLanguageLogit {
            index,
            value: logits[index],
        });
    }

    let (token_logits, duration_logits) = logits.split_at(blank_id + 1);
    let duration_frames = policy
        .deployment
        .map_duration_bin(first_argmax(duration_logits))?;
    let token_id = first_argmax(token_logits);
    let mut step = DecodedJointStep {
        token_id,
        duration_frames,
        confidence: softmax_at(token_logits, token_logits.iter().copied(), token_id),
    };

    let Some(mask) = mask.filter(|_| token_id != blank_id) else {
        return Ok(step);
    };

    let top = top_k_ids(token_logits, mask.top_k());
    // first strict maximum in rank order, as the host filter picks it
    let mut filtered: Option<usize> = None;
    for id in top.iter().copied().filter(|id| mask.allowed[*id]) {
        if filtered.is_none_or(|best| token_logits[id] > token_logits[best]) {
            filtered = Some(id);
        }
    }
    if let Some(filtered) = filtered {
        step.token_id = filtered;
        step.confidence = softmax_at(
            token_logits,
            top.iter().map(|id| token_logits[*id]),
            filtered,
        );
    }

    Ok(step)
}

fn first_argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// Ids of the `k` largest logits, ties broken by lower id
fn top_k_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then(left.cmp(right))
    });
    ids.truncate(k);
    ids
}

/// Softmax of `logits[id]` over the `support` logits
fn softmax_at(logits: &[f32], support: impl Iterator<Item = f32> + Clone, id: usize) -> f32 {
    let max = support.clone().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = support.map(|value| (value - max).exp()).sum();
    ((logits[id] - max).exp() / sum).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::{DecodePolicy, decide_from_logits, top_k_ids};
    use crate::deployment::ParakeetDeployment;
    use crate::language::{LanguageSelection, ParakeetLanguage, TopKCandidates};
    use crate::model_io::JointOutputs;
    use crate::vocab::Vocabulary;

    /// Mixed-script vocabulary large enough that a 64-wide top-K excludes ids
    fn mixed_vocab() -> Vocabulary {
        let mut tokens: Vec<String> = vec!["<unk>".to_owned()];
        for index in 0..299 {
            tokens.push(match index % 3 {
                0 => format!("\u{2581}w{index}"),
                1 => format!("к{index}"),
                _ => format!("{index}"),
            });
        }
        tokens.push("<blk>".to_owned());
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        Vocabulary::from_tokens(&refs)
    }

    /// Joint outputs as the published Core ML joint computes them from logits
    fn coreml_joint(logits: &[f32], blank_id: usize, top_k: usize) -> JointOutputs {
        let (tokens, durations) = logits.split_at(blank_id + 1);
        let token_id = super::first_argmax(tokens);
        let ids = top_k_ids(tokens, top_k);
        JointOutputs {
            token_id,
            token_prob: super::softmax_at(tokens, tokens.iter().copied(), token_id),
            duration_bin: super::first_argmax(durations),
            top_k: Some(
                TopKCandidates::from_parallel(
                    ids.clone(),
                    ids.iter().map(|id| tokens[*id]).collect(),
                )
                .unwrap(),
            ),
        }
    }

    /// Deterministic logits with a few large spikes so filtering has to choose
    fn logits(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                if unit > 0.97 {
                    unit * 12.0
                } else {
                    unit * 4.0 - 2.0
                }
            })
            .collect()
    }

    #[test]
    fn device_rule_matches_the_host_top_k_filter() {
        let deployment = ParakeetDeployment::v3();
        let vocab = mixed_vocab();
        let blank_id = vocab.blank_id();
        let languages = [
            LanguageSelection::Auto,
            LanguageSelection::Language(ParakeetLanguage::Polish),
            LanguageSelection::Language(ParakeetLanguage::Russian),
            LanguageSelection::Language(ParakeetLanguage::Greek),
        ];

        let mut filtered_steps = 0;
        for language in languages {
            let policy = DecodePolicy::new(&deployment, &vocab, language).unwrap();
            let mask = policy.language_mask();
            assert_eq!(mask.is_some(), language != LanguageSelection::Auto);

            for seed in 0..400 {
                let logits = logits(seed, blank_id + 1 + deployment.duration_bins().len());
                let joint = coreml_joint(&logits, blank_id, 64);
                let host = policy.decide(&joint).unwrap();
                let device = decide_from_logits(&policy, &logits, mask.as_ref()).unwrap();

                assert_eq!(host.token_id, device.token_id, "seed {seed} {language:?}");
                assert_eq!(host.duration_frames, device.duration_frames);
                assert!((host.confidence - device.confidence).abs() < 1e-6);
                if host.token_id != joint.token_id {
                    filtered_steps += 1;
                }
            }
        }

        // the fixture must exercise the filter, not only the plain argmax path
        assert!(filtered_steps > 50, "only {filtered_steps} filtered steps");
    }
}

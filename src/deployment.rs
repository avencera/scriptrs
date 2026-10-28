use crate::constants::{
    DECODER_HIDDEN_SIZE, DECODER_LAYERS, DEFAULT_HARD_SAMPLE_MAX, DURATION_BINS,
    ENCODER_HIDDEN_SIZE, LANGUAGE_FILTER_TOP_K, MEL_HOP_SAMPLES, SAMPLE_RATE,
    SAMPLES_PER_ENCODER_FRAME,
};
use crate::error::TranscriptionError;
use crate::language::{LanguageSelection, ParakeetLanguage, V3_LANGUAGES};
use crate::vocab::Vocabulary;

const V2_ENCODER: &str = "parakeet-v2/encoder.mlmodelc";
const V2_DECODER: &str = "parakeet-v2/decoder.mlmodelc";
const V2_JOINT: &str = "parakeet-v2/joint-decision.mlmodelc";
const V2_VOCAB: &str = "parakeet-v2/vocab.txt";

const V3_ENCODER: &str = "parakeet-v3/encoder.mlmodelc";
const V3_DECODER: &str = "parakeet-v3/decoder.mlmodelc";
const V3_JOINT: &str = "parakeet-v3/joint-decision.mlmodelc";
const V3_VOCAB: &str = "parakeet-v3/parakeet_vocab.json";

pub(crate) const V2_BLANK_ID: usize = 1_024;
pub(crate) const V2_VOCAB_TOKEN_COUNT: usize = V2_BLANK_ID + 1;
pub(crate) const V3_BLANK_ID: usize = 8_192;
pub(crate) const V3_VOCAB_TOKEN_COUNT: usize = V3_BLANK_ID;
pub(crate) const V3_LAST_VOCAB_TOKEN: &str = "Œ";

/// Parakeet TDT runtime version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParakeetVersion {
    /// English-only TDT v2
    V2,
    /// Multilingual TDT v3
    V3,
}

/// Tokenizer file format owned by a deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VocabKind {
    /// SentencePiece `vocab.txt` used by v2
    SentencePieceTxt,
    /// JSON tokenizer used by v3 (`parakeet_vocab.json`)
    JsonTokenizer,
}

/// Relative model layout for one Parakeet version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelLayout {
    encoder: &'static str,
    decoder: &'static str,
    joint_decision: &'static str,
    vocab: &'static str,
    vocab_kind: VocabKind,
}

impl ModelLayout {
    /// Encoder `.mlmodelc` path relative to the models root
    pub const fn encoder(self) -> &'static str {
        self.encoder
    }

    /// Decoder `.mlmodelc` path relative to the models root
    pub const fn decoder(self) -> &'static str {
        self.decoder
    }

    /// Joint-decision `.mlmodelc` path relative to the models root
    pub const fn joint_decision(self) -> &'static str {
        self.joint_decision
    }

    /// Vocabulary file path relative to the models root
    pub const fn vocab(self) -> &'static str {
        self.vocab
    }

    /// Vocabulary parser required by this layout
    pub const fn vocab_kind(self) -> VocabKind {
        self.vocab_kind
    }
}

/// Encoder, decoder, and joint tensor names for one deployment
///
/// Live Core ML names for the unpublished v3 bundle are isolated here so tests
/// can inject [`crate::model_io::JointOutputs`] without loading a `.mlmodelc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelIo {
    encoder: EncoderIo,
    decoder: DecoderIo,
    joint: JointIo,
}

/// Encoder input and output names
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderIo {
    /// Mel feature input
    pub input_mel: &'static str,
    /// Mel length input
    pub input_length: &'static str,
    /// Encoder hidden output
    pub output_encoder: &'static str,
    /// Encoder length output
    pub output_length: &'static str,
}

/// Decoder LSTM input and output names
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderIo {
    /// Token target input
    pub targets: &'static str,
    /// Target length input
    pub target_length: &'static str,
    /// Hidden-state input
    pub hidden_in: &'static str,
    /// Cell-state input
    pub cell_in: &'static str,
    /// Decoder projection output
    pub decoder: &'static str,
    /// Hidden-state output
    pub hidden_out: &'static str,
    /// Cell-state output
    pub cell_out: &'static str,
}

/// Joint-decision input and output names
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JointIo {
    /// Encoder step input
    pub encoder_step: &'static str,
    /// Decoder step input
    pub decoder_step: &'static str,
    /// Argmax token id
    pub token_id: &'static str,
    /// Token probability
    pub token_prob: &'static str,
    /// Duration bin
    pub duration: &'static str,
    /// Optional top-K token ids used by v3 language filtering
    pub top_k_ids: Option<&'static str>,
    /// Optional top-K logits used by v3 language filtering
    pub top_k_logits: Option<&'static str>,
}

impl ModelIo {
    /// Encoder names
    pub const fn encoder(self) -> EncoderIo {
        self.encoder
    }

    /// Decoder names
    pub const fn decoder(self) -> DecoderIo {
        self.decoder
    }

    /// Joint-decision names
    pub const fn joint(self) -> JointIo {
        self.joint
    }

    const fn shared() -> Self {
        Self {
            encoder: EncoderIo {
                input_mel: "mel",
                input_length: "mel_length",
                output_encoder: "encoder",
                output_length: "encoder_length",
            },
            decoder: DecoderIo {
                targets: "targets",
                target_length: "target_length",
                hidden_in: "h_in",
                cell_in: "c_in",
                decoder: "decoder",
                hidden_out: "h_out",
                cell_out: "c_out",
            },
            joint: JointIo {
                encoder_step: "encoder_step",
                decoder_step: "decoder_step",
                token_id: "token_id",
                token_prob: "token_prob",
                duration: "duration",
                top_k_ids: None,
                top_k_logits: None,
            },
        }
    }

    const fn v2() -> Self {
        Self::shared()
    }

    const fn v3() -> Self {
        let mut io = Self::shared();
        io.joint.top_k_ids = Some("top_k_ids");
        io.joint.top_k_logits = Some("top_k_logits");
        io
    }
}

/// LSTM and encoder hidden sizes for one deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderDimensions {
    encoder_hidden: usize,
    decoder_hidden: usize,
    decoder_layers: usize,
}

impl DecoderDimensions {
    /// Encoder hidden size
    pub const fn encoder_hidden(self) -> usize {
        self.encoder_hidden
    }

    /// Decoder LSTM hidden size
    pub const fn decoder_hidden(self) -> usize {
        self.decoder_hidden
    }

    /// Decoder LSTM layer count
    pub const fn decoder_layers(self) -> usize {
        self.decoder_layers
    }

    const fn tdt_0_6b() -> Self {
        Self {
            encoder_hidden: ENCODER_HIDDEN_SIZE,
            decoder_hidden: DECODER_HIDDEN_SIZE,
            decoder_layers: DECODER_LAYERS,
        }
    }
}

/// Frontend numbers owned by a deployment
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontendSpec {
    sample_rate: usize,
    feature_size: usize,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    preemphasis: f32,
    samples_per_encoder_frame: usize,
}

impl FrontendSpec {
    /// Sample rate expected by the frontend
    pub const fn sample_rate(self) -> usize {
        self.sample_rate
    }

    /// Mel feature count
    pub const fn feature_size(self) -> usize {
        self.feature_size
    }

    /// FFT window size
    pub const fn n_fft(self) -> usize {
        self.n_fft
    }

    /// STFT window length
    pub const fn win_length(self) -> usize {
        self.win_length
    }

    /// STFT hop length
    pub const fn hop_length(self) -> usize {
        self.hop_length
    }

    /// Preemphasis coefficient
    pub const fn preemphasis(self) -> f32 {
        self.preemphasis
    }

    /// Encoder frame size in samples
    pub const fn samples_per_encoder_frame(self) -> usize {
        self.samples_per_encoder_frame
    }

    const fn parakeet_defaults() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            feature_size: 128,
            n_fft: 512,
            win_length: 400,
            hop_length: MEL_HOP_SAMPLES,
            preemphasis: 0.97,
            samples_per_encoder_frame: SAMPLES_PER_ENCODER_FRAME,
        }
    }
}

/// Typed owner of a Parakeet v2 or v3 runtime
///
/// Construct this with [`Self::v2`] or [`Self::v3`]. Mixed layouts, language
/// sets, and decoder sizes cannot be represented.
#[derive(Debug, Clone, PartialEq)]
pub struct ParakeetDeployment {
    version: ParakeetVersion,
    layout: ModelLayout,
    frontend: FrontendSpec,
    decoder: DecoderDimensions,
    languages: &'static [ParakeetLanguage],
    blank_id: usize,
    hard_sample_max: usize,
    duration_bins: &'static [usize],
    language_filter_top_k: Option<usize>,
    model_io: ModelIo,
}

impl ParakeetDeployment {
    /// English-only Parakeet TDT v2 deployment
    pub fn v2() -> Self {
        Self {
            version: ParakeetVersion::V2,
            layout: ModelLayout {
                encoder: V2_ENCODER,
                decoder: V2_DECODER,
                joint_decision: V2_JOINT,
                vocab: V2_VOCAB,
                vocab_kind: VocabKind::SentencePieceTxt,
            },
            frontend: FrontendSpec::parakeet_defaults(),
            decoder: DecoderDimensions::tdt_0_6b(),
            languages: &[ParakeetLanguage::English],
            blank_id: V2_BLANK_ID,
            hard_sample_max: DEFAULT_HARD_SAMPLE_MAX,
            duration_bins: &DURATION_BINS,
            language_filter_top_k: None,
            model_io: ModelIo::v2(),
        }
    }

    /// Multilingual Parakeet TDT v3 deployment
    pub fn v3() -> Self {
        Self {
            version: ParakeetVersion::V3,
            layout: ModelLayout {
                encoder: V3_ENCODER,
                decoder: V3_DECODER,
                joint_decision: V3_JOINT,
                vocab: V3_VOCAB,
                vocab_kind: VocabKind::JsonTokenizer,
            },
            frontend: FrontendSpec::parakeet_defaults(),
            decoder: DecoderDimensions::tdt_0_6b(),
            languages: V3_LANGUAGES,
            blank_id: V3_BLANK_ID,
            hard_sample_max: DEFAULT_HARD_SAMPLE_MAX,
            duration_bins: &DURATION_BINS,
            language_filter_top_k: Some(LANGUAGE_FILTER_TOP_K),
            model_io: ModelIo::v3(),
        }
    }

    /// Runtime version
    pub const fn version(&self) -> ParakeetVersion {
        self.version
    }

    /// Relative model layout
    pub const fn layout(&self) -> ModelLayout {
        self.layout
    }

    /// Frontend numbers
    pub const fn frontend(&self) -> FrontendSpec {
        self.frontend
    }

    /// Decoder and encoder hidden sizes
    pub const fn decoder_dimensions(&self) -> DecoderDimensions {
        self.decoder
    }

    /// Languages this deployment may emit or detect
    pub const fn languages(&self) -> &'static [ParakeetLanguage] {
        self.languages
    }

    /// RNNT blank token ID emitted by this deployment's joint model
    pub const fn blank_id(&self) -> usize {
        self.blank_id
    }

    /// Validate that a vocabulary matches this deployment's tokenizer contract
    pub fn validate_vocabulary(&self, vocabulary: &Vocabulary) -> Result<(), TranscriptionError> {
        if vocabulary.blank_id() != self.blank_id {
            return Err(TranscriptionError::InvalidVocabulary(format!(
                "{version:?} deployment requires RNNT blank id {expected}, found {found}",
                version = self.version,
                expected = self.blank_id,
                found = vocabulary.blank_id(),
            )));
        }

        let expected_token_count = match self.version {
            ParakeetVersion::V2 => V2_VOCAB_TOKEN_COUNT,
            ParakeetVersion::V3 => V3_VOCAB_TOKEN_COUNT,
        };
        if vocabulary.len() != expected_token_count {
            return Err(TranscriptionError::InvalidVocabulary(format!(
                "{version:?} deployment requires {expected} vocabulary tokens, found {found}",
                version = self.version,
                expected = expected_token_count,
                found = vocabulary.len(),
            )));
        }

        match self.version {
            ParakeetVersion::V2 => {
                if !matches!(vocabulary.token(self.blank_id), Some("<blk>" | "<blank>")) {
                    return Err(TranscriptionError::InvalidVocabulary(format!(
                        "v2 vocabulary must contain `<blk>` or `<blank>` at blank id {}",
                        self.blank_id
                    )));
                }
            }
            ParakeetVersion::V3 => {
                if vocabulary.token(self.blank_id).is_some() {
                    return Err(TranscriptionError::InvalidVocabulary(format!(
                        "v3 blank id {} must be one past the tokenizer table",
                        self.blank_id
                    )));
                }
                if vocabulary.token(self.blank_id - 1) != Some(V3_LAST_VOCAB_TOKEN) {
                    return Err(TranscriptionError::InvalidVocabulary(format!(
                        "v3 vocabulary token {} must be `{V3_LAST_VOCAB_TOKEN}`",
                        self.blank_id - 1
                    )));
                }
            }
        }

        Ok(())
    }

    /// Hard sample maximum owned by this deployment
    pub const fn hard_sample_max(&self) -> usize {
        self.hard_sample_max
    }

    /// Duration bin mapping owned by this deployment
    pub const fn duration_bins(&self) -> &'static [usize] {
        self.duration_bins
    }

    /// Joint top-K width that explicit-language filtering searches
    ///
    /// `None` when the deployment has no language filter (v2 is English-only).
    pub const fn language_filter_top_k(&self) -> Option<usize> {
        self.language_filter_top_k
    }

    /// Core ML tensor names for this deployment
    pub const fn model_io(&self) -> ModelIo {
        self.model_io
    }

    /// Vocabulary parser required by this deployment
    pub const fn vocab_kind(&self) -> VocabKind {
        self.layout.vocab_kind
    }

    /// Whether `language` is in this deployment's allowlist
    pub fn supports_language(&self, language: ParakeetLanguage) -> bool {
        self.languages.contains(&language)
    }

    /// Reject a language that is not in this deployment's allowlist
    pub fn select_language(
        &self,
        selection: LanguageSelection,
    ) -> Result<LanguageSelection, TranscriptionError> {
        let selection = selection.validate(self.languages, self.version)?;
        match (self.version, selection) {
            (ParakeetVersion::V2, LanguageSelection::Auto) => {
                Ok(LanguageSelection::Language(ParakeetLanguage::English))
            }
            (_, resolved) => Ok(resolved),
        }
    }

    /// Map a joint duration bin to encoder frames
    pub fn map_duration_bin(&self, bin: usize) -> Result<usize, TranscriptionError> {
        self.duration_bins
            .get(bin)
            .copied()
            .ok_or(TranscriptionError::InvalidDurationBin {
                bin,
                len: self.duration_bins.len(),
            })
    }

    pub(crate) fn sibling_vocab_path(&self) -> &'static str {
        match self.version {
            ParakeetVersion::V2 => V3_VOCAB,
            ParakeetVersion::V3 => V2_VOCAB,
        }
    }

    pub(crate) fn sibling_version(&self) -> ParakeetVersion {
        match self.version {
            ParakeetVersion::V2 => ParakeetVersion::V3,
            ParakeetVersion::V3 => ParakeetVersion::V2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ParakeetDeployment, ParakeetVersion, VocabKind};
    use crate::error::TranscriptionError;
    use crate::language::{LanguageSelection, ParakeetLanguage};

    #[test]
    fn v2_layout_keeps_existing_fixture_paths() {
        let deployment = ParakeetDeployment::v2();
        assert_eq!(deployment.version(), ParakeetVersion::V2);
        assert_eq!(deployment.blank_id(), 1_024);
        assert_eq!(
            deployment.layout().encoder(),
            "parakeet-v2/encoder.mlmodelc"
        );
        assert_eq!(
            deployment.layout().decoder(),
            "parakeet-v2/decoder.mlmodelc"
        );
        assert_eq!(
            deployment.layout().joint_decision(),
            "parakeet-v2/joint-decision.mlmodelc"
        );
        assert_eq!(deployment.layout().vocab(), "parakeet-v2/vocab.txt");
        assert_eq!(deployment.vocab_kind(), VocabKind::SentencePieceTxt);
        assert_eq!(deployment.languages(), &[ParakeetLanguage::English]);
        assert_eq!(deployment.hard_sample_max(), 240_000);
        assert_eq!(deployment.decoder_dimensions().encoder_hidden(), 1_024);
        assert_eq!(deployment.decoder_dimensions().decoder_hidden(), 640);
        assert_eq!(deployment.decoder_dimensions().decoder_layers(), 2);
        assert!(deployment.model_io().joint().top_k_ids.is_none());
    }

    #[test]
    fn v3_layout_uses_json_tokenizer_and_language_set() {
        let deployment = ParakeetDeployment::v3();
        assert_eq!(deployment.version(), ParakeetVersion::V3);
        assert_eq!(deployment.blank_id(), 8_192);
        assert_eq!(
            deployment.layout().encoder(),
            "parakeet-v3/encoder.mlmodelc"
        );
        assert_eq!(
            deployment.layout().vocab(),
            "parakeet-v3/parakeet_vocab.json"
        );
        assert_eq!(deployment.vocab_kind(), VocabKind::JsonTokenizer);
        assert_eq!(deployment.hard_sample_max(), 240_000);
        assert!(deployment.supports_language(ParakeetLanguage::Polish));
        assert!(deployment.supports_language(ParakeetLanguage::English));
        assert_eq!(deployment.model_io().joint().top_k_ids, Some("top_k_ids"));
        assert_eq!(
            deployment.model_io().joint().top_k_logits,
            Some("top_k_logits")
        );
    }

    #[test]
    fn v2_rejects_non_english_selection() {
        let error = ParakeetDeployment::v2()
            .select_language(LanguageSelection::Language(ParakeetLanguage::French))
            .expect_err("v2 is english-only");
        match error {
            TranscriptionError::UnsupportedLanguage { language, version } => {
                assert_eq!(language, "fr");
                assert_eq!(version, ParakeetVersion::V2);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn duration_bins_are_owned_by_the_deployment() {
        let deployment = ParakeetDeployment::v3();
        assert_eq!(deployment.map_duration_bin(0).unwrap(), 0);
        assert_eq!(deployment.map_duration_bin(4).unwrap(), 4);
        let error = deployment
            .map_duration_bin(5)
            .expect_err("bin 5 is outside the mapping");
        match error {
            TranscriptionError::InvalidDurationBin { bin, len } => {
                assert_eq!(bin, 5);
                assert_eq!(len, 5);
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}

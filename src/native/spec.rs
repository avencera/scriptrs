use serde_json::Value;

use crate::constants::MAX_TOKENS_PER_STEP;
use crate::deployment::{ParakeetDeployment, ParakeetVersion};
use crate::error::TranscriptionError;

/// Parakeet TDT architecture read from a native weight file's config
///
/// The native engines size every buffer and loop from this, not from crate
/// constants. Parsing rejects NeMo options the engines do not implement, so a
/// checkpoint with biased linears or local attention fails at load instead of
/// computing a wrong transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParakeetSpec {
    pub version: ParakeetVersion,
    pub frontend: FrontendConfig,
    pub encoder: EncoderSpec,
    pub predictor: PredictorSpec,
    pub joint: JointSpec,
    /// Encoder frames each duration logit advances
    pub durations: Vec<usize>,
    /// NeMo `max_symbols`: tokens one frame may emit with duration 0
    pub max_symbols: usize,
}

/// STFT geometry the checkpoint's mel filterbank was trained with
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrontendConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
}

/// FastConformer encoder with 8x depthwise-striding subsampling and rel-pos attention
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EncoderSpec {
    /// Mel features per frame
    pub features: usize,
    pub layers: usize,
    pub d_model: usize,
    pub heads: usize,
    /// Hidden width of each feed-forward module
    pub ff_hidden: usize,
    /// Depthwise convolution kernel in the conv module
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
    /// Longest sequence the relative position table covers before it is extended
    pub pos_emb_max_len: usize,
}

/// LSTM prediction network
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PredictorSpec {
    /// Tokenizer size; the blank id equals it and embeds to zeros
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
}

/// Joint network
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JointSpec {
    pub hidden: usize,
    /// Token logits (vocabulary plus blank) followed by duration logits
    pub outputs: usize,
}

impl ParakeetSpec {
    /// Parse the `config` metadata entry written by `export_native.py`
    pub(crate) fn from_config_json(config: &str) -> Result<Self, TranscriptionError> {
        let root: Value = serde_json::from_str(config)
            .map_err(|error| invalid(format!("config is not JSON: {error}")))?;
        let config = Config(&root);

        let version = match config.str("version")? {
            "v2" => ParakeetVersion::V2,
            "v3" => ParakeetVersion::V3,
            other => return Err(invalid(format!("unknown Parakeet version `{other}`"))),
        };

        let pre = config.object("preprocessor")?;
        pre.expect_str("normalize", "per_feature")?;
        pre.expect_str("window", "hann")?;
        pre.expect_bool("log", true)?;
        pre.expect_usize("frame_splicing", 1)?;
        pre.expect_usize("pad_to", 0)?;
        let sample_rate = pre.usize("sample_rate")?;
        let frontend = FrontendConfig {
            sample_rate,
            n_fft: pre.usize("n_fft")?,
            win_length: pre.seconds_as_samples("window_size", sample_rate)?,
            hop_length: pre.seconds_as_samples("window_stride", sample_rate)?,
        };

        let enc = config.object("encoder")?;
        enc.expect_str("subsampling", "dw_striding")?;
        enc.expect_str("self_attention_model", "rel_pos")?;
        enc.expect_str("conv_norm_type", "batch_norm")?;
        enc.expect_bool("use_bias", false)?;
        enc.expect_bool("xscaling", false)?;
        enc.expect_bool("untie_biases", true)?;
        let context = enc.array("att_context_size")?;
        if context.len() != 2 || context.iter().any(|value| value.as_i64() != Some(-1)) {
            return Err(invalid(format!(
                "encoder att_context_size {context:?} is not full context [-1, -1]"
            )));
        }

        let d_model = enc.usize("d_model")?;
        let heads = enc.usize("n_heads")?;
        if heads == 0 || d_model % heads != 0 {
            return Err(invalid(format!(
                "d_model {d_model} does not split into {heads} heads"
            )));
        }
        let features = enc.usize("feat_in")?;
        if features != pre.usize("features")? {
            return Err(invalid(
                "encoder feat_in disagrees with the preprocessor".to_owned(),
            ));
        }
        let encoder = EncoderSpec {
            features,
            layers: enc.usize("n_layers")?,
            d_model,
            heads,
            ff_hidden: d_model * enc.usize("ff_expansion_factor")?,
            conv_kernel: enc.usize("conv_kernel_size")?,
            subsampling_factor: enc.usize("subsampling_factor")?,
            subsampling_channels: enc.usize("subsampling_conv_channels")?,
            pos_emb_max_len: enc.usize("pos_emb_max_len")?,
        };
        if encoder.subsampling_factor != 8 {
            return Err(invalid(format!(
                "subsampling factor {} is not the 8x dw_striding stack",
                encoder.subsampling_factor
            )));
        }
        if encoder.conv_kernel % 2 == 0 {
            return Err(invalid(format!(
                "conv kernel {} has no centered 'same' padding",
                encoder.conv_kernel
            )));
        }

        let dec = config.object("decoder")?;
        dec.expect_bool("blank_as_pad", true)?;
        let predictor = PredictorSpec {
            vocab_size: dec.usize("vocab_size")?,
            hidden: dec.usize("pred_hidden")?,
            layers: dec.usize("pred_rnn_layers")?,
        };

        let joint_config = config.object("joint")?;
        joint_config.expect_str("activation", "relu")?;
        let decoding = config.object("decoding")?;
        decoding.expect_str("model_type", "tdt")?;
        let durations = decoding
            .array("durations")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .map(|value| value as usize)
                    .ok_or_else(|| invalid(format!("duration {value} is not a frame count")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let classes = joint_config.usize("num_classes")?;
        if classes != predictor.vocab_size {
            return Err(invalid(format!(
                "joint has {classes} classes for a {} token vocabulary",
                predictor.vocab_size
            )));
        }
        let extra = joint_config.usize("num_extra_outputs")?;
        if extra != durations.len() {
            return Err(invalid(format!(
                "joint has {extra} extra outputs for {} durations",
                durations.len()
            )));
        }
        let joint = JointSpec {
            hidden: joint_config.usize("joint_hidden")?,
            outputs: classes + 1 + extra,
        };

        Ok(Self {
            version,
            frontend,
            encoder,
            predictor,
            joint,
            durations,
            max_symbols: decoding.usize("max_symbols")?,
        })
    }

    /// Reject weights that belong to another deployment or break its contract
    pub(crate) fn validate_for(
        &self,
        deployment: &ParakeetDeployment,
    ) -> Result<(), TranscriptionError> {
        if self.version != deployment.version() {
            return Err(TranscriptionError::LayoutMismatch {
                expected: deployment.version(),
                found: self.version,
            });
        }

        let dims = deployment.decoder_dimensions();
        let frontend = deployment.frontend();
        let checks = [
            (
                "sample rate",
                self.frontend.sample_rate,
                frontend.sample_rate(),
            ),
            ("FFT size", self.frontend.n_fft, frontend.n_fft()),
            (
                "window length",
                self.frontend.win_length,
                frontend.win_length(),
            ),
            (
                "hop length",
                self.frontend.hop_length,
                frontend.hop_length(),
            ),
            ("blank id", self.predictor.vocab_size, deployment.blank_id()),
            (
                "encoder hidden",
                self.encoder.d_model,
                dims.encoder_hidden(),
            ),
            (
                "predictor hidden",
                self.predictor.hidden,
                dims.decoder_hidden(),
            ),
            (
                "predictor layers",
                self.predictor.layers,
                dims.decoder_layers(),
            ),
            (
                "mel features",
                self.encoder.features,
                frontend.feature_size(),
            ),
            ("max symbols", self.max_symbols, MAX_TOKENS_PER_STEP),
        ];
        for (name, found, expected) in checks {
            if found != expected {
                return Err(invalid(format!(
                    "{name} is {found} in the weights but {expected} in the {:?} deployment",
                    deployment.version()
                )));
            }
        }

        if self.durations != deployment.duration_bins() {
            return Err(invalid(format!(
                "durations {:?} differ from the deployment's {:?}",
                self.durations,
                deployment.duration_bins()
            )));
        }

        Ok(())
    }
}

fn invalid(message: String) -> TranscriptionError {
    TranscriptionError::InvalidWeights(message)
}

/// Typed access to one JSON object in the exported config
#[derive(Clone, Copy)]
struct Config<'a>(&'a Value);

impl<'a> Config<'a> {
    fn field(self, key: &str) -> Result<&'a Value, TranscriptionError> {
        self.0
            .get(key)
            .ok_or_else(|| invalid(format!("config is missing `{key}`")))
    }

    fn object(self, key: &str) -> Result<Config<'a>, TranscriptionError> {
        let value = self.field(key)?;
        if !value.is_object() {
            return Err(invalid(format!("config `{key}` is not an object")));
        }
        Ok(Config(value))
    }

    fn str(self, key: &str) -> Result<&'a str, TranscriptionError> {
        self.field(key)?
            .as_str()
            .ok_or_else(|| invalid(format!("config `{key}` is not a string")))
    }

    fn usize(self, key: &str) -> Result<usize, TranscriptionError> {
        self.field(key)?
            .as_u64()
            .map(|value| value as usize)
            .ok_or_else(|| invalid(format!("config `{key}` is not a non-negative integer")))
    }

    /// NeMo stores STFT windows in seconds; anything off the sample grid is rejected
    fn seconds_as_samples(
        self,
        key: &str,
        sample_rate: usize,
    ) -> Result<usize, TranscriptionError> {
        let seconds = self
            .field(key)?
            .as_f64()
            .ok_or_else(|| invalid(format!("config `{key}` is not a number")))?;
        let samples = seconds * sample_rate as f64;
        let rounded = samples.round();
        if rounded < 1.0 || (samples - rounded).abs() > 1e-6 {
            return Err(invalid(format!(
                "config `{key}` {seconds}s is not a whole number of samples"
            )));
        }
        Ok(rounded as usize)
    }

    fn array(self, key: &str) -> Result<&'a [Value], TranscriptionError> {
        self.field(key)?
            .as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| invalid(format!("config `{key}` is not an array")))
    }

    fn expect_str(self, key: &str, expected: &str) -> Result<(), TranscriptionError> {
        let found = self.str(key)?;
        if found != expected {
            return Err(invalid(format!(
                "config `{key}` is `{found}`; the native engine implements `{expected}`"
            )));
        }
        Ok(())
    }

    fn expect_bool(self, key: &str, expected: bool) -> Result<(), TranscriptionError> {
        let found = self
            .field(key)?
            .as_bool()
            .ok_or_else(|| invalid(format!("config `{key}` is not a boolean")))?;
        if found != expected {
            return Err(invalid(format!(
                "config `{key}` is {found}; the native engine implements {expected}"
            )));
        }
        Ok(())
    }

    fn expect_usize(self, key: &str, expected: usize) -> Result<(), TranscriptionError> {
        let found = self.usize(key)?;
        if found != expected {
            return Err(invalid(format!(
                "config `{key}` is {found}; the native engine implements {expected}"
            )));
        }
        Ok(())
    }
}

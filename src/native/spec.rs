use std::collections::BTreeMap;

use serde_json::Value;

use crate::constants::MAX_TOKENS_PER_STEP;
use crate::deployment::{ParakeetDeployment, ParakeetVersion};
use crate::error::TranscriptionError;

/// Parakeet TDT architecture read from a native weight file's config
///
/// The native engines size every buffer and loop from this, not from crate
/// constants. Parsing rejects NeMo options the engines do not implement, so a
/// checkpoint with biased linears or local attention fails at load instead of
/// computing a wrong transcript
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParakeetSpec {
    pub version: ParakeetVersion,
    pub frontend: FrontendConfig,
    pub encoder: EncoderSpec,
    pub transducer: TransducerSpec,
}

/// STFT geometry the checkpoint's mel filterbank was trained with
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FrontendConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub preemphasis: f32,
}

/// FastConformer encoder with 8x depthwise-striding subsampling and rel-pos attention
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EncoderSpec {
    /// Causal subsampling and depthwise convolution: each output frame sees only the
    /// frames up to it, as cache-aware streaming models are trained
    pub causal: bool,
    pub conv_norm: ConvNorm,
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

/// The normalization between the conv module's depthwise convolution and its SiLU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvNorm {
    /// `BatchNorm1d` with running statistics, folded at load
    Batch,
    /// `LayerNorm` over the channels of each frame
    Layer,
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

/// The predictor, joint and greedy rule a transducer decoder is built from
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransducerSpec {
    pub predictor: PredictorSpec,
    pub joint: JointSpec,
    /// Width of the encoder frames the joint reads
    pub encoder_hidden: usize,
    /// Encoder frames each duration logit advances (TDT); empty for RNNT, which
    /// moves to the next frame on a blank or after `max_symbols` tokens
    pub durations: Vec<usize>,
    /// NeMo `max_symbols`: tokens one frame may emit before the cursor moves on
    pub max_symbols: usize,
}

/// A cache-aware streaming FastConformer RNNT read from a native weight file's config
///
/// Each layer attends to `left_context` cached frames plus its current chunk of
/// `lookahead + 1` frames (NeMo's `chunked_limited` attention). Subsampling and the
/// depthwise convolutions are causal, so a chunk needs no audio past its own end
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StreamingSpec {
    /// The checkpoint's name, such as `nemotron-speech-streaming-en-0.6b`
    pub model: String,
    pub frontend: FrontendConfig,
    pub encoder: EncoderSpec,
    pub transducer: TransducerSpec,
    /// Encoder frames of left context each layer caches, a nonzero multiple of
    /// every chunk length
    pub left_context: usize,
    /// Lookaheads (right contexts) the model was trained with, in encoder frames
    pub lookaheads: Vec<usize>,
    /// The language prompt of a multilingual checkpoint
    pub prompt: Option<PromptSpec>,
}

/// Language prompt table: the encoder output is joined with a one-hot prompt and
/// projected back to `d_model` before the joint
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptSpec {
    /// Width of the one-hot prompt
    pub prompts: usize,
    /// Prompt index of each language tag, such as `en-US`
    pub languages: BTreeMap<String, usize>,
}

impl ParakeetSpec {
    /// Parse the `config` metadata entry written by `export_native.py`
    pub(crate) fn from_config_json(config: &str) -> Result<Self, TranscriptionError> {
        let root = parse_json(config)?;
        let config = Config(&root);

        let version = match config.str("version")? {
            "v2" => ParakeetVersion::V2,
            "v3" => ParakeetVersion::V3,
            other => return Err(invalid(format!("unknown Parakeet version `{other}`"))),
        };

        let frontend = frontend(config, "per_feature")?;
        let enc = config.object("encoder")?;
        enc.expect_str("conv_norm_type", "batch_norm")?;
        enc.expect_bool("causal_downsampling", false)?;
        // null resolves to the symmetric `(kernel - 1) / 2` padding the conv module uses
        enc.expect_null("conv_context_size")?;
        let context = enc.array("att_context_size")?;
        if context.len() != 2 || context.iter().any(|value| value.as_i64() != Some(-1)) {
            return Err(invalid(format!(
                "encoder att_context_size {context:?} is not full context [-1, -1]"
            )));
        }
        let encoder = encoder(config, ConvNorm::Batch, false)?;
        if encoder.conv_kernel % 2 == 0 {
            return Err(invalid(format!(
                "conv kernel {} has no centered 'same' padding",
                encoder.conv_kernel
            )));
        }

        Ok(Self {
            version,
            frontend,
            encoder,
            transducer: transducer(config, "tdt", encoder.d_model)?,
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
        let transducer = &self.transducer;
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
            (
                "blank id",
                transducer.predictor.vocab_size,
                deployment.blank_id(),
            ),
            (
                "encoder hidden",
                self.encoder.d_model,
                dims.encoder_hidden(),
            ),
            (
                "predictor hidden",
                transducer.predictor.hidden,
                dims.decoder_hidden(),
            ),
            (
                "predictor layers",
                transducer.predictor.layers,
                dims.decoder_layers(),
            ),
            (
                "mel features",
                self.encoder.features,
                frontend.feature_size(),
            ),
            ("max symbols", transducer.max_symbols, MAX_TOKENS_PER_STEP),
        ];
        for (name, found, expected) in checks {
            if found != expected {
                return Err(invalid(format!(
                    "{name} is {found} in the weights but {expected} in the {:?} deployment",
                    deployment.version()
                )));
            }
        }

        if self.frontend.preemphasis != frontend.preemphasis() {
            return Err(invalid(format!(
                "preemphasis is {} in the weights but {} in the {:?} deployment",
                self.frontend.preemphasis,
                frontend.preemphasis(),
                deployment.version()
            )));
        }

        if transducer.durations != deployment.duration_bins() {
            return Err(invalid(format!(
                "durations {:?} differ from the deployment's {:?}",
                transducer.durations,
                deployment.duration_bins()
            )));
        }

        Ok(())
    }
}

impl StreamingSpec {
    /// Parse the `config` metadata entry `export_native.py` writes for a streaming
    /// checkpoint
    pub(crate) fn from_config_json(config: &str) -> Result<Self, TranscriptionError> {
        let root = parse_json(config)?;
        let config = Config(&root);
        let model = config.str("model")?.to_owned();

        // streaming features are not normalized per utterance, which a stream could
        // not do before its end
        let frontend = frontend(config, "NA")?;
        let enc = config.object("encoder")?;
        enc.expect_str("conv_norm_type", "layer_norm")?;
        enc.expect_bool("causal_downsampling", true)?;
        // "causal" pads the depthwise convolution with `kernel - 1` past frames only
        enc.expect_str("conv_context_size", "causal")?;
        enc.expect_str("att_context_style", "chunked_limited")?;
        let encoder = encoder(config, ConvNorm::Layer, true)?;

        let mut left_context = None;
        let mut lookaheads = Vec::new();
        for context in enc.array("att_context_size")? {
            let pair = context
                .as_array()
                .filter(|pair| pair.len() == 2)
                .and_then(|pair| Some((pair[0].as_u64()?, pair[1].as_u64()?)))
                .ok_or_else(|| {
                    invalid(format!(
                        "att_context_size entry {context} is not a [left, right] frame pair"
                    ))
                })?;
            let (left, lookahead) = (pair.0 as usize, pair.1 as usize);
            let chunk = lookahead.checked_add(1).ok_or_else(|| {
                invalid(format!(
                    "att_context_size lookahead {lookahead} is too large"
                ))
            })?;
            if *left_context.get_or_insert(left) != left {
                return Err(invalid(
                    "att_context_size entries disagree on the left context".to_owned(),
                ));
            }
            // chunked_limited attention sees whole chunks of `lookahead + 1` frames
            // to its left, so the cache is a whole number of chunks, and at least
            // one: the cache is a ring indexed modulo its length
            if left == 0 || left % chunk != 0 {
                return Err(invalid(format!(
                    "left context {left} is not a whole, nonzero number of \
                     {chunk}-frame chunks"
                )));
            }
            lookaheads.push(lookahead);
        }
        let left_context =
            left_context.ok_or_else(|| invalid("att_context_size lists no contexts".to_owned()))?;

        let transducer = transducer(config, "rnnt", encoder.d_model)?;
        let prompt = match root.get("prompt") {
            Some(_) => Some(prompt(config.object("prompt")?)?),
            None => None,
        };

        Ok(Self {
            model,
            frontend,
            encoder,
            transducer,
            left_context,
            lookaheads,
            prompt,
        })
    }
}

fn parse_json(config: &str) -> Result<Value, TranscriptionError> {
    serde_json::from_str(config).map_err(|error| invalid(format!("config is not JSON: {error}")))
}

/// The preprocessor settings both families share, with the feature normalization
/// each expects
fn frontend(config: Config<'_>, normalize: &str) -> Result<FrontendConfig, TranscriptionError> {
    let pre = config.object("preprocessor")?;
    pre.expect_str("normalize", normalize)?;
    pre.expect_str("window", "hann")?;
    pre.expect_bool("log", true)?;
    pre.expect_usize("frame_splicing", 1)?;
    pre.expect_usize("pad_to", 0)?;
    pre.expect_f32("pad_value", 0.0)?;
    // the featurizer squares magnitudes, adds 2^-24 before the log and builds a
    // slaney filterbank from 0 Hz to Nyquist with centered frames
    pre.expect_f32("mag_power", 2.0)?;
    pre.expect_str("log_zero_guard_type", "add")?;
    pre.expect_f32("log_zero_guard_value", 2f32.powi(-24))?;
    pre.expect_bool("exact_pad", false)?;
    pre.expect_f32("lowfreq", 0.0)?;
    pre.expect_null("highfreq")?;
    pre.expect_str("mel_norm", "slaney")?;
    let sample_rate = pre.usize("sample_rate")?;
    Ok(FrontendConfig {
        sample_rate,
        n_fft: pre.usize("n_fft")?,
        win_length: pre.seconds_as_samples("window_size", sample_rate)?,
        hop_length: pre.seconds_as_samples("window_stride", sample_rate)?,
        preemphasis: pre.f64("preemph")? as f32,
    })
}

/// The encoder settings both families share; the caller checks the attention
/// context, conv padding and conv norm its engine implements
fn encoder(
    config: Config<'_>,
    conv_norm: ConvNorm,
    causal: bool,
) -> Result<EncoderSpec, TranscriptionError> {
    let enc = config.object("encoder")?;
    enc.expect_str("subsampling", "dw_striding")?;
    enc.expect_str("self_attention_model", "rel_pos")?;
    enc.expect_bool("use_bias", false)?;
    enc.expect_bool("xscaling", false)?;
    enc.expect_bool("untie_biases", true)?;
    enc.expect_i64("feat_out", -1)?;
    enc.expect_usize("global_tokens", 0)?;
    enc.expect_null("reduction")?;

    let d_model = enc.usize("d_model")?;
    let heads = enc.usize("n_heads")?;
    if heads == 0 || d_model % heads != 0 {
        return Err(invalid(format!(
            "d_model {d_model} does not split into {heads} heads"
        )));
    }
    let features = enc.usize("feat_in")?;
    if features != config.object("preprocessor")?.usize("features")? {
        return Err(invalid(
            "encoder feat_in disagrees with the preprocessor".to_owned(),
        ));
    }
    let encoder = EncoderSpec {
        causal,
        conv_norm,
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
    Ok(encoder)
}

/// The predictor, joint and decoding rule, for `model_type` `tdt` or `rnnt`
fn transducer(
    config: Config<'_>,
    model_type: &str,
    encoder_hidden: usize,
) -> Result<TransducerSpec, TranscriptionError> {
    let dec = config.object("decoder")?;
    dec.expect_bool("blank_as_pad", true)?;
    dec.expect_null("normalization_mode")?;
    let predictor = PredictorSpec {
        vocab_size: dec.usize("vocab_size")?,
        hidden: dec.usize("pred_hidden")?,
        layers: dec.usize("pred_rnn_layers")?,
    };
    // a positive rnn_hidden_size other than pred_hidden adds an LSTM projection
    let rnn_hidden = dec.i64("rnn_hidden_size")?;
    if rnn_hidden > 0 && rnn_hidden as usize != predictor.hidden {
        return Err(invalid(format!(
            "decoder rnn_hidden_size {rnn_hidden} projects the {} wide LSTM",
            predictor.hidden
        )));
    }

    let joint_config = config.object("joint")?;
    joint_config.expect_str("activation", "relu")?;
    let decoding = config.object("decoding")?;
    decoding.expect_str("model_type", model_type)?;
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
    // an RNNT joint has no duration head, and a TDT joint without one is malformed
    if (model_type == "rnnt") != durations.is_empty() {
        return Err(invalid(format!(
            "a {model_type} model with durations {durations:?}"
        )));
    }
    let classes = joint_config.usize("num_classes")?;
    if classes != predictor.vocab_size {
        return Err(invalid(format!(
            "joint has {classes} classes for a {} token vocabulary",
            predictor.vocab_size
        )));
    }
    let max_symbols = decoding.usize("max_symbols")?;
    // the decoder gives each frame `max_symbols` output slots, so zero leaves no
    // room for a token
    if max_symbols == 0 {
        return Err(invalid("decoding max_symbols is 0".to_owned()));
    }
    let extra = joint_config.usize("num_extra_outputs")?;
    if extra != durations.len() {
        return Err(invalid(format!(
            "joint has {extra} extra outputs for {} durations",
            durations.len()
        )));
    }

    Ok(TransducerSpec {
        predictor,
        joint: JointSpec {
            hidden: joint_config.usize("joint_hidden")?,
            outputs: classes + 1 + extra,
        },
        encoder_hidden,
        durations,
        max_symbols,
    })
}

fn prompt(config: Config<'_>) -> Result<PromptSpec, TranscriptionError> {
    let prompts = config.usize("num_prompts")?;
    let dictionary = config
        .field("dictionary")?
        .as_object()
        .ok_or_else(|| invalid("prompt dictionary is not an object".to_owned()))?;
    let languages = dictionary
        .iter()
        .map(|(language, index)| match index.as_u64() {
            Some(index) if (index as usize) < prompts => Ok((language.clone(), index as usize)),
            _ => Err(invalid(format!(
                "prompt `{language}` is {index}, not one of {prompts} prompts"
            ))),
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    Ok(PromptSpec { prompts, languages })
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

    fn i64(self, key: &str) -> Result<i64, TranscriptionError> {
        self.field(key)?
            .as_i64()
            .ok_or_else(|| invalid(format!("config `{key}` is not an integer")))
    }

    fn f64(self, key: &str) -> Result<f64, TranscriptionError> {
        self.field(key)?
            .as_f64()
            .ok_or_else(|| invalid(format!("config `{key}` is not a number")))
    }

    /// NeMo stores STFT windows in seconds; anything off the sample grid is rejected
    fn seconds_as_samples(
        self,
        key: &str,
        sample_rate: usize,
    ) -> Result<usize, TranscriptionError> {
        let seconds = self.f64(key)?;
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

    fn expect_i64(self, key: &str, expected: i64) -> Result<(), TranscriptionError> {
        let found = self.i64(key)?;
        if found != expected {
            return Err(invalid(format!(
                "config `{key}` is {found}; the native engine implements {expected}"
            )));
        }
        Ok(())
    }

    /// Exact match at the frontend's f32 precision; serde_json's float parsing can
    /// land one f64 ulp off the value Python wrote
    fn expect_f32(self, key: &str, expected: f32) -> Result<(), TranscriptionError> {
        let found = self.f64(key)?;
        if found as f32 != expected {
            return Err(invalid(format!(
                "config `{key}` is {found}; the native engine implements {expected}"
            )));
        }
        Ok(())
    }

    fn expect_null(self, key: &str) -> Result<(), TranscriptionError> {
        let found = self.field(key)?;
        if !found.is_null() {
            return Err(invalid(format!(
                "config `{key}` is {found}; the native engine implements only null"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ConvNorm, ParakeetSpec, StreamingSpec};
    use crate::deployment::ParakeetDeployment;
    use crate::error::TranscriptionError;

    /// The config `export_native.py` writes for parakeet-tdt-0.6b-v3
    fn v3_config() -> Value {
        json!({
            "version": "v3",
            "preprocessor": {
                "sample_rate": 16000, "normalize": "per_feature", "window_size": 0.025,
                "window_stride": 0.01, "window": "hann", "features": 128, "n_fft": 512,
                "log": true, "frame_splicing": 1, "pad_to": 0, "pad_value": 0.0,
                "preemph": 0.97, "mag_power": 2.0, "log_zero_guard_type": "add",
                "log_zero_guard_value": 5.960464477539063e-08, "exact_pad": false,
                "lowfreq": 0, "highfreq": null, "mel_norm": "slaney"
            },
            "encoder": {
                "feat_in": 128, "feat_out": -1, "n_layers": 24, "d_model": 1024,
                "use_bias": false, "subsampling": "dw_striding", "subsampling_factor": 8,
                "subsampling_conv_channels": 256, "causal_downsampling": false,
                "ff_expansion_factor": 4, "self_attention_model": "rel_pos", "n_heads": 8,
                "att_context_size": [-1, -1], "xscaling": false, "untie_biases": true,
                "pos_emb_max_len": 5000, "conv_kernel_size": 9, "conv_norm_type": "batch_norm",
                "conv_context_size": null, "reduction": null, "global_tokens": 0
            },
            "decoder": {
                "vocab_size": 8192, "blank_as_pad": true, "normalization_mode": null,
                "pred_hidden": 640, "pred_rnn_layers": 2, "rnn_hidden_size": -1
            },
            "joint": {
                "num_classes": 8192, "num_extra_outputs": 5, "joint_hidden": 640,
                "activation": "relu"
            },
            "decoding": { "model_type": "tdt", "durations": [0, 1, 2, 3, 4], "max_symbols": 10 }
        })
    }

    fn load(config: &Value) -> Result<ParakeetSpec, TranscriptionError> {
        let spec = ParakeetSpec::from_config_json(&config.to_string())?;
        spec.validate_for(&ParakeetDeployment::v3())?;
        Ok(spec)
    }

    #[test]
    fn exported_v3_config_loads_for_v3() {
        let spec = load(&v3_config()).unwrap();
        assert_eq!(spec.encoder.layers, 24);
        assert_eq!(spec.frontend.win_length, 400);
        assert_eq!(spec.transducer.joint.outputs, 8192 + 1 + 5);
        assert_eq!(spec.encoder.conv_norm, ConvNorm::Batch);
        assert!(!spec.encoder.causal);
    }

    #[test]
    fn settings_that_change_the_numbers_are_rejected() {
        let changes = [
            ("preprocessor", "preemph", json!(0.0)),
            ("preprocessor", "mag_power", json!(1.0)),
            ("preprocessor", "log_zero_guard_type", json!("clamp")),
            ("preprocessor", "log_zero_guard_value", json!(1e-5)),
            ("preprocessor", "exact_pad", json!(true)),
            ("preprocessor", "lowfreq", json!(20)),
            ("preprocessor", "highfreq", json!(7600)),
            ("preprocessor", "mel_norm", json!(null)),
            ("preprocessor", "pad_value", json!(-1.0)),
            ("preprocessor", "pad_to", json!(16)),
            ("encoder", "causal_downsampling", json!(true)),
            ("encoder", "feat_out", json!(512)),
            ("encoder", "conv_context_size", json!([8, 0])),
            ("encoder", "reduction", json!("pooling")),
            ("encoder", "global_tokens", json!(1)),
            ("encoder", "att_context_size", json!([128, 128])),
            ("encoder", "use_bias", json!(true)),
            ("decoder", "normalization_mode", json!("layer")),
            ("decoder", "rnn_hidden_size", json!(1024)),
            ("joint", "activation", json!("tanh")),
            ("decoding", "max_symbols", json!(5)),
            ("decoding", "model_type", json!("rnnt")),
        ];
        for (section, key, value) in changes {
            let mut config = v3_config();
            config[section][key] = value.clone();
            assert!(
                matches!(load(&config), Err(TranscriptionError::InvalidWeights(_))),
                "{section}.{key} = {value} was accepted"
            );

            let mut config = v3_config();
            config[section].as_object_mut().unwrap().remove(key);
            assert!(
                load(&config).is_err(),
                "missing {section}.{key} was accepted"
            );
        }
    }

    #[test]
    fn rnn_hidden_size_equal_to_pred_hidden_is_no_projection() {
        let mut config = v3_config();
        config["decoder"]["rnn_hidden_size"] = json!(640);
        load(&config).unwrap();
    }

    /// The config `export_native.py` writes for nemotron-3.5-asr-streaming-0.6b,
    /// with the prompt dictionary cut to two languages
    fn streaming_config() -> Value {
        let mut config = v3_config();
        let root = config.as_object_mut().unwrap();
        root.remove("version");
        root.insert("model".into(), json!("nemotron-3.5-asr-streaming-0.6b"));
        root.insert(
            "prompt".into(),
            json!({ "num_prompts": 128, "dictionary": { "en-US": 0, "auto": 101 } }),
        );
        config["preprocessor"]["normalize"] = json!("NA");
        let encoder = config["encoder"].as_object_mut().unwrap();
        encoder.insert(
            "att_context_size".into(),
            json!([[56, 3], [56, 0], [56, 6], [56, 13]]),
        );
        encoder.insert("att_context_style".into(), json!("chunked_limited"));
        encoder.insert("causal_downsampling".into(), json!(true));
        encoder.insert("conv_context_size".into(), json!("causal"));
        encoder.insert("conv_norm_type".into(), json!("layer_norm"));
        config["decoder"]["vocab_size"] = json!(13087);
        config["joint"]["num_classes"] = json!(13087);
        config["joint"]["num_extra_outputs"] = json!(0);
        config["decoding"] = json!({ "model_type": "rnnt", "durations": [], "max_symbols": 10 });
        config
    }

    fn load_streaming(config: &Value) -> Result<StreamingSpec, TranscriptionError> {
        StreamingSpec::from_config_json(&config.to_string())
    }

    #[test]
    fn exported_streaming_config_loads() {
        let spec = load_streaming(&streaming_config()).unwrap();
        assert_eq!(spec.left_context, 56);
        assert_eq!(spec.lookaheads, [3, 0, 6, 13]);
        assert!(spec.encoder.causal);
        assert_eq!(spec.encoder.conv_norm, ConvNorm::Layer);
        assert!(spec.transducer.durations.is_empty());
        assert_eq!(spec.transducer.joint.outputs, 13088);
        let prompt = spec.prompt.unwrap();
        assert_eq!(prompt.prompts, 128);
        assert_eq!(prompt.languages["auto"], 101);

        let mut config = streaming_config();
        config.as_object_mut().unwrap().remove("prompt");
        assert_eq!(load_streaming(&config).unwrap().prompt, None);
    }

    #[test]
    fn streaming_settings_the_engine_lacks_are_rejected() {
        let changes = [
            ("preprocessor", "normalize", json!("per_feature")),
            ("encoder", "causal_downsampling", json!(false)),
            ("encoder", "conv_context_size", json!(null)),
            ("encoder", "conv_norm_type", json!("batch_norm")),
            ("encoder", "att_context_style", json!("regular")),
            ("encoder", "att_context_size", json!([[56, 3], [70, 13]])),
            ("encoder", "att_context_size", json!([[56, 4]])),
            ("encoder", "att_context_size", json!([[0, 3]])),
            ("encoder", "att_context_size", json!([[-1, -1]])),
            ("encoder", "att_context_size", json!([])),
            ("encoder", "att_context_size", json!([[70, u64::MAX]])),
            ("decoding", "model_type", json!("tdt")),
            ("decoding", "max_symbols", json!(0)),
            ("decoding", "durations", json!([0, 1])),
            ("prompt", "dictionary", json!({ "en-US": 128 })),
        ];
        for (section, key, value) in changes {
            let mut config = streaming_config();
            config[section][key] = value.clone();
            assert!(
                matches!(
                    load_streaming(&config),
                    Err(TranscriptionError::InvalidWeights(_))
                ),
                "{section}.{key} = {value} was accepted"
            );
        }
    }
}

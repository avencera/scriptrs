pub(crate) const SAMPLE_RATE: usize = 16_000;
pub(crate) const MAX_MODEL_SAMPLES: usize = 240_000;
pub(crate) const MEL_HOP_SAMPLES: usize = 160;
pub(crate) const ENCODER_SUBSAMPLING: usize = 8;
pub(crate) const SAMPLES_PER_ENCODER_FRAME: usize = MEL_HOP_SAMPLES * ENCODER_SUBSAMPLING;
#[cfg(feature = "long-form")]
pub(crate) const VAD_WINDOW_SAMPLES: usize = 512;
#[cfg(feature = "long-form")]
pub(crate) const VAD_CONTEXT_SAMPLES: usize = 64;
#[cfg(feature = "long-form")]
pub(crate) const VAD_STATE_SIZE: usize = 128;
pub(crate) const SECONDS_PER_ENCODER_FRAME: f64 =
    SAMPLES_PER_ENCODER_FRAME as f64 / SAMPLE_RATE as f64;
pub(crate) const ENCODER_HIDDEN_SIZE: usize = 1_024;
pub(crate) const DECODER_HIDDEN_SIZE: usize = 640;
pub(crate) const DECODER_LAYERS: usize = 2;
pub(crate) const MAX_TOKENS_PER_STEP: usize = 10;

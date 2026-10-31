//! Silero VAD v5 (16 kHz branch) in plain Rust on the CPU
//!
//! Everything before the LSTM sees one window and no recurrent state, so the STFT,
//! the four convolutions and the LSTM input gates run as matrix multiplies over
//! tasks of windows on worker threads. The calling thread runs the LSTM recurrence
//! window by window over the tasks already encoded, so the encoder hides behind it.
//!
//! It runs on the CPU because the recurrence is the critical path either way, and
//! on a GPU each step is a dependent pair of launches. On an RTX 5090 a
//! graph-captured cuBLAS product plus cell kernel took 3.2 µs a step, against
//! 2.6 µs for the whole CPU step, and the CPU path leaves the GPU to Parakeet: a
//! recording takes about 0.3 s per hour of audio

use std::path::Path;
use std::sync::{Arc, mpsc};

use ndarray::linalg::general_mat_mul;
use ndarray::{ArrayView2, ArrayViewMut2};

use crate::constants::{VAD_CONTEXT_SAMPLES, VAD_STATE_SIZE, VAD_WINDOW_SAMPLES};
use crate::error::TranscriptionError;
use crate::native::weights::WeightFile;

/// Samples in one model input: the carried context plus one chunk
pub(crate) const VAD_INPUT_SAMPLES: usize = VAD_CONTEXT_SAMPLES + VAD_WINDOW_SAMPLES;

const MODEL: &str = "silero-vad-v5";
const SAMPLE_RATE: u64 = 16_000;

/// STFT as Silero runs it: a conv1d over a reflect-padded window with a DFT basis
const FILTER: usize = 256;
const HOP: usize = 128;
const BINS: usize = FILTER / 2 + 1;
const REFLECT_PAD: usize = 64;
const FRAMES: usize = (VAD_INPUT_SAMPLES + REFLECT_PAD - FILTER) / HOP + 1;

const HIDDEN: usize = VAD_STATE_SIZE;
const GATES: usize = 4 * HIDDEN;

/// Encoder convolutions as `(in channels, out channels, stride)`, all kernel 3 and
/// padding 1, each followed by ReLU
const CONVS: [(usize, usize, usize); 4] =
    [(BINS, 128, 1), (128, 64, 2), (64, 64, 2), (64, HIDDEN, 1)];
const KERNEL: usize = 3;

/// Windows per encoder task: enough rows for efficient matrix multiplies, few
/// enough that the recurrence starts soon after a block does
const WINDOWS_PER_TASK: usize = 128;

/// Encoder tasks per worker in one block, so the wait for a block's first task is
/// short next to the recurrence over the whole block
const TASKS_PER_WORKER: usize = 4;

/// Encoder threads at most; the recurrence on the calling thread is the critical
/// path, and a few workers already encode faster than it consumes
const MAX_WORKERS: usize = 6;

/// Silero VAD v5 weights, shared by clones
#[derive(Debug, Clone)]
pub(crate) struct SileroModel(Arc<Weights>);

#[derive(Debug)]
struct Weights {
    /// DFT basis `[2 * BINS, FILTER]`, real rows then imaginary rows
    basis: Vec<f32>,
    convs: [Conv; 4],
    /// `weight_ih` as PyTorch stores it, `[GATES, HIDDEN]`
    input_gates: Vec<f32>,
    /// `bias_ih + bias_hh`
    gate_bias: Vec<f32>,
    /// `weight_hh` in tiles of `TILE` gates, `[GATES / TILE, HIDDEN, TILE]`
    recurrent_tiles: Vec<f32>,
    head: Vec<f32>,
    head_bias: f32,
    /// Encoder threads beside the one running the recurrence
    workers: usize,
    /// Whether the recurrent product can run its AVX build
    #[cfg(target_arch = "x86_64")]
    avx: bool,
}

#[derive(Debug)]
struct Conv {
    inputs: usize,
    outputs: usize,
    stride: usize,
    /// `[outputs, KERNEL * inputs]`, tap-major to match the unfolded input rows
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// Recurrent state carried across the blocks of one recording
#[derive(Debug, Clone)]
pub(crate) struct SileroState {
    h: Vec<f32>,
    c: Vec<f32>,
}

impl SileroState {
    pub(crate) fn new() -> Self {
        Self {
            h: vec![0.0; HIDDEN],
            c: vec![0.0; HIDDEN],
        }
    }
}

impl SileroModel {
    /// Load `silero-vad-v5.safetensors` as `dev-tools/nemo/export_native.py` writes it
    pub(crate) fn open(path: &Path) -> Result<Self, TranscriptionError> {
        let file = WeightFile::open(path)?;
        file.expect_metadata("model", MODEL)?;
        check_config(&file)?;

        let basis = file.tensor("stft.forward_basis_buffer", &[2 * BINS, 1, FILTER])?;
        let conv = |index: usize| -> Result<Conv, TranscriptionError> {
            let (inputs, outputs, stride) = CONVS[index];
            let prefix = format!("encoder.{index}.reparam_conv");
            let weight = file.tensor(&format!("{prefix}.weight"), &[outputs, inputs, KERNEL])?;
            let bias = file.tensor(&format!("{prefix}.bias"), &[outputs])?;
            // PyTorch's `[out, in, tap]` becomes `[out, tap, in]`
            let mut reordered = vec![0.0; weight.len()];
            for out in 0..outputs {
                for input in 0..inputs {
                    for tap in 0..KERNEL {
                        reordered[(out * KERNEL + tap) * inputs + input] =
                            weight[(out * inputs + input) * KERNEL + tap];
                    }
                }
            }
            Ok(Conv {
                inputs,
                outputs,
                stride,
                weight: reordered,
                bias: bias.to_vec(),
            })
        };
        let convs = [conv(0)?, conv(1)?, conv(2)?, conv(3)?];

        let bias_ih = file.tensor("decoder.rnn.bias_ih", &[GATES])?;
        let bias_hh = file.tensor("decoder.rnn.bias_hh", &[GATES])?;
        let recurrent = file.tensor("decoder.rnn.weight_hh", &[GATES, HIDDEN])?;
        // `[GATES / TILE, HIDDEN, TILE]`: each tile of gates streams its weights
        // in one contiguous run
        let mut recurrent_tiles = vec![0.0; recurrent.len()];
        for gate in 0..GATES {
            for unit in 0..HIDDEN {
                recurrent_tiles[((gate / TILE) * HIDDEN + unit) * TILE + gate % TILE] =
                    recurrent[gate * HIDDEN + unit];
            }
        }

        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let workers = (threads - 1).min(MAX_WORKERS);
        let weights = Weights {
            basis: basis.to_vec(),
            convs,
            input_gates: file
                .tensor("decoder.rnn.weight_ih", &[GATES, HIDDEN])?
                .to_vec(),
            gate_bias: bias_ih.iter().zip(bias_hh).map(|(a, b)| a + b).collect(),
            recurrent_tiles,
            head: file
                .tensor("decoder.decoder.2.weight", &[1, HIDDEN, 1])?
                .to_vec(),
            head_bias: file.tensor("decoder.decoder.2.bias", &[1])?[0],
            workers,
            #[cfg(target_arch = "x86_64")]
            avx: std::arch::is_x86_feature_detected!("avx"),
        };
        weights.check_finite()?;
        Ok(Self(Arc::new(weights)))
    }

    /// Windows to hand [`Self::classify`] per call
    pub(crate) fn block_windows(&self) -> usize {
        self.0.workers.max(1) * TASKS_PER_WORKER * WINDOWS_PER_TASK
    }

    /// Speech probability of each `[VAD_INPUT_SAMPLES]` window in `windows`, in
    /// order, advancing `state` across them
    ///
    /// Workers encode tasks of windows while this thread runs the recurrence over
    /// the tasks already encoded, in order
    pub(crate) fn classify(
        &self,
        windows: &[f32],
        state: &mut SileroState,
        probabilities: &mut Vec<f32>,
    ) {
        debug_assert_eq!(windows.len() % VAD_INPUT_SAMPLES, 0);
        let count = windows.len() / VAD_INPUT_SAMPLES;
        let mut gates = vec![0.0; count * GATES];
        probabilities.reserve(count);
        let tasks: Vec<_> = windows
            .chunks(WINDOWS_PER_TASK * VAD_INPUT_SAMPLES)
            .zip(gates.chunks_mut(WINDOWS_PER_TASK * GATES))
            .collect();

        let workers = self.0.workers.min(tasks.len());
        if workers <= 1 {
            for (windows, gates) in tasks {
                self.0.input_gates(windows, gates);
                self.0.recur(gates, state, probabilities);
            }
            return;
        }

        let task_count = tasks.len();
        let mut queues: Vec<Vec<_>> = (0..workers).map(|_| Vec::new()).collect();
        for (index, (windows, gates)) in tasks.into_iter().enumerate() {
            queues[index % workers].push((index, windows, gates));
        }

        std::thread::scope(|scope| {
            let (encoded, finished) = mpsc::channel();
            for queue in queues {
                let encoded = encoded.clone();
                scope.spawn(move || {
                    for (index, windows, gates) in queue {
                        self.0.input_gates(windows, gates);
                        if encoded.send((index, gates)).is_err() {
                            return;
                        }
                    }
                });
            }
            drop(encoded);

            // tasks finish out of order; each runs once every task before it has
            let mut ready: Vec<Option<&mut [f32]>> = (0..task_count).map(|_| None).collect();
            let mut next = 0;
            for (index, gates) in finished {
                ready[index] = Some(gates);
                while let Some(gates) = ready.get_mut(next).and_then(Option::take) {
                    self.0.recur(gates, state, probabilities);
                    next += 1;
                }
            }
            // a worker that panicked leaves tasks undone, and the scope re-raises
            // its panic on exit
        });
    }
}

impl Weights {
    /// A NaN or infinite coefficient makes every probability NaN, which plans no
    /// speech and returns an empty transcript instead of failing
    fn check_finite(&self) -> Result<(), TranscriptionError> {
        let convs = self
            .convs
            .iter()
            .flat_map(|conv| [&conv.weight, &conv.bias]);
        let finite = [
            &self.basis,
            &self.input_gates,
            &self.gate_bias,
            &self.recurrent_tiles,
            &self.head,
        ]
        .into_iter()
        .chain(convs)
        .flatten()
        .chain([&self.head_bias])
        .all(|value| value.is_finite());
        if finite {
            Ok(())
        } else {
            Err(TranscriptionError::InvalidWeights(
                "Silero VAD weights hold a non-finite value".to_owned(),
            ))
        }
    }

    /// LSTM input gates (`x·W_ihᵀ + b_ih + b_hh`) of each window, `[windows, GATES]`
    fn input_gates(&self, windows: &[f32], gates: &mut [f32]) {
        let count = windows.len() / VAD_INPUT_SAMPLES;
        let features = self.encode(windows);
        matmul_t(&features, &self.input_gates, gates, count, HIDDEN, GATES);
        for row in gates.as_chunks_mut::<GATES>().0 {
            for (gate, bias) in row.iter_mut().zip(&self.gate_bias) {
                *gate += bias;
            }
        }
    }

    /// Encoder output of each window, `[windows, HIDDEN]`
    fn encode(&self, windows: &[f32]) -> Vec<f32> {
        let count = windows.len() / VAD_INPUT_SAMPLES;
        let mut features = self.magnitude(windows);
        let mut steps = FRAMES;
        for conv in &self.convs {
            (features, steps) = conv.forward(&features, count, steps);
        }
        debug_assert_eq!(steps, 1);
        features
    }

    /// STFT magnitude of each window, `[windows, FRAMES, BINS]`, channels last for
    /// the convolutions
    fn magnitude(&self, windows: &[f32]) -> Vec<f32> {
        let count = windows.len() / VAD_INPUT_SAMPLES;

        // frames of the right-reflect-padded window, `[count * FRAMES, FILTER]`
        let mut frames = vec![0.0; count * FRAMES * FILTER];
        let mut padded = [0.0f32; VAD_INPUT_SAMPLES + REFLECT_PAD];
        for (window, frames) in windows
            .as_chunks::<VAD_INPUT_SAMPLES>()
            .0
            .iter()
            .zip(frames.as_chunks_mut::<{ FRAMES * FILTER }>().0)
        {
            padded[..VAD_INPUT_SAMPLES].copy_from_slice(window);
            for offset in 0..REFLECT_PAD {
                // reflection excludes the edge sample, as `ReflectionPad1d` does
                padded[VAD_INPUT_SAMPLES + offset] = window[VAD_INPUT_SAMPLES - 2 - offset];
            }
            for (frame, out) in frames.as_chunks_mut::<FILTER>().0.iter_mut().enumerate() {
                out.copy_from_slice(&padded[frame * HOP..frame * HOP + FILTER]);
            }
        }

        let mut spectrum = vec![0.0; count * FRAMES * 2 * BINS];
        matmul_t(
            &frames,
            &self.basis,
            &mut spectrum,
            count * FRAMES,
            FILTER,
            2 * BINS,
        );
        spectrum
            .as_chunks::<{ 2 * BINS }>()
            .0
            .iter()
            .flat_map(|row| {
                let (real, imaginary) = row.split_at(BINS);
                real.iter()
                    .zip(imaginary)
                    .map(|(re, im)| (re * re + im * im).sqrt())
            })
            .collect()
    }

    /// Recurrence over the input gates of consecutive windows, `[windows, GATES]`
    fn recur(&self, gates: &mut [f32], state: &mut SileroState, probabilities: &mut Vec<f32>) {
        for gates in gates.as_chunks_mut::<GATES>().0 {
            probabilities.push(self.step(gates, state));
        }
    }

    /// One LSTM step on precomputed input gates, then the speech head
    fn step(&self, gates: &mut [f32], state: &mut SileroState) -> f32 {
        #[cfg(target_arch = "x86_64")]
        let logit = if self.avx {
            // SAFETY: `avx` is only set when the CPU reports AVX
            unsafe { lstm_step_avx(gates, state, &self.recurrent_tiles, &self.head) }
        } else {
            lstm_step(gates, state, &self.recurrent_tiles, &self.head)
        };
        #[cfg(not(target_arch = "x86_64"))]
        let logit = lstm_step(gates, state, &self.recurrent_tiles, &self.head);
        sigmoid(logit + self.head_bias)
    }
}

impl Conv {
    /// Conv1d (kernel 3, padding 1) and ReLU over `[count, steps, inputs]`
    fn forward(&self, x: &[f32], count: usize, steps: usize) -> (Vec<f32>, usize) {
        let out_steps = (steps + 2 - KERNEL) / self.stride + 1;
        let row = KERNEL * self.inputs;

        // unfold each output step's three taps into one row, zero where the
        // padding falls outside the window
        let mut unfolded = vec![0.0; count * out_steps * row];
        for (window, rows) in x
            .chunks_exact(steps * self.inputs)
            .zip(unfolded.chunks_exact_mut(out_steps * row))
        {
            for (out_step, taps) in rows.chunks_exact_mut(row).enumerate() {
                for (tap, slot) in taps.chunks_exact_mut(self.inputs).enumerate() {
                    let Some(step) = (out_step * self.stride + tap).checked_sub(1) else {
                        continue;
                    };
                    if step < steps {
                        slot.copy_from_slice(&window[step * self.inputs..][..self.inputs]);
                    }
                }
            }
        }

        let mut out = vec![0.0; count * out_steps * self.outputs];
        matmul_t(
            &unfolded,
            &self.weight,
            &mut out,
            count * out_steps,
            row,
            self.outputs,
        );
        for values in out.chunks_exact_mut(self.outputs) {
            for (value, bias) in values.iter_mut().zip(&self.bias) {
                *value = (*value + bias).max(0.0);
            }
        }
        (out, out_steps)
    }
}

/// One LSTM cell update on input gates, returning the speech head's logit
/// without its bias
///
/// Every value comes from separate multiplies and adds in a fixed order, the same
/// in every build, so the AVX and baseline results are identical
#[inline(always)]
fn lstm_step(
    gates: &mut [f32],
    state: &mut SileroState,
    recurrent_tiles: &[f32],
    head: &[f32],
) -> f32 {
    recurrent(gates, &state.h, recurrent_tiles);

    // PyTorch gate order: input, forget, cell, output
    let (input, rest) = gates.split_at_mut(HIDDEN);
    let (forget, rest) = rest.split_at_mut(HIDDEN);
    let (cell, output) = rest.split_at_mut(HIDDEN);
    for gate in input.iter_mut() {
        *gate = sigmoid(*gate);
    }
    for gate in forget.iter_mut() {
        *gate = sigmoid(*gate);
    }
    for gate in output.iter_mut() {
        *gate = sigmoid(*gate);
    }
    for gate in cell.iter_mut() {
        *gate = tanh(*gate);
    }

    for unit in 0..HIDDEN {
        let c = forget[unit] * state.c[unit] + input[unit] * cell[unit];
        state.c[unit] = c;
        state.h[unit] = output[unit] * tanh(c);
    }

    // the head reads ReLU(h); the state keeps h itself
    let mut logit = 0.0;
    for (h, weight) in state.h.iter().zip(head) {
        logit += h.max(0.0) * weight;
    }
    logit
}

/// [`lstm_step`] built for AVX, which the x86-64 baseline target leaves out
///
/// # Safety
///
/// The CPU must support AVX
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn lstm_step_avx(
    gates: &mut [f32],
    state: &mut SileroState,
    recurrent_tiles: &[f32],
    head: &[f32],
) -> f32 {
    lstm_step(gates, state, recurrent_tiles, head)
}

/// Gates per register tile of the recurrent product: 8 AVX or 16 SSE registers
const TILE: usize = 64;

/// `gates += W_hh · h`, one tile of gates at a time
///
/// Each gate sums its terms in hidden-unit order. Keeping a tile in registers
/// saves storing the gates back once per hidden unit, which made the step four
/// times slower
#[inline(always)]
fn recurrent(gates: &mut [f32], h: &[f32], recurrent_tiles: &[f32]) {
    for (tile_index, tile) in gates.as_chunks_mut::<TILE>().0.iter_mut().enumerate() {
        let mut sums = [0.0f32; TILE];
        sums.copy_from_slice(tile);
        let tile_weights = &recurrent_tiles[tile_index * HIDDEN * TILE..][..HIDDEN * TILE];
        for (h, weights) in h.iter().zip(tile_weights.as_chunks::<TILE>().0) {
            for (sum, weight) in sums.iter_mut().zip(weights) {
                *sum += h * weight;
            }
        }
        tile.copy_from_slice(&sums);
    }
}

/// Adding and subtracting 1.5 · 2²³ rounds a float below 2²² to the nearest
/// integer without SSE4.1's `roundps`
const ROUND: f32 = 12_582_912.0;

/// `eˣ` to within a couple of ulp with the Cephes `expf` polynomial
///
/// Branch-free so the activation loops vectorize: glibc's scalar `expf` and
/// `tanhf` took half of each step
#[inline(always)]
fn exp(x: f32) -> f32 {
    // keeps 2ⁿ below a normal float's largest exponent
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E + ROUND) - ROUND;
    // x - n·ln 2 in two parts, the first exact in f32
    let r = x - n * 0.693_359_4 - n * -2.121_944_4e-4;
    let p = ((((1.987_569_1e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r
        + 0.166_666_65)
        * r
        + 0.5;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

/// Through `exp` rather than libm's `tanhf`, exact to about 1e-7 absolute
#[inline(always)]
fn tanh(x: f32) -> f32 {
    1.0 - 2.0 / (1.0 + exp(2.0 * x))
}

/// `c = a · bᵀ` for row-major `a [m, k]` and `b [n, k]`
fn matmul_t(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let a = ArrayView2::from_shape((m, k), a).expect("left operand shape");
    let b = ArrayView2::from_shape((n, k), b).expect("right operand shape");
    let mut c = ArrayViewMut2::from_shape((m, n), c).expect("output shape");
    general_mat_mul(1.0, &a, &b.t(), 0.0, &mut c);
}

/// Reject a file exported for different window settings than the ones built in
fn check_config(file: &WeightFile) -> Result<(), TranscriptionError> {
    let invalid = |message: String| TranscriptionError::InvalidWeights(message);
    let config = file
        .metadata("config")
        .ok_or_else(|| invalid("metadata `config` is missing".to_owned()))?;
    let config: serde_json::Value = serde_json::from_str(config)
        .map_err(|error| invalid(format!("metadata `config`: {error}")))?;
    for (key, expected) in [
        ("sample_rate", SAMPLE_RATE),
        ("window", VAD_WINDOW_SAMPLES as u64),
        ("context", VAD_CONTEXT_SAMPLES as u64),
        ("state", VAD_STATE_SIZE as u64),
    ] {
        let found = config.get(key).and_then(serde_json::Value::as_u64);
        if found != Some(expected) {
            return Err(invalid(format!(
                "VAD config `{key}` is {found:?}, expected {expected}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BINS, FRAMES, HIDDEN, SileroModel, SileroState, VAD_INPUT_SAMPLES, exp, sigmoid, tanh,
    };
    #[cfg(target_arch = "x86_64")]
    use super::{GATES, lstm_step, lstm_step_avx};
    use crate::cuda::test_support::native_root;
    use crate::test_support::NemoReference;

    fn model() -> SileroModel {
        SileroModel::open(&native_root().join("vad/silero-vad-v5.safetensors"))
            .expect("exported Silero weights")
    }

    /// Largest difference, NaN if any value is NaN so a broken stage cannot pass
    fn max_abs_diff(a: impl IntoIterator<Item = f32>, b: impl IntoIterator<Item = f32>) -> f32 {
        let (a, b): (Vec<_>, Vec<_>) = (a.into_iter().collect(), b.into_iter().collect());
        assert_eq!(a.len(), b.len(), "compared outputs differ in length");
        a.iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, |worst, diff| {
                if diff > worst || diff.is_nan() {
                    diff
                } else {
                    worst
                }
            })
    }

    #[test]
    fn activations_track_libm() {
        let mut worst = [0.0f32; 3];
        // past ±87 `exp` clamps, which only sigmoid and tanh rely on
        for step in -40_000..=40_000 {
            let x = step as f32 * 2.5e-3;
            let exp_error = if x.abs() < 87.0 {
                ((exp(x) - x.exp()) / x.exp()).abs()
            } else {
                0.0
            };
            let sigmoid_error = (sigmoid(x) - 1.0 / (1.0 + (-x).exp())).abs();
            let tanh_error = (tanh(x) - x.tanh()).abs();
            for (worst, error) in worst.iter_mut().zip([exp_error, sigmoid_error, tanh_error]) {
                *worst = worst.max(error);
            }
        }
        assert!(worst[0] < 4e-7, "exp relative error {}", worst[0]);
        assert!(worst[1] < 2e-7, "sigmoid error {}", worst[1]);
        assert!(worst[2] < 4e-7, "tanh error {}", worst[2]);

        // saturates instead of overflowing past the clamp
        assert!(sigmoid(-200.0) < 1e-30);
        assert_eq!(sigmoid(200.0), 1.0);
        assert_eq!(tanh(200.0), 1.0);
        assert_eq!(tanh(-200.0), -1.0);
    }

    /// The AVX build runs the same operations in the same order as the baseline
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx_step_matches_the_baseline_bit_for_bit() {
        if !std::arch::is_x86_feature_detected!("avx") {
            return;
        }
        let values = |seed: usize, len: usize| -> Vec<f32> {
            (0..len)
                .map(|index| (((index * 7919 + seed * 104_729) % 2001) as f32 - 1000.0) * 2e-3)
                .collect()
        };
        let weights = values(1, GATES * HIDDEN);
        let head = values(2, HIDDEN);
        let mut baseline_state = SileroState::new();
        let mut avx_state = SileroState::new();

        for step in 0..32 {
            let mut baseline_gates = values(step + 3, GATES);
            let mut avx_gates = baseline_gates.clone();
            let baseline = lstm_step(&mut baseline_gates, &mut baseline_state, &weights, &head);
            // SAFETY: AVX support was checked above
            let avx = unsafe { lstm_step_avx(&mut avx_gates, &mut avx_state, &weights, &head) };
            assert_eq!(baseline.to_bits(), avx.to_bits(), "logit at step {step}");
            assert_eq!(baseline_state.h, avx_state.h, "h at step {step}");
            assert_eq!(baseline_state.c, avx_state.c, "c at step {step}");
        }
        assert!(baseline_state.h.iter().any(|h| *h != 0.0));
    }

    /// Window 0's STFT magnitude and encoder output match Silero's own modules
    #[test]
    #[ignore = "reads Silero references from /shared/datasets/parakeet-ref"]
    fn nemo_parity_silero_encoder_stages() {
        let reference = NemoReference::silero();
        let window = reference.f32("window0.in");
        let window = window.as_slice().expect("contiguous window");
        assert_eq!(window.len(), VAD_INPUT_SAMPLES);
        let model = model();

        // the reference is `[1, BINS, FRAMES]`; ours is frames first
        let stft = reference.f32("window0.stft");
        let magnitude = model.0.magnitude(window);
        assert_eq!(magnitude.len(), FRAMES * BINS);
        let ours = (0..BINS).flat_map(|bin| (0..FRAMES).map(move |frame| (bin, frame)));
        let diff = max_abs_diff(
            ours.map(|(bin, frame)| magnitude[frame * BINS + bin]),
            stft.iter().copied(),
        );
        let scale = stft.iter().copied().fold(0.0, f32::max);
        assert!(
            diff <= 1e-5 * scale.max(1.0),
            "STFT max diff {diff} (peak {scale})"
        );

        let encoder = reference.f32("window0.encoder");
        let features = model.0.encode(window);
        assert_eq!(features.len(), HIDDEN);
        let diff = max_abs_diff(features, encoder.iter().copied());
        assert!(diff <= 1e-4, "encoder max diff {diff}");
    }

    /// Per-window probabilities over a whole file stay within 1e-4 of Silero's, and
    /// the block size used for the batched encoder does not change them
    #[test]
    #[ignore = "reads Silero references from /shared/datasets/parakeet-ref"]
    fn nemo_parity_silero_probabilities() {
        let reference = NemoReference::silero();
        let windows = reference.f32("windows");
        let windows = windows.as_slice().expect("contiguous windows");
        let expected = reference.f32("probs");
        let model = model();

        let run = |block: usize| {
            let mut state = SileroState::new();
            let mut probabilities = Vec::new();
            for block in windows.chunks(block * VAD_INPUT_SAMPLES) {
                model.classify(block, &mut state, &mut probabilities);
            }
            probabilities
        };
        let whole = run(windows.len());
        assert_eq!(whole.len(), expected.len());
        let diff = max_abs_diff(whole.iter().copied(), expected.iter().copied());
        eprintln!(
            "silero probability max abs diff {diff:e} over {} windows",
            whole.len()
        );
        assert!(diff <= 1e-5, "probability max diff {diff}");
        assert_eq!(run(7), whole, "block size changed the probabilities");
    }
}

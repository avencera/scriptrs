//! Launchers for the transducer decode step in the kernel crate's `decode` area

use cudakit::cudarc::driver::{
    CudaFunction, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg,
};
use cudakit::{Error, LoadedKernels, check_len, element_count};

use super::{CudaRuntime, KernelModule, len, len_mut, rows, u32_of};

/// Threads per block of `decode_decide`, one block per batch row: a decode step
/// has only `batch` blocks, so a wide block hides the latency of each pass over
/// the class logits
const DECIDE_THREADS: u32 = 1024;

/// Class logits each `decode_decide` thread holds in registers, the kernel's
/// `DECIDE_VALUES`
const DECIDE_VALUES: usize = 13;

/// Threads per block of `decode_lstm` and `decode_project`, one block per output,
/// the kernels' `ROW_WARPS` warps
const WEIGHT_ROW_THREADS: u32 = 4 * 32;

/// Weight columns each of those threads holds in registers, the kernels'
/// `ROW_STEPS`, which bounds the predictor's hidden width
const WEIGHT_ROW_STEPS: usize = 8;

/// Fails unless a weight row of `hidden` fits the registers of one
/// `decode_lstm` or `decode_project` block
fn check_weight_row(hidden: usize) -> Result<(), Error> {
    let limit = WEIGHT_ROW_STEPS * WEIGHT_ROW_THREADS as usize;
    if hidden == 0 || hidden > limit {
        return Err(Error::BufferLength {
            context: "predictor hidden width",
            expected: limit,
            actual: hidden,
        });
    }
    Ok(())
}

/// Per-row decode state, `[batch]` each
pub(crate) struct DecodeState<'a> {
    pub frame: CudaViewMut<'a, u32>,
    pub emitted: CudaViewMut<'a, u32>,
    pub label: CudaViewMut<'a, u32>,
    pub emit: CudaViewMut<'a, u32>,
    pub counts: CudaViewMut<'a, u32>,
    pub faults: CudaViewMut<'a, u32>,
}

/// Emitted tokens, `[batch, capacity]` each
pub(crate) struct DecodeOutput<'a> {
    pub tokens: CudaViewMut<'a, u32>,
    pub frames: CudaViewMut<'a, u32>,
    pub durations: CudaViewMut<'a, u32>,
    pub confidences: CudaViewMut<'a, f32>,
}

/// Widest language filter `decode_decide` takes, its `MAX_FILTER_TOP_K`
pub(crate) const MAX_FILTER_TOP_K: usize = 256;

/// Language filter of a decode step, as `decode_decide` reads it
pub(crate) struct DecodeFilter<'a> {
    /// `[classes]`, nonzero for the token ids the filter allows
    pub allowed: CudaView<'a, u32>,
    /// `[1]`, the filter's top-K width, or 0 for no filter
    pub top_k: CudaView<'a, u32>,
}

/// Joint and duration geometry of one decode step
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodeShape {
    pub batch: usize,
    /// Token classes including the blank, which is the last
    pub classes: usize,
    pub max_symbols: usize,
    /// Output slots per row
    pub capacity: usize,
}

/// One predictor LSTM layer's weights, `[4 * hidden, hidden]` each and the summed
/// biases `[4 * hidden]`
pub(crate) struct LstmWeights<'a> {
    pub weight_ih: CudaView<'a, f32>,
    pub weight_hh: CudaView<'a, f32>,
    pub bias: CudaView<'a, f32>,
}

/// Where a predictor layer reads its input rows
pub(crate) enum LstmInput<'a> {
    /// Row `label[b]` of the embedding table `[classes, hidden]`
    Embedded {
        table: CudaView<'a, f32>,
        label: CudaView<'a, u32>,
    },
    /// Row `b` of the layer below's new state, `[batch, hidden]`
    Below(CudaView<'a, f32>),
}

/// One predictor layer's state, `[batch, hidden]` each: the state before the step,
/// the new state, and the cell state updated in place
pub(crate) struct LstmState<'a> {
    pub h: CudaView<'a, f32>,
    pub h_next: CudaViewMut<'a, f32>,
    pub c: CudaViewMut<'a, f32>,
}

/// The buffers `decode_project` reads and writes
pub(crate) struct Projection<'a> {
    /// `[joint_hidden, hidden]` and `[joint_hidden]`
    pub weight: CudaView<'a, f32>,
    pub bias: CudaView<'a, f32>,
    /// The top layer's new state, `[batch, hidden]`
    pub top: CudaView<'a, f32>,
    /// `[batch, frames, joint_hidden]`
    pub enc_proj: CudaView<'a, f32>,
    pub frame: CudaView<'a, u32>,
    pub lens: CudaView<'a, u32>,
    /// `[outputs]`
    pub joint_bias: CudaView<'a, f32>,
    /// Every layer's new and current state, `[layers, batch, hidden]`
    pub h_next: CudaView<'a, f32>,
    pub h: CudaViewMut<'a, f32>,
    /// `[batch, joint_hidden]`
    pub pred_proj: CudaViewMut<'a, f32>,
    pub joint_in: CudaViewMut<'a, f32>,
    /// `[batch, outputs]`
    pub logits: CudaViewMut<'a, f32>,
}

/// The `decode` area's kernels
pub(crate) struct DecodeKernels {
    decide: CudaFunction,
    lstm: CudaFunction,
    project: CudaFunction,
}

impl DecodeKernels {
    pub(crate) fn load(runtime: &CudaRuntime) -> Result<Self, Error> {
        let module: LoadedKernels<KernelModule> = runtime.load_embedded(KernelModule::Decode)?;
        Ok(Self {
            decide: module.function("decode_decide")?,
            lstm: module.function("decode_lstm")?,
            project: module.function("decode_project")?,
        })
    }

    /// Token decision and frame advance for every unfinished row
    ///
    /// A nonzero `filter.top_k` must be at most `shape.classes` and
    /// [`MAX_FILTER_TOP_K`]; the kernel clamps it to stay in bounds, so a wider one
    /// would search fewer candidates than asked
    #[expect(
        clippy::too_many_arguments,
        reason = "launchers mirror their kernel's parameter list so each launch reads against it"
    )]
    pub(crate) fn decide(
        &self,
        stream: &CudaStream,
        logits: &CudaView<'_, f32>,
        lens_frames: &CudaView<'_, u32>,
        durations: &CudaView<'_, u32>,
        filter: &DecodeFilter<'_>,
        state: &mut DecodeState<'_>,
        output: &mut DecodeOutput<'_>,
        shape: DecodeShape,
    ) -> Result<(), Error> {
        let batch = shape.batch;
        let row = shape.classes + durations.len();
        check_len(
            "joint logits",
            element_count("joint logits", &[batch, row])?,
            logits.len(),
        )?;
        check_len("frame lengths", batch, lens_frames.len())?;
        check_len("allowed tokens", shape.classes, filter.allowed.len())?;
        check_len("language filter", 1, filter.top_k.len())?;
        for (context, buffer) in [
            ("frame cursors", &state.frame),
            ("emitted counts", &state.emitted),
            ("labels", &state.label),
            ("emit flags", &state.emit),
            ("token counts", &state.counts),
            ("faults", &state.faults),
        ] {
            check_len(context, batch, buffer.len())?;
        }
        let slots = element_count("decode output", &[batch, shape.capacity])?;
        for (context, buffer) in [
            ("tokens", &output.tokens),
            ("token frames", &output.frames),
            ("token durations", &output.durations),
        ] {
            check_len(context, slots, buffer.len())?;
        }
        check_len("confidences", slots, output.confidences.len())?;
        if shape.classes == 0 || durations.is_empty() {
            return Err(Error::BufferLength {
                context: "joint classes and durations",
                expected: 1,
                actual: 0,
            });
        }
        // each thread holds its class logits in registers, and one lane of warp 0
        // reads each duration logit, so wider rows would drop some
        let widest = DECIDE_VALUES * DECIDE_THREADS as usize;
        if shape.classes > widest {
            return Err(Error::BufferLength {
                context: "joint classes",
                expected: widest,
                actual: shape.classes,
            });
        }
        if durations.len() > 32 {
            return Err(Error::BufferLength {
                context: "joint durations",
                expected: 32,
                actual: durations.len(),
            });
        }

        let config = LaunchConfig {
            block_dim: (DECIDE_THREADS, 1, 1),
            ..rows(batch)?
        };
        let lens = [
            len(logits),
            len(lens_frames),
            len(durations),
            len(&filter.allowed),
            len(&filter.top_k),
            len_mut(&state.frame),
            len_mut(&state.emitted),
            len_mut(&state.label),
            len_mut(&state.emit),
            len_mut(&state.counts),
            len_mut(&state.faults),
            len_mut(&output.tokens),
            len_mut(&output.frames),
            len_mut(&output.durations),
            len_mut(&output.confidences),
        ];
        let dims = [
            u32_of("classes", shape.classes)?,
            u32_of("blank", shape.classes - 1)?,
            u32_of("max symbols", shape.max_symbols)?,
            u32_of("capacity", shape.capacity)?,
        ];
        let mut launch = stream.launch_builder(&self.decide);
        launch
            .arg(logits)
            .arg(&lens[0])
            .arg(lens_frames)
            .arg(&lens[1])
            .arg(durations)
            .arg(&lens[2]);
        launch
            .arg(&filter.allowed)
            .arg(&lens[3])
            .arg(&filter.top_k)
            .arg(&lens[4]);
        launch
            .arg(&mut state.frame)
            .arg(&lens[5])
            .arg(&mut state.emitted)
            .arg(&lens[6])
            .arg(&mut state.label)
            .arg(&lens[7]);
        launch
            .arg(&mut state.emit)
            .arg(&lens[8])
            .arg(&mut state.counts)
            .arg(&lens[9])
            .arg(&mut state.faults)
            .arg(&lens[10]);
        launch
            .arg(&mut output.tokens)
            .arg(&lens[11])
            .arg(&mut output.frames)
            .arg(&lens[12])
            .arg(&mut output.durations)
            .arg(&lens[13]);
        launch.arg(&mut output.confidences).arg(&lens[14]);
        for dim in &dims {
            launch.arg(dim);
        }
        // SAFETY: matches `decode_decide`; one block per row writes only its row's
        // state slots and its `capacity` output slots, all sized above
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// One predictor LSTM layer for the rows with `emit` set; the others copy their
    /// state to `state.h_next`
    pub(crate) fn lstm(
        &self,
        stream: &CudaStream,
        weights: &LstmWeights<'_>,
        input: &LstmInput<'_>,
        emit: &CudaView<'_, u32>,
        state: &mut LstmState<'_>,
        hidden: usize,
    ) -> Result<(), Error> {
        check_weight_row(hidden)?;
        let batch = emit.len();
        let n = element_count("lstm state", &[batch, hidden])?;
        let gates = element_count("lstm gates", &[4, hidden])?;
        let weight = element_count("lstm weight", &[gates, hidden])?;
        check_len("lstm input weight", weight, weights.weight_ih.len())?;
        check_len("lstm state weight", weight, weights.weight_hh.len())?;
        check_len("lstm bias", gates, weights.bias.len())?;
        check_len("lstm hidden", n, state.h.len())?;
        check_len("lstm next hidden", n, state.h_next.len())?;
        check_len("lstm cell", n, state.c.len())?;
        // the kernel's input reads are checked slice indexes, so a label past the
        // embedding table traps instead of reading out of bounds
        let (rows, label, embedded) = match input {
            LstmInput::Embedded { table, label } => {
                if hidden == 0 || table.len() % hidden != 0 {
                    return Err(Error::BufferLength {
                        context: "embedding table",
                        expected: hidden.max(1) * table.len().div_ceil(hidden.max(1)),
                        actual: table.len(),
                    });
                }
                check_len("labels", batch, label.len())?;
                (table, label, 1u32)
            }
            // the label is unread without `embedded`; `emit` stands in for it
            LstmInput::Below(below) => {
                check_len("lstm input", n, below.len())?;
                (below, emit, 0u32)
            }
        };

        let config = LaunchConfig {
            grid_dim: (u32_of("lstm units", hidden)?, 1, 1),
            block_dim: (WEIGHT_ROW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let lens = [
            len(&weights.weight_ih),
            len(&weights.weight_hh),
            len(&weights.bias),
            len(rows),
            len(label),
            len(emit),
            len(&state.h),
            len_mut(&state.h_next),
            len_mut(&state.c),
        ];
        let hidden = u32_of("hidden", hidden)?;
        let mut launch = stream.launch_builder(&self.lstm);
        launch
            .arg(&weights.weight_ih)
            .arg(&lens[0])
            .arg(&weights.weight_hh)
            .arg(&lens[1])
            .arg(&weights.bias)
            .arg(&lens[2]);
        launch
            .arg(rows)
            .arg(&lens[3])
            .arg(label)
            .arg(&lens[4])
            .arg(emit)
            .arg(&lens[5]);
        launch
            .arg(&state.h)
            .arg(&lens[6])
            .arg(&mut state.h_next)
            .arg(&lens[7])
            .arg(&mut state.c)
            .arg(&lens[8]);
        launch.arg(&hidden).arg(&embedded);
        // SAFETY: matches `decode_lstm`; one block per unit writes only that unit's
        // column of `h_next` and `c`, which are sized above; its unchecked reads are
        // within the weights checked above and the rows it slices with a check
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// The predictor's joint projection for the rows with `emit` set, the next
    /// step's joint input for every row, the joint bias into `logits`, and every
    /// layer's new state back into `h`
    pub(crate) fn project(
        &self,
        stream: &CudaStream,
        buffers: &mut Projection<'_>,
        emit: &CudaView<'_, u32>,
        frames: usize,
        hidden: usize,
    ) -> Result<(), Error> {
        check_weight_row(hidden)?;
        let batch = emit.len();
        let joint_hidden = buffers.bias.len();
        let outputs = buffers.joint_bias.len();
        let rows = element_count("joint input", &[batch, joint_hidden])?;
        if joint_hidden == 0 || outputs == 0 {
            return Err(Error::BufferLength {
                context: "joint widths",
                expected: 1,
                actual: 0,
            });
        }
        check_len(
            "predictor projection weight",
            element_count("predictor projection weight", &[joint_hidden, hidden])?,
            buffers.weight.len(),
        )?;
        check_len(
            "top state",
            element_count("top state", &[batch, hidden])?,
            buffers.top.len(),
        )?;
        check_len(
            "encoder projection",
            element_count("encoder projection", &[rows, frames])?,
            buffers.enc_proj.len(),
        )?;
        check_len("frame cursors", batch, buffers.frame.len())?;
        check_len("frame lengths", batch, buffers.lens.len())?;
        check_len("predictor states", buffers.h.len(), buffers.h_next.len())?;
        check_len("predictor projection", rows, buffers.pred_proj.len())?;
        check_len("joint input", rows, buffers.joint_in.len())?;
        check_len(
            "joint logits",
            element_count("joint logits", &[batch, outputs])?,
            buffers.logits.len(),
        )?;

        let config = LaunchConfig {
            grid_dim: (u32_of("projection grid", joint_hidden)?, 1, 1),
            block_dim: (WEIGHT_ROW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let lens = [
            len(&buffers.weight),
            len(&buffers.bias),
            len(&buffers.top),
            len(emit),
            len(&buffers.enc_proj),
            len(&buffers.frame),
            len(&buffers.lens),
            len(&buffers.joint_bias),
            len(&buffers.h_next),
            len_mut(&buffers.h),
            len_mut(&buffers.pred_proj),
            len_mut(&buffers.joint_in),
            len_mut(&buffers.logits),
        ];
        let dims = [u32_of("frames", frames)?, u32_of("hidden", hidden)?];
        let mut launch = stream.launch_builder(&self.project);
        launch
            .arg(&buffers.weight)
            .arg(&lens[0])
            .arg(&buffers.bias)
            .arg(&lens[1])
            .arg(&buffers.top)
            .arg(&lens[2]);
        launch
            .arg(emit)
            .arg(&lens[3])
            .arg(&buffers.enc_proj)
            .arg(&lens[4])
            .arg(&buffers.frame)
            .arg(&lens[5]);
        launch
            .arg(&buffers.lens)
            .arg(&lens[6])
            .arg(&buffers.joint_bias)
            .arg(&lens[7])
            .arg(&buffers.h_next)
            .arg(&lens[8]);
        launch
            .arg(&mut buffers.h)
            .arg(&lens[9])
            .arg(&mut buffers.pred_proj)
            .arg(&lens[10])
            .arg(&mut buffers.joint_in)
            .arg(&lens[11]);
        launch.arg(&mut buffers.logits).arg(&lens[12]);
        launch.arg(&dims[0]).arg(&dims[1]);
        // SAFETY: matches `decode_project`; each block writes only its column of
        // `pred_proj` and `joint_in`, the side copies stay below `logits.len()` and
        // `h.len()`, every buffer is sized above, and its unchecked weight reads are
        // within the weight checked above
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use cudakit::Options;
    use cudakit::cudarc::driver::CudaSlice;

    use super::{CudaRuntime, DecodeKernels, LstmInput, LstmState, LstmWeights, Projection};

    /// Predictor sizes for the decode kernel test: 17 rows cross two 8-row passes,
    /// and a hidden width that is not a multiple of a block's 128 threads leaves
    /// some threads a column short
    const BATCH: usize = 17;
    const HIDDEN: usize = 200;
    const LAYERS: usize = 2;
    const CLASSES: usize = 11;
    const JOINT_HIDDEN: usize = 24;
    const OUTPUTS: usize = 7;
    const FRAMES: usize = 3;

    /// Deterministic values in `[-scale, scale)`
    struct Values(u64);

    impl Values {
        fn next(&mut self, scale: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        }

        fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
            (0..len).map(|_| self.next(scale)).collect()
        }

        fn below(&mut self, bound: usize) -> u32 {
            ((self.next(1.0) + 1.0) * 0.5 * bound as f32) as u32 % bound as u32
        }
    }

    /// The predictor's weights and the state a step carries over, on the host
    struct Predictor {
        weight_ih: Vec<Vec<f32>>,
        weight_hh: Vec<Vec<f32>>,
        bias: Vec<Vec<f32>>,
        embed: Vec<f32>,
        proj_weight: Vec<f32>,
        proj_bias: Vec<f32>,
        enc_proj: Vec<f32>,
        joint_bias: Vec<f32>,
        h: Vec<f32>,
        c: Vec<f32>,
        pred_proj: Vec<f32>,
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    impl Predictor {
        /// One step on the host: the LSTM layers and the projection for the rows
        /// with `emit` set, then every row's joint input and bias-filled logits
        fn step(&mut self, emit: &[u32], label: &[u32], frame: &[u32], lens: &[u32]) -> Step {
            let n = BATCH * HIDDEN;
            let mut h_next = vec![0.0; LAYERS * n];
            for layer in 0..LAYERS {
                for b in 0..BATCH {
                    let row = layer * n + b * HIDDEN;
                    if emit[b] == 0 {
                        h_next[row..row + HIDDEN].copy_from_slice(&self.h[row..row + HIDDEN]);
                        continue;
                    }

                    let x: Vec<f32> = if layer == 0 {
                        let at = label[b] as usize * HIDDEN;
                        self.embed[at..at + HIDDEN].to_vec()
                    } else {
                        let at = (layer - 1) * n + b * HIDDEN;
                        h_next[at..at + HIDDEN].to_vec()
                    };
                    let gate = |g: usize, unit: usize| {
                        let j = g * HIDDEN + unit;
                        (0..HIDDEN)
                            .map(|k| {
                                self.weight_ih[layer][j * HIDDEN + k] * x[k]
                                    + self.weight_hh[layer][j * HIDDEN + k] * self.h[row + k]
                            })
                            .sum::<f32>()
                            + self.bias[layer][j]
                    };
                    for unit in 0..HIDDEN {
                        let cell = sigmoid(gate(1, unit)) * self.c[row + unit]
                            + sigmoid(gate(0, unit)) * gate(2, unit).tanh();
                        self.c[row + unit] = cell;
                        h_next[row + unit] = sigmoid(gate(3, unit)) * cell.tanh();
                    }
                }
            }

            let top = (LAYERS - 1) * n;
            let mut joint_in = vec![0.0; BATCH * JOINT_HIDDEN];
            for b in 0..BATCH {
                let t = frame[b].min(lens[b] - 1) as usize;
                for col in 0..JOINT_HIDDEN {
                    let i = b * JOINT_HIDDEN + col;
                    if emit[b] != 0 {
                        self.pred_proj[i] = (0..HIDDEN)
                            .map(|k| {
                                self.proj_weight[col * HIDDEN + k] * h_next[top + b * HIDDEN + k]
                            })
                            .sum::<f32>()
                            + self.proj_bias[col];
                    }
                    let encoded = self.enc_proj[(b * FRAMES + t) * JOINT_HIDDEN + col];
                    joint_in[i] = (encoded + self.pred_proj[i]).max(0.0);
                }
            }
            self.h = h_next;

            Step {
                h: self.h.clone(),
                c: self.c.clone(),
                pred_proj: self.pred_proj.clone(),
                joint_in,
                logits: (0..BATCH * OUTPUTS)
                    .map(|i| self.joint_bias[i % OUTPUTS])
                    .collect(),
            }
        }
    }

    /// The state a step leaves for the next one
    struct Step {
        h: Vec<f32>,
        c: Vec<f32>,
        pred_proj: Vec<f32>,
        joint_in: Vec<f32>,
        logits: Vec<f32>,
    }

    /// Holds a device buffer of rows `width` wide to the host's, within float
    /// rounding of a different summation order
    fn assert_close(name: &str, step: usize, width: usize, device: &[f32], host: &[f32]) {
        assert_eq!(device.len(), host.len(), "{name} length");
        for (i, (d, h)) in device.iter().zip(host).enumerate() {
            assert!(
                (d - h).abs() <= 1e-5 + 1e-4 * h.abs(),
                "step {step} {name}[{i}] (row {}): device {d}, host {h}",
                (i / width) % BATCH
            );
        }
    }

    /// `decode_lstm` and `decode_project` over two steps against a host predictor:
    /// rows past the first 8-row pass, rows that skip a step keeping their state and
    /// projection, and the new state copied back for the next step
    #[test]
    #[ignore = "needs a CUDA GPU; `just gpu-parity` runs it on gpu0"]
    fn gpu_predictor_step_matches_the_host() {
        let runtime = CudaRuntime::open(0, Options::default()).expect("CUDA runtime");
        let kernels = DecodeKernels::load(&runtime).expect("kernels");
        let stream = runtime.stream();
        let mut values = Values(7);
        let gates = 4 * HIDDEN;
        let n = BATCH * HIDDEN;
        let mut host = Predictor {
            weight_ih: (0..LAYERS)
                .map(|_| values.vec(gates * HIDDEN, 0.1))
                .collect(),
            weight_hh: (0..LAYERS)
                .map(|_| values.vec(gates * HIDDEN, 0.1))
                .collect(),
            bias: (0..LAYERS).map(|_| values.vec(gates, 0.5)).collect(),
            embed: values.vec(CLASSES * HIDDEN, 1.0),
            proj_weight: values.vec(JOINT_HIDDEN * HIDDEN, 0.1),
            proj_bias: values.vec(JOINT_HIDDEN, 0.5),
            enc_proj: values.vec(BATCH * FRAMES * JOINT_HIDDEN, 1.0),
            joint_bias: values.vec(OUTPUTS, 1.0),
            h: values.vec(LAYERS * n, 0.5),
            c: values.vec(LAYERS * n, 0.5),
            pred_proj: values.vec(BATCH * JOINT_HIDDEN, 0.5),
        };

        let upload = |data: &[f32]| stream.clone_htod(data).expect("upload");
        let weight_ih: Vec<CudaSlice<f32>> = host.weight_ih.iter().map(|w| upload(w)).collect();
        let weight_hh: Vec<CudaSlice<f32>> = host.weight_hh.iter().map(|w| upload(w)).collect();
        let bias: Vec<CudaSlice<f32>> = host.bias.iter().map(|b| upload(b)).collect();
        let embed = upload(&host.embed);
        let proj_weight = upload(&host.proj_weight);
        let proj_bias = upload(&host.proj_bias);
        let enc_proj = upload(&host.enc_proj);
        let joint_bias = upload(&host.joint_bias);
        let mut h = upload(&host.h);
        let mut c = upload(&host.c);
        let mut pred_proj = upload(&host.pred_proj);
        // stale values a step must overwrite
        let mut h_next = upload(&vec![9.0; LAYERS * n]);
        let mut joint_in = upload(&vec![9.0; BATCH * JOINT_HIDDEN]);
        let mut logits = upload(&vec![9.0; BATCH * OUTPUTS]);

        // different rows emit on each step, and the last row of the second pass
        // emits on both
        let emits: [fn(usize) -> bool; 2] = [|b| b % 2 == 0, |b| b % 3 == 0 || b == BATCH - 1];
        for (step, emits) in emits.into_iter().enumerate() {
            let emit: Vec<u32> = (0..BATCH).map(|b| u32::from(emits(b))).collect();
            let label: Vec<u32> = (0..BATCH).map(|_| values.below(CLASSES)).collect();
            let lens: Vec<u32> = (0..BATCH).map(|_| values.below(FRAMES) + 1).collect();
            // past the last frame on some rows, which the projection clamps
            let frame: Vec<u32> = (0..BATCH).map(|_| values.below(FRAMES + 1)).collect();
            let expected = host.step(&emit, &label, &frame, &lens);

            let emit_device = stream.clone_htod(&emit).expect("upload");
            let label_device = stream.clone_htod(&label).expect("upload");
            let frame_device = stream.clone_htod(&frame).expect("upload");
            let lens_device = stream.clone_htod(&lens).expect("upload");
            for layer in 0..LAYERS {
                let weights = LstmWeights {
                    weight_ih: weight_ih[layer].slice(..),
                    weight_hh: weight_hh[layer].slice(..),
                    bias: bias[layer].slice(..),
                };
                let (below, mut rest) = h_next.split_at_mut(layer * n);
                let input = match layer {
                    0 => LstmInput::Embedded {
                        table: embed.slice(..),
                        label: label_device.slice(..),
                    },
                    _ => LstmInput::Below(below.slice((layer - 1) * n..)),
                };
                let mut state = LstmState {
                    h: h.slice(layer * n..(layer + 1) * n),
                    h_next: rest.slice_mut(..n),
                    c: c.slice_mut(layer * n..(layer + 1) * n),
                };
                kernels
                    .lstm(
                        stream,
                        &weights,
                        &input,
                        &emit_device.slice(..),
                        &mut state,
                        HIDDEN,
                    )
                    .expect("lstm");
            }

            let mut buffers = Projection {
                weight: proj_weight.slice(..),
                bias: proj_bias.slice(..),
                top: h_next.slice((LAYERS - 1) * n..),
                enc_proj: enc_proj.slice(..),
                frame: frame_device.slice(..),
                lens: lens_device.slice(..),
                joint_bias: joint_bias.slice(..),
                h_next: h_next.slice(..),
                h: h.slice_mut(..),
                pred_proj: pred_proj.slice_mut(..),
                joint_in: joint_in.slice_mut(..),
                logits: logits.slice_mut(..),
            };
            kernels
                .project(stream, &mut buffers, &emit_device.slice(..), FRAMES, HIDDEN)
                .expect("project");

            let download = |data: &CudaSlice<f32>| stream.clone_dtoh(data).expect("download");
            let device = Step {
                h: download(&h),
                c: download(&c),
                pred_proj: download(&pred_proj),
                joint_in: download(&joint_in),
                logits: download(&logits),
            };
            assert_close("h", step, HIDDEN, &device.h, &expected.h);
            assert_close("c", step, HIDDEN, &device.c, &expected.c);
            assert_close(
                "pred_proj",
                step,
                JOINT_HIDDEN,
                &device.pred_proj,
                &expected.pred_proj,
            );
            assert_close(
                "joint_in",
                step,
                JOINT_HIDDEN,
                &device.joint_in,
                &expected.joint_in,
            );
            assert_close("logits", step, OUTPUTS, &device.logits, &expected.logits);
        }
    }
}

//! The device language filter against the host decision rule on synthetic logits

use cudakit::cudarc::driver::CudaSlice;

use super::runtime;
use crate::backend::{DecodePolicy, LanguageMask, decide_from_logits};
use crate::cuda::decoder::{CudaDecoder, DecodeWorkspace};
use crate::cuda::kernels::CudaRuntime;
use crate::cuda::test_support::{native_root, weights};
use crate::deployment::{ParakeetDeployment, ParakeetVersion};
use crate::language::{LanguageSelection, ParakeetLanguage};
use crate::models::ModelBundle;

/// Rows in the language filter fixture, 48 of each of its eight kinds
const FILTER_ROWS: usize = 384;

/// Seeded generator, so a failing fixture row reproduces
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 40
    }

    /// Uniform in `[low, high)`
    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * (self.next() as f32 / (1u64 << 24) as f32)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// `count` distinct entries of `ids`
    fn pick(&mut self, ids: &[usize], count: usize) -> Vec<usize> {
        let mut ids = ids.to_vec();
        for i in 0..count {
            let j = i + self.below(ids.len() - i);
            ids.swap(i, j);
        }
        ids.truncate(count);
        ids
    }
}

/// Joint logits, one row per decision, built around `mask` so the filter meets
/// its edge cases: ties at the max and at the K-th rank, a best allowed id
/// ranked K-1 and K, nothing allowed in the top K, and a blank argmax
fn filter_fixture(mask: &LanguageMask, bins: usize, seed: u64) -> Vec<Vec<f32>> {
    let flags = mask.allowed();
    let classes = flags.len();
    let blank = classes - 1;
    let k = mask.top_k();
    let allowed: Vec<usize> = (0..blank).filter(|id| flags[*id]).collect();
    let others: Vec<usize> = (0..blank).filter(|id| !flags[*id]).collect();
    assert!(
        allowed.len() > k + 32 && others.len() > k + 32,
        "{} allowed and {} other ids are too few for the fixture",
        allowed.len(),
        others.len()
    );

    let mut rng = Lcg(seed);
    (0..FILTER_ROWS)
        .map(|row| {
            let mut logits = vec![0.0f32; classes + bins];
            let tokens = &mut logits[..classes];
            let noise = |rng: &mut Lcg, tokens: &mut [f32]| {
                tokens.iter_mut().for_each(|v| *v = rng.uniform(-2.0, 2.0));
            };
            let floor = |rng: &mut Lcg, tokens: &mut [f32]| {
                tokens
                    .iter_mut()
                    .for_each(|v| *v = rng.uniform(-10.0, -5.0));
            };
            match row % 8 {
                // noise with rare spikes, the shape of the host rule's own test
                0 => tokens.iter_mut().for_each(|v| {
                    let unit = rng.uniform(0.0, 1.0);
                    *v = if unit > 0.97 {
                        unit * 12.0
                    } else {
                        unit * 4.0 - 2.0
                    };
                }),
                // half steps, so ties fall everywhere, the max and the K-th included
                1 => tokens
                    .iter_mut()
                    .for_each(|v| *v = (rng.uniform(-6.0, 6.0) * 2.0).round() / 2.0),
                // a blank argmax, which the filter leaves alone
                2 => {
                    noise(&mut rng, tokens);
                    tokens[blank] = 9.0;
                }
                // the best allowed id ranks K-1, the last place the filter takes,
                // or K, the first it does not
                3 | 4 => {
                    floor(&mut rng, tokens);
                    let ahead = if row % 8 == 3 { k - 1 } else { k };
                    for id in rng.pick(&others, ahead) {
                        tokens[id] = rng.uniform(5.0, 10.0);
                    }
                    tokens[allowed[rng.below(allowed.len())]] = 4.0;
                }
                // twelve equal logits straddle the K-th rank, so the top K takes
                // the four lowest ids, which may or may not hold an allowed one
                5 => {
                    floor(&mut rng, tokens);
                    let picked = rng.pick(&others, k - 4 + 8);
                    let (high, tied) = picked.split_at(k - 4);
                    for id in high {
                        tokens[*id] = rng.uniform(6.0, 10.0);
                    }
                    for id in tied.iter().chain(&rng.pick(&allowed, 4)) {
                        tokens[*id] = 5.0;
                    }
                }
                // no allowed id anywhere near the top K
                6 => {
                    floor(&mut rng, tokens);
                    for id in rng.pick(&others, k + 20) {
                        tokens[id] = rng.uniform(0.0, 10.0);
                    }
                }
                // an allowed and another id tie at the max, the lower id winning
                // the argmax
                _ => {
                    noise(&mut rng, tokens);
                    tokens[allowed[rng.below(allowed.len())]] = 9.0;
                    tokens[others[rng.below(others.len())]] = 9.0;
                }
            }
            // whole numbers, so duration ties come up too
            for v in &mut logits[classes..] {
                *v = rng.uniform(-3.0, 3.0).round();
            }
            logits
        })
        .collect()
}

/// Device state after one decision per row
struct Decided {
    counts: Vec<u32>,
    faults: Vec<u32>,
    frame: Vec<u32>,
    tokens: Vec<u32>,
    durations: Vec<u32>,
    confidences: Vec<f32>,
}

/// One `decide` launch over `rows`, each with a single frame left, under `mask`
fn decide_once(
    runtime: &CudaRuntime,
    decoder: &CudaDecoder,
    ws: &mut DecodeWorkspace,
    rows: &[Vec<f32>],
    mask: Option<&LanguageMask>,
) -> Decided {
    let stream = runtime.stream();
    let flat: Vec<f32> = rows.concat();
    stream.memcpy_htod(&flat, &mut ws.logits).expect("logits");
    stream
        .memcpy_htod(&vec![1u32; ws.batch], &mut ws.lens)
        .expect("lens");
    // `decide` only writes the label and emit flags this also sets
    decoder
        .reset_cursors(runtime, ws, &vec![0; ws.batch])
        .expect("reset");
    decoder.set_language(runtime, ws, mask).expect("filter");
    decoder.decide(runtime, ws).expect("decide");

    let read = |buffer: &CudaSlice<u32>| stream.clone_dtoh(buffer).expect("read back");
    Decided {
        counts: read(&ws.counts),
        faults: read(&ws.faults),
        frame: read(&ws.frame),
        tokens: read(&ws.tokens),
        durations: read(&ws.token_durations),
        confidences: stream.clone_dtoh(&ws.confidences).expect("read back"),
    }
}

/// One decision per row of synthetic v3 logits, with each language's filter and
/// with none, against the host rule `decide_from_logits`: the same token and
/// duration, and the same confidence within the sum-order tolerances. Each
/// fixture also runs under the next language's mask straight after its own, at
/// the same K, so a mask left over from the previous call picks the wrong ids.
/// Rows with a non-finite class logit fault under the filter, as the host rule
/// rejects them
#[test]
#[ignore = "needs a CUDA GPU and the native weights; `just gpu-parity` runs it on gpu0"]
fn gpu_language_filter_matches_the_host_rule() {
    // the device adds the top-K softmax terms in the host's order, so only `exp`
    // rounding (2 ulp on the device) separates the two
    const TOP_K_TOLERANCE: f32 = 1e-6;
    // the full softmax adds 8193 terms one after another on the host but strided
    // and then as a tree on the device; on peaked rows the host's running sum
    // drops terms below half an ulp, about 4e-6 at worst on this fixture when
    // both orders are simulated in f32
    const FULL_SOFTMAX_TOLERANCE: f32 = 2e-5;

    let runtime = runtime();
    let deployment = ParakeetDeployment::v3();
    let file = weights(ParakeetVersion::V3);
    let spec = file.parakeet_spec(&deployment).expect("spec");
    let decoder = CudaDecoder::load(&runtime, &file, &spec.transducer).expect("decoder");
    let vocab = ModelBundle::from_native_dir(native_root(), deployment.clone())
        .vocabulary()
        .expect("vocabulary");
    let blank = vocab.blank_id();
    assert_eq!(blank + 1, decoder.classes, "vocabulary and joint classes");
    let bins = deployment.duration_bins().len();
    let auto = DecodePolicy::new(&deployment, &vocab, LanguageSelection::Auto).expect("policy");
    let mut ws = decoder
        .workspace(&runtime, FILTER_ROWS, 1)
        .expect("workspace");

    let (mut retargeted, mut rescored, mut outside) = (0, 0, 0);
    let languages = [
        ParakeetLanguage::Polish,
        ParakeetLanguage::Russian,
        ParakeetLanguage::Greek,
    ];
    let policies: Vec<DecodePolicy<'_>> = languages
        .iter()
        .map(|language| {
            DecodePolicy::new(&deployment, &vocab, LanguageSelection::Language(*language))
                .expect("policy")
        })
        .collect();
    let masks: Vec<LanguageMask> = policies
        .iter()
        .map(|policy| {
            policy
                .language_mask()
                .expect("v3 filters an explicit language")
        })
        .collect();
    for (seed, language) in languages.into_iter().enumerate() {
        let rows = filter_fixture(&masks[seed], bins, seed as u64);
        let next = (seed + 1) % languages.len();

        for active in [Some(seed), Some(next), None] {
            let (policy, filter) = match active {
                Some(index) => (&policies[index], Some(&masks[index])),
                None => (&policies[seed], None),
            };
            let decided = decide_once(&runtime, &decoder, &mut ws, &rows, filter);
            for (row, logits) in rows.iter().enumerate() {
                let context = format!(
                    "{language:?} row {row}, filter {:?}",
                    active.map(|index| languages[index])
                );
                let host = decide_from_logits(policy, logits, filter).expect("host rule");
                let plain = decide_from_logits(&auto, logits, None).expect("host rule");
                assert_eq!(decided.faults[row], 0, "{context}: fault");
                if host.token_id == blank {
                    assert_eq!(decided.counts[row], 0, "{context}: emitted for a blank");
                    assert_eq!(
                        decided.frame[row] as usize,
                        host.duration_frames.max(1),
                        "{context}: advance"
                    );
                    continue;
                }

                let slot = row * ws.capacity;
                assert_eq!(decided.counts[row], 1, "{context}: emitted");
                assert_eq!(
                    decided.tokens[slot] as usize, host.token_id,
                    "{context}: token"
                );
                assert_eq!(
                    decided.durations[slot] as usize, host.duration_frames,
                    "{context}: duration"
                );
                // an allowed argmax ranks first, so the top-K softmax scores it, as
                // it does a filtered token; otherwise no allowed id made the top K
                let top_k = filter.is_some_and(|mask| {
                    host.token_id != plain.token_id || mask.allowed()[plain.token_id]
                });
                let tolerance = if top_k {
                    TOP_K_TOLERANCE
                } else {
                    FULL_SOFTMAX_TOLERANCE
                };
                let confidence = decided.confidences[slot];
                assert!(
                    (confidence - host.confidence).abs() <= tolerance,
                    "{context}: confidence {confidence} against {}",
                    host.confidence
                );

                match (filter.is_some(), top_k) {
                    (true, true) if host.token_id != plain.token_id => retargeted += 1,
                    (true, true) => rescored += 1,
                    (true, false) => outside += 1,
                    (false, _) => {}
                }
            }
        }
    }

    // non-finite class logits: a low -inf under a finite max, the max at +inf, a
    // NaN, a -inf beside a blank max, which the filter itself skips, and fewer
    // finite classes than K
    let mut rows = filter_fixture(&masks[0], bins, 7);
    let finite = blank / 2;
    rows[0][3] = f32::NEG_INFINITY;
    rows[1][finite] = f32::INFINITY;
    rows[2][finite] = f32::NAN;
    rows[3][blank] = 50.0;
    rows[3][3] = f32::NEG_INFINITY;
    rows[4][..blank].fill(f32::NEG_INFINITY);
    rows[4][finite..finite + 10].fill(1.0);
    let decided = decide_once(&runtime, &decoder, &mut ws, &rows, Some(&masks[0]));
    for (row, logits) in rows.iter().enumerate().take(5) {
        assert!(
            decide_from_logits(&policies[0], logits, Some(&masks[0])).is_err(),
            "row {row}: the host rule accepted a non-finite logit"
        );
        assert_ne!(decided.faults[row], 0, "non-finite row {row} did not fault");
    }
    assert!(
        decoder.output(&runtime, &mut ws).is_err(),
        "faulted rows reached the output"
    );

    // the fixture must reach every branch of the filter, not only the argmax
    println!("{retargeted} retargeted, {rescored} rescored, {outside} outside the top K");
    assert!(
        retargeted > 100 && rescored > 50 && outside > 100,
        "{retargeted} retargeted, {rescored} rescored, {outside} outside the top K"
    );
}

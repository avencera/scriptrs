use super::LongFormConfig;
#[cfg(feature = "vad")]
use crate::constants::VAD_WINDOW_SAMPLES;
use crate::constants::{SAMPLE_RATE, SAMPLES_PER_ENCODER_FRAME};
use crate::deployment::ParakeetDeployment;
use crate::error::TranscriptionError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SampleRange {
    pub start: usize,
    pub end: usize,
}

/// Segmentation parameters for converting VAD probabilities into speech regions
///
/// The model hard sample maximum is not stored here. Pass
/// [`ParakeetDeployment::hard_sample_max`] into the planner
#[cfg(feature = "vad")]
#[derive(Debug, Clone)]
pub struct VadSegmentationConfig {
    /// Speech probability at or above which a VAD window counts as speech
    pub threshold: f32,
    /// Minimum speech span kept in the output
    pub min_speech_duration: f64,
    /// Minimum silence span required to end a speech region
    pub min_silence_duration: f64,
    /// Padding added around speech regions
    pub speech_padding: f64,
    /// Maximum probability considered a valid silence split candidate
    pub silence_threshold_for_split: f32,
    /// Optional override for the negative hysteresis threshold
    pub negative_threshold: Option<f32>,
    /// Offset applied when deriving the negative threshold from the positive threshold
    pub negative_threshold_offset: f32,
    /// Minimum silence duration considered valid when splitting long speech
    pub min_silence_at_max_speech: f64,
}

#[cfg(feature = "vad")]
impl Default for VadSegmentationConfig {
    fn default() -> Self {
        Self {
            // Silero's own default; 0.85 dropped low-confidence phone and far-field
            // speech, raising long-form WER on every set measured (README)
            threshold: 0.5,
            min_speech_duration: 0.15,
            min_silence_duration: 0.75,
            // Silero crosses the threshold up to ~220 ms into a soft onset, so 0.1 s
            // clipped short first words ("a cold" became "cold"); 0.2 s lowered
            // long-form WER on every set measured, larger values cost AMI headset
            // accuracy (README)
            speech_padding: 0.2,
            silence_threshold_for_split: 0.3,
            negative_threshold: None,
            negative_threshold_offset: 0.15,
            min_silence_at_max_speech: 0.098,
        }
    }
}

#[cfg(feature = "vad")]
impl VadSegmentationConfig {
    pub(crate) fn validate(&self) -> Result<(), TranscriptionError> {
        validate_probability("vad.threshold", self.threshold)?;
        validate_positive_duration("vad.min_speech_duration", self.min_speech_duration)?;
        validate_positive_duration("vad.min_silence_duration", self.min_silence_duration)?;
        validate_positive_duration("vad.speech_padding", self.speech_padding)?;
        validate_probability(
            "vad.silence_threshold_for_split",
            self.silence_threshold_for_split,
        )?;
        validate_probability(
            "vad.negative_threshold_offset",
            self.negative_threshold_offset,
        )?;
        validate_positive_duration(
            "vad.min_silence_at_max_speech",
            self.min_silence_at_max_speech,
        )?;

        if let Some(negative_threshold) = self.negative_threshold {
            validate_probability("vad.negative_threshold", negative_threshold)?;
            if negative_threshold > self.threshold {
                return Err(TranscriptionError::InvalidConfiguration(format!(
                    "vad.negative_threshold={negative_threshold} exceeds vad.threshold={}",
                    self.threshold
                )));
            }
        }

        Ok(())
    }

    fn effective_negative_threshold(&self) -> f32 {
        if let Some(override_threshold) = self.negative_threshold {
            return override_threshold;
        }
        (self.threshold - self.negative_threshold_offset).max(0.01)
    }
}

/// Overlap-window settings used when VAD cannot find a safe silence split
#[derive(Debug, Clone)]
pub struct OverlapChunkConfig {
    /// Desired overlap between adjacent fallback windows
    pub overlap_seconds: f64,
    /// Left-context samples prepended to non-first fallback windows
    pub context_samples: usize,
    /// Maximum samples accepted by the base Parakeet model
    pub max_model_samples: usize,
}

impl Default for OverlapChunkConfig {
    fn default() -> Self {
        Self::from_deployment(&ParakeetDeployment::v2())
    }
}

impl OverlapChunkConfig {
    /// Build overlap settings from a deployment hard sample maximum
    pub fn from_deployment(deployment: &ParakeetDeployment) -> Self {
        Self {
            overlap_seconds: 2.0,
            context_samples: deployment.frontend().samples_per_encoder_frame(),
            max_model_samples: deployment.hard_sample_max(),
        }
    }

    /// Validate overlap settings against the deployment hard maximum
    pub fn validate_for(&self, deployment: &ParakeetDeployment) -> Result<(), TranscriptionError> {
        if self.max_model_samples == 0 {
            return Err(TranscriptionError::InvalidConfiguration(
                "overlap max_model_samples must be positive".to_owned(),
            ));
        }
        if self.max_model_samples > deployment.hard_sample_max() {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "overlap max_model_samples={} exceeds deployment hard sample maximum {}",
                self.max_model_samples,
                deployment.hard_sample_max()
            )));
        }
        if self.context_samples > deployment.hard_sample_max() {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "overlap context_samples={} exceeds deployment hard sample maximum {}",
                self.context_samples,
                deployment.hard_sample_max()
            )));
        }
        if !self.overlap_seconds.is_finite() || self.overlap_seconds < 0.0 {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "overlap overlap_seconds={} must be finite and non-negative",
                self.overlap_seconds
            )));
        }

        let max_overlap_seconds =
            deployment.hard_sample_max() as f64 / deployment.frontend().sample_rate() as f64;
        if self.overlap_seconds > max_overlap_seconds {
            return Err(TranscriptionError::InvalidConfiguration(format!(
                "overlap overlap_seconds={} exceeds deployment hard duration {max_overlap_seconds}",
                self.overlap_seconds
            )));
        }

        Ok(())
    }

    pub(crate) fn chunk_samples(&self) -> usize {
        let max_actual_chunk = self.max_model_samples.saturating_sub(self.context_samples);
        let raw = max_actual_chunk
            .saturating_sub(crate::constants::MEL_HOP_SAMPLES)
            .max(SAMPLES_PER_ENCODER_FRAME);
        raw / SAMPLES_PER_ENCODER_FRAME * SAMPLES_PER_ENCODER_FRAME
    }

    pub(crate) fn overlap_samples(&self) -> usize {
        let requested = (self.overlap_seconds * SAMPLE_RATE as f64) as usize;
        let capped = requested.min(self.chunk_samples() / 2);
        capped / SAMPLES_PER_ENCODER_FRAME * SAMPLES_PER_ENCODER_FRAME
    }

    pub(crate) fn stride_samples(&self) -> usize {
        let raw = self.chunk_samples().saturating_sub(self.overlap_samples());
        raw.max(SAMPLES_PER_ENCODER_FRAME) / SAMPLES_PER_ENCODER_FRAME * SAMPLES_PER_ENCODER_FRAME
    }

    pub(crate) fn plan(&self, range: SampleRange) -> Vec<SampleRange> {
        let mut chunks = Vec::new();
        let mut start = range.start;
        while start < range.end {
            let end = (start + self.chunk_samples()).min(range.end);
            chunks.push(SampleRange { start, end });
            if end == range.end {
                break;
            }
            start += self.stride_samples();
        }
        chunks
    }
}

#[cfg(feature = "vad")]
fn validate_probability(field: &str, value: f32) -> Result<(), TranscriptionError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        return Ok(());
    }

    Err(TranscriptionError::InvalidConfiguration(format!(
        "{field}={value} must be finite and in the range 0..=1"
    )))
}

#[cfg(feature = "vad")]
fn validate_positive_duration(field: &str, value: f64) -> Result<(), TranscriptionError> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }

    Err(TranscriptionError::InvalidConfiguration(format!(
        "{field}={value} must be finite and positive"
    )))
}

#[cfg(feature = "vad")]
pub(crate) fn detect_speech_regions(
    probabilities: &[f32],
    audio_length_samples: usize,
    config: &VadSegmentationConfig,
) -> Vec<SampleRange> {
    if probabilities.is_empty() || audio_length_samples == 0 {
        return Vec::new();
    }

    let hop_size = VAD_WINDOW_SAMPLES;
    let min_speech_samples = (config.min_speech_duration * SAMPLE_RATE as f64) as usize;
    let min_silence_samples = (config.min_silence_duration * SAMPLE_RATE as f64) as usize;
    let speech_pad_samples = (config.speech_padding * SAMPLE_RATE as f64) as usize;
    let threshold = config.threshold;
    let negative_threshold = config.effective_negative_threshold();

    let mut triggered = false;
    let mut current_speech_start = 0usize;
    let mut temp_end = None;
    let mut speeches = Vec::new();

    for (index, probability) in probabilities.iter().copied().enumerate() {
        let frame_start = index * hop_size;
        if probability >= threshold {
            temp_end = None;
            if !triggered {
                triggered = true;
                current_speech_start = frame_start;
            }
            continue;
        }

        if probability < negative_threshold && triggered {
            if temp_end.is_none() {
                temp_end = Some(frame_start);
            }
            if let Some(start_silence) = temp_end
                && frame_start.saturating_sub(start_silence) >= min_silence_samples
            {
                if start_silence.saturating_sub(current_speech_start) >= min_speech_samples {
                    speeches.push(SampleRange {
                        start: current_speech_start,
                        end: start_silence,
                    });
                }
                triggered = false;
                temp_end = None;
            }
        }
    }

    if triggered && audio_length_samples.saturating_sub(current_speech_start) >= min_speech_samples
    {
        speeches.push(SampleRange {
            start: current_speech_start,
            end: audio_length_samples,
        });
    }

    if speeches.is_empty() {
        return Vec::new();
    }

    let mut adjusted = speeches;
    for index in 0..adjusted.len() {
        if index == 0 {
            adjusted[index].start = adjusted[index].start.saturating_sub(speech_pad_samples);
        }
        if index < adjusted.len() - 1 {
            let silence = adjusted[index + 1]
                .start
                .saturating_sub(adjusted[index].end);
            if silence < 2 * speech_pad_samples {
                let half = silence / 2;
                adjusted[index].end = (adjusted[index].end + half).min(audio_length_samples);
                adjusted[index + 1].start = adjusted[index + 1].start.saturating_sub(half);
            } else {
                adjusted[index].end =
                    (adjusted[index].end + speech_pad_samples).min(audio_length_samples);
                adjusted[index + 1].start =
                    adjusted[index + 1].start.saturating_sub(speech_pad_samples);
            }
        } else {
            adjusted[index].end =
                (adjusted[index].end + speech_pad_samples).min(audio_length_samples);
        }
    }

    adjusted.retain(|range| range.end > range.start);
    adjusted
}

#[cfg(feature = "vad")]
pub(crate) fn region_probability_slice(probabilities: &[f32], region: SampleRange) -> &[f32] {
    let frame_start = region.start / VAD_WINDOW_SAMPLES;
    let frame_end = region.end.div_ceil(VAD_WINDOW_SAMPLES);
    &probabilities[frame_start.min(probabilities.len())..frame_end.min(probabilities.len())]
}

#[cfg(feature = "vad")]
pub(crate) fn plan_region_subsegments(
    region: SampleRange,
    region_probabilities: &[f32],
    config: &VadSegmentationConfig,
    max_chunk_samples: usize,
) -> Option<Vec<SampleRange>> {
    if region.end.saturating_sub(region.start) <= max_chunk_samples {
        return Some(vec![region]);
    }

    let min_silence_samples = (config.min_silence_at_max_speech * SAMPLE_RATE as f64) as usize;
    let speech_pad_samples = (config.speech_padding * SAMPLE_RATE as f64) as usize;
    let silence_threshold = config.silence_threshold_for_split;
    let mut spans = silence_spans(region, region_probabilities, silence_threshold);
    spans.retain(|span| span.end.saturating_sub(span.start) >= min_silence_samples);

    let mut segments = Vec::new();
    let mut cursor = region.start;

    while region.end.saturating_sub(cursor) > max_chunk_samples {
        let search_end = cursor + max_chunk_samples;
        let best_span = spans
            .iter()
            .copied()
            .filter(|span| span.start > cursor && span.start < search_end)
            .max_by_key(|span| span.end.saturating_sub(span.start));
        let span = best_span?;

        // pad both sides of the split like region edges, since Silero lags a soft
        // onset here too; half the gap at most keeps the segments apart, and the
        // window limit still caps the segment before it
        let pad = speech_pad_samples.min(span.end.saturating_sub(span.start) / 2);
        segments.push(SampleRange {
            start: cursor,
            end: (span.start + pad).min(search_end),
        });
        cursor = if span.end >= region.end {
            region.end
        } else {
            span.end - pad
        };
    }

    if cursor < region.end {
        segments.push(SampleRange {
            start: cursor,
            end: region.end,
        });
    }
    Some(segments)
}

#[cfg(feature = "vad")]
fn silence_spans(
    region: SampleRange,
    region_probabilities: &[f32],
    threshold: f32,
) -> Vec<SampleRange> {
    let region_frame_start = region.start / VAD_WINDOW_SAMPLES;
    let mut spans = Vec::new();
    let mut start = None;

    for (index, probability) in region_probabilities.iter().copied().enumerate() {
        let absolute_frame = region_frame_start + index;
        let sample_start = absolute_frame * VAD_WINDOW_SAMPLES;
        if probability <= threshold {
            start.get_or_insert(sample_start);
            continue;
        }
        if let Some(span_start) = start.take() {
            spans.push(SampleRange {
                start: span_start.max(region.start),
                end: sample_start.min(region.end),
            });
        }
    }

    if let Some(span_start) = start {
        spans.push(SampleRange {
            start: span_start.max(region.start),
            end: region.end,
        });
    }

    spans
}

#[cfg(feature = "vad")]
pub(super) fn build_vad_execution_plan(
    regions: &[SampleRange],
    probabilities: &[f32],
    config: &LongFormConfig,
) -> LongFormExecutionPlan {
    let mut tasks = Vec::new();
    let mut region_tasks = Vec::with_capacity(regions.len());
    for region in regions.iter().copied() {
        region_tasks.push(build_region_plan(region, probabilities, config, &mut tasks));
    }
    LongFormExecutionPlan {
        tasks,
        regions: region_tasks,
    }
}

pub(super) fn build_fast_execution_plan(
    audio_len: usize,
    config: &LongFormConfig,
) -> LongFormExecutionPlan {
    let mut tasks = Vec::new();
    let full_audio = SampleRange {
        start: 0,
        end: audio_len,
    };
    let task_indices = config
        .overlap
        .plan(full_audio)
        .into_iter()
        .map(|chunk| push_overlap_task(&mut tasks, chunk, config))
        .collect();
    let regions = vec![RegionTask {
        task_indices,
        kind: RegionTaskKind::Overlap,
    }];
    LongFormExecutionPlan { tasks, regions }
}

#[derive(Debug)]
pub(super) struct LongFormExecutionPlan {
    pub(super) tasks: Vec<ChunkTask>,
    pub(super) regions: Vec<RegionTask>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ChunkTask {
    pub(super) audio_start: usize,
    pub(super) audio_end: usize,
    pub(super) global_sample_offset: usize,
    pub(super) context_samples: usize,
}

#[derive(Debug)]
pub(super) struct RegionTask {
    pub(super) task_indices: Vec<usize>,
    pub(super) kind: RegionTaskKind,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum RegionTaskKind {
    #[cfg(feature = "vad")]
    Segments,
    Overlap,
}

#[cfg(feature = "vad")]
pub(super) fn build_region_plan(
    region: SampleRange,
    probabilities: &[f32],
    config: &LongFormConfig,
    tasks: &mut Vec<ChunkTask>,
) -> RegionTask {
    let region_len = region.end.saturating_sub(region.start);
    let max_chunk_samples = config.overlap.max_model_samples;
    if region_len <= max_chunk_samples {
        return RegionTask {
            task_indices: vec![push_segment_task(tasks, region)],
            kind: RegionTaskKind::Segments,
        };
    }

    if let Some(subsegments) = plan_region_subsegments(
        region,
        region_probability_slice(probabilities, region),
        &config.segmentation,
        max_chunk_samples,
    ) {
        return RegionTask {
            task_indices: subsegments
                .into_iter()
                .map(|subsegment| push_segment_task(tasks, subsegment))
                .collect(),
            kind: RegionTaskKind::Segments,
        };
    }

    RegionTask {
        task_indices: config
            .overlap
            .plan(region)
            .into_iter()
            .map(|chunk| push_overlap_task(tasks, chunk, config))
            .collect(),
        kind: RegionTaskKind::Overlap,
    }
}

#[cfg(feature = "vad")]
fn push_segment_task(tasks: &mut Vec<ChunkTask>, segment: SampleRange) -> usize {
    let task_idx = tasks.len();
    tasks.push(ChunkTask {
        audio_start: segment.start,
        audio_end: segment.end,
        global_sample_offset: 0,
        context_samples: 0,
    });
    task_idx
}

fn push_overlap_task(
    tasks: &mut Vec<ChunkTask>,
    chunk: SampleRange,
    config: &LongFormConfig,
) -> usize {
    let task_idx = tasks.len();
    let context_start = chunk.start.saturating_sub(config.overlap.context_samples);
    tasks.push(ChunkTask {
        audio_start: context_start,
        audio_end: chunk.end,
        global_sample_offset: chunk.start,
        context_samples: chunk.start - context_start,
    });
    task_idx
}

#[cfg(test)]
mod tests {
    use super::{OverlapChunkConfig, RegionTaskKind, SampleRange, build_fast_execution_plan};
    #[cfg(feature = "vad")]
    use super::{
        VadSegmentationConfig, build_region_plan, detect_speech_regions, plan_region_subsegments,
    };
    #[cfg(feature = "vad")]
    use crate::constants::VAD_WINDOW_SAMPLES;
    use crate::deployment::ParakeetDeployment;
    use crate::long_form::{LongFormConfig, LongFormMode};

    /// Default segmentation with speech at or above `threshold`
    #[cfg(feature = "vad")]
    fn segmentation_at(threshold: f32) -> VadSegmentationConfig {
        VadSegmentationConfig {
            threshold,
            ..VadSegmentationConfig::default()
        }
    }

    #[cfg(feature = "vad")]
    #[test]
    fn vad_defaults_lock_threshold_and_segmentation() {
        let segmentation = VadSegmentationConfig::default();
        assert_eq!(segmentation.threshold, 0.5);
        assert_eq!(segmentation.min_speech_duration, 0.15);
        assert_eq!(segmentation.min_silence_duration, 0.75);
        assert_eq!(segmentation.speech_padding, 0.2);
    }

    #[cfg(feature = "vad")]
    #[test]
    fn vad_rejects_non_finite_or_out_of_range_probabilities() {
        for threshold in [f32::NAN, -0.01, 1.01] {
            assert!(
                VadSegmentationConfig {
                    threshold,
                    ..VadSegmentationConfig::default()
                }
                .validate()
                .is_err()
            );
        }

        let segmentation = VadSegmentationConfig {
            silence_threshold_for_split: 1.01,
            ..VadSegmentationConfig::default()
        };
        assert!(segmentation.validate().is_err());

        let segmentation = VadSegmentationConfig {
            negative_threshold: Some(f32::NAN),
            ..VadSegmentationConfig::default()
        };
        assert!(segmentation.validate().is_err());

        let segmentation = VadSegmentationConfig {
            negative_threshold_offset: -0.01,
            ..VadSegmentationConfig::default()
        };
        assert!(segmentation.validate().is_err());
    }

    #[cfg(feature = "vad")]
    #[test]
    fn vad_rejects_non_positive_or_non_finite_durations() {
        for duration in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            let segmentation = VadSegmentationConfig {
                min_speech_duration: duration,
                ..VadSegmentationConfig::default()
            };
            assert!(segmentation.validate().is_err());

            let segmentation = VadSegmentationConfig {
                min_silence_duration: duration,
                ..VadSegmentationConfig::default()
            };
            assert!(segmentation.validate().is_err());

            let segmentation = VadSegmentationConfig {
                speech_padding: duration,
                ..VadSegmentationConfig::default()
            };
            assert!(segmentation.validate().is_err());

            let segmentation = VadSegmentationConfig {
                min_silence_at_max_speech: duration,
                ..VadSegmentationConfig::default()
            };
            assert!(segmentation.validate().is_err());
        }
    }

    #[cfg(feature = "vad")]
    #[test]
    fn speech_regions_trim_short_silence() {
        let probabilities = vec![0.9, 0.9, 0.1, 0.9, 0.9];
        let regions = detect_speech_regions(
            &probabilities,
            probabilities.len() * VAD_WINDOW_SAMPLES,
            &segmentation_at(0.85),
        );
        assert_eq!(regions.len(), 1);
    }

    #[cfg(feature = "vad")]
    #[test]
    fn min_speech_duration_drops_short_regions() {
        let probabilities = vec![0.9, 0.9, 0.9, 0.9];
        let audio_len = probabilities.len() * VAD_WINDOW_SAMPLES;
        assert!(audio_len as f64 / 16_000.0 < 0.15);
        let regions = detect_speech_regions(&probabilities, audio_len, &segmentation_at(0.85));
        assert!(regions.is_empty());
    }

    #[cfg(feature = "vad")]
    #[test]
    fn end_silence_of_point_75_seconds_closes_a_region() {
        let mut probabilities = vec![0.9; 8];
        probabilities.extend(std::iter::repeat_n(0.1, 25));
        probabilities.extend(std::iter::repeat_n(0.9, 8));
        let regions = detect_speech_regions(
            &probabilities,
            probabilities.len() * VAD_WINDOW_SAMPLES,
            &segmentation_at(0.85),
        );
        assert_eq!(regions.len(), 2);
    }

    #[cfg(feature = "vad")]
    #[test]
    fn speech_padding_is_two_tenths_of_a_second() {
        let mut probabilities = vec![0.9; 16];
        probabilities.extend(std::iter::repeat_n(0.1, 25));
        let audio_len = probabilities.len() * VAD_WINDOW_SAMPLES;
        let regions = detect_speech_regions(&probabilities, audio_len, &segmentation_at(0.85));
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start, 0);
        assert_eq!(regions[0].end, 16 * VAD_WINDOW_SAMPLES + 3_200);
    }

    #[cfg(feature = "vad")]
    #[test]
    fn silence_split_prefers_internal_gap() {
        let probabilities = vec![0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9];
        let region = SampleRange {
            start: 0,
            end: probabilities.len() * VAD_WINDOW_SAMPLES,
        };
        let segments = plan_region_subsegments(
            region,
            &probabilities,
            &VadSegmentationConfig::default(),
            5 * VAD_WINDOW_SAMPLES,
        )
        .unwrap();
        assert_eq!(segments.len(), 2);
        assert!(segments[0].end <= 5 * VAD_WINDOW_SAMPLES);
    }

    #[cfg(feature = "vad")]
    #[test]
    fn silence_split_pads_both_sides_of_the_gap() {
        // speech, a 0.32 s gap, speech: the region is just over the window limit
        let mut probabilities = vec![0.9; 40];
        probabilities.extend(std::iter::repeat_n(0.1, 10));
        probabilities.extend(std::iter::repeat_n(0.9, 40));
        let region = SampleRange {
            start: 0,
            end: probabilities.len() * VAD_WINDOW_SAMPLES,
        };
        let max_chunk_samples = region.end - VAD_WINDOW_SAMPLES;
        let config = VadSegmentationConfig::default();

        let segments =
            plan_region_subsegments(region, &probabilities, &config, max_chunk_samples).unwrap();

        // half the 5_120-sample gap caps the 3_200-sample padding
        let gap = SampleRange {
            start: 40 * VAD_WINDOW_SAMPLES,
            end: 50 * VAD_WINDOW_SAMPLES,
        };
        assert_eq!(
            segments,
            vec![
                SampleRange {
                    start: 0,
                    end: gap.start + 2_560,
                },
                SampleRange {
                    start: gap.end - 2_560,
                    end: region.end,
                },
            ]
        );
    }

    #[cfg(feature = "vad")]
    #[test]
    fn no_silence_returns_none_for_overlap_fallback() {
        let probabilities = vec![0.9; 10];
        let region = SampleRange {
            start: 0,
            end: probabilities.len() * VAD_WINDOW_SAMPLES,
        };
        assert!(
            plan_region_subsegments(
                region,
                &probabilities,
                &VadSegmentationConfig::default(),
                4 * VAD_WINDOW_SAMPLES
            )
            .is_none()
        );
    }

    #[cfg(feature = "vad")]
    #[test]
    fn exact_hard_max_region_is_one_accepted_segment() {
        let hard_max = ParakeetDeployment::v3().hard_sample_max();
        let region = SampleRange {
            start: 0,
            end: hard_max,
        };
        let segments =
            plan_region_subsegments(region, &[], &VadSegmentationConfig::default(), hard_max)
                .unwrap();
        assert_eq!(segments, vec![region]);
    }

    #[test]
    fn overlap_fallback_uses_two_seconds() {
        let v2 = OverlapChunkConfig::from_deployment(&ParakeetDeployment::v2());
        let v3 = OverlapChunkConfig::from_deployment(&ParakeetDeployment::v3());
        assert_eq!(v2.overlap_seconds, 2.0);
        assert_eq!(v3.overlap_seconds, 2.0);
        assert_eq!(v2.overlap_samples(), 32_000);
        assert_eq!(
            v3.max_model_samples,
            ParakeetDeployment::v3().hard_sample_max()
        );
    }

    #[test]
    fn overlap_plan_advances_with_stride() {
        let config = OverlapChunkConfig::default();
        let chunks = config.plan(SampleRange {
            start: 0,
            end: 600_000,
        });
        assert!(chunks.len() > 1);
        assert!(chunks[1].start < chunks[0].end);
    }

    #[test]
    fn overlap_rejects_a_maximum_above_the_deployment_hard_limit() {
        let deployment = ParakeetDeployment::v3();
        let mut config = OverlapChunkConfig::from_deployment(&deployment);
        config.max_model_samples = deployment.hard_sample_max() + 1;

        let error = config
            .validate_for(&deployment)
            .expect_err("overlap must not exceed the deployment hard maximum");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidConfiguration(_)
        ));
    }

    #[test]
    fn overlap_rejects_a_duration_above_the_deployment_hard_limit() {
        let deployment = ParakeetDeployment::v3();
        let mut config = OverlapChunkConfig::from_deployment(&deployment);
        config.overlap_seconds = 16.0;

        assert!(config.validate_for(&deployment).is_err());
    }

    #[cfg(feature = "vad")]
    #[test]
    fn short_region_stays_as_a_single_segment_task() {
        let config = LongFormConfig::default();
        let mut tasks = Vec::new();
        let region = SampleRange {
            start: 16_000,
            end: 32_000,
        };

        let plan = build_region_plan(region, &[], &config, &mut tasks);

        assert!(matches!(plan.kind, RegionTaskKind::Segments));
        assert_eq!(plan.task_indices, vec![0]);
        assert_eq!(tasks.len(), 1);
        assert_eq!((tasks[0].audio_start, tasks[0].audio_end), (16_000, 32_000));
    }

    #[cfg(feature = "vad")]
    #[test]
    fn overlap_plan_keeps_window_order_and_context_offsets() {
        let mut config = LongFormConfig::default();
        config.overlap.max_model_samples = 48_000;
        config.segmentation.min_silence_at_max_speech = 10.0;

        let mut tasks = Vec::new();
        let region = SampleRange {
            start: 0,
            end: 96_000,
        };
        let probabilities = vec![1.0; region.end.div_ceil(512)];

        let plan = build_region_plan(region, &probabilities, &config, &mut tasks);

        assert!(matches!(plan.kind, RegionTaskKind::Overlap));
        assert_eq!(plan.task_indices.len(), tasks.len());
        assert!(
            tasks
                .windows(2)
                .all(|window| window[0].global_sample_offset <= window[1].global_sample_offset)
        );
        assert!(
            tasks
                .iter()
                .all(|task| task.context_samples <= task.global_sample_offset)
        );
    }

    #[test]
    fn fast_execution_plan_uses_one_overlap_region_for_full_audio() {
        let config = LongFormConfig {
            mode: LongFormMode::Fast,
            ..LongFormConfig::default()
        };
        let plan = build_fast_execution_plan(96_000, &config);

        assert_eq!(plan.regions.len(), 1);
        assert!(matches!(plan.regions[0].kind, RegionTaskKind::Overlap));
        assert_eq!(plan.regions[0].task_indices.len(), plan.tasks.len());
        assert!(!plan.tasks.is_empty());
    }

    #[cfg(feature = "vad")]
    #[test]
    fn hook_source_range_keeps_vad_segment_position_separate_from_decoder_offset() {
        let config = LongFormConfig::default();
        let mut tasks = Vec::new();
        let plan = build_region_plan(
            SampleRange {
                start: 16_000,
                end: 32_000,
            },
            &[],
            &config,
            &mut tasks,
        );
        let task = tasks[plan.task_indices[0]];

        assert_eq!(task.audio_start, 16_000);
        assert_eq!(task.audio_end, 32_000);
        assert_eq!(task.global_sample_offset, 0);
        assert_eq!(task.context_samples, 0);
    }
}

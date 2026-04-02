use crate::constants::{MAX_MODEL_SAMPLES, SAMPLE_RATE, SAMPLES_PER_ENCODER_FRAME};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SampleRange {
    pub start: usize,
    pub end: usize,
}

/// Runtime threshold configuration for the VAD model
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Positive speech threshold applied to per-window probabilities
    pub default_threshold: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            default_threshold: 0.85,
        }
    }
}

/// Segmentation parameters for converting VAD probabilities into speech regions
#[derive(Debug, Clone)]
pub struct VadSegmentationConfig {
    /// Minimum speech span kept in the output
    pub min_speech_duration: f64,
    /// Minimum silence span required to end a speech region
    pub min_silence_duration: f64,
    /// Soft maximum speech duration before the planner looks for an internal split
    pub max_speech_duration: f64,
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
    /// Whether to use the best available silence even if it does not cross the main split threshold
    pub use_max_possible_silence_at_max_speech: bool,
}

impl Default for VadSegmentationConfig {
    fn default() -> Self {
        Self {
            min_speech_duration: 0.15,
            min_silence_duration: 0.75,
            max_speech_duration: 14.0,
            speech_padding: 0.1,
            silence_threshold_for_split: 0.3,
            negative_threshold: None,
            negative_threshold_offset: 0.15,
            min_silence_at_max_speech: 0.098,
            use_max_possible_silence_at_max_speech: true,
        }
    }
}

impl VadSegmentationConfig {
    pub(crate) fn threshold(&self, default_threshold: f32) -> f32 {
        default_threshold
    }

    fn effective_negative_threshold(&self, base_threshold: f32) -> f32 {
        if let Some(override_threshold) = self.negative_threshold {
            return override_threshold;
        }
        (base_threshold - self.negative_threshold_offset).max(0.01)
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
        Self {
            overlap_seconds: 2.0,
            context_samples: SAMPLES_PER_ENCODER_FRAME,
            max_model_samples: MAX_MODEL_SAMPLES,
        }
    }
}

impl OverlapChunkConfig {
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

pub(crate) fn detect_speech_regions(
    probabilities: &[f32],
    audio_length_samples: usize,
    threshold: f32,
    config: &VadSegmentationConfig,
) -> Vec<SampleRange> {
    if probabilities.is_empty() || audio_length_samples == 0 {
        return Vec::new();
    }

    let hop_size = 4096usize;
    let min_speech_samples = (config.min_speech_duration * SAMPLE_RATE as f64) as usize;
    let min_silence_samples = (config.min_silence_duration * SAMPLE_RATE as f64) as usize;
    let speech_pad_samples = (config.speech_padding * SAMPLE_RATE as f64) as usize;
    let negative_threshold = config.effective_negative_threshold(threshold);

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

pub(crate) fn region_probability_slice(probabilities: &[f32], region: SampleRange) -> &[f32] {
    let frame_start = region.start / 4096;
    let frame_end = region.end.div_ceil(4096);
    &probabilities[frame_start.min(probabilities.len())..frame_end.min(probabilities.len())]
}

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
        segments.push(SampleRange {
            start: cursor,
            end: span.start,
        });
        cursor = span.end;
    }

    if cursor < region.end {
        segments.push(SampleRange {
            start: cursor,
            end: region.end,
        });
    }
    Some(segments)
}

fn silence_spans(
    region: SampleRange,
    region_probabilities: &[f32],
    threshold: f32,
) -> Vec<SampleRange> {
    let region_frame_start = region.start / 4096;
    let mut spans = Vec::new();
    let mut start = None;

    for (index, probability) in region_probabilities.iter().copied().enumerate() {
        let absolute_frame = region_frame_start + index;
        let sample_start = absolute_frame * 4096;
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

#[cfg(test)]
mod tests {
    use super::{
        OverlapChunkConfig, SampleRange, VadSegmentationConfig, detect_speech_regions,
        plan_region_subsegments,
    };

    #[test]
    fn speech_regions_trim_short_silence() {
        let probabilities = vec![0.9, 0.9, 0.1, 0.9, 0.9];
        let regions = detect_speech_regions(
            &probabilities,
            probabilities.len() * 4096,
            0.85,
            &VadSegmentationConfig::default(),
        );
        assert_eq!(regions.len(), 1);
    }

    #[test]
    fn silence_split_prefers_internal_gap() {
        let probabilities = vec![0.9, 0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.9];
        let region = SampleRange {
            start: 0,
            end: probabilities.len() * 4096,
        };
        let segments = plan_region_subsegments(
            region,
            &probabilities,
            &VadSegmentationConfig::default(),
            4 * 4096,
        )
        .unwrap();
        assert_eq!(segments.len(), 2);
        assert!(segments[0].end <= 4 * 4096);
    }

    #[test]
    fn no_silence_returns_none_for_overlap_fallback() {
        let probabilities = vec![0.9; 10];
        let region = SampleRange {
            start: 0,
            end: probabilities.len() * 4096,
        };
        assert!(
            plan_region_subsegments(
                region,
                &probabilities,
                &VadSegmentationConfig::default(),
                4 * 4096
            )
            .is_none()
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
}

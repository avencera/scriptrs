use crate::constants::{SAMPLE_RATE, SAMPLES_PER_ENCODER_FRAME, SECONDS_PER_ENCODER_FRAME};
use crate::types::{TimedToken, TranscriptChunk, TranscriptionResult};
use crate::vocab::Vocabulary;

/// One token a decoder emitted
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RawToken {
    pub id: u32,
    /// Encoder frame the token was emitted at
    pub frame: usize,
    /// Encoder frames the token spans
    pub duration: usize,
    pub confidence: f32,
}

/// A decoder's emitted tokens, in order
#[derive(Debug, Clone)]
pub(crate) struct RawTranscription {
    pub tokens: Vec<RawToken>,
}

impl RawTranscription {
    pub(crate) fn empty() -> Self {
        Self { tokens: Vec::new() }
    }

    #[cfg(feature = "long-form")]
    pub(crate) fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Drops the tokens in the first `context_frames` frames, then moves the rest
    /// so frame `context_frames` lands on `frame_offset`
    pub(crate) fn drop_context_frames(&mut self, context_frames: usize, frame_offset: usize) {
        self.tokens.retain_mut(|token| {
            if token.frame < context_frames {
                return false;
            }
            token.frame = token.frame - context_frames + frame_offset;
            true
        });
    }

    /// Appends `other`'s tokens with their frames moved `frame_offset` later
    #[cfg(feature = "cuda")]
    pub(crate) fn extend_shifted(&mut self, other: Self, frame_offset: usize) {
        self.tokens
            .extend(other.tokens.into_iter().map(|token| RawToken {
                frame: frame_offset + token.frame,
                ..token
            }));
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtDecoder {
    vocab: Vocabulary,
}

impl ParakeetTdtDecoder {
    pub(crate) fn new(vocab: Vocabulary) -> Self {
        Self { vocab }
    }

    pub(crate) fn decode(
        &self,
        raw: &RawTranscription,
        duration_seconds: f64,
    ) -> TranscriptionResult {
        if raw.tokens.is_empty() {
            return TranscriptionResult::empty(duration_seconds);
        }

        let mut text = String::new();
        let mut tokens = Vec::with_capacity(raw.tokens.len());

        for raw_token in &raw.tokens {
            let token_id = raw_token.id;
            let Some(token_text) = self.vocab.token(token_id as usize) else {
                continue;
            };
            if is_special_token(token_text) {
                continue;
            }

            let start = frame_to_seconds(raw_token.frame);
            let end = token_end_seconds(start, raw_token.duration, duration_seconds);

            let mut display_text = token_text.replace('▁', " ");
            if needs_digit_spacing(&text, &display_text) {
                display_text.insert(0, ' ');
            }

            text.push_str(&display_text);
            tokens.push(TimedToken {
                token_id,
                text: display_text,
                start,
                end,
                confidence: raw_token.confidence,
            });
        }

        let text = text.trim().to_owned();
        let chunks = if text.is_empty() || tokens.is_empty() {
            Vec::new()
        } else {
            vec![TranscriptChunk {
                start: tokens.first().map(|token| token.start).unwrap_or(0.0),
                end: tokens
                    .last()
                    .map(|token| token.end)
                    .unwrap_or(duration_seconds),
                text: text.clone(),
            }]
        };

        TranscriptionResult {
            text,
            chunks,
            tokens,
            duration_seconds,
        }
    }
}

fn frame_to_seconds(frame: usize) -> f64 {
    (frame * SAMPLES_PER_ENCODER_FRAME) as f64 / SAMPLE_RATE as f64
}

fn token_end_seconds(start: f64, duration_frames: usize, duration_seconds: f64) -> f64 {
    let span = if duration_frames == 0 {
        SECONDS_PER_ENCODER_FRAME
    } else {
        duration_frames as f64 * SECONDS_PER_ENCODER_FRAME
    };
    (start + span).min(duration_seconds).max(start)
}

fn is_special_token(token: &str) -> bool {
    token.starts_with('<') && token.ends_with('>') && token != "<unk>"
}

fn needs_digit_spacing(existing_text: &str, display_text: &str) -> bool {
    if existing_text.is_empty() || display_text.starts_with(' ') {
        return false;
    }
    if !display_text
        .chars()
        .all(|character| character.is_ascii_digit())
    {
        return false;
    }

    let trailing_letters = existing_text
        .chars()
        .rev()
        .take_while(|character| character.is_alphabetic())
        .count();
    let last_char = existing_text.chars().last();
    let is_article_a = trailing_letters == 1 && last_char == Some('a');

    trailing_letters > 1 || is_article_a
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{RawToken, RawTranscription};

    /// Builds a transcription from per-token columns
    pub(crate) fn raw_from_columns(
        ids: &[u32],
        frames: &[usize],
        durations: &[usize],
        confidences: &[f32],
    ) -> RawTranscription {
        assert!(
            [frames.len(), durations.len(), confidences.len()]
                .iter()
                .all(|len| *len == ids.len()),
            "every column needs one entry per token"
        );
        let tokens = (0..ids.len())
            .map(|index| RawToken {
                id: ids[index],
                frame: frames[index],
                duration: durations[index],
                confidence: confidences[index],
            })
            .collect();
        RawTranscription { tokens }
    }

    impl RawTranscription {
        pub(crate) fn ids(&self) -> Vec<u32> {
            self.tokens.iter().map(|token| token.id).collect()
        }

        pub(crate) fn frames(&self) -> Vec<usize> {
            self.tokens.iter().map(|token| token.frame).collect()
        }

        pub(crate) fn durations(&self) -> Vec<usize> {
            self.tokens.iter().map(|token| token.duration).collect()
        }

        #[cfg(feature = "cuda")]
        pub(crate) fn confidences(&self) -> Vec<f32> {
            self.tokens.iter().map(|token| token.confidence).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::TimedToken;

    use super::test_support::raw_from_columns;
    use super::{ParakeetTdtDecoder, SECONDS_PER_ENCODER_FRAME};
    use crate::vocab::Vocabulary;

    fn decoder(tokens: &[&str]) -> ParakeetTdtDecoder {
        ParakeetTdtDecoder::new(Vocabulary::from_tokens(tokens))
    }

    #[test]
    fn digit_spacing_matches_parakeet_rs() {
        let decoder = decoder(&["▁like", "1", "0", "0", "<blk>"]);
        let raw = raw_from_columns(&[0, 1, 2, 3], &[0, 1, 2, 3], &[1, 1, 1, 1], &[0.9; 4]);
        let result = decoder.decode(&raw, 1.0);
        assert_eq!(result.text, "like 100");
        assert_eq!(result.tokens[1].text, " 1");
    }

    #[test]
    fn chunk_spans_the_token_range() {
        let decoder = decoder(&["▁hello", "▁world", "<blk>"]);
        let raw = raw_from_columns(&[0, 1], &[0, 4], &[1, 1], &[0.9; 2]);
        let result = decoder.decode(&raw, 1.0);
        let chunk = &result.chunks[0];
        assert_eq!(chunk.text, "hello world");
        assert_eq!(chunk.start, 0.0);
        assert!(chunk.end > chunk.start);
    }

    #[test]
    fn duration_decode_fixture_uses_raw_durations() {
        let decoder = decoder(&["▁hello", "▁world", "<blk>"]);
        let raw = raw_from_columns(&[0, 1], &[0, 2], &[2, 3], &[0.8, 0.7]);
        let result = decoder.decode(&raw, 2.0);
        assert_eq!(result.tokens[0].start, 0.0);
        assert_eq!(result.tokens[0].end, 2.0 * SECONDS_PER_ENCODER_FRAME);
        assert_eq!(result.tokens[1].start, 2.0 * SECONDS_PER_ENCODER_FRAME);
        assert_eq!(
            result.tokens[1].end,
            2.0 * SECONDS_PER_ENCODER_FRAME + 3.0 * SECONDS_PER_ENCODER_FRAME
        );
    }

    #[test]
    fn token_type_is_publicly_usable() {
        let token = TimedToken {
            token_id: 1,
            text: "hello".to_owned(),
            start: 0.0,
            end: 0.1,
            confidence: 0.9,
        };
        assert_eq!(token.text, "hello");
    }
}

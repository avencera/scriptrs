use crate::constants::{MEL_HOP_SAMPLES, SAMPLE_RATE, SECONDS_PER_ENCODER_FRAME};
use crate::types::{TimedToken, TranscriptChunk, TranscriptionResult};
use crate::vocab::Vocabulary;

#[derive(Debug, Clone)]
pub(crate) struct RawTranscription {
    pub token_ids: Vec<u32>,
    pub frame_indices: Vec<usize>,
    pub durations: Vec<usize>,
    pub confidences: Vec<f32>,
}

impl RawTranscription {
    pub(crate) fn empty() -> Self {
        Self {
            token_ids: Vec::new(),
            frame_indices: Vec::new(),
            durations: Vec::new(),
            confidences: Vec::new(),
        }
    }

    #[cfg(feature = "long-form")]
    pub(crate) fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
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
        if raw.token_ids.is_empty() {
            return TranscriptionResult::empty(duration_seconds);
        }

        let mut text = String::new();
        let mut tokens = Vec::with_capacity(raw.token_ids.len());

        for (index, token_id) in raw.token_ids.iter().copied().enumerate() {
            let Some(token_text) = self.vocab.token(token_id as usize) else {
                continue;
            };
            if is_special_token(token_text) {
                continue;
            }

            let frame = raw.frame_indices[index];
            let start = frame_to_seconds(frame);
            let end = raw
                .frame_indices
                .get(index + 1)
                .copied()
                .map(frame_to_seconds)
                .unwrap_or_else(|| (start + SECONDS_PER_ENCODER_FRAME).min(duration_seconds));

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
                confidence: raw.confidences.get(index).copied().unwrap_or(0.0),
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
    (frame * MEL_HOP_SAMPLES * 8) as f64 / SAMPLE_RATE as f64
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
mod tests {
    use crate::types::TimedToken;

    use super::{ParakeetTdtDecoder, RawTranscription};
    use crate::vocab::Vocabulary;

    fn decoder(tokens: &[&str]) -> ParakeetTdtDecoder {
        let vocab = Vocabulary {
            id_to_token: tokens.iter().map(|token| token.to_string()).collect(),
            blank_id: tokens.len().saturating_sub(1),
        };
        ParakeetTdtDecoder::new(vocab)
    }

    #[test]
    fn digit_spacing_matches_parakeet_rs() {
        let decoder = decoder(&["▁like", "1", "0", "0", "<blk>"]);
        let raw = RawTranscription {
            token_ids: vec![0, 1, 2, 3],
            frame_indices: vec![0, 1, 2, 3],
            durations: vec![1, 1, 1, 1],
            confidences: vec![0.9; 4],
        };
        let result = decoder.decode(&raw, 1.0);
        assert_eq!(result.text, "like 100");
        assert_eq!(result.tokens[1].text, " 1");
    }

    #[test]
    fn chunk_spans_the_token_range() {
        let decoder = decoder(&["▁hello", "▁world", "<blk>"]);
        let raw = RawTranscription {
            token_ids: vec![0, 1],
            frame_indices: vec![0, 4],
            durations: vec![1, 1],
            confidences: vec![0.9; 2],
        };
        let result = decoder.decode(&raw, 1.0);
        let chunk = &result.chunks[0];
        assert_eq!(chunk.text, "hello world");
        assert_eq!(chunk.start, 0.0);
        assert!(chunk.end > chunk.start);
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

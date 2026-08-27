use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::deployment::VocabKind;
use crate::error::TranscriptionError;
use crate::language::ParakeetLanguage;

/// Vocabulary owner for v2 `vocab.txt` and v3 JSON tokenizers
///
/// SentencePiece files include an explicit `<blk>` token. NVIDIA Parakeet TDT
/// v3 JSON tables omit blank; the RNNT blank id is one past the last tokenizer
/// entry (`8192` for the published 0..8191 table).
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub(crate) id_to_token: Vec<String>,
    pub(crate) blank_id: usize,
}

impl Vocabulary {
    /// Parse a SentencePiece `vocab.txt` file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TranscriptionError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.splitn(2, ' ');
            let Some(token) = parts.next() else {
                continue;
            };
            let Some(id) = parts.next() else {
                continue;
            };
            let id: usize = id.parse().map_err(|error| {
                TranscriptionError::InvalidVocabulary(format!("invalid token id `{id}`: {error}"))
            })?;
            entries.push((id, token.to_owned()));
        }

        finish_entries(entries, MissingBlank::LastToken)
    }

    /// Parse a Parakeet JSON tokenizer such as `parakeet_vocab.json`
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self, TranscriptionError> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json_str(&data)
    }

    /// Load a vocabulary using the parser owned by a deployment
    pub fn load(path: impl AsRef<Path>, kind: VocabKind) -> Result<Self, TranscriptionError> {
        match kind {
            VocabKind::SentencePieceTxt => Self::from_file(path),
            VocabKind::JsonTokenizer => Self::from_json(path),
        }
    }

    /// Parse JSON tokenizer bytes without a filesystem path
    pub fn from_json_str(data: &str) -> Result<Self, TranscriptionError> {
        let value: serde_json::Value = serde_json::from_str(data).map_err(|error| {
            TranscriptionError::InvalidVocabulary(format!("invalid tokenizer json: {error}"))
        })?;
        match value {
            serde_json::Value::Array(tokens) => {
                let mut id_to_token = Vec::with_capacity(tokens.len());
                let mut blank_id = None;
                for (id, token) in tokens.into_iter().enumerate() {
                    let Some(token) = token.as_str() else {
                        return Err(TranscriptionError::InvalidVocabulary(format!(
                            "tokenizer array entry {id} was not a string"
                        )));
                    };
                    if is_blank_token(token) {
                        blank_id = Some(id);
                    }
                    id_to_token.push(token.to_owned());
                }

                finish(id_to_token, blank_id, MissingBlank::VocabSize)
            }
            serde_json::Value::Object(map) => parse_json_object(map),
            _ => Err(TranscriptionError::InvalidVocabulary(
                "tokenizer json must be an object or array".to_owned(),
            )),
        }
    }

    /// Look up a token string by id
    pub fn token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(String::as_str)
    }

    /// Blank token id owned by this vocabulary
    ///
    /// This may be one past the last tokenizer entry when the JSON table omits
    /// an explicit blank token.
    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    /// Number of tokenizer entries
    ///
    /// This does not count an RNNT blank that lives one past the table.
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Whether the tokenizer table has no entries
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// Language encoded by this token, if it is a language tag
    pub fn language_of_token(&self, id: usize) -> Option<ParakeetLanguage> {
        ParakeetLanguage::from_token(self.token(id)?)
    }

    #[cfg(test)]
    pub(crate) fn from_tokens(tokens: &[&str]) -> Self {
        let mut blank_id = tokens.len().saturating_sub(1);
        let id_to_token = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| {
                if is_blank_token(token) {
                    blank_id = id;
                }
                (*token).to_owned()
            })
            .collect();
        Self {
            id_to_token,
            blank_id,
        }
    }
}

fn parse_json_object(
    map: serde_json::Map<String, serde_json::Value>,
) -> Result<Vocabulary, TranscriptionError> {
    let keys_are_ids = map.keys().any(|key| key.parse::<usize>().is_ok());
    let mut entries = Vec::with_capacity(map.len());
    if keys_are_ids {
        for (key, value) in map {
            let id: usize = key.parse().map_err(|error| {
                TranscriptionError::InvalidVocabulary(format!(
                    "invalid tokenizer id `{key}`: {error}"
                ))
            })?;
            let Some(token) = value.as_str() else {
                return Err(TranscriptionError::InvalidVocabulary(format!(
                    "tokenizer value for id {id} was not a string"
                )));
            };
            entries.push((id, token.to_owned()));
        }
    } else {
        for (token, value) in map {
            let id = value.as_u64().ok_or_else(|| {
                TranscriptionError::InvalidVocabulary(format!(
                    "tokenizer id for `{token}` was not an integer"
                ))
            })?;
            let id = usize::try_from(id).map_err(|_| {
                TranscriptionError::InvalidVocabulary(format!(
                    "tokenizer id for `{token}` was outside the platform range"
                ))
            })?;
            entries.push((id, token));
        }
    }

    finish_entries(entries, MissingBlank::VocabSize)
}

fn finish_entries(
    mut entries: Vec<(usize, String)>,
    missing_blank: MissingBlank,
) -> Result<Vocabulary, TranscriptionError> {
    entries.sort_unstable_by_key(|(id, _)| *id);

    let mut id_to_token = Vec::with_capacity(entries.len());
    let mut blank_id = None;
    for (expected_id, (id, token)) in entries.into_iter().enumerate() {
        if id != expected_id {
            return Err(TranscriptionError::InvalidVocabulary(format!(
                "tokenizer ids must be contiguous from 0: expected {expected_id}, found {id}"
            )));
        }
        if is_blank_token(&token) {
            blank_id = Some(id);
        }
        id_to_token.push(token);
    }

    finish(id_to_token, blank_id, missing_blank)
}

fn finish(
    id_to_token: Vec<String>,
    blank_id: Option<usize>,
    missing_blank: MissingBlank,
) -> Result<Vocabulary, TranscriptionError> {
    if id_to_token.is_empty() {
        return Err(TranscriptionError::InvalidVocabulary(
            "vocabulary was empty".to_owned(),
        ));
    }

    Ok(Vocabulary {
        blank_id: blank_id.unwrap_or_else(|| missing_blank.fallback_id(id_to_token.len())),
        id_to_token,
    })
}

fn is_blank_token(token: &str) -> bool {
    matches!(token, "<blk>" | "<blank>")
}

enum MissingBlank {
    LastToken,
    VocabSize,
}

impl MissingBlank {
    fn fallback_id(self, token_count: usize) -> usize {
        match self {
            Self::LastToken => token_count.saturating_sub(1),
            Self::VocabSize => token_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Vocabulary;
    use crate::language::ParakeetLanguage;
    use std::path::PathBuf;

    fn fixture_path(relative: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
    }

    #[test]
    fn blank_defaults_to_last_token() {
        let vocab = Vocabulary::from_tokens(&["a", "b"]);
        assert_eq!(vocab.blank_id(), 1);
    }

    #[test]
    fn v2_vocab_txt_blank_is_blk() {
        let vocab = Vocabulary::from_file(fixture_path("fixtures/models/parakeet-v2/vocab.txt"))
            .expect("v2 vocab fixture should parse");
        assert_eq!(vocab.blank_id(), 1_024);
        assert_eq!(vocab.token(vocab.blank_id()), Some("<blk>"));
    }

    const V3_TOKENIZER_FIXTURE: &str = r#"{
  "0": "<unk>",
  "1": "▁hello",
  "2": "▁world",
  "3": "<en>",
  "4": "<pl>",
  "5": "<ru>",
  "6": "cat",
  "7": "кот",
  "8": "<blk>"
}"#;

    #[test]
    fn v3_json_tokenizer_fixture_has_language_tokens_and_blank() {
        let vocab = Vocabulary::from_json_str(V3_TOKENIZER_FIXTURE)
            .expect("v3 tokenizer fixture should parse");
        assert_eq!(vocab.token(vocab.blank_id()), Some("<blk>"));
        assert_eq!(vocab.language_of_token(3), Some(ParakeetLanguage::English));
        assert_eq!(vocab.language_of_token(4), Some(ParakeetLanguage::Polish));
        assert_eq!(vocab.language_of_token(5), Some(ParakeetLanguage::Russian));

        let path = fixture_path("fixtures/models/parakeet-v3/parakeet_vocab.json");
        if path.exists() {
            let from_file = Vocabulary::from_json(&path).expect("on-disk v3 fixture should parse");
            assert_eq!(from_file.blank_id(), vocab.blank_id());
            assert_eq!(from_file.token(3), vocab.token(3));
        }
    }

    #[test]
    fn json_tokenizer_without_blank_uses_vocab_size() {
        let vocab = Vocabulary::from_json_str(r#"{"0":"<unk>","1":"hello","2":"world"}"#)
            .expect("json tokenizer without blank should parse");
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.blank_id(), 3);
        assert_eq!(vocab.token(2), Some("world"));
        assert_eq!(vocab.token(3), None);
    }

    #[test]
    fn sparse_json_ids_are_rejected_without_placeholder_tokens() {
        let error = Vocabulary::from_json_str(r#"{"0":"<unk>","8191":"Œ"}"#)
            .expect_err("sparse tokenizer ids must not create empty placeholders");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));
    }

    #[test]
    fn sparse_token_to_id_maps_are_rejected() {
        let error = Vocabulary::from_json_str(r#"{"<unk>":0,"Œ":8191}"#)
            .expect_err("sparse token-to-id maps must be rejected at the parser boundary");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));
    }
}

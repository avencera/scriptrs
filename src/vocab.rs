use std::collections::BTreeMap;
use std::path::Path;

use crate::deployment::{ParakeetDeployment, ParakeetVersion};
use crate::error::TranscriptionError;
use crate::language::ParakeetLanguage;

/// Vocabulary owner for v2 `vocab.txt` and v3 JSON tokenizers
///
/// SentencePiece files include an explicit `<blk>` token. NVIDIA Parakeet TDT
/// v3 JSON tables omit blank; the deployment supplies the RNNT blank id one
/// past the last tokenizer entry
#[derive(Debug, Clone)]
pub struct Vocabulary {
    id_to_token: Vec<String>,
    blank_id: usize,
}

impl Vocabulary {
    /// Load and validate the vocabulary file a deployment uses
    ///
    /// v2 reads a SentencePiece `vocab.txt` and v3 a JSON tokenizer such as
    /// `parakeet_vocab.json`, whose table has no blank, so the deployment
    /// supplies it
    pub fn load(
        path: impl AsRef<Path>,
        deployment: &ParakeetDeployment,
    ) -> Result<Self, TranscriptionError> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data, deployment)
    }

    /// Parse and validate tokenizer bytes in the format a deployment uses
    pub fn from_bytes(
        data: &[u8],
        deployment: &ParakeetDeployment,
    ) -> Result<Self, TranscriptionError> {
        let blank_id = Some(deployment.blank_id());
        let vocabulary = match deployment.version() {
            ParakeetVersion::V2 => {
                parse_sentencepiece_text(utf8(data, "SentencePiece")?, blank_id)?
            }
            ParakeetVersion::V3 => parse_json_str(utf8(data, "JSON")?, blank_id)?,
        };
        deployment.validate_vocabulary(&vocabulary)?;
        Ok(vocabulary)
    }

    /// Parse a SentencePiece `vocab.txt` with an explicit blank token, for models
    /// outside the Parakeet deployments
    #[cfg(feature = "cuda")]
    pub(crate) fn from_sentencepiece_file(
        path: impl AsRef<Path>,
    ) -> Result<Self, TranscriptionError> {
        let data = std::fs::read(path)?;
        parse_sentencepiece_text(utf8(&data, "SentencePiece")?, None)
    }

    /// Look up a token string by id
    pub fn token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(String::as_str)
    }

    /// Resolved RNNT blank token ID
    ///
    /// Deployment-aware loading verifies this value against the joint model
    /// contract before it can be used for decoding
    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    /// Number of tokenizer entries
    ///
    /// This does not count an RNNT blank that lives one past the table
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
        let Some(blank_id) = tokens.iter().position(|token| is_blank_token(token)) else {
            panic!("test vocabulary must include an explicit blank token");
        };
        let id_to_token = tokens.iter().map(|token| (*token).to_owned()).collect();
        Self {
            id_to_token,
            blank_id,
        }
    }
}

/// The tokenizer file as text, naming its `format` when it is not UTF-8
fn utf8<'a>(data: &'a [u8], format: &str) -> Result<&'a str, TranscriptionError> {
    std::str::from_utf8(data).map_err(|error| {
        TranscriptionError::InvalidVocabulary(format!("{format} tokenizer is not UTF-8: {error}"))
    })
}

fn parse_sentencepiece_text(
    text: &str,
    expected_blank_id: Option<usize>,
) -> Result<Vocabulary, TranscriptionError> {
    let mut entries = Vec::new();

    for line in text.lines() {
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

    finish_entries(entries, expected_blank_id)
}

fn parse_json_str(
    data: &str,
    expected_blank_id: Option<usize>,
) -> Result<Vocabulary, TranscriptionError> {
    let trimmed = data.trim_start();
    if trimmed.starts_with('[') {
        let tokens: Vec<String> = serde_json::from_str(data).map_err(|error| {
            TranscriptionError::InvalidVocabulary(format!("invalid tokenizer json: {error}"))
        })?;
        let entries = tokens.into_iter().enumerate().collect();
        return finish_entries(entries, expected_blank_id);
    }

    if trimmed.starts_with('{') {
        if let Ok(map) = serde_json::from_str::<BTreeMap<String, String>>(data) {
            let entries = map
                .into_iter()
                .map(|(key, token)| {
                    let id = key.parse::<usize>().map_err(|error| {
                        TranscriptionError::InvalidVocabulary(format!(
                            "invalid tokenizer id `{key}`: {error}"
                        ))
                    })?;
                    Ok((id, token))
                })
                .collect::<Result<Vec<_>, TranscriptionError>>()?;
            return finish_entries(entries, expected_blank_id);
        }

        if let Ok(map) = serde_json::from_str::<BTreeMap<String, u64>>(data) {
            let entries = map
                .into_iter()
                .map(|(token, id)| {
                    let id = usize::try_from(id).map_err(|_| {
                        TranscriptionError::InvalidVocabulary(format!(
                            "tokenizer id for `{token}` was outside the platform range"
                        ))
                    })?;
                    Ok((id, token))
                })
                .collect::<Result<Vec<_>, TranscriptionError>>()?;
            return finish_entries(entries, expected_blank_id);
        }

        return Err(TranscriptionError::InvalidVocabulary(
            "tokenizer object must map ids to strings or tokens to integer ids".to_owned(),
        ));
    }

    Err(TranscriptionError::InvalidVocabulary(
        "tokenizer json must be an object or array".to_owned(),
    ))
}

fn finish_entries(
    mut entries: Vec<(usize, String)>,
    expected_blank_id: Option<usize>,
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
        if is_blank_token(&token) && blank_id.replace(id).is_some() {
            return Err(TranscriptionError::InvalidVocabulary(
                "tokenizer contained multiple blank tokens".to_owned(),
            ));
        }
        id_to_token.push(token);
    }

    finish(id_to_token, blank_id.or(expected_blank_id))
}

fn finish(
    id_to_token: Vec<String>,
    blank_id: Option<usize>,
) -> Result<Vocabulary, TranscriptionError> {
    if id_to_token.is_empty() {
        return Err(TranscriptionError::InvalidVocabulary(
            "vocabulary was empty".to_owned(),
        ));
    }

    let Some(blank_id) = blank_id else {
        return Err(TranscriptionError::InvalidVocabulary(
            "tokenizer did not contain an explicit blank token".to_owned(),
        ));
    };

    Ok(Vocabulary {
        blank_id,
        id_to_token,
    })
}

/// Whether `token` is one of the tokenizer spellings of the RNNT blank
pub(crate) fn is_blank_token(token: &str) -> bool {
    matches!(token, "<blk>" | "<blank>")
}

#[cfg(test)]
mod tests {
    use super::{Vocabulary, parse_json_str};
    use crate::deployment::ParakeetDeployment;
    use crate::language::ParakeetLanguage;
    use std::collections::BTreeMap;

    #[test]
    fn blank_is_taken_from_the_explicit_token() {
        let vocab = Vocabulary::from_tokens(&["a", "<blk>"]);
        assert_eq!(vocab.blank_id(), 1);
    }

    #[test]
    fn v2_vocab_txt_blank_is_blk() {
        // the v2 `vocab.txt` shape: 1024 SentencePiece pieces, then the blank
        let mut data: String = (0..1_024).map(|id| format!("▁t{id} {id}\n")).collect();
        data.push_str("<blk> 1024\n");
        let path = tempfile::NamedTempFile::new().expect("temporary vocabulary should exist");
        std::fs::write(path.path(), data).expect("temporary vocabulary should be written");

        let vocab = Vocabulary::load(path.path(), &ParakeetDeployment::v2())
            .expect("v2 vocab should match the deployment contract");
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
        let vocab =
            parse_json_str(V3_TOKENIZER_FIXTURE, None).expect("v3 tokenizer fixture should parse");
        assert_eq!(vocab.token(vocab.blank_id()), Some("<blk>"));
        assert_eq!(vocab.language_of_token(3), Some(ParakeetLanguage::English));
        assert_eq!(vocab.language_of_token(4), Some(ParakeetLanguage::Polish));
        assert_eq!(vocab.language_of_token(5), Some(ParakeetLanguage::Russian));
    }

    #[test]
    fn json_tokenizer_without_blank_is_rejected() {
        let error = parse_json_str(r#"{"0":"<unk>","1":"hello","2":"world"}"#, None)
            .expect_err("an arbitrary tokenizer must not infer its blank id");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));
    }

    #[test]
    fn sparse_json_ids_are_rejected_without_placeholder_tokens() {
        let error = parse_json_str(r#"{"0":"<unk>","8191":"Œ"}"#, None)
            .expect_err("sparse tokenizer ids must not create empty placeholders");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));
    }

    #[test]
    fn sparse_token_to_id_maps_are_rejected() {
        let error = parse_json_str(r#"{"<unk>":0,"Œ":8191}"#, None)
            .expect_err("sparse token-to-id maps must be rejected at the parser boundary");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));
    }

    fn published_v3_tokens() -> BTreeMap<String, String> {
        (0..8_192)
            .map(|id| {
                let token = if id == 8_191 {
                    "Œ".to_owned()
                } else {
                    format!("token-{id}")
                };
                (id.to_string(), token)
            })
            .collect()
    }

    #[test]
    fn published_v3_json_shape_uses_deployment_blank_8192() {
        let data = serde_json::to_string(&published_v3_tokens())
            .expect("published v3 tokenizer fixture should serialize");
        let generic_error = parse_json_str(&data, None)
            .expect_err("generic parsing must not infer a deployment blank");
        assert!(matches!(
            generic_error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));

        let vocabulary = Vocabulary::from_bytes(data.as_bytes(), &ParakeetDeployment::v3())
            .expect("published v3 tokenizer should match the deployment contract");
        assert_eq!(vocabulary.len(), 8_192);
        assert_eq!(vocabulary.blank_id(), 8_192);
        assert_eq!(vocabulary.token(8_191), Some("Œ"));
        assert_eq!(vocabulary.token(8_192), None);
    }

    #[test]
    fn wrong_blank_id_is_rejected_by_the_deployment() {
        let mut tokens = published_v3_tokens();
        tokens.insert("8191".to_owned(), "<blk>".to_owned());
        let data = serde_json::to_string(&tokens).expect("wrong-blank fixture should serialize");
        let error = Vocabulary::from_bytes(data.as_bytes(), &ParakeetDeployment::v3())
            .expect_err("v3 must reject a blank inside the tokenizer table");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));
    }

    #[test]
    fn missing_blank_is_rejected_for_sentence_piece_vocabularies() {
        let data = "a 0\nb 1\n";
        let path = tempfile::NamedTempFile::new().expect("temporary vocabulary should exist");
        std::fs::write(path.path(), data).expect("temporary vocabulary should be written");
        let error = Vocabulary::load(path.path(), &ParakeetDeployment::v2())
            .expect_err("v2 must require its explicit blank token");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));
    }

    #[test]
    fn wrong_v3_vocabulary_size_is_rejected() {
        let tokens: BTreeMap<_, _> = (0..8_191)
            .map(|id| (id.to_string(), format!("token-{id}")))
            .collect();
        let data = serde_json::to_string(&tokens).expect("wrong-size fixture should serialize");
        let error = Vocabulary::from_bytes(data.as_bytes(), &ParakeetDeployment::v3())
            .expect_err("v3 must require the published tokenizer size");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));
    }
}

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::deployment::{ParakeetDeployment, VocabKind};
use crate::error::TranscriptionError;
use crate::language::ParakeetLanguage;

/// Vocabulary owner for v2 `vocab.txt` and v3 JSON tokenizers
///
/// SentencePiece files include an explicit `<blk>` token. NVIDIA Parakeet TDT
/// v3 JSON tables omit blank; the deployment supplies the RNNT blank id one
/// past the last tokenizer entry.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub(crate) id_to_token: Vec<String>,
    pub(crate) blank_id: usize,
}

impl Vocabulary {
    /// Parse a SentencePiece `vocab.txt` file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TranscriptionError> {
        parse_file(path, None)
    }

    /// Parse a Parakeet JSON tokenizer such as `parakeet_vocab.json`
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self, TranscriptionError> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json_str(&data)
    }

    /// Load a vocabulary using the parser owned by a deployment
    pub fn load(path: impl AsRef<Path>, kind: VocabKind) -> Result<Self, TranscriptionError> {
        let deployment = match kind {
            VocabKind::SentencePieceTxt => ParakeetDeployment::v2(),
            VocabKind::JsonTokenizer => ParakeetDeployment::v3(),
        };
        Self::load_for_deployment(path, &deployment)
    }

    /// Load and validate the vocabulary required by a deployment
    pub fn load_for_deployment(
        path: impl AsRef<Path>,
        deployment: &ParakeetDeployment,
    ) -> Result<Self, TranscriptionError> {
        match deployment.vocab_kind() {
            VocabKind::SentencePieceTxt => Self::from_file_for_deployment(path, deployment),
            VocabKind::JsonTokenizer => {
                let data = std::fs::read_to_string(path)?;
                Self::from_json_str_for_deployment(&data, deployment)
            }
        }
    }

    /// Parse JSON tokenizer bytes without a filesystem path
    pub fn from_json_str(data: &str) -> Result<Self, TranscriptionError> {
        parse_json_str(data, None)
    }

    /// Parse a tokenizer using a deployment's exact blank contract
    pub fn from_json_str_for_deployment(
        data: &str,
        deployment: &ParakeetDeployment,
    ) -> Result<Self, TranscriptionError> {
        if deployment.vocab_kind() != VocabKind::JsonTokenizer {
            return Err(TranscriptionError::InvalidVocabulary(
                "JSON tokenizer is not valid for this deployment".to_owned(),
            ));
        }

        let vocabulary = parse_json_str(data, Some(deployment.blank_id()))?;
        deployment.validate_vocabulary(&vocabulary)?;
        Ok(vocabulary)
    }

    /// Parse a `vocab.txt` file using a deployment's exact blank contract
    pub fn from_file_for_deployment(
        path: impl AsRef<Path>,
        deployment: &ParakeetDeployment,
    ) -> Result<Self, TranscriptionError> {
        if deployment.vocab_kind() != VocabKind::SentencePieceTxt {
            return Err(TranscriptionError::InvalidVocabulary(
                "SentencePiece tokenizer is not valid for this deployment".to_owned(),
            ));
        }

        let vocabulary = parse_file(path, Some(deployment.blank_id()))?;
        deployment.validate_vocabulary(&vocabulary)?;
        Ok(vocabulary)
    }

    /// Look up a token string by id
    pub fn token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(String::as_str)
    }

    /// Resolved RNNT blank token ID
    ///
    /// Deployment-aware loading verifies this value against the joint model
    /// contract before it can be used for decoding.
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

fn parse_file(
    path: impl AsRef<Path>,
    expected_blank_id: Option<usize>,
) -> Result<Vocabulary, TranscriptionError> {
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
        return finish_json_entries(entries, expected_blank_id);
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
            return finish_json_entries(entries, expected_blank_id);
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
            return finish_json_entries(entries, expected_blank_id);
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

fn finish_json_entries(
    entries: Vec<(usize, String)>,
    expected_blank_id: Option<usize>,
) -> Result<Vocabulary, TranscriptionError> {
    finish_entries(entries, expected_blank_id)
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

fn is_blank_token(token: &str) -> bool {
    matches!(token, "<blk>" | "<blank>")
}

#[cfg(test)]
mod tests {
    use super::Vocabulary;
    use crate::deployment::ParakeetDeployment;
    use crate::language::ParakeetLanguage;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn fixture_path(relative: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
    }

    #[test]
    fn blank_is_taken_from_the_explicit_token() {
        let vocab = Vocabulary::from_tokens(&["a", "<blk>"]);
        assert_eq!(vocab.blank_id(), 1);
    }

    #[test]
    fn v2_vocab_txt_blank_is_blk() {
        let vocab = Vocabulary::from_file(fixture_path("fixtures/models/parakeet-v2/vocab.txt"))
            .expect("v2 vocab fixture should parse");
        assert_eq!(vocab.blank_id(), 1_024);
        assert_eq!(vocab.token(vocab.blank_id()), Some("<blk>"));
        ParakeetDeployment::v2()
            .validate_vocabulary(&vocab)
            .expect("v2 fixture should match the deployment contract");
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
    fn json_tokenizer_without_blank_is_rejected() {
        let error = Vocabulary::from_json_str(r#"{"0":"<unk>","1":"hello","2":"world"}"#)
            .expect_err("an arbitrary tokenizer must not infer its blank id");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));
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
        let generic_error = Vocabulary::from_json_str(&data)
            .expect_err("generic parsing must not infer a deployment blank");
        assert!(matches!(
            generic_error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));

        let vocabulary = Vocabulary::from_json_str_for_deployment(&data, &ParakeetDeployment::v3())
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
        let error = Vocabulary::from_json_str_for_deployment(&data, &ParakeetDeployment::v3())
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
        let error = Vocabulary::from_file_for_deployment(path.path(), &ParakeetDeployment::v2())
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
        let error = Vocabulary::from_json_str_for_deployment(&data, &ParakeetDeployment::v3())
            .expect_err("v3 must require the published tokenizer size");
        assert!(matches!(
            error,
            crate::error::TranscriptionError::InvalidVocabulary(_)
        ));
    }
}

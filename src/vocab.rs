use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::TranscriptionError;

/// Vocabulary parser for `vocab.txt`
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub(crate) id_to_token: Vec<String>,
    pub(crate) blank_id: usize,
}

impl Vocabulary {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TranscriptionError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut id_to_token = Vec::new();
        let mut blank_id = None;

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
            if id >= id_to_token.len() {
                id_to_token.resize(id + 1, String::new());
            }
            id_to_token[id] = token.to_owned();
            if matches!(token, "<blk>" | "<blank>") {
                blank_id = Some(id);
            }
        }

        if id_to_token.is_empty() {
            return Err(TranscriptionError::InvalidVocabulary(
                "vocabulary was empty".to_owned(),
            ));
        }

        Ok(Self {
            blank_id: blank_id.unwrap_or(id_to_token.len() - 1),
            id_to_token,
        })
    }

    pub fn token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(String::as_str)
    }

    pub fn blank_id(&self) -> usize {
        self.blank_id
    }
}

#[cfg(test)]
mod tests {
    use super::Vocabulary;

    #[test]
    fn blank_defaults_to_last_token() {
        let vocab = Vocabulary {
            id_to_token: vec!["a".to_owned(), "b".to_owned()],
            blank_id: 1,
        };
        assert_eq!(vocab.blank_id(), 1);
    }
}

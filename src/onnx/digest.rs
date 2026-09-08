use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// SHA-256 digest of one exported bundle file
///
/// The canonical text form is 64 lowercase hexadecimal characters. Uppercase
/// or shorter prefixes are rejected so one file has exactly one identity string.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Hash the full contents of a file
    pub fn of_file(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];

        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }

        Ok(Self(hasher.finalize().into()))
    }

    /// Hash an in-memory byte string
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Raw digest bytes
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sha256Digest({self})")
    }
}

/// Reason a digest string was not canonical SHA-256 hex
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid SHA-256 digest `{value}`: {reason}")]
pub struct DigestParseError {
    value: String,
    reason: &'static str,
}

impl FromStr for Sha256Digest {
    type Err = DigestParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let invalid = |reason| DigestParseError {
            value: value.to_owned(),
            reason,
        };
        if value.len() != 64 {
            return Err(invalid("expected 64 hexadecimal characters"));
        }

        let mut bytes = [0u8; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = hex_nibble(pair[0]).ok_or_else(|| invalid("expected lowercase hex"))?;
            let low = hex_nibble(pair[1]).ok_or_else(|| invalid("expected lowercase hex"))?;
            bytes[index] = (high << 4) | low;
        }

        Ok(Self(bytes))
    }
}

impl TryFrom<String> for Sha256Digest {
    type Error = DigestParseError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<Sha256Digest> for String {
    fn from(digest: Sha256Digest) -> Self {
        digest.to_string()
    }
}

fn hex_nibble(character: u8) -> Option<u8> {
    match character {
        b'0'..=b'9' => Some(character - b'0'),
        b'a'..=b'f' => Some(character - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::Sha256Digest;

    #[test]
    fn digest_round_trips_through_canonical_hex() {
        let digest = Sha256Digest::of_bytes(b"abc");
        assert_eq!(
            digest.to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(digest.to_string().parse::<Sha256Digest>().unwrap(), digest);
    }

    #[test]
    fn digest_rejects_uppercase_short_and_non_hex_text() {
        let upper = "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD";
        assert!(upper.parse::<Sha256Digest>().is_err());
        assert!("ba7816bf".parse::<Sha256Digest>().is_err());
        let bad = "zz7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(bad.parse::<Sha256Digest>().is_err());
    }
}

use crate::deployment::ParakeetVersion;
use crate::error::TranscriptionError;
use crate::vocab::Vocabulary;

/// Languages in the NVIDIA Parakeet TDT v3 allowlist
///
/// Automatic detection and explicit selection may only use this set. v2
/// deployments accept English only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParakeetLanguage {
    /// Bulgarian (`bg`)
    Bulgarian,
    /// Croatian (`hr`)
    Croatian,
    /// Czech (`cs`)
    Czech,
    /// Danish (`da`)
    Danish,
    /// Dutch (`nl`)
    Dutch,
    /// English (`en`)
    English,
    /// Estonian (`et`)
    Estonian,
    /// Finnish (`fi`)
    Finnish,
    /// French (`fr`)
    French,
    /// German (`de`)
    German,
    /// Greek (`el`)
    Greek,
    /// Hungarian (`hu`)
    Hungarian,
    /// Italian (`it`)
    Italian,
    /// Latvian (`lv`)
    Latvian,
    /// Lithuanian (`lt`)
    Lithuanian,
    /// Maltese (`mt`)
    Maltese,
    /// Polish (`pl`)
    Polish,
    /// Portuguese (`pt`)
    Portuguese,
    /// Romanian (`ro`)
    Romanian,
    /// Russian (`ru`)
    Russian,
    /// Slovak (`sk`)
    Slovak,
    /// Slovenian (`sl`)
    Slovenian,
    /// Spanish (`es`)
    Spanish,
    /// Swedish (`sv`)
    Swedish,
    /// Ukrainian (`uk`)
    Ukrainian,
}

/// NVIDIA Parakeet TDT v3 language allowlist, including English
pub const V3_LANGUAGES: &[ParakeetLanguage] = &[
    ParakeetLanguage::Bulgarian,
    ParakeetLanguage::Croatian,
    ParakeetLanguage::Czech,
    ParakeetLanguage::Danish,
    ParakeetLanguage::Dutch,
    ParakeetLanguage::English,
    ParakeetLanguage::Estonian,
    ParakeetLanguage::Finnish,
    ParakeetLanguage::French,
    ParakeetLanguage::German,
    ParakeetLanguage::Greek,
    ParakeetLanguage::Hungarian,
    ParakeetLanguage::Italian,
    ParakeetLanguage::Latvian,
    ParakeetLanguage::Lithuanian,
    ParakeetLanguage::Maltese,
    ParakeetLanguage::Polish,
    ParakeetLanguage::Portuguese,
    ParakeetLanguage::Romanian,
    ParakeetLanguage::Russian,
    ParakeetLanguage::Slovak,
    ParakeetLanguage::Slovenian,
    ParakeetLanguage::Spanish,
    ParakeetLanguage::Swedish,
    ParakeetLanguage::Ukrainian,
];

/// Writing script used by top-K language filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    /// Latin script, including Latin Extended blocks
    Latin,
    /// Cyrillic script
    Cyrillic,
    /// Greek script
    Greek,
}

/// Language selection for a transcription run
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LanguageSelection {
    /// Detect a language, but only inside the deployment allowlist
    #[default]
    Auto,
    /// Use one allowlisted language
    Language(ParakeetLanguage),
}

/// Token chosen by top-K language filtering
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilteredToken {
    /// Vocabulary id of the selected token
    pub token_id: usize,
    /// Softmax over the supplied top-K logits only
    pub probability: f32,
}

impl ParakeetLanguage {
    /// ISO 639-1 code for this language
    pub const fn code(self) -> &'static str {
        match self {
            Self::Bulgarian => "bg",
            Self::Croatian => "hr",
            Self::Czech => "cs",
            Self::Danish => "da",
            Self::Dutch => "nl",
            Self::English => "en",
            Self::Estonian => "et",
            Self::Finnish => "fi",
            Self::French => "fr",
            Self::German => "de",
            Self::Greek => "el",
            Self::Hungarian => "hu",
            Self::Italian => "it",
            Self::Latvian => "lv",
            Self::Lithuanian => "lt",
            Self::Maltese => "mt",
            Self::Polish => "pl",
            Self::Portuguese => "pt",
            Self::Romanian => "ro",
            Self::Russian => "ru",
            Self::Slovak => "sk",
            Self::Slovenian => "sl",
            Self::Spanish => "es",
            Self::Swedish => "sv",
            Self::Ukrainian => "uk",
        }
    }

    /// Parse an ISO 639-1 code into the NVIDIA v3 allowlist
    pub fn from_code(code: &str) -> Option<Self> {
        V3_LANGUAGES
            .iter()
            .copied()
            .find(|language| language.code() == code)
    }

    /// Parse a tokenizer language token such as `<en>` or `<|pl|>`
    pub fn from_token(token: &str) -> Option<Self> {
        let code = token
            .strip_prefix("<|")
            .and_then(|value| value.strip_suffix("|>"))
            .or_else(|| {
                token
                    .strip_prefix('<')
                    .and_then(|value| value.strip_suffix('>'))
            })?;
        Self::from_code(code)
    }

    /// Script used by this language
    pub const fn script(self) -> Script {
        match self {
            Self::Bulgarian | Self::Russian | Self::Ukrainian => Script::Cyrillic,
            Self::Greek => Script::Greek,
            Self::Croatian
            | Self::Czech
            | Self::Danish
            | Self::Dutch
            | Self::English
            | Self::Estonian
            | Self::Finnish
            | Self::French
            | Self::German
            | Self::Hungarian
            | Self::Italian
            | Self::Latvian
            | Self::Lithuanian
            | Self::Maltese
            | Self::Polish
            | Self::Portuguese
            | Self::Romanian
            | Self::Slovak
            | Self::Slovenian
            | Self::Spanish
            | Self::Swedish => Script::Latin,
        }
    }
}

impl LanguageSelection {
    /// Reject a language that is not in `allowlist`
    pub fn validate(
        self,
        allowlist: &[ParakeetLanguage],
        version: ParakeetVersion,
    ) -> Result<Self, TranscriptionError> {
        match self {
            Self::Auto => Ok(Self::Auto),
            Self::Language(language) => {
                if allowlist.contains(&language) {
                    Ok(self)
                } else {
                    Err(TranscriptionError::UnsupportedLanguage {
                        language: language.code().to_owned(),
                        version,
                    })
                }
            }
        }
    }

    /// Return the explicit language when one was selected
    pub const fn explicit(self) -> Option<ParakeetLanguage> {
        match self {
            Self::Auto => None,
            Self::Language(language) => Some(language),
        }
    }
}

/// Select the highest-logit language that is inside `allowlist`
///
/// `languages` and `logits` are parallel. Entries whose language is not in the
/// allowlist are ignored, so automatic detection cannot leave the typed set.
/// Non-finite logits are rejected.
pub fn select_language_from_logits(
    languages: &[ParakeetLanguage],
    logits: &[f32],
    allowlist: &[ParakeetLanguage],
) -> Result<Option<ParakeetLanguage>, TranscriptionError> {
    validate_candidate_logits(logits)?;
    let count = languages.len().min(logits.len());
    let mut best_language = None;
    let mut best_logit = f32::NEG_INFINITY;
    for index in 0..count {
        let language = languages[index];
        if !allowlist.contains(&language) {
            continue;
        }
        let logit = logits[index];
        if best_language.is_none() || logit > best_logit {
            best_logit = logit;
            best_language = Some(language);
        }
    }
    Ok(best_language)
}

/// Detect a language token inside decoded ids, limited to `allowlist`
pub fn detect_language_from_tokens(
    token_ids: &[u32],
    vocabulary: &Vocabulary,
    allowlist: &[ParakeetLanguage],
) -> Option<ParakeetLanguage> {
    for token_id in token_ids {
        let Some(token) = vocabulary.token(*token_id as usize) else {
            continue;
        };
        let Some(language) = ParakeetLanguage::from_token(token) else {
            continue;
        };
        if allowlist.contains(&language) {
            return Some(language);
        }
    }
    None
}

/// Pick the highest-logit top-K token whose script matches `language`
///
/// Blank tokens are never replaced. Returned probability is softmax over the
/// supplied top-K logits only, not the full vocabulary. Non-finite logits are
/// rejected.
pub fn filter_top_k_language(
    top_k_ids: &[usize],
    top_k_logits: &[f32],
    vocabulary: &Vocabulary,
    language: ParakeetLanguage,
    blank_id: usize,
) -> Result<Option<FilteredToken>, TranscriptionError> {
    validate_candidate_logits(top_k_logits)?;
    let count = top_k_ids.len().min(top_k_logits.len());
    if count == 0 {
        return Ok(None);
    }

    let preferred_script = language.script();
    let mut best_idx = None;
    let mut best_logit = f32::NEG_INFINITY;
    for index in 0..count {
        let token_id = top_k_ids[index];
        if token_id == blank_id {
            continue;
        }
        let Some(token_text) = vocabulary.token(token_id) else {
            continue;
        };
        if !token_matches_script(token_text, preferred_script) {
            continue;
        }
        let logit = top_k_logits[index];
        if best_idx.is_none() || logit > best_logit {
            best_logit = logit;
            best_idx = Some(index);
        }
    }
    let Some(best_idx) = best_idx else {
        return Ok(None);
    };

    let mut max_logit = f32::NEG_INFINITY;
    for logit in top_k_logits.iter().copied().take(count) {
        if logit > max_logit {
            max_logit = logit;
        }
    }
    if !max_logit.is_finite() {
        return Ok(Some(FilteredToken {
            token_id: top_k_ids[best_idx],
            probability: 0.0,
        }));
    }

    let mut sum_exp = 0.0f32;
    for logit in top_k_logits.iter().copied().take(count) {
        sum_exp += (logit - max_logit).exp();
    }
    if sum_exp <= 0.0 {
        return Ok(Some(FilteredToken {
            token_id: top_k_ids[best_idx],
            probability: 0.0,
        }));
    }

    let probability = ((top_k_logits[best_idx] - max_logit).exp() / sum_exp).clamp(0.0, 1.0);
    Ok(Some(FilteredToken {
        token_id: top_k_ids[best_idx],
        probability,
    }))
}

pub(crate) fn validate_candidate_logits(logits: &[f32]) -> Result<(), TranscriptionError> {
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(TranscriptionError::InvalidLanguageLogit { index, value });
        }
    }
    Ok(())
}

fn token_matches_script(token: &str, script: Script) -> bool {
    let cleaned = token.replace('\u{2581}', "");
    if cleaned.is_empty() {
        return true;
    }

    cleaned
        .chars()
        .all(|character| char_matches_script(character, script))
}

fn char_matches_script(character: char, script: Script) -> bool {
    let value = character as u32;
    match script {
        Script::Latin => {
            (0x0020..=0x007F).contains(&value)
                || (0x00A0..=0x00FF).contains(&value)
                || (0x0100..=0x017F).contains(&value)
                || (0x0180..=0x024F).contains(&value)
                || (0x0300..=0x036F).contains(&value)
                || (0x1E00..=0x1EFF).contains(&value)
        }
        Script::Cyrillic => {
            if (0x0400..=0x04FF).contains(&value) {
                return true;
            }
            if (0x0020..=0x007F).contains(&value) {
                return !character.is_ascii_alphabetic();
            }
            false
        }
        Script::Greek => {
            if (0x0370..=0x03FF).contains(&value) || (0x1F00..=0x1FFF).contains(&value) {
                return true;
            }
            if (0x0300..=0x036F).contains(&value) {
                return true;
            }
            if (0x0020..=0x007F).contains(&value) {
                return !character.is_ascii_alphabetic();
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LanguageSelection, ParakeetLanguage, V3_LANGUAGES, detect_language_from_tokens,
        filter_top_k_language, select_language_from_logits,
    };
    use crate::deployment::ParakeetVersion;
    use crate::error::TranscriptionError;
    use crate::vocab::Vocabulary;

    fn vocab(tokens: &[&str]) -> Vocabulary {
        Vocabulary::from_tokens(tokens)
    }

    #[test]
    fn unknown_language_code_is_outside_the_allowlist() {
        assert!(ParakeetLanguage::from_code("zh").is_none());
        assert!(ParakeetLanguage::from_code("no").is_none());
        assert!(ParakeetLanguage::from_code("tr").is_none());
    }

    #[test]
    fn v3_allowlist_has_twenty_five_languages() {
        assert_eq!(V3_LANGUAGES.len(), 25);
        assert!(V3_LANGUAGES.contains(&ParakeetLanguage::English));
        assert!(V3_LANGUAGES.contains(&ParakeetLanguage::Maltese));
        assert!(V3_LANGUAGES.contains(&ParakeetLanguage::Ukrainian));
    }

    #[test]
    fn selecting_a_non_allowlisted_language_fails() {
        let error = LanguageSelection::Language(ParakeetLanguage::Polish)
            .validate(&[ParakeetLanguage::English], ParakeetVersion::V2)
            .expect_err("polish is not valid on v2");
        match error {
            TranscriptionError::UnsupportedLanguage { language, version } => {
                assert_eq!(language, "pl");
                assert_eq!(version, ParakeetVersion::V2);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn auto_selection_is_always_valid() {
        let selection = LanguageSelection::Auto
            .validate(&[ParakeetLanguage::English], ParakeetVersion::V2)
            .unwrap();
        assert_eq!(selection, LanguageSelection::Auto);
    }

    #[test]
    fn auto_logits_cannot_leave_the_allowlist() {
        let languages = [
            ParakeetLanguage::English,
            ParakeetLanguage::Polish,
            ParakeetLanguage::Russian,
        ];
        let logits = [0.1, 0.2, 4.0];
        let selected = select_language_from_logits(
            &languages,
            &logits,
            &[ParakeetLanguage::English, ParakeetLanguage::Polish],
        )
        .unwrap();
        assert_eq!(selected, Some(ParakeetLanguage::Polish));
    }

    #[test]
    fn non_finite_language_logits_are_rejected() {
        let languages = [ParakeetLanguage::English];
        let logits = [f32::NAN];
        assert!(matches!(
            select_language_from_logits(&languages, &logits, &languages),
            Err(TranscriptionError::InvalidLanguageLogit { index: 0, value })
                if value.is_nan()
        ));

        let vocabulary = vocab(&["<unk>", "cat", "<blk>"]);
        assert!(matches!(
            filter_top_k_language(
                &[1],
                &[f32::INFINITY],
                &vocabulary,
                ParakeetLanguage::English,
                2,
            ),
            Err(TranscriptionError::InvalidLanguageLogit { index: 0, value })
                if value.is_infinite()
        ));
    }

    #[test]
    fn top_k_filter_maps_known_logits_to_known_language_ids() {
        let vocabulary = vocab(&["<unk>", "<pl>", "<ru>", "kot", "кот", "<blk>"]);
        let filtered = filter_top_k_language(
            &[4, 3, 1],
            &[9.0, 2.0, 1.0],
            &vocabulary,
            ParakeetLanguage::Polish,
            5,
        )
        .unwrap()
        .unwrap();
        assert_eq!(filtered.token_id, 3);
    }

    #[test]
    fn top_k_filter_keeps_cyrillic_for_russian() {
        let vocabulary = vocab(&["<unk>", "cat", "кот", "<blk>"]);
        let filtered = filter_top_k_language(
            &[1, 2],
            &[8.0, 1.5],
            &vocabulary,
            ParakeetLanguage::Russian,
            3,
        )
        .unwrap()
        .unwrap();
        assert_eq!(filtered.token_id, 2);
    }

    #[test]
    fn detect_language_skips_tokens_outside_the_allowlist() {
        let vocabulary = vocab(&["<zh>", "<pl>", "<blk>"]);
        let detected = detect_language_from_tokens(&[0, 1], &vocabulary, V3_LANGUAGES);
        assert_eq!(detected, Some(ParakeetLanguage::Polish));
    }

    #[test]
    fn language_tokens_parse_angle_and_pipe_forms() {
        assert_eq!(
            ParakeetLanguage::from_token("<en>"),
            Some(ParakeetLanguage::English)
        );
        assert_eq!(
            ParakeetLanguage::from_token("<|uk|>"),
            Some(ParakeetLanguage::Ukrainian)
        );
        assert_eq!(ParakeetLanguage::from_token("<zh>"), None);
    }
}

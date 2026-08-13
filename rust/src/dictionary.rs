//! Lazy, bounded access to the optional human-language dictionary.
//!
//! The source contains more than one hundred thousand entries.  It is loaded
//! only when a human prompt needs it, and each lookup returns a tiny, typed
//! evidence bundle rather than allowing a full gloss to become a sentence
//! template or a world fact.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeaningEvidence {
    pub word: String,
    pub category_tokens: Vec<String>,
    pub definition_tokens: Vec<String>,
    pub category_bigrams: Vec<(String, String)>,
    pub synonyms: Vec<String>,
    pub antonyms: Vec<String>,
}

#[derive(Debug)]
pub struct HumanDictionary {
    path: Option<PathBuf>,
    entries: Option<BTreeMap<String, Value>>,
}

impl HumanDictionary {
    pub fn discover(workspace_root: &Path) -> Self {
        let candidates = [
            workspace_root.join("python/filtered.json"),
            workspace_root.join("filtered.json"),
        ];
        Self {
            path: candidates.into_iter().find(|candidate| candidate.is_file()),
            entries: None,
        }
    }

    #[cfg(test)]
    pub fn at(path: PathBuf) -> Self {
        Self {
            path: Some(path),
            entries: None,
        }
    }

    pub fn available(&mut self) -> bool {
        self.load().is_some()
    }

    pub fn meaning_evidence(&mut self, token: &str) -> Option<MeaningEvidence> {
        let word = normalise(token)?;
        let entry = self.entry(&word)?;
        let meanings = entry.get("MEANINGS")?.as_array()?;
        let mut category_tokens = Vec::new();
        let mut definition_tokens = Vec::new();
        let mut category_bigrams = Vec::new();
        for meaning in meanings {
            let Some(fields) = meaning.as_array() else {
                continue;
            };
            if let Some(definition) = fields.get(1).and_then(Value::as_str) {
                definition_tokens.extend(words(definition));
            }
            for category in fields
                .get(2)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                let phrase = words(category);
                category_bigrams.extend(
                    phrase
                        .windows(2)
                        .map(|pair| (pair[0].clone(), pair[1].clone())),
                );
                category_tokens.extend(phrase);
            }
        }
        let synonyms = values_to_words(entry.get("SYNONYMS"));
        let antonyms = values_to_words(entry.get("ANTONYMS"));
        Some(MeaningEvidence {
            word: word.clone(),
            category_tokens: unique(category_tokens, &word, 12),
            definition_tokens: unique(definition_tokens, &word, 12),
            category_bigrams: unique_bigrams(category_bigrams, &word, 8),
            synonyms: unique(synonyms, &word, 8),
            antonyms: unique(antonyms, &word, 8),
        })
    }

    fn entry(&mut self, word: &str) -> Option<&Value> {
        let entries = self.load()?;
        let candidates = inflections(word);
        candidates.into_iter().find_map(|candidate| {
            entries
                .get(&candidate.to_ascii_uppercase())
                .or_else(|| entries.get(&candidate))
                .filter(|entry| entry.get("MEANINGS").is_some())
        })
    }

    fn load(&mut self) -> Option<&BTreeMap<String, Value>> {
        if self.entries.is_none() {
            let path = self.path.as_ref()?;
            let text = fs::read_to_string(path).ok()?;
            self.entries = serde_json::from_str::<BTreeMap<String, Value>>(&text).ok();
        }
        self.entries.as_ref()
    }
}

fn inflections(word: &str) -> Vec<String> {
    let mut values = vec![word.to_string()];
    if word.ends_with("ies") && word.len() > 4 {
        values.push(format!("{}y", &word[..word.len() - 3]));
    } else if word.ends_with('s') && word.len() > 3 && !word.ends_with("ss") {
        values.push(word[..word.len() - 1].to_string());
    }
    values
}

fn words(text: &str) -> Vec<String> {
    text.split(|character: char| {
        !character.is_ascii_alphabetic() && character != '\'' && character != '-'
    })
    .filter_map(normalise)
    .collect()
}

fn normalise(token: &str) -> Option<String> {
    let token = token.trim().to_ascii_lowercase();
    (!token.is_empty() && token.len() <= 32).then_some(token)
}

fn values_to_words(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .flat_map(words)
        .collect()
}

fn unique(values: Vec<String>, exclude: &str, limit: usize) -> Vec<String> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .filter(|value| value != exclude && seen.insert(value.clone()))
        .take(limit)
        .collect()
}

fn unique_bigrams(
    values: Vec<(String, String)>,
    exclude: &str,
    limit: usize,
) -> Vec<(String, String)> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .filter(|(left, right)| {
            left != exclude && right != exclude && seen.insert((left.clone(), right.clone()))
        })
        .take(limit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::HumanDictionary;
    use std::fs;

    #[test]
    fn extracts_bounded_category_and_definition_evidence() {
        let path = std::env::temp_dir().join("mas-rust-dictionary-test.json");
        fs::write(
            &path,
            r#"{"APPLE":{"MEANINGS":[["Noun","an edible red fruit",["Edible fruit","Plant food"],[]]],"SYNONYMS":["pome"],"ANTONYMS":[]}}"#,
        )
        .unwrap();
        let mut dictionary = HumanDictionary::at(path.clone());
        let evidence = dictionary.meaning_evidence("apples").unwrap();
        assert!(evidence.category_tokens.contains(&"edible".to_string()));
        assert!(evidence.definition_tokens.contains(&"fruit".to_string()));
        assert_eq!(evidence.synonyms, vec!["pome"]);
        let _ = fs::remove_file(path);
    }
}

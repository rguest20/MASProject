use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::learning::{CONTEXT_LIMIT, Observation};
use crate::model::Rng;

const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from", "has", "have",
    "he", "her", "his", "i", "in", "is", "it", "its", "me", "my", "not", "of", "on", "or", "our",
    "she", "so", "that", "the", "their", "them", "then", "there", "they", "this", "to", "was",
    "we", "were", "when", "will", "with", "would", "you", "your",
];

#[derive(Debug, Default)]
pub struct ReadingBridge {
    sentences: Vec<Vec<String>>,
    cursor: usize,
    token_sentences: BTreeMap<String, BTreeSet<usize>>,
    contexts: BTreeMap<String, BTreeSet<String>>,
    transitions: BTreeMap<String, BTreeMap<String, usize>>,
    promoted: BTreeSet<String>,
    masked_profiles: BTreeMap<usize, BTreeMap<String, u32>>,
    masked_contexts: BTreeMap<(usize, String, String), BTreeMap<String, u32>>,
    role_positions: BTreeMap<String, BTreeSet<usize>>,
    role_left: BTreeMap<String, BTreeSet<String>>,
    role_right: BTreeMap<String, BTreeSet<String>>,
    inquiry_index: Option<BTreeMap<String, Vec<(usize, usize)>>>,
    inquiry_locations: BTreeMap<u64, (usize, usize)>,
    source_hashes: Vec<u64>,
}

impl ReadingBridge {
    /// Fingerprints canonical corpus content and order, independently of cursors.
    pub fn inquiry_fingerprint(&self) -> u64 {
        self.sentences
            .iter()
            .fold(0xcbf29ce484222325, |hash, words| {
                (hash ^ source_hash(words)).wrapping_mul(0x100000001b3)
            })
    }

    /// Validate checkpoint evidence against the corpus, once at startup. Saved
    /// text is never trusted as a new observation or used to choose a partition.
    pub fn inquiry_sources(&self, requested: &BTreeSet<u64>) -> BTreeMap<u64, (Observation, bool)> {
        let mut found = BTreeMap::new();
        for words in &self.sentences {
            let hash = source_hash(words);
            for position in 1..words.len() {
                let source = hash.wrapping_mul(31).wrapping_add(position as u64);
                if requested.contains(&source) {
                    found.insert(
                        source,
                        (
                            Observation {
                                source,
                                context: words[position.saturating_sub(CONTEXT_LIMIT)..position]
                                    .to_vec(),
                                outcome: words[position].clone(),
                            },
                            hash.is_multiple_of(5),
                        ),
                    );
                }
            }
        }
        found
    }

    #[cfg(test)]
    pub fn inquiry_fixture(sentences: Vec<Vec<String>>) -> Self {
        Self {
            sentences,
            ..Self::default()
        }
    }

    pub fn inquiry_examples(words: &[String]) -> Vec<Observation> {
        let hash = source_hash(words);
        if hash.is_multiple_of(5) {
            return Vec::new();
        }
        (1..words.len())
            .map(|position| Observation {
                source: hash.wrapping_mul(31).wrapping_add(position as u64),
                context: words[position.saturating_sub(CONTEXT_LIMIT)..position].to_vec(),
                outcome: words[position].clone(),
            })
            .collect()
    }

    /// Look up one source-labelled episode after a topic is chosen. The
    /// evaluation partition never supplies inquiry training or testimony.
    pub fn inquiry_observation(
        &mut self,
        context: &[String],
        excluded: &BTreeSet<u64>,
        evaluation: bool,
        rng: &mut Rng,
    ) -> Option<Observation> {
        if self.sentences.is_empty() {
            return None;
        }
        self.ensure_inquiry_index();
        let candidates = context
            .last()
            .and_then(|word| self.inquiry_index.as_ref()?.get(word));
        if !context.is_empty() && candidates.is_none() {
            return None;
        }
        for _ in 0..64 {
            let (sentence_id, position) = if let Some(candidates) = candidates {
                candidates[rng.index(candidates.len())]
            } else {
                let sentence_id = rng.index(self.sentences.len());
                let length = self.sentences[sentence_id].len();
                if length < 2 {
                    continue;
                }
                (sentence_id, 1 + rng.index(length - 1))
            };
            // Partition by content hash, keeping duplicate sentences together.
            if self.source_hashes[sentence_id].is_multiple_of(5) != evaluation {
                continue;
            }
            let words = &self.sentences[sentence_id];
            if !words[..position].ends_with(context) {
                continue;
            }
            let source = self.source_hashes[sentence_id]
                .wrapping_mul(31)
                .wrapping_add(position as u64);
            if excluded.contains(&source) {
                continue;
            }
            return Some(Observation {
                source,
                context: words[position.saturating_sub(CONTEXT_LIMIT)..position].to_vec(),
                outcome: words[position].clone(),
            });
        }
        None
    }

    /// Display-only source continuation. Inquiry learns only the first
    /// observed outcome; callers must not feed this additional text into the
    /// learner or use it for selection/reward.
    pub fn inquiry_source_continuation(&mut self, source: u64, limit: usize) -> Vec<String> {
        self.ensure_inquiry_index();
        let Some(&(sentence_id, position)) = self.inquiry_locations.get(&source) else {
            return Vec::new();
        };
        self.sentences[sentence_id][position..]
            .iter()
            .take(limit)
            .cloned()
            .collect()
    }

    fn ensure_inquiry_index(&mut self) {
        if self.inquiry_index.is_some() {
            return;
        }
        let mut index: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
        self.source_hashes = self
            .sentences
            .iter()
            .map(|words| source_hash(words))
            .collect();
        for (sentence_id, words) in self.sentences.iter().enumerate() {
            for position in 1..words.len() {
                let source = self.source_hashes[sentence_id]
                    .wrapping_mul(31)
                    .wrapping_add(position as u64);
                self.inquiry_locations
                    .insert(source, (sentence_id, position));
                index
                    .entry(words[position - 1].clone())
                    .or_default()
                    .push((sentence_id, position));
            }
        }
        self.inquiry_index = Some(index);
    }

    pub fn memory_value(&self) -> Value {
        json!({"cursor": self.cursor, "profiles": self.masked_profiles, "contexts": self.masked_contexts.iter().map(|((p,l,r), values)| json!([p,l,r,values])).collect::<Vec<_>>()})
    }

    pub fn restore_memory_value(&mut self, value: &Value) {
        self.cursor = value.get("cursor").and_then(Value::as_u64).unwrap_or(0) as usize;
        if let Some(profiles) = value.get("profiles") {
            self.masked_profiles = serde_json::from_value(profiles.clone()).unwrap_or_default();
        }
        if let Some(contexts) = value.get("contexts").and_then(Value::as_array) {
            self.masked_contexts.clear();
            for item in contexts {
                let Some(parts) = item.as_array() else {
                    continue;
                };
                if parts.len() != 4 {
                    continue;
                }
                let (Some(position), Some(left), Some(right)) =
                    (parts[0].as_u64(), parts[1].as_str(), parts[2].as_str())
                else {
                    continue;
                };
                let values = serde_json::from_value(parts[3].clone()).unwrap_or_default();
                self.masked_contexts.insert(
                    (position as usize, left.to_string(), right.to_string()),
                    values,
                );
            }
        }
    }
    fn masked_holdout(index: usize) -> bool {
        index.is_multiple_of(5)
    }
    pub fn load(workspace_root: &Path) -> Self {
        let candidates = [
            workspace_root.join("python/clean_merged_fairy_tales_without_eos.txt"),
            workspace_root.join("python/cleaned_merged_fairy_tales_without_eos.txt"),
            workspace_root.join("clean_merged_fairy_tales_without_eos.txt"),
            workspace_root.join("cleaned_merged_fairy_tales_without_eos.txt"),
        ];
        let text = candidates
            .iter()
            .find_map(|path| fs::read_to_string(path).ok());
        let sentences = text.as_deref().map(split_sentences).unwrap_or_default();
        Self {
            sentences,
            ..Self::default()
        }
    }

    pub fn read_passage(&mut self) -> Vec<Vec<String>> {
        if self.sentences.is_empty() {
            return Vec::new();
        }
        let mut passage = Vec::new();
        for _ in 0..2 {
            let index = self.cursor % self.sentences.len();
            // The corpus is millions of characters long. A wide stride keeps
            // a run from spending thousands of generations in its opening
            // story while remaining deterministic and reproducible.
            self.cursor = self.cursor.saturating_add(7919);
            let words = self.sentences[index].clone();
            let distinct: BTreeSet<_> = words.iter().cloned().collect();
            for word in &distinct {
                self.token_sentences
                    .entry(word.clone())
                    .or_default()
                    .insert(index);
            }
            for (position, word) in words.iter().enumerate() {
                self.role_positions
                    .entry(word.clone())
                    .or_default()
                    .insert(position.min(15));
                if let Some(left) = position.checked_sub(1).and_then(|i| words.get(i)) {
                    self.role_left
                        .entry(word.clone())
                        .or_default()
                        .insert(left.clone());
                }
                if let Some(right) = words.get(position + 1) {
                    self.role_right
                        .entry(word.clone())
                        .or_default()
                        .insert(right.clone());
                }
                let context = self.contexts.entry(word.clone()).or_default();
                for neighbour in words.iter().skip(position.saturating_sub(3)).take(7) {
                    if neighbour != word && !is_stop_word(neighbour) {
                        context.insert(neighbour.clone());
                    }
                }
            }
            for pair in words.windows(2) {
                *self
                    .transitions
                    .entry(pair[0].clone())
                    .or_default()
                    .entry(pair[1].clone())
                    .or_default() += 1;
            }
            if !Self::masked_holdout(index) {
                for (position, word) in words.iter().enumerate() {
                    *self
                        .masked_profiles
                        .entry(position % 16)
                        .or_default()
                        .entry(word.clone())
                        .or_default() += 1;
                    let left = position
                        .checked_sub(1)
                        .and_then(|i| words.get(i))
                        .cloned()
                        .unwrap_or_default();
                    let right = words.get(position + 1).cloned().unwrap_or_default();
                    *self
                        .masked_contexts
                        .entry((position % 16, left, right))
                        .or_default()
                        .entry(word.clone())
                        .or_default() += 1;
                }
            }
            passage.push(words);
        }
        self.refresh_promotions();
        passage
    }

    pub fn relevant_words(&self, anchors: &BTreeSet<String>) -> BTreeSet<String> {
        self.relevant_word_scores(anchors).into_keys().collect()
    }

    /// Reading vocabulary remains quarantined until it is both promoted and
    /// contextually relevant to the current human/topic words.  Scores are
    /// intentionally small so story text can assist a reply but cannot drown
    /// out the human conversation graph.
    pub fn relevant_word_scores(&self, anchors: &BTreeSet<String>) -> BTreeMap<String, f64> {
        let anchors: BTreeSet<_> = anchors
            .iter()
            .filter(|word| !is_stop_word(word))
            .cloned()
            .collect();
        self.promoted
            .iter()
            .filter_map(|word| {
                let contextual_overlap = self
                    .contexts
                    .get(word)
                    .map(|context| context.intersection(&anchors).count())
                    .unwrap_or_default();
                (anchors.contains(word) || contextual_overlap > 0).then(|| {
                    (
                        word.clone(),
                        0.10 + (0.04 * contextual_overlap as f64).min(0.16),
                    )
                })
            })
            .collect()
    }

    pub fn relevant_continuations(&self, word: &str, allowed: &BTreeSet<String>) -> Vec<String> {
        self.transitions
            .get(word)
            .into_iter()
            .flat_map(|continuations| continuations.keys())
            .filter(|candidate| allowed.contains(*candidate))
            .cloned()
            .collect()
    }

    pub fn promoted_count(&self) -> usize {
        self.promoted.len()
    }

    /// Positional substitution profiles are learned from masked reading
    /// contexts. They are distributional evidence only, not world claims.
    pub fn masked_slot_families(&self) -> Vec<BTreeSet<String>> {
        self.masked_profiles
            .values()
            .filter_map(|profile| {
                let members: BTreeSet<_> = profile
                    .iter()
                    .filter(|(_, count)| **count >= 3)
                    .map(|(word, _)| word.clone())
                    .collect();
                (members.len() >= 2).then_some(members)
            })
            .collect()
    }

    pub fn masked_trial(
        &self,
        index: usize,
    ) -> Option<(usize, Vec<String>, String, String, String)> {
        if self.sentences.is_empty() {
            return None;
        }
        let sentence = self.sentences.get(index % self.sentences.len())?;
        if sentence.len() < 3 {
            return None;
        }
        let position = (index / self.sentences.len()) % sentence.len();
        let target = sentence[position].clone();
        let context = sentence
            .iter()
            .enumerate()
            .filter_map(|(i, word)| (i != position).then_some(word.clone()))
            .collect();
        let left = position
            .checked_sub(1)
            .and_then(|i| sentence.get(i))
            .cloned()
            .unwrap_or_default();
        let right = sentence.get(position + 1).cloned().unwrap_or_default();
        Some((position % 16, context, target, left, right))
    }

    pub fn masked_prediction(&self, position: usize) -> Option<String> {
        self.masked_profiles
            .get(&position)?
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(word, _)| word.clone())
    }

    pub fn contextual_masked_prediction(
        &self,
        position: usize,
        left: &str,
        right: &str,
    ) -> Option<String> {
        self.masked_contexts
            .get(&(position, left.to_string(), right.to_string()))
            .and_then(|profile| {
                profile
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(word, _)| word.clone())
            })
            .or_else(|| self.masked_prediction(position))
    }

    pub fn has_masked_context(&self, position: usize, left: &str, right: &str) -> bool {
        self.masked_contexts
            .contains_key(&(position, left.to_string(), right.to_string()))
    }

    pub fn masked_holdout_trial(&self, index: usize) -> bool {
        if self.sentences.is_empty() {
            return false;
        }
        self.sentences
            .get(index % self.sentences.len())
            .is_some_and(|_| Self::masked_holdout(index % self.sentences.len()))
    }

    pub fn masked_candidates(&self, position: usize) -> Vec<String> {
        self.masked_profiles
            .get(&position)
            .map(|profile| profile.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Unlabelled distributional role signatures: positional spread and
    /// diversity of incoming/outgoing neighbours.
    pub fn role_signature_count(&self) -> usize {
        let mut signatures = BTreeSet::new();
        for token in self.role_positions.keys() {
            let p = self.role_positions[token].len().min(15) as u8;
            let l = self.role_left.get(token).map_or(0, |v| v.len().min(15)) as u8;
            let r = self.role_right.get(token).map_or(0, |v| v.len().min(15)) as u8;
            if p > 0 {
                signatures.insert((p / 3, l / 3, r / 3));
            }
        }
        signatures.len()
    }

    /// An unlabelled distributional signature.  It records where a token has
    /// occurred and the diversity of its observed neighbours; callers must
    /// not interpret the buckets as supplied grammatical categories.
    pub fn role_signature(&self, token: &str) -> Option<(u8, u8, u8)> {
        let positions = self.role_positions.get(token)?;
        Some((
            (positions.len().min(15) as u8) / 3,
            (self.role_left.get(token).map_or(0, BTreeSet::len).min(15) as u8) / 3,
            (self.role_right.get(token).map_or(0, BTreeSet::len).min(15) as u8) / 3,
        ))
    }

    pub fn masked_context_family(
        &self,
        position: usize,
        left: &str,
        right: &str,
    ) -> BTreeSet<String> {
        self.masked_contexts
            .get(&(position, left.to_string(), right.to_string()))
            .map(|p| p.keys().cloned().collect())
            .unwrap_or_else(|| {
                self.masked_slot_families()
                    .into_iter()
                    .next()
                    .unwrap_or_default()
            })
    }

    /// Story words that have earned enough recurring contextual evidence.
    /// Exploration ranks this frontier by each agent's semantic connectivity;
    /// normal conversation still admits story vocabulary only through
    /// `relevant_word_scores`.
    pub fn exploration_candidates(&self) -> Vec<String> {
        self.promoted.iter().cloned().collect()
    }

    /// Corpus-held evidence for an inquiry claim. This is deliberately a
    /// count, rather than a truth value: story co-occurrence can corroborate
    /// a proposal but cannot prove a fact about the physical world.
    pub fn inquiry_evidence(&self, left: &str, right: &str) -> usize {
        let Some(left_sentences) = self.token_sentences.get(left) else {
            return 0;
        };
        let Some(right_sentences) = self.token_sentences.get(right) else {
            return 0;
        };
        left_sentences.intersection(right_sentences).count()
    }

    pub fn inquiry_neighbours(&self, word: &str) -> Vec<String> {
        self.contexts
            .get(word)
            .into_iter()
            .flat_map(|words| words.iter())
            .filter(|word| self.promoted.contains(*word))
            .cloned()
            .collect()
    }

    fn refresh_promotions(&mut self) {
        for (word, sentence_ids) in &self.token_sentences {
            let contexts = self
                .contexts
                .get(word)
                .map(BTreeSet::len)
                .unwrap_or_default();
            if word.len() >= 3
                && word.chars().all(char::is_alphabetic)
                && !is_stop_word(word)
                && sentence_ids.len() >= 4
                && contexts >= 3
            {
                self.promoted.insert(word.clone());
            }
        }
    }
}

fn split_sentences(text: &str) -> Vec<Vec<String>> {
    text.split(['.', '!', '?', '\n'])
        .map(tokenise)
        .filter(|words| (4..=30).contains(&words.len()))
        .collect()
}

fn source_hash(words: &[String]) -> u64 {
    words
        .iter()
        .flat_map(|word| word.bytes().chain(std::iter::once(0)))
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ byte as u64).wrapping_mul(0x100000001b3)
        })
}

pub fn tokenise(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_ascii_alphabetic() && character != '\'')
        .filter(|word| !word.is_empty())
        .map(|word| word.to_ascii_lowercase())
        .take(30)
        .collect()
}

fn is_stop_word(word: &str) -> bool {
    STOP_WORDS.contains(&word)
}

#[cfg(test)]
mod inquiry_tests {
    use super::*;

    #[test]
    fn source_sampling_keeps_training_and_test_evidence_separate() {
        let mut reading = ReadingBridge {
            sentences: (0..100)
                .map(|i| vec![format!("s{i}"), "cue".into(), "outcome".into()])
                .collect(),
            ..ReadingBridge::default()
        };
        let mut rng = Rng::new(904);
        let mut seen = BTreeSet::new();
        let context = vec!["cue".into()];
        for _ in 0..30 {
            let source = reading
                .inquiry_observation(&context, &seen, false, &mut rng)
                .unwrap();
            assert!(source.context.ends_with(&context));
            assert!(seen.insert(source.source));
        }
        for _ in 0..30 {
            let source = reading
                .inquiry_observation(&context, &BTreeSet::new(), true, &mut rng)
                .unwrap();
            assert!(!seen.contains(&source.source));
            assert!(
                !reading
                    .sentences
                    .iter()
                    .flat_map(|words| ReadingBridge::inquiry_examples(words))
                    .any(|example| example.source == source.source)
            );
        }
        assert!(
            reading
                .inquiry_observation(&["unknown".into()], &seen, false, &mut rng)
                .is_none()
        );
    }

    #[test]
    fn no_corpus_is_a_supported_empty_experiment() {
        let mut reading = ReadingBridge::default();
        assert!(reading.masked_trial(1).is_none());
        assert!(
            reading
                .inquiry_observation(&[], &BTreeSet::new(), false, &mut Rng::new(1))
                .is_none()
        );
    }

    #[test]
    fn source_continuation_is_available_for_display_without_creating_evidence() {
        let words = vec!["she".into(), "had".into(), "become".into(), "quiet".into()];
        let source = source_hash(&words).wrapping_mul(31).wrapping_add(1);
        let mut reading = ReadingBridge::inquiry_fixture(vec![words]);
        assert_eq!(
            reading.inquiry_source_continuation(source, 4),
            vec!["had", "become", "quiet"]
        );
        assert!(
            reading
                .inquiry_sources(&BTreeSet::from([source]))
                .contains_key(&source)
        );
    }
}

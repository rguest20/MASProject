use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

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
}

impl ReadingBridge {
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
            self.cursor += 1;
            let words = self.sentences[index].clone();
            let distinct: BTreeSet<_> = words.iter().cloned().collect();
            for word in &distinct {
                self.token_sentences
                    .entry(word.clone())
                    .or_default()
                    .insert(index);
            }
            for (position, word) in words.iter().enumerate() {
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

    /// Story words that have earned enough recurring contextual evidence.
    /// Exploration ranks this frontier by each agent's semantic connectivity;
    /// normal conversation still admits story vocabulary only through
    /// `relevant_word_scores`.
    pub fn exploration_candidates(&self) -> Vec<String> {
        self.promoted.iter().cloned().collect()
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

//! Community-owned language conventions.
//!
//! A convention is deliberately an outcome of successful interaction, not a
//! bootstrap dictionary.  This mirrors `python/evolution/community_lexicon.py`:
//! agents retain private alternatives until the same mapping has enough
//! evidence and a sufficiently clear majority to become public.

use std::collections::BTreeMap;

use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct CommunityLexicon {
    min_successes: u32,
    min_confidence: f64,
    replacement_margin: u32,
    numeric_evidence: BTreeMap<u32, BTreeMap<String, u32>>,
    referential_evidence: BTreeMap<String, BTreeMap<String, u32>>,
    action_evidence: BTreeMap<String, BTreeMap<String, u32>>,
    base_evidence: BTreeMap<u32, u32>,
    grammar_evidence: BTreeMap<String, BTreeMap<Vec<String>, u32>>,
    numeric_promoted: BTreeMap<u32, String>,
    referential_promoted: BTreeMap<String, String>,
    action_promoted: BTreeMap<String, String>,
    base_promoted: Option<u32>,
    grammar_promoted: BTreeMap<String, Vec<String>>,
}

impl Default for CommunityLexicon {
    fn default() -> Self {
        Self::new(2, 0.60, 2)
    }
}

impl CommunityLexicon {
    pub fn new(min_successes: u32, min_confidence: f64, replacement_margin: u32) -> Self {
        Self {
            min_successes,
            min_confidence,
            replacement_margin,
            numeric_evidence: BTreeMap::new(),
            referential_evidence: BTreeMap::new(),
            action_evidence: BTreeMap::new(),
            base_evidence: BTreeMap::new(),
            grammar_evidence: BTreeMap::new(),
            numeric_promoted: BTreeMap::new(),
            referential_promoted: BTreeMap::new(),
            action_promoted: BTreeMap::new(),
            base_promoted: None,
            grammar_promoted: BTreeMap::new(),
        }
    }

    pub fn observe_numeric_success(&mut self, digit: u32, token: &str) -> bool {
        let Some(token) = clean_token(token) else {
            return false;
        };
        let previous = self.numeric_token(digit).map(str::to_owned);
        *self
            .numeric_evidence
            .entry(digit)
            .or_default()
            .entry(token)
            .or_default() += 1;
        self.refresh_numeric(digit) != previous
    }

    pub fn observe_referential_success(&mut self, referent: &str, token: &str) -> bool {
        let (Some(referent), Some(token)) = (clean_token(referent), clean_token(token)) else {
            return false;
        };
        let previous = self.referential_signal(&referent).map(str::to_owned);
        *self
            .referential_evidence
            .entry(referent.clone())
            .or_default()
            .entry(token)
            .or_default() += 1;
        self.refresh_referent(&referent) != previous
    }

    pub fn observe_action_success(&mut self, action: &str, token: &str) -> bool {
        let (Some(action), Some(token)) = (clean_token(action), clean_token(token)) else {
            return false;
        };
        let previous = self.action_signal(&action).map(str::to_owned);
        *self
            .action_evidence
            .entry(action.clone())
            .or_default()
            .entry(token)
            .or_default() += 1;
        self.refresh_action(&action) != previous
    }

    pub fn observe_base_success(&mut self, base: u32) -> bool {
        if base < 2 {
            return false;
        }
        let previous = self.community_base();
        *self.base_evidence.entry(base).or_default() += 1;
        self.refresh_base() != previous
    }

    pub fn observe_grammar_success(&mut self, task_type: &str, order: &[String]) -> bool {
        let Some(task_type) = clean_token(task_type) else {
            return false;
        };
        if order.is_empty() || order.iter().any(|part| clean_token(part).is_none()) {
            return false;
        }
        let order: Vec<String> = order.iter().filter_map(|part| clean_token(part)).collect();
        let previous = self.grammar_order(&task_type).map(|value| value.to_vec());
        *self
            .grammar_evidence
            .entry(task_type.clone())
            .or_default()
            .entry(order)
            .or_default() += 1;
        self.refresh_grammar(&task_type) != previous
    }

    pub fn numeric_token(&self, digit: u32) -> Option<&str> {
        self.numeric_promoted.get(&digit).map(String::as_str)
    }

    pub fn referential_signal(&self, referent: &str) -> Option<&str> {
        self.referential_promoted
            .get(&referent.to_ascii_lowercase())
            .map(String::as_str)
    }

    pub fn action_signal(&self, action: &str) -> Option<&str> {
        self.action_promoted
            .get(&action.to_ascii_lowercase())
            .map(String::as_str)
    }

    pub fn community_base(&self) -> Option<u32> {
        self.base_promoted
    }

    pub fn grammar_order(&self, task_type: &str) -> Option<&[String]> {
        self.grammar_promoted
            .get(&task_type.to_ascii_lowercase())
            .map(Vec::as_slice)
    }

    pub fn numeric_support(&self, digit: u32) -> u32 {
        self.numeric_evidence
            .get(&digit)
            .map(|evidence| evidence.values().sum())
            .unwrap_or(0)
    }

    pub fn numeric_confidence(&self, digit: u32) -> f64 {
        confidence(self.numeric_evidence.get(&digit))
    }

    pub fn referential_support(&self, referent: &str) -> u32 {
        self.referential_evidence
            .get(&referent.to_ascii_lowercase())
            .map(|evidence| evidence.values().sum())
            .unwrap_or(0)
    }

    pub fn action_support(&self, action: &str) -> u32 {
        self.action_evidence
            .get(&action.to_ascii_lowercase())
            .map(|evidence| evidence.values().sum())
            .unwrap_or(0)
    }

    pub fn numeric_conventions(&self) -> BTreeMap<u32, String> {
        self.numeric_promoted.clone()
    }

    pub fn referential_conventions(&self) -> BTreeMap<String, String> {
        self.referential_promoted.clone()
    }

    pub fn action_conventions(&self) -> BTreeMap<String, String> {
        self.action_promoted.clone()
    }

    pub fn grammar_conventions(&self) -> BTreeMap<String, Vec<String>> {
        self.grammar_promoted.clone()
    }

    pub fn referent_for_signal(&self, token: &str) -> Option<&str> {
        let token = clean_token(token)?;
        let mut matches = self
            .referential_promoted
            .iter()
            .filter_map(|(referent, signal)| (signal == &token).then_some(referent.as_str()));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }

    pub fn action_for_signal(&self, token: &str) -> Option<&str> {
        let token = clean_token(token)?;
        let mut matches = self
            .action_promoted
            .iter()
            .filter_map(|(action, signal)| (signal == &token).then_some(action.as_str()));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }

    /// Snapshot only the evidence behind public conventions. Private agent
    /// overlays stay with the run that produced them.
    pub fn memory_value(&self) -> Value {
        let grammar = self
            .grammar_evidence
            .iter()
            .map(|(task, evidence)| {
                (
                    task.clone(),
                    Value::Array(
                        evidence
                            .iter()
                            .map(|(order, support)| json!({"order": order, "support": support}))
                            .collect(),
                    ),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        json!({
            "numeric_evidence": self.numeric_evidence,
            "referential_evidence": self.referential_evidence,
            "action_evidence": self.action_evidence,
            "base_evidence": self.base_evidence,
            "grammar_evidence": grammar,
        })
    }

    pub fn restore_memory_value(&mut self, value: &Value) {
        let Some(value) = value.as_object() else {
            return;
        };
        restore_numeric_evidence(value.get("numeric_evidence"), &mut self.numeric_evidence);
        restore_string_evidence(
            value.get("referential_evidence"),
            &mut self.referential_evidence,
        );
        restore_string_evidence(value.get("action_evidence"), &mut self.action_evidence);
        if let Some(entries) = value.get("base_evidence").and_then(Value::as_object) {
            for (base, support) in entries {
                let Ok(base) = base.parse::<u32>() else {
                    continue;
                };
                let support = support.as_u64().unwrap_or(0).min(1_000_000) as u32;
                if base >= 2 && support > 0 {
                    *self.base_evidence.entry(base).or_default() += support;
                }
            }
        }
        if let Some(tasks) = value.get("grammar_evidence").and_then(Value::as_object) {
            for (task, records) in tasks {
                let Some(task) = clean_token(task) else {
                    continue;
                };
                for record in records.as_array().into_iter().flatten() {
                    let Some(order) = record.get("order").and_then(Value::as_array) else {
                        continue;
                    };
                    let order: Vec<_> = order
                        .iter()
                        .filter_map(Value::as_str)
                        .filter_map(clean_token)
                        .collect();
                    let support = record
                        .get("support")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                        .min(1_000_000) as u32;
                    if !order.is_empty() && support > 0 {
                        *self
                            .grammar_evidence
                            .entry(task.clone())
                            .or_default()
                            .entry(order)
                            .or_default() += support;
                    }
                }
            }
        }
        for digit in self.numeric_evidence.keys().copied().collect::<Vec<_>>() {
            self.refresh_numeric(digit);
        }
        for referent in self
            .referential_evidence
            .keys()
            .cloned()
            .collect::<Vec<_>>()
        {
            self.refresh_referent(&referent);
        }
        for action in self.action_evidence.keys().cloned().collect::<Vec<_>>() {
            self.refresh_action(&action);
        }
        self.refresh_base();
        for task in self.grammar_evidence.keys().cloned().collect::<Vec<_>>() {
            self.refresh_grammar(&task);
        }
    }

    fn refresh_numeric(&mut self, digit: u32) -> Option<String> {
        refresh_convention(
            &self.numeric_evidence,
            &mut self.numeric_promoted,
            digit,
            self.min_successes,
            self.min_confidence,
            self.replacement_margin,
        )
    }

    fn refresh_referent(&mut self, referent: &str) -> Option<String> {
        refresh_convention(
            &self.referential_evidence,
            &mut self.referential_promoted,
            referent.to_string(),
            self.min_successes,
            self.min_confidence,
            self.replacement_margin,
        )
    }

    fn refresh_action(&mut self, action: &str) -> Option<String> {
        refresh_convention(
            &self.action_evidence,
            &mut self.action_promoted,
            action.to_string(),
            self.min_successes,
            self.min_confidence,
            self.replacement_margin,
        )
    }

    fn refresh_base(&mut self) -> Option<u32> {
        let (candidate, support, total) = top_candidate(&self.base_evidence)?;
        match self.base_promoted {
            None if support >= self.min_successes
                && support as f64 / total as f64 >= self.min_confidence =>
            {
                self.base_promoted = Some(candidate);
            }
            Some(current) if candidate != current => {
                let current_support = self.base_evidence.get(&current).copied().unwrap_or(0);
                if support >= current_support + self.replacement_margin
                    && support as f64 / total as f64 >= self.min_confidence
                {
                    self.base_promoted = Some(candidate);
                }
            }
            _ => {}
        }
        self.base_promoted
    }

    fn refresh_grammar(&mut self, task_type: &str) -> Option<Vec<String>> {
        refresh_convention(
            &self.grammar_evidence,
            &mut self.grammar_promoted,
            task_type.to_string(),
            self.min_successes,
            self.min_confidence,
            self.replacement_margin,
        )
    }
}

fn restore_numeric_evidence(
    value: Option<&Value>,
    target: &mut BTreeMap<u32, BTreeMap<String, u32>>,
) {
    let Some(entries) = value.and_then(Value::as_object) else {
        return;
    };
    for (key, votes) in entries {
        let Ok(key) = key.parse::<u32>() else {
            continue;
        };
        restore_votes(votes, target.entry(key).or_default());
    }
}

fn restore_string_evidence(
    value: Option<&Value>,
    target: &mut BTreeMap<String, BTreeMap<String, u32>>,
) {
    let Some(entries) = value.and_then(Value::as_object) else {
        return;
    };
    for (key, votes) in entries {
        let Some(key) = clean_token(key) else {
            continue;
        };
        restore_votes(votes, target.entry(key).or_default());
    }
}

fn restore_votes(value: &Value, target: &mut BTreeMap<String, u32>) {
    let Some(votes) = value.as_object() else {
        return;
    };
    for (token, support) in votes {
        let Some(token) = clean_token(token) else {
            continue;
        };
        let support = support.as_u64().unwrap_or(0).min(1_000_000) as u32;
        if support > 0 {
            *target.entry(token).or_default() += support;
        }
    }
}

fn clean_token(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();
    (!value.is_empty()).then_some(value)
}

fn confidence<T: Ord>(evidence: Option<&BTreeMap<T, u32>>) -> f64 {
    let Some(evidence) = evidence else {
        return 0.0;
    };
    let total: u32 = evidence.values().sum();
    let support = evidence.values().copied().max().unwrap_or(0);
    if total == 0 {
        0.0
    } else {
        support as f64 / total as f64
    }
}

fn top_candidate<T: Ord + Clone>(evidence: &BTreeMap<T, u32>) -> Option<(T, u32, u32)> {
    let total: u32 = evidence.values().sum();
    let (candidate, support) = evidence
        .iter()
        .max_by_key(|(_, support)| *support)
        .map(|(candidate, support)| (candidate.clone(), *support))?;
    Some((candidate, support, total))
}

fn refresh_convention<K: Ord + Clone, V: Ord + Clone>(
    all_evidence: &BTreeMap<K, BTreeMap<V, u32>>,
    promoted: &mut BTreeMap<K, V>,
    key: K,
    min_successes: u32,
    min_confidence: f64,
    replacement_margin: u32,
) -> Option<V> {
    let evidence = all_evidence.get(&key)?;
    let (candidate, support, total) = top_candidate(evidence)?;
    let current = promoted.get(&key).cloned();
    match current {
        None if support >= min_successes && support as f64 / total as f64 >= min_confidence => {
            promoted.insert(key, candidate.clone());
            Some(candidate)
        }
        Some(current) if candidate != current => {
            let current_support = evidence.get(&current).copied().unwrap_or(0);
            if support >= current_support + replacement_margin
                && support as f64 / total as f64 >= min_confidence
            {
                promoted.insert(key, candidate.clone());
                Some(candidate)
            } else {
                Some(current)
            }
        }
        Some(current) => Some(current),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::CommunityLexicon;

    #[test]
    fn convention_requires_repeated_majority_evidence() {
        let mut lexicon = CommunityLexicon::default();
        assert!(!lexicon.observe_numeric_success(3, "Bel"));
        assert_eq!(lexicon.numeric_token(3), None);
        assert!(lexicon.observe_numeric_success(3, "bel"));
        assert_eq!(lexicon.numeric_token(3), Some("bel"));
        assert_eq!(lexicon.numeric_support(3), 2);
        assert_eq!(lexicon.numeric_confidence(3), 1.0);
    }

    #[test]
    fn convention_does_not_flip_on_a_single_competitor() {
        let mut lexicon = CommunityLexicon::default();
        lexicon.observe_referential_success("r3", "muk");
        lexicon.observe_referential_success("r3", "muk");
        assert_eq!(lexicon.referential_signal("r3"), Some("muk"));
        lexicon.observe_referential_success("r3", "tol");
        assert_eq!(lexicon.referential_signal("r3"), Some("muk"));
        lexicon.observe_referential_success("r3", "tol");
        lexicon.observe_referential_success("r3", "tol");
        lexicon.observe_referential_success("r3", "tol");
        assert_eq!(lexicon.referential_signal("r3"), Some("tol"));
    }

    #[test]
    fn inverse_lookup_fails_closed_for_ambiguous_signal() {
        let mut lexicon = CommunityLexicon::default();
        for referent in ["r1", "r2"] {
            lexicon.observe_referential_success(referent, "bel");
            lexicon.observe_referential_success(referent, "bel");
        }
        assert_eq!(lexicon.referent_for_signal("bel"), None);
    }

    #[test]
    fn grammar_and_base_are_learned_not_preseeded() {
        let mut lexicon = CommunityLexicon::default();
        assert_eq!(lexicon.community_base(), None);
        lexicon.observe_base_success(8);
        lexicon.observe_base_success(8);
        assert_eq!(lexicon.community_base(), Some(8));

        let order = vec!["action".into(), "referent".into(), "number".into()];
        lexicon.observe_grammar_success("referent_action_number", &order);
        lexicon.observe_grammar_success("referent_action_number", &order);
        assert_eq!(
            lexicon.grammar_order("referent_action_number"),
            Some(order.as_slice())
        );
    }
}

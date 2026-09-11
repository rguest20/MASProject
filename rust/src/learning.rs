//! Bounded, task-independent context → outcome evidence.
//! Only observed outcomes enter this store; predictions and repeated testimony
//! are not additional evidence. No word classes or world facts are supplied.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const MEMORY_LIMIT: usize = 384;
pub const CONTEXT_LIMIT: usize = 3;
pub const CONSOLIDATED_LIMIT: usize = 384;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub source: u64,
    pub context: Vec<String>,
    pub outcome: String,
}

#[derive(Clone, Debug)]
pub struct EvidenceMemory {
    examples: VecDeque<Observation>,
    consolidated: VecDeque<Observation>,
    memory_limit: usize,
    consolidated_limit: usize,
    per_context_limit: usize,
    // Decaying excess pre-observation loss relative to the frequency expert.
    regret: [f64; CONTEXT_LIMIT + 1],
    counts: BTreeMap<Vec<String>, BTreeMap<String, u32>>,
    sources: BTreeSet<u64>,
}

impl Default for EvidenceMemory {
    fn default() -> Self {
        Self::with_limits(MEMORY_LIMIT, CONSOLIDATED_LIMIT, 8)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Prediction {
    pub probabilities: BTreeMap<String, f64>,
    pub evidence: usize,
}

impl Prediction {
    pub fn best(&self) -> Option<&str> {
        self.probabilities
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(word, _)| word.as_str())
    }

    /// Multiclass Brier loss, including an outcome not yet in the vocabulary.
    /// Unknown is an abstention with loss 1; confidently wrong approaches 2.
    pub fn loss(&self, outcome: &str) -> f64 {
        1.0 - 2.0 * self.probabilities.get(outcome).copied().unwrap_or(0.0)
            + self.probabilities.values().map(|p| p * p).sum::<f64>()
    }

    pub fn disagreement(&self, other: &Self) -> f64 {
        let mut difference = 0.0;
        for (word, p) in &self.probabilities {
            difference += (p - other.probabilities.get(word).unwrap_or(&0.0)).abs();
        }
        for (word, p) in &other.probabilities {
            if !self.probabilities.contains_key(word) {
                difference += p;
            }
        }
        difference * 0.5
    }

    /// Combine independently sourced estimates.  The caller supplies the
    /// evidence-dependent weight of `prior`; this type does not assign any
    /// semantic meaning to either predictor.
    pub fn blend(&self, prior: &Self, prior_weight: f64) -> Self {
        let prior_weight = prior_weight.clamp(0.0, 1.0);
        let mut probabilities = BTreeMap::new();
        for (word, probability) in &self.probabilities {
            *probabilities.entry(word.clone()).or_default() += (1.0 - prior_weight) * probability;
        }
        for (word, probability) in &prior.probabilities {
            *probabilities.entry(word.clone()).or_default() += prior_weight * probability;
        }
        Self {
            probabilities,
            evidence: self.evidence.max(prior.evidence),
        }
    }
}

impl EvidenceMemory {
    pub fn with_limits(
        memory_limit: usize,
        consolidated_limit: usize,
        per_context_limit: usize,
    ) -> Self {
        Self {
            memory_limit,
            consolidated_limit,
            per_context_limit,
            examples: VecDeque::new(),
            consolidated: VecDeque::new(),
            regret: [0.0; CONTEXT_LIMIT + 1],
            counts: BTreeMap::new(),
            sources: BTreeSet::new(),
        }
    }

    /// Original evidence in eviction order; derived counts are rebuilt on load.
    pub fn observations(&self) -> impl Iterator<Item = &Observation> {
        self.consolidated.iter().chain(self.examples.iter())
    }

    pub fn contains(&self, source: u64) -> bool {
        self.sources.contains(&source)
    }

    pub fn observe(&mut self, observation: &Observation) -> bool {
        if observation.context.is_empty()
            || observation.outcome.is_empty()
            || self.contains(observation.source)
        {
            return false;
        }
        let mut observation = observation.clone();
        if observation.context.len() > CONTEXT_LIMIT {
            observation
                .context
                .drain(..observation.context.len() - CONTEXT_LIMIT);
        }
        let experts = self.experts(&observation.context);
        let baseline = experts[0].loss(&observation.outcome);
        for (regret, expert) in self.regret.iter_mut().zip(&experts) {
            *regret = (0.995 * *regret + expert.loss(&observation.outcome) - baseline)
                .clamp(-100.0, 100.0);
        }
        self.adjust(&observation, true);
        self.sources.insert(observation.source);
        self.examples.push_back(observation);
        if self.examples.len() > self.memory_limit {
            let oldest = self.examples.pop_front().expect("over capacity");
            let cue = &oldest.context[oldest.context.len() - 1..];
            if self.context_evidence(cue) >= 4 {
                // Keep a small source-backed sample of recurring contexts.
                // A frequent context cannot fill the entire long-term store.
                if self
                    .consolidated
                    .iter()
                    .filter(|e| e.context.ends_with(cue))
                    .count()
                    >= self.per_context_limit
                {
                    let index = self
                        .consolidated
                        .iter()
                        .position(|e| e.context.ends_with(cue))
                        .unwrap();
                    let retired = self.consolidated.remove(index).unwrap();
                    self.adjust(&retired, false);
                    self.sources.remove(&retired.source);
                }
                self.consolidated.push_back(oldest);
                if self.consolidated.len() > self.consolidated_limit {
                    let retired = self.consolidated.pop_front().unwrap();
                    self.adjust(&retired, false);
                    self.sources.remove(&retired.source);
                }
            } else {
                self.adjust(&oldest, false);
                self.sources.remove(&oldest.source);
            }
        }
        true
    }

    fn adjust(&mut self, observation: &Observation, add: bool) {
        for start in 0..=observation.context.len() {
            let context = &observation.context[start..];
            if add {
                *self
                    .counts
                    .entry(context.to_vec())
                    .or_default()
                    .entry(observation.outcome.clone())
                    .or_default() += 1;
            } else if let Some(row) = self.counts.get_mut(context) {
                if let Some(count) = row.get_mut(&observation.outcome) {
                    *count -= 1;
                    if *count == 0 {
                        row.remove(&observation.outcome);
                    }
                }
                if row.is_empty() {
                    self.counts.remove(context);
                }
            }
        }
    }

    pub fn consolidated_len(&self) -> usize {
        self.consolidated.len()
    }
    pub fn trust_state(&self) -> [f64; CONTEXT_LIMIT + 1] {
        self.regret
    }

    /// Restore validated source episodes without learning from them a second time.
    pub fn restore(
        observations: &[Observation],
        consolidated: usize,
        regret: [f64; CONTEXT_LIMIT + 1],
    ) -> Self {
        Self::restore_with_limits(
            observations,
            consolidated,
            regret,
            MEMORY_LIMIT,
            CONSOLIDATED_LIMIT,
            8,
        )
    }

    pub fn restore_with_limits(
        observations: &[Observation],
        consolidated: usize,
        regret: [f64; CONTEXT_LIMIT + 1],
        memory_limit: usize,
        consolidated_limit: usize,
        per_context_limit: usize,
    ) -> Self {
        let mut memory = Self::with_limits(memory_limit, consolidated_limit, per_context_limit);
        memory.regret = regret;
        for (index, observation) in observations.iter().enumerate() {
            memory.adjust(observation, true);
            memory.sources.insert(observation.source);
            if index < consolidated {
                memory.consolidated.push_back(observation.clone());
            } else {
                memory.examples.push_back(observation.clone());
            }
        }
        memory
    }

    pub fn trust_weights(&self) -> [f64; CONTEXT_LIMIT + 1] {
        let best = self.regret.iter().copied().fold(f64::INFINITY, f64::min);
        let raw = self.regret.map(|r| (-4.0 * (r - best)).exp());
        let total: f64 = raw.iter().sum();
        raw.map(|w| 0.01 + 0.96 * w / total)
    }

    pub fn predict(&self, context: &[String]) -> Prediction {
        let experts = self.experts(context);
        let mut result = Prediction::default();
        for (expert, weight) in experts.iter().zip(self.trust_weights()) {
            for (word, probability) in &expert.probabilities {
                *result.probabilities.entry(word.clone()).or_default() += weight * probability;
            }
            result.evidence = result.evidence.max(expert.evidence);
        }
        result
    }

    fn experts(&self, context: &[String]) -> Vec<Prediction> {
        // Mix specific contexts with shorter ones; scarce specific evidence
        // cannot erase a well-supported general expectation.
        let mut prediction = Prediction::default();
        let mut experts = Vec::new();
        for length in 0..=context.len().min(CONTEXT_LIMIT) {
            let key = &context[context.len() - length..];
            let Some(row) = self.counts.get(key) else {
                experts.push(prediction.clone());
                continue;
            };
            let total = row.values().sum::<u32>() as f64;
            let alpha = if prediction.evidence == 0 {
                1.0
            } else {
                total / (total + 8.0)
            };
            for p in prediction.probabilities.values_mut() {
                *p *= 1.0 - alpha;
            }
            for (outcome, count) in row {
                *prediction.probabilities.entry(outcome.clone()).or_default() +=
                    alpha * *count as f64 / total;
            }
            prediction.evidence = total as usize;
            experts.push(prediction.clone());
        }
        while experts.len() <= CONTEXT_LIMIT {
            experts.push(prediction.clone());
        }
        experts
    }

    pub fn lesson_for(&self, learner: &Self, context: &[String]) -> Option<Observation> {
        self.observations()
            .find(|example| example.context.ends_with(context) && !learner.contains(example.source))
            .cloned()
    }

    pub fn context_evidence(&self, context: &[String]) -> usize {
        self.counts
            .get(context)
            .map_or(0, |row| row.values().sum::<u32>() as usize)
    }

    /// A proposal assembled from learned conditional distributions. The caller
    /// must keep this distinct from an observation until it has been tested.
    pub fn compose(&self, context: &[String], limit: usize) -> Vec<String> {
        let mut words = context.to_vec();
        for _ in 0..limit.min(6) {
            let Some(next) = self.informative_next(&words) else {
                break;
            };
            words.push(next.to_string());
        }
        words
    }

    /// Choose an outcome because its local evidence exceeds its general
    /// frequency. A language's most common glue token is useful as a baseline,
    /// but must not be emitted merely because it is common everywhere.
    fn informative_next(&self, context: &[String]) -> Option<String> {
        let tail = &context[context.len().saturating_sub(CONTEXT_LIMIT)..];
        let (row, total) = (1..=tail.len()).rev().find_map(|length| {
            let row = self.counts.get(&tail[tail.len() - length..])?;
            let total = row.values().sum::<u32>() as usize;
            (total >= 3).then_some((row, total))
        })?;
        let global = self.counts.get(&Vec::new())?;
        let global_total = global.values().sum::<u32>() as f64;
        let support = (total as f64 / (total as f64 + 8.0)).sqrt();
        let mut candidate: Option<(&String, f64)> = None;
        for (word, count) in row {
            // Do not repeat either of the immediately preceding tokens. This
            // is an output constraint, not negative evidence about a word.
            if context
                .iter()
                .rev()
                .take(2)
                .any(|previous| previous == word)
            {
                continue;
            }
            let local = *count as f64 / total as f64;
            let background = global.get(word).copied().unwrap_or(0) as f64 / global_total;
            let score = (local - background) * support;
            if score > 0.03 && candidate.is_none_or(|(_, best)| score > best) {
                candidate = Some((word, score));
            }
        }
        candidate.map(|(word, _)| word.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn example(source: u64, context: &[&str], outcome: &str) -> Observation {
        Observation {
            source,
            context: context.iter().map(|s| s.to_string()).collect(),
            outcome: outcome.into(),
        }
    }

    #[test]
    fn recurring_evidence_survives_short_term_eviction_without_recounting() {
        let mut memory = EvidenceMemory::default();
        for id in 0..8 {
            memory.observe(&example(id, &["signal"], "result"));
        }
        for id in 8..(MEMORY_LIMIT as u64 + 8) {
            memory.observe(&example(id, &[&format!("unrelated{id}")], "other"));
        }
        assert_eq!(memory.context_evidence(&["signal".into()]), 8);
        assert_eq!(memory.consolidated_len(), 8);
        let saved: Vec<_> = memory.observations().cloned().collect();
        let mut restored =
            EvidenceMemory::restore(&saved, memory.consolidated_len(), memory.trust_state());
        for id in 500..1000 {
            let event = example(id, &["later"], "event");
            memory.observe(&event);
            restored.observe(&event);
        }
        assert_eq!(
            memory.observations().collect::<Vec<_>>(),
            restored.observations().collect::<Vec<_>>()
        );
        assert_eq!(memory.trust_state(), restored.trust_state());
        assert_eq!(
            memory.predict(&["signal".into()]).probabilities,
            restored.predict(&["signal".into()]).probabilities
        );
        let state = memory.trust_state();
        assert!(!memory.observe(&example(0, &["signal"], "result")));
        assert_eq!(memory.trust_state(), state);
        assert_eq!(memory.context_evidence(&["signal".into()]), 8);
    }

    #[test]
    fn trust_distinguishes_predictive_context_from_noise() {
        let mut useful = EvidenceMemory::default();
        let mut noisy = EvidenceMemory::default();
        let mut rng = crate::model::Rng::new(818);
        for id in 0..2000 {
            let cue = if rng.index(2) == 0 { "x" } else { "y" };
            useful.observe(&example(id, &[cue], if cue == "x" { "r" } else { "s" }));
            noisy.observe(&example(
                id,
                &[&format!("cue{}", rng.index(64))],
                if rng.index(2) == 0 { "r" } else { "s" },
            ));
        }
        assert!(
            useful.trust_weights()[0] < 0.1,
            "{:?}",
            useful.trust_weights()
        );
        assert!(
            noisy.trust_weights()[0] > 0.7,
            "{:?}",
            noisy.trust_weights()
        );
        assert!(useful.predict(&["x".into()]).loss("r") < useful.predict(&[]).loss("r"));
        let frozen = useful.trust_state();
        for _ in 0..100 {
            useful.predict(&["x".into()]);
        }
        assert_eq!(
            useful.trust_state(),
            frozen,
            "evaluation must not train trust"
        );
    }

    #[test]
    fn composition_requires_contextual_lift_and_does_not_echo_the_context() {
        let mut memory = EvidenceMemory::default();
        // `the` is globally common, but it carries no extra information after
        // `bridge`; `river` does.
        for id in 0..20 {
            memory.observe(&example(id, &["ordinary"], "the"));
        }
        for id in 20..28 {
            memory.observe(&example(id, &["bridge"], "river"));
        }
        for id in 28..32 {
            memory.observe(&example(id, &["bridge"], "the"));
        }
        let bridge = memory.compose(&["bridge".into()], 3);
        assert_eq!(bridge, vec!["bridge", "river"]);
        let mut loop_memory = EvidenceMemory::default();
        for id in 0..12 {
            loop_memory.observe(&example(id, &["the"], "the"));
        }
        assert_eq!(loop_memory.compose(&["the".into()], 3), vec!["the"]);
    }

    /// Fixed data, equal training exposure; no adaptive agenda or evolution.
    /// Run explicitly to measure the method independently of simulation noise.
    #[test]
    #[ignore = "corpus benchmark; run explicitly with --ignored --nocapture"]
    fn corpus_learning_comparison() {
        use crate::{model::Rng, reading::ReadingBridge};
        use std::{collections::BTreeSet, path::Path};
        let mut reading =
            ReadingBridge::load(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap());
        for seed in [71, 72, 73] {
            let mut rng = Rng::new(seed);
            let mut test_rng = Rng::new(901_449);
            let mut test_sources = BTreeSet::new();
            let mut probes = Vec::new();
            for _ in 0..256 {
                let probe = reading
                    .inquiry_observation(&[], &test_sources, true, &mut test_rng)
                    .unwrap();
                test_sources.insert(probe.source);
                probes.push(probe);
            }
            let mut new = EvidenceMemory::default();
            let mut recent = EvidenceMemory::default();
            let mut sources = BTreeSet::new();
            for step in 1..=3000 {
                let example = reading
                    .inquiry_observation(&[], &sources, false, &mut rng)
                    .unwrap();
                assert!(!test_sources.contains(&example.source));
                sources.insert(example.source);
                new.observe(&example);
                recent.observe(&example);
                // Equal-size recent-only ablation; old predictor reads its
                // fixed interpolation expert, adaptive ablation its mixture.
                while let Some(old) = recent.consolidated.pop_front() {
                    recent.adjust(&old, false);
                }
                if step % 1000 == 0 {
                    let mut losses = [0.0; 5];
                    for probe in &probes {
                        let predictions = [
                            recent.experts(&probe.context).pop().unwrap(),
                            recent.predict(&probe.context),
                            new.experts(&probe.context).pop().unwrap(),
                            new.predict(&probe.context),
                            new.predict(&[]),
                        ];
                        for (loss, prediction) in losses.iter_mut().zip(predictions) {
                            *loss += prediction.loss(&probe.outcome) / probes.len() as f64;
                        }
                    }
                    println!(
                        "{}",
                        serde_json::json!({"seed":seed,"training_sources":step,"test_sources":probes.len(),
                        "old_fixed_recent":losses[0],"adaptive_recent":losses[1],"fixed_consolidated":losses[2],
                        "adaptive_consolidated":losses[3],"consolidated_unigram":losses[4],
                        "trust":new.trust_weights(),"retained_sources":new.observations().count()})
                    );
                }
            }
        }
    }

    #[test]
    fn teaching_same_source_and_inheritance_do_not_multiply_evidence() {
        let event = example(1, &["turn"], "pull");
        let mut parent = EvidenceMemory::default();
        assert!(parent.observe(&event));
        let mut child = parent.clone();
        assert!(!child.observe(&event));
        assert_eq!(child.predict(&event.context).evidence, 1);
        assert_eq!(child.context_evidence(&["pull".into()]), 0);
    }

    #[test]
    fn learns_distributions_and_longer_contexts_without_grammar_labels() {
        let mut memory = EvidenceMemory::default();
        for id in 0..80 {
            memory.observe(&example(
                id,
                &[if id % 4 < 2 { "x" } else { "y" }, "q"],
                if id % 4 < 2 {
                    "r"
                } else if id % 4 == 2 {
                    "s"
                } else {
                    "t"
                },
            ));
        }
        let general = memory.predict(&["q".into()]);
        assert!((general.probabilities["r"] - 0.5).abs() < 1e-12);
        assert!((general.probabilities["s"] - 0.25).abs() < 1e-12);
        assert!(memory.predict(&["x".into(), "q".into()]).loss("r") < general.loss("r"));
    }

    #[test]
    fn forgotten_examples_are_also_removed_from_counts() {
        let mut memory = EvidenceMemory::default();
        memory.observe(&example(0, &["old"], "gone"));
        for id in 1..=MEMORY_LIMIT as u64 {
            memory.observe(&example(id, &["new"], "here"));
        }
        assert_eq!(memory.examples.len(), MEMORY_LIMIT);
        assert_eq!(memory.context_evidence(&["old".into()]), 0);
    }

    #[test]
    fn sourced_peer_lesson_improves_a_different_unseen_example() {
        let mut teacher = EvidenceMemory::default();
        let mut learner = EvidenceMemory::default();
        for id in 0..8 {
            learner.observe(&example(id, &["other"], "s"));
            teacher.observe(&example(100 + id, &["x", "cue"], "r"));
        }
        let held_out = example(999, &["y", "cue"], "r");
        let before = learner.predict(&held_out.context).loss(&held_out.outcome);
        let lesson = teacher.lesson_for(&learner, &["cue".into()]).unwrap();
        assert!(learner.observe(&lesson));
        let after = learner.predict(&held_out.context).loss(&held_out.outcome);
        assert!(after < before);
        assert!(!learner.contains(held_out.source));
        assert!(!teacher.contains(held_out.source));
        assert!(!learner.observe(&lesson));
        assert_eq!(
            learner.predict(&held_out.context).loss(&held_out.outcome),
            after
        );
    }
}

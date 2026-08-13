use std::collections::{BTreeMap, BTreeSet};

use crate::cognition::{
    Action, Needs, SocialMemory, State, Traits, TrustMap, TrustProfile, decay_social_memory,
    decay_state, decide_action, remember_interaction, state_event, update_needs,
};
use crate::language::LanguageProfile;
use crate::numeric::NumericSystem;
use crate::phase3::ProgramInstruction;
use crate::semantics::SemanticStore;

#[derive(Clone, Debug)]
pub struct Agent {
    pub id: usize,
    /// Stable biological/cultural identity. `id` is a reusable population
    /// slot; this value is never reused and is safe for lineage accounting.
    pub lineage_id: u64,
    pub identity_token: String,
    pub energy: f64,
    pub vocabulary: BTreeSet<String>,
    pub semantics: SemanticStore,
    pub numeric: NumericSystem,
    pub referent_lexicon: BTreeMap<String, String>,
    pub action_lexicon: BTreeMap<String, String>,
    pub fitness: f64,
    pub frustration: f64,
    pub tasks_solved: usize,
    pub tasks_failed: usize,
    pub traits: Traits,
    pub state: State,
    pub needs: Needs,
    pub trust: TrustMap,
    pub social_memory: SocialMemory,
    pub total_fitness: f64,
    pub cooperation_bonus: f64,
    pub novelty_bonus: f64,
    /// Slow, bounded evidence that this agent's descendants are functioning.
    /// Unlike task fitness it survives a generation, then naturally fades.
    pub lineage_score: f64,
    /// Typed, bounded arithmetic genome used by the Phase-3 behaviour
    /// evaluator.  It is deliberately data, never dynamically executed code.
    pub program: Vec<ProgramInstruction>,
    pub language: LanguageProfile,
    /// Birth data is deliberately small and immutable. It lets reports show
    /// whether selection is preserving useful lineages without turning the
    /// simulation into an unbounded genealogy database.
    pub birth_generation: u64,
    pub parent_lineages: Option<[u64; 3]>,
}

impl Agent {
    pub fn new(id: usize, rng: &mut Rng) -> Self {
        Self {
            id,
            lineage_id: id as u64,
            identity_token: format!("agent_{id}"),
            energy: 100.0,
            vocabulary: BTreeSet::new(),
            semantics: SemanticStore::new(),
            numeric: NumericSystem::new(rng),
            referent_lexicon: private_signal_map("r", 6, rng),
            action_lexicon: private_signal_map("a", 4, rng),
            fitness: 0.0,
            frustration: 0.0,
            tasks_solved: 0,
            tasks_failed: 0,
            traits: Traits::random(rng),
            state: State::default(),
            needs: Needs::default(),
            trust: TrustMap::new(),
            social_memory: SocialMemory::new(),
            total_fitness: 0.0,
            cooperation_bonus: 0.0,
            novelty_bonus: 0.0,
            lineage_score: 0.0,
            program: Vec::new(),
            language: LanguageProfile::new(rng),
            birth_generation: 0,
            parent_lineages: None,
        }
    }

    pub fn observe(&mut self, words: &[String], gain: f32, rng: &mut Rng) {
        self.vocabulary.extend(words.iter().cloned());
        self.semantics.observe(words, gain, rng);
    }

    pub fn referent_signal(&self, referent: &str) -> String {
        self.referent_lexicon
            .get(referent)
            .cloned()
            .unwrap_or_else(|| referent.to_string())
    }

    pub fn action_signal(&self, action: &str) -> String {
        self.action_lexicon
            .get(action)
            .cloned()
            .unwrap_or_else(|| action.to_string())
    }

    pub fn referent_for(&self, signal: &str) -> Option<&str> {
        unique_inverse(&self.referent_lexicon, signal)
    }

    pub fn action_for(&self, signal: &str) -> Option<&str> {
        unique_inverse(&self.action_lexicon, signal)
    }

    pub fn learn_referent(&mut self, referent: &str, signal: &str) {
        self.referent_lexicon
            .insert(referent.to_string(), signal.to_ascii_lowercase());
    }

    pub fn learn_action(&mut self, action: &str, signal: &str) {
        self.action_lexicon
            .insert(action.to_string(), signal.to_ascii_lowercase());
    }

    pub fn semantic_tick(&mut self, rng: &mut Rng) {
        self.semantics.tick(rng);
    }

    pub fn ensure_numeric_semantics(&mut self, rng: &mut Rng) {
        let base = self.numeric.base;
        let symbols: Vec<_> = self
            .numeric
            .symbols()
            .iter()
            .map(|(digit, token)| (*digit, token.clone()))
            .collect();
        for (digit, token) in symbols {
            self.semantics
                .ensure_numeric_token(&token, digit, base, rng);
            self.vocabulary.insert(token);
        }
    }

    pub fn decide_action(&self, rng: &mut Rng) -> Action {
        decide_action(&self.state, &self.traits, self.energy, rng)
    }

    pub fn update_motivations(&mut self) {
        update_needs(&mut self.needs, &self.state, self.energy);
        decay_state(&mut self.state);
    }

    pub fn state_event(&mut self, event: &str) {
        state_event(&mut self.state, event);
    }

    pub fn adjust_trust(&mut self, partner: usize, reward: f64) {
        self.trust.entry(partner).or_default().update(reward);
    }

    pub fn teaching_willingness(&self, partner: usize) -> f64 {
        self.trust
            .get(&partner)
            .map(TrustProfile::teaching_willingness)
            .unwrap_or(0.0)
    }

    pub fn remember_interaction(&mut self, partner: usize, outcome: f64, generation: u64) {
        remember_interaction(&mut self.social_memory, partner, outcome, generation);
        self.adjust_trust(partner, outcome);
    }

    pub fn decay_social_memory(&mut self, generation: u64) {
        let rate = (1.0 - self.traits.stability_weight).clamp(0.02, 0.40);
        decay_social_memory(&mut self.social_memory, rate, generation);
        self.traits.trust_threshold = (self.traits.trust_threshold
            + (0.35 - self.traits.trust_threshold) * 0.08)
            .clamp(0.10, 0.80);
    }

    pub fn apply_action_result(&mut self, action: Action, success: bool) {
        let event = match (action, success) {
            (Action::Social, true) => "successful_comm",
            (Action::Social, false) => "failed_comm",
            (Action::Teach, true) => "teaching_success",
            (Action::Teach, false) => "teaching_failure",
            (Action::Learn, true) => "learning_success",
            (Action::Learn, false) => "learning_failure",
            (Action::Explore, _) | (Action::Math, _) | (Action::Language, _) => "curiosity_boost",
            (Action::Reorganise, _) => "concept_cleanup",
            _ => "",
        };
        if !event.is_empty() {
            self.state_event(event);
        }
        if action != Action::Idle {
            self.energy = (self.energy - 0.5).max(0.0);
        }
    }

    pub fn teach(&mut self, student: &mut Self, generation: u64, rng: &mut Rng) -> bool {
        if self.energy <= 0.5 || self.traits.teaching_drive < rng.unit() {
            self.apply_action_result(Action::Teach, false);
            return false;
        }
        let tokens = self.semantics.token_names();
        if tokens.is_empty() {
            self.apply_action_result(Action::Teach, false);
            return false;
        }
        let token = &tokens[rng.index(tokens.len())];
        let Some(teacher_vector) = self.semantics.vector(token).cloned() else {
            return false;
        };
        let student_vector = student.semantics.vector(token);
        let similarity = student_vector
            .map(|vector| {
                let denom = teacher_vector.dot(&teacher_vector).sqrt() * vector.dot(vector).sqrt();
                if denom > f32::EPSILON {
                    teacher_vector.dot(vector) / denom
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        student
            .semantics
            .blend_from(token, &teacher_vector, 0.22, rng);
        let reward = if similarity > 0.75 {
            0.06
        } else if similarity > 0.45 {
            0.03
        } else {
            -0.06
        };
        self.adjust_trust(student.id, reward);
        student.adjust_trust(self.id, reward * 0.5);
        self.remember_interaction(student.id, reward, generation);
        student.remember_interaction(self.id, reward * 0.5, generation);
        self.energy = (self.energy + reward * 0.3).clamp(0.0, 100.0);
        student.energy = (student.energy + reward * 0.2).clamp(0.0, 100.0);
        self.apply_action_result(Action::Teach, reward > 0.0);
        student.state_event(if reward > 0.0 {
            "learning_success"
        } else {
            "learning_failure"
        });
        reward > 0.0
    }
}

fn private_signal_map(prefix: &str, count: usize, rng: &mut Rng) -> BTreeMap<String, String> {
    const VOWELS: &[u8] = b"aeiou";
    const CONSONANTS: &[u8] = b"bcdfghjklmnpqrstvwxyz";
    let mut used = BTreeSet::new();
    (0..count)
        .map(|index| {
            let token = loop {
                let candidate = format!(
                    "{}{}",
                    CONSONANTS[rng.index(CONSONANTS.len())] as char,
                    VOWELS[rng.index(VOWELS.len())] as char,
                );
                if used.insert(candidate.clone()) {
                    break candidate;
                }
            };
            (format!("{prefix}{index}"), token)
        })
        .collect()
}

fn unique_inverse<'a>(lexicon: &'a BTreeMap<String, String>, signal: &str) -> Option<&'a str> {
    let mut matches = lexicon
        .iter()
        .filter_map(|(meaning, candidate)| (candidate == signal).then_some(meaning.as_str()));
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

#[derive(Clone, Debug, Default)]
pub struct WorldModel {
    facts: BTreeMap<String, BTreeMap<String, f64>>,
}

impl WorldModel {
    pub fn observe(&mut self, subject: &str, property: &str, confidence: f64) {
        if subject.is_empty() || property.is_empty() || subject == property {
            return;
        }
        *self
            .facts
            .entry(subject.to_string())
            .or_default()
            .entry(property.to_string())
            .or_default() += confidence;
    }

    pub fn fact(&self, subject: &str) -> Option<&str> {
        self.facts
            .get(subject)?
            .iter()
            .max_by(|left, right| {
                left.1
                    .partial_cmp(right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(property, _)| property.as_str())
    }

    pub fn count(&self) -> usize {
        self.facts.values().map(BTreeMap::len).sum()
    }
}

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn index(&mut self, upper_bound: usize) -> usize {
        if upper_bound == 0 {
            0
        } else {
            (self.next_u64() as usize) % upper_bound
        }
    }

    pub fn unit(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }
}

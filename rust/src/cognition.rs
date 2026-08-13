//! Lightweight agent affect, needs, decisions, and social learning.
//!
//! This is the Rust home for the stable behaviour behind Python's emotion,
//! needs, decision, social, and teaching mixins.  It intentionally keeps
//! each state vector compact and typed so millions of agent updates remain
//! cheap.

use std::collections::BTreeMap;

use crate::model::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Idle,
    Learn,
    Math,
    Language,
    Social,
    Teach,
    Explore,
    Reorganise,
}

#[derive(Clone, Debug)]
pub struct Traits {
    pub mutation_rate: f64,
    pub cooperation_weight: f64,
    pub novelty_weight: f64,
    pub stability_weight: f64,
    pub trust_threshold: f64,
    pub chattiness: f64,
    pub patience: f64,
    pub teaching_drive: f64,
    pub curiosity: f64,
    pub expressiveness: f64,
}

impl Traits {
    pub fn random(rng: &mut Rng) -> Self {
        Self {
            mutation_rate: between(rng, 0.05, 0.4),
            cooperation_weight: rng.unit(),
            novelty_weight: rng.unit(),
            stability_weight: rng.unit(),
            trust_threshold: rng.unit(),
            chattiness: between(rng, 0.2, 0.8),
            patience: between(rng, 0.2, 0.8),
            teaching_drive: between(rng, 0.2, 0.8),
            curiosity: between(rng, 0.2, 0.8),
            expressiveness: between(rng, 0.2, 0.8),
        }
    }

    pub fn compatibility_distance(&self, other: &Self) -> f64 {
        [
            self.cooperation_weight - other.cooperation_weight,
            self.novelty_weight - other.novelty_weight,
            self.stability_weight - other.stability_weight,
            self.mutation_rate - other.mutation_rate,
            self.trust_threshold - other.trust_threshold,
        ]
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
    }

    pub fn child_from(parents: [&Self; 3], rng: &mut Rng) -> Self {
        let pick = |rng: &mut Rng, get: fn(&Traits) -> f64| {
            let inherited = get(parents[rng.index(parents.len())]);
            clamp01(inherited + (0.5 - inherited) * 0.08 + signed(rng, 0.01))
        };
        Self {
            mutation_rate: pick(rng, |trait_| trait_.mutation_rate),
            cooperation_weight: pick(rng, |trait_| trait_.cooperation_weight),
            novelty_weight: pick(rng, |trait_| trait_.novelty_weight),
            stability_weight: pick(rng, |trait_| trait_.stability_weight),
            trust_threshold: pick(rng, |trait_| trait_.trust_threshold),
            chattiness: pick(rng, |trait_| trait_.chattiness),
            patience: pick(rng, |trait_| trait_.patience),
            teaching_drive: pick(rng, |trait_| trait_.teaching_drive),
            curiosity: pick(rng, |trait_| trait_.curiosity),
            expressiveness: pick(rng, |trait_| trait_.expressiveness),
        }
    }
}

#[derive(Clone, Debug)]
pub struct State {
    pub happiness: f64,
    pub loneliness: f64,
    pub confidence: f64,
    pub frustration: f64,
    pub curiosity: f64,
    pub satisfaction: f64,
    pub purpose: f64,
    pub expressiveness: f64,
    pub intrinsic_discomfort: f64,
}

impl Default for State {
    fn default() -> Self {
        Self {
            happiness: 0.5,
            loneliness: 0.5,
            confidence: 0.5,
            frustration: 0.5,
            curiosity: 0.5,
            satisfaction: 0.5,
            purpose: 0.5,
            expressiveness: 0.5,
            intrinsic_discomfort: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Needs {
    pub energy: f64,
    pub social: f64,
    pub curiosity: f64,
    pub esteem: f64,
    pub play: f64,
}

impl Default for Needs {
    fn default() -> Self {
        Self {
            energy: 1.0,
            social: 0.8,
            curiosity: 0.7,
            esteem: 0.5,
            play: 0.6,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrustProfile {
    pub affinity: f64,
    pub reliability: f64,
    pub generosity: f64,
    pub competence: f64,
    pub consistency: f64,
    pub collaboration: f64,
    pub safety: f64,
}

impl TrustProfile {
    pub fn update(&mut self, reward: f64) {
        let reward = reward.clamp(-1.0, 1.0);
        self.affinity = clamp_trust(self.affinity + 0.08 * reward);
        self.reliability = clamp_trust(self.reliability + 0.096 * reward);
        self.generosity = clamp_trust(self.generosity + 0.048 * reward);
        self.competence = clamp_trust(self.competence + 0.112 * reward);
        self.consistency = clamp_trust(self.consistency + 0.040 * reward);
        self.collaboration = clamp_trust(self.collaboration + 0.08 * reward);
        self.safety = clamp_trust(self.safety + 0.056 * reward);
    }

    pub fn teaching_willingness(&self) -> f64 {
        0.40 * self.competence
            + 0.30 * self.reliability
            + 0.20 * self.affinity
            + 0.10 * self.collaboration
    }
}

#[derive(Clone, Debug, Default)]
pub struct SocialRecord {
    pub interactions: u32,
    pub mean_outcome: f64,
    pub last_outcome: f64,
    pub trust_delta: f64,
    pub offspring_success: f64,
    pub last_seen_generation: u64,
}

pub type TrustMap = BTreeMap<usize, TrustProfile>;
pub type SocialMemory = BTreeMap<usize, SocialRecord>;

pub fn decide_action(state: &State, traits: &Traits, energy: f64, rng: &mut Rng) -> Action {
    let actions = [
        Action::Idle,
        Action::Learn,
        Action::Math,
        Action::Language,
        Action::Social,
        Action::Teach,
        Action::Explore,
        Action::Reorganise,
    ];
    let discomfort = state.intrinsic_discomfort.clamp(0.0, 1.0);
    let mut weights = [
        0.1,
        0.6 * state.curiosity + 0.4 * traits.curiosity + 0.70 * discomfort,
        0.7 * state.curiosity + 0.3 * state.confidence,
        0.6 * traits.expressiveness + 0.4 * traits.curiosity + 0.25 * discomfort,
        0.8 * state.loneliness + 0.4 * traits.cooperation_weight,
        0.6 * traits.teaching_drive + 0.2 * state.satisfaction + 0.2 * state.purpose,
        0.3 * state.curiosity
            + 0.3 * (1.0 - state.satisfaction)
            + 0.2 * state.frustration
            + 0.2 * state.purpose
            + 0.75 * discomfort,
        0.5 * traits.curiosity + 0.5 * (1.0 - state.satisfaction) + 0.55 * discomfort,
    ];
    if energy < 20.0 {
        weights[0] += 2.0;
        weights[1] *= 0.4;
        weights[2] *= 0.2;
        weights[3] *= 0.5;
        weights[5] *= 0.3;
    }
    if state.frustration > 0.7 {
        weights[4] += 0.5;
        weights[7] += 0.4;
    }
    if state.happiness > 0.7 && state.purpose > 0.6 {
        weights[5] += 0.5;
    }
    actions[weighted_index(&weights, rng)]
}

pub fn update_needs(needs: &mut Needs, state: &State, energy: f64) {
    let target_energy = 1.0 - (energy / 100.0).clamp(0.0, 1.0);
    let targets = [
        target_energy,
        state.loneliness,
        state.curiosity,
        (0.5 * state.confidence + 0.5 * state.happiness).clamp(0.0, 1.0),
        (state.happiness - 0.5 * state.frustration).clamp(0.0, 1.0),
    ];
    for (current, target) in [
        &mut needs.energy,
        &mut needs.social,
        &mut needs.curiosity,
        &mut needs.esteem,
        &mut needs.play,
    ]
    .into_iter()
    .zip(targets)
    {
        *current = (0.5 * *current + 0.5 * target).clamp(0.0, 1.0);
    }
}

pub fn decay_state(state: &mut State) {
    for value in [
        &mut state.happiness,
        &mut state.loneliness,
        &mut state.confidence,
        &mut state.curiosity,
        &mut state.satisfaction,
        &mut state.purpose,
        &mut state.expressiveness,
    ] {
        *value = (*value + 0.01 * (0.5 - *value)).clamp(0.0, 1.0);
    }
    state.frustration = (state.frustration + 0.02 * (0.5 - state.frustration)).clamp(0.0, 1.0);
}

pub fn state_event(state: &mut State, event: &str) {
    let (happiness, loneliness, confidence, frustration, curiosity, satisfaction, purpose) =
        match event {
            "successful_comm" => (0.05, -0.05, 0.04, -0.03, 0.0, 0.0, 0.0),
            "failed_comm" => (-0.03, 0.03, -0.04, 0.07, 0.0, 0.0, 0.0),
            "teaching_success" => (0.06, -0.03, 0.08, -0.02, 0.0, 0.0, 0.04),
            "teaching_failure" => (-0.04, 0.0, -0.06, 0.06, 0.0, 0.0, 0.0),
            "learning_success" => (0.03, 0.0, 0.04, -0.03, 0.06, 0.0, 0.03),
            "learning_failure" => (0.0, 0.0, -0.03, 0.05, 0.01, 0.0, 0.0),
            "concept_cleanup" => (0.0, 0.0, 0.0, -0.12, 0.0, 0.05, 0.04),
            "curiosity_boost" => (0.0, 0.0, 0.0, -0.02, 0.03, 0.0, 0.0),
            _ => return,
        };
    state.happiness = clamp01(state.happiness + happiness);
    state.loneliness = clamp01(state.loneliness + loneliness);
    state.confidence = clamp01(state.confidence + confidence);
    state.frustration = clamp01(state.frustration + frustration);
    state.curiosity = clamp01(state.curiosity + curiosity);
    state.satisfaction = clamp01(state.satisfaction + satisfaction);
    state.purpose = clamp01(state.purpose + purpose);
}

pub fn remember_interaction(
    memory: &mut SocialMemory,
    partner: usize,
    outcome: f64,
    generation: u64,
) {
    let record = memory.entry(partner).or_default();
    record.interactions += 1;
    record.last_outcome = outcome;
    record.mean_outcome = 0.7 * record.mean_outcome + 0.3 * outcome;
    record.trust_delta = (record.trust_delta + 0.5 * outcome).clamp(-1000.0, 1000.0);
    record.last_seen_generation = generation;
}

pub fn decay_social_memory(memory: &mut SocialMemory, rate: f64, generation: u64) {
    memory.retain(|_, record| {
        record.mean_outcome *= 1.0 - 0.5 * rate;
        record.trust_delta *= 1.0 - rate;
        let tiny =
            record.mean_outcome.abs() + record.trust_delta.abs() + record.offspring_success.abs()
                < 0.02;
        !(generation.saturating_sub(record.last_seen_generation) > 50 && tiny)
    });
}

fn weighted_index(weights: &[f64], rng: &mut Rng) -> usize {
    let total: f64 = weights.iter().map(|weight| weight.max(0.0)).sum();
    if total <= f64::EPSILON {
        return 0;
    }
    let mut threshold = rng.unit() * total;
    for (index, weight) in weights.iter().enumerate() {
        threshold -= weight.max(0.0);
        if threshold <= 0.0 {
            return index;
        }
    }
    weights.len() - 1
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}
fn clamp_trust(value: f64) -> f64 {
    value.clamp(-2.0, 2.0)
}
fn between(rng: &mut Rng, low: f64, high: f64) -> f64 {
    low + rng.unit() * (high - low)
}
fn signed(rng: &mut Rng, scale: f64) -> f64 {
    (rng.unit() * 2.0 - 1.0) * scale
}

#[cfg(test)]
mod tests {
    use super::{Action, State, Traits, decide_action, update_needs};
    use crate::model::Rng;

    #[test]
    fn low_energy_strongly_biases_idle_decisions() {
        let mut rng = Rng::new(2);
        let traits = Traits::random(&mut rng);
        let state = State::default();
        let idle = (0..100)
            .filter(|_| decide_action(&state, &traits, 1.0, &mut rng) == Action::Idle)
            .count();
        assert!(idle > 20);
    }

    #[test]
    fn needs_follow_energy_and_emotional_state_smoothly() {
        let mut needs = super::Needs::default();
        let state = State {
            loneliness: 1.0,
            curiosity: 1.0,
            ..State::default()
        };
        update_needs(&mut needs, &state, 0.0);
        assert!(needs.energy > 0.9 && needs.social > 0.8 && needs.curiosity > 0.8);
    }
}

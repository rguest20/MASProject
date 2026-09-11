//! Grounded tasks that create cultural pressure for shared conventions.
//!
//! Python schedules a broad family of tasks.  This first Rust task engine
//! ports the grounding core on which the higher-order task families depend:
//! numeric reconciliation, referential signals, action signals, and
//! compositional action grammar.  Trials are deliberately revisited after a
//! correction; without recurrence, a learner cannot demonstrate that it has
//! actually adopted a peer's signal.

use std::collections::VecDeque;

use crate::lexicon::CommunityLexicon;
use crate::model::{Agent, Rng, two_agents};

const REFERENTS: [&str; 6] = ["r0", "r1", "r2", "r3", "r4", "r5"];
const ACTIONS: [&str; 4] = ["a0", "a1", "a2", "a3"];

#[derive(Clone, Debug, Default)]
pub struct TaskMetrics {
    pub generated: usize,
    pub attempted: usize,
    pub solved: usize,
    pub failed: usize,
    pub numeric_successes: usize,
    pub referential_successes: usize,
    pub action_successes: usize,
    pub grammar_successes: usize,
    pub comparison_successes: usize,
    pub quantity_successes: usize,
    pub concept_attempts: usize,
    pub concept_successes: usize,
    pub concept_repairs: usize,
    pub definition_consensus: usize,
    pub narrative_successes: usize,
    pub prediction_successes: usize,
    pub role_successes: usize,
    pub reconstruction_successes: usize,
    pub property_successes: usize,
    pub compatibility_successes: usize,
    pub misunderstanding_repairs: usize,
}

#[derive(Clone, Debug)]
struct NumericTrial {
    speaker: usize,
    listener: usize,
    value: u32,
}

#[derive(Clone, Debug)]
struct SignalTrial {
    speaker: usize,
    listener: usize,
    meaning: String,
}

#[derive(Clone, Debug)]
struct ConceptTrial {
    anchor: String,
    support: String,
    distractor: String,
}

#[derive(Clone, Copy, Debug)]
enum StructuredPractice {
    Narrative,
    Prediction,
    Role,
    Reconstruction,
    Property,
    Compatibility,
    Misunderstanding,
}

impl StructuredPractice {
    fn from_generation(generation: usize) -> Self {
        match (generation / 4) % 7 {
            0 => Self::Narrative,
            1 => Self::Prediction,
            2 => Self::Role,
            3 => Self::Reconstruction,
            4 => Self::Property,
            5 => Self::Compatibility,
            _ => Self::Misunderstanding,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TaskEngine {
    next_id: u64,
    generation: usize,
    numeric_memory: VecDeque<NumericTrial>,
    referential_memory: VecDeque<SignalTrial>,
    action_memory: VecDeque<SignalTrial>,
    concept_memory: VecDeque<ConceptTrial>,
}

impl TaskEngine {
    pub fn run_generation(
        &mut self,
        agents: &mut [Agent],
        lexicon: &mut CommunityLexicon,
        rng: &mut Rng,
    ) -> TaskMetrics {
        let mut metrics = TaskMetrics::default();
        if agents.len() < 2 {
            return metrics;
        }
        self.generation += 1;
        // This mirrors Python's focused grounding curriculum.  Randomly
        // mixing every task from the beginning made it possible to ask for
        // grammar before the agents had public words worth composing.
        let numeric_ready = lexicon.numeric_conventions().len() >= 3;
        let language_ready = numeric_ready && lexicon.referential_conventions().len() >= 3;
        let two_slot_ready = lexicon.grammar_order("referent_quantity").is_some();
        let action_ready = lexicon.action_conventions().len() >= 2;
        for slot in 0..4 {
            metrics.generated += 1;
            self.next_id += 1;
            match slot {
                0 => self.numeric_reconciliation(agents, lexicon, rng, &mut metrics),
                1 if numeric_ready && !self.generation.is_multiple_of(6) => {
                    self.translate_quantity(agents, lexicon, rng, &mut metrics)
                }
                1 => self.compare_numbers(agents, lexicon, rng, &mut metrics),
                2 => self.referential_signal(agents, lexicon, rng, &mut metrics),
                _ if action_ready => self.compositional_action(agents, lexicon, rng, &mut metrics),
                _ if two_slot_ready => self.action_signal(agents, lexicon, rng, &mut metrics),
                _ if language_ready => {
                    self.compositional_signal(agents, lexicon, rng, &mut metrics)
                }
                _ => self.referential_signal(agents, lexicon, rng, &mut metrics),
            }
        }
        // Higher-order tasks supplement the grounded core rather than
        // replacing it.  They start only after repeated semantic experience,
        // then revisit a small memory of disagreements so a repair can be
        // demonstrated later instead of being a one-shot correction.
        if self.generation.is_multiple_of(3) {
            if self.semantic_similarity_debate(agents, rng, &mut metrics) {
                metrics.generated += 1;
            }
            if self.generation.is_multiple_of(6)
                && self.definition_exchange(agents, rng, &mut metrics)
            {
                metrics.generated += 1;
            }
        }
        if self.generation.is_multiple_of(4)
            && self.structured_concept_practice(agents, rng, &mut metrics)
        {
            metrics.generated += 1;
        }
        metrics
    }

    fn structured_concept_practice(
        &mut self,
        agents: &mut [Agent],
        rng: &mut Rng,
        metrics: &mut TaskMetrics,
    ) -> bool {
        let Some(trial) = self.concept_trial(agents, rng) else {
            return false;
        };
        let kind = StructuredPractice::from_generation(self.generation);
        let participants = sample_indices(agents.len(), agents.len().min(5), rng);
        if participants.len() < 3 {
            return false;
        }
        // The ordered prompt is the task's structure.  It does not include
        // an oracle relation: members decide whether the final candidate is
        // locally compatible with the anchor and the shared context.
        let prompt = practice_prompt(kind, &trial);
        let mut responses = Vec::new();
        for index in participants {
            let agent = &mut agents[index];
            agent.observe(&prompt, 0.055, rng);
            let support_score = practice_score(agent, kind, &trial, true);
            let distractor_score = practice_score(agent, kind, &trial, false);
            responses.push((index, support_score >= distractor_score));
        }
        let support_votes = responses.iter().filter(|(_, choice)| *choice).count();
        let consensus = support_votes * 2 > responses.len();
        metrics.attempted += responses.len();
        metrics.concept_attempts += responses.len();
        if consensus {
            for (index, preferred) in responses {
                let agent = &mut agents[index];
                if preferred {
                    agent.fitness += 0.075;
                    agent.tasks_solved += 1;
                    metrics.solved += 1;
                    record_structured_success(metrics, kind);
                } else {
                    // In every task family a losing proposal is repaired by
                    // contextual co-occurrence with the consensus candidate,
                    // not overwritten by an externally named definition.
                    agent
                        .semantics
                        .link(&trial.anchor, &trial.support, 0.10, rng);
                    agent
                        .semantics
                        .link(&trial.support, &trial.distractor, 0.035, rng);
                    metrics.concept_repairs += 1;
                    if matches!(kind, StructuredPractice::Misunderstanding) {
                        metrics.misunderstanding_repairs += 1;
                    }
                }
            }
        } else {
            for (index, _) in responses {
                let agent = &mut agents[index];
                agent.tasks_failed += 1;
                agent.frustration = (agent.frustration + 0.012).min(1.0);
            }
            metrics.failed += 1;
        }
        remember(&mut self.concept_memory, trial, 120);
        true
    }

    fn semantic_similarity_debate(
        &mut self,
        agents: &mut [Agent],
        rng: &mut Rng,
        metrics: &mut TaskMetrics,
    ) -> bool {
        let Some(trial) = self.concept_trial(agents, rng) else {
            return false;
        };
        let participants = sample_indices(agents.len(), agents.len().min(6), rng);
        if participants.len() < 3 {
            return false;
        }
        let context = vec![
            trial.anchor.clone(),
            trial.support.clone(),
            trial.distractor.clone(),
        ];
        let mut responses = Vec::new();
        for index in participants {
            let agent = &mut agents[index];
            // A shared prompt makes the terms available, but it does not say
            // which relation is preferred. Each agent still evaluates with
            // its own learned geometry.
            agent.observe(&context, 0.045, rng);
            let support = agent
                .semantics
                .cosine_similarity(&trial.anchor, &trial.support)
                .unwrap_or(-1.0);
            let distractor = agent
                .semantics
                .cosine_similarity(&trial.anchor, &trial.distractor)
                .unwrap_or(-1.0);
            responses.push((index, support >= distractor));
        }
        let support_votes = responses.iter().filter(|(_, choice)| *choice).count();
        let agreed = support_votes * 2 > responses.len();
        metrics.attempted += responses.len();
        metrics.concept_attempts += responses.len();
        if agreed {
            for (index, choice) in responses {
                let agent = &mut agents[index];
                if choice {
                    agent.fitness += 0.11;
                    agent.tasks_solved += 1;
                    metrics.solved += 1;
                    metrics.concept_successes += 1;
                } else {
                    // The majority relation is supplied as corrective
                    // evidence, not a large penalty. This preserves agent
                    // diversity while giving disagreement a learnable path.
                    agent
                        .semantics
                        .link(&trial.anchor, &trial.support, 0.12, rng);
                    agent.fitness += 0.02;
                    metrics.concept_repairs += 1;
                }
            }
        } else {
            for (index, _) in responses {
                let agent = &mut agents[index];
                agent.tasks_failed += 1;
                agent.frustration = (agent.frustration + 0.015).min(1.0);
            }
            metrics.failed += 1;
        }
        remember(&mut self.concept_memory, trial, 120);
        true
    }

    fn definition_exchange(
        &mut self,
        agents: &mut [Agent],
        rng: &mut Rng,
        metrics: &mut TaskMetrics,
    ) -> bool {
        let Some(trial) = self.concept_trial(agents, rng) else {
            return false;
        };
        let participants = sample_indices(agents.len(), agents.len().min(5), rng);
        if participants.len() < 3 {
            return false;
        }
        let mut responses = Vec::new();
        for index in participants {
            let agent = &mut agents[index];
            // A definition is an explicitly inspectable local proposal: the
            // anchor plus the candidate the agent considers most compatible.
            // It can be exchanged and corrected without inventing an English
            // gloss for an emergent token.
            let preferred = agent
                .semantics
                .cosine_similarity(&trial.anchor, &trial.support)
                .unwrap_or(-1.0)
                >= agent
                    .semantics
                    .cosine_similarity(&trial.anchor, &trial.distractor)
                    .unwrap_or(-1.0);
            responses.push((index, preferred));
        }
        let support_votes = responses.iter().filter(|(_, choice)| *choice).count();
        let consensus = support_votes * 2 > responses.len();
        metrics.attempted += responses.len();
        metrics.concept_attempts += responses.len();
        if consensus {
            metrics.definition_consensus += 1;
            for (index, preferred) in responses {
                let agent = &mut agents[index];
                let definition = if preferred {
                    vec![trial.anchor.clone(), trial.support.clone()]
                } else {
                    vec![trial.anchor.clone(), trial.distractor.clone()]
                };
                agent.observe(&definition, 0.09, rng);
                if preferred {
                    agent.fitness += 0.09;
                    agent.tasks_solved += 1;
                    metrics.solved += 1;
                    metrics.concept_successes += 1;
                } else {
                    agent
                        .semantics
                        .link(&trial.anchor, &trial.support, 0.14, rng);
                    metrics.concept_repairs += 1;
                }
            }
        } else {
            for (index, _) in responses {
                agents[index].tasks_failed += 1;
            }
            metrics.failed += 1;
        }
        remember(&mut self.concept_memory, trial, 120);
        true
    }

    fn concept_trial(&mut self, agents: &[Agent], rng: &mut Rng) -> Option<ConceptTrial> {
        if !self.concept_memory.is_empty() && rng.unit() < 0.55 {
            return Some(self.concept_memory[rng.index(self.concept_memory.len())].clone());
        }
        let source = &agents[rng.index(agents.len())];
        let candidates = source.semantics.reasoning_candidates(3);
        if candidates.len() < 3 {
            return None;
        }
        let anchor = candidates[rng.index(candidates.len())].clone();
        let support = source
            .semantics
            .nearest(&anchor, candidates.len())
            .into_iter()
            .find(|candidate| candidates.contains(candidate))?;
        let distractors: Vec<_> = candidates
            .into_iter()
            .filter(|candidate| candidate != &anchor && candidate != &support)
            .collect();
        let distractor = distractors.get(rng.index(distractors.len()))?.clone();
        Some(ConceptTrial {
            anchor,
            support,
            distractor,
        })
    }

    fn compare_numbers(
        &mut self,
        agents: &mut [Agent],
        lexicon: &mut CommunityLexicon,
        rng: &mut Rng,
        metrics: &mut TaskMetrics,
    ) {
        let (speaker_id, listener_id) = rng.pair(agents.len());
        let (speaker, listener) = two_agents(agents, speaker_id, listener_id);
        let base = speaker.numeric.base;
        let left_value = rng.index((base * base).max(2) as usize) as u32;
        let right_value = rng.index((base * base).max(2) as usize) as u32;
        let left = speaker.numeric.speak_number(left_value, lexicon, rng);
        let right = speaker.numeric.speak_number(right_value, lexicon, rng);
        let understood_left = listener.numeric.decode_phrase(&left, lexicon);
        let understood_right = listener.numeric.decode_phrase(&right, lexicon);
        metrics.attempted += 1;
        if understood_left == Some(left_value) && understood_right == Some(right_value) {
            // The exact relation is intentionally local task evidence.  It
            // rewards interoperable number systems without creating a public
            // English relation word.
            reward(speaker, listener);
            speaker.fitness += 0.05;
            listener.fitness += 0.05;
            metrics.solved += 1;
            metrics.comparison_successes += 1;
        } else {
            failure(speaker, listener);
            metrics.failed += 1;
        }
    }

    fn translate_quantity(
        &mut self,
        agents: &mut [Agent],
        lexicon: &mut CommunityLexicon,
        rng: &mut Rng,
        metrics: &mut TaskMetrics,
    ) {
        let (speaker_id, listener_id) = rng.pair(agents.len());
        let (speaker, listener) = two_agents(agents, speaker_id, listener_id);
        let base = lexicon.community_base().unwrap_or(speaker.numeric.base);
        let value = rng.index((base * base + base * 4).max(2) as usize) as u32;
        let phrase = speaker.numeric.speak_number(value, lexicon, rng);
        metrics.attempted += 1;
        if listener.numeric.decode_phrase(&phrase, lexicon) == Some(value) {
            lexicon.observe_base_success(base);
            reward(speaker, listener);
            speaker.fitness += 0.10;
            listener.fitness += 0.10;
            metrics.solved += 1;
            metrics.quantity_successes += 1;
        } else {
            failure(speaker, listener);
            metrics.failed += 1;
        }
    }

    fn compositional_signal(
        &mut self,
        agents: &mut [Agent],
        lexicon: &mut CommunityLexicon,
        rng: &mut Rng,
        metrics: &mut TaskMetrics,
    ) {
        let referent = REFERENTS[rng.index(REFERENTS.len())];
        let value = rng.index(4) as u32;
        let (Some(referent_signal), Some(number_signal)) = (
            lexicon.referential_signal(referent).map(str::to_owned),
            lexicon.numeric_token(value).map(str::to_owned),
        ) else {
            metrics.failed += 1;
            return;
        };
        let (speaker_id, listener_id) = rng.pair(agents.len());
        let (speaker, listener) = two_agents(agents, speaker_id, listener_id);
        let phrase = format!("{referent_signal} {number_signal}");
        speaker.observe(std::slice::from_ref(&phrase), 0.12, rng);
        listener.observe(&[phrase], 0.12, rng);
        let order = ["referent".to_string(), "number".to_string()];
        lexicon.observe_grammar_success("referent_quantity", &order);
        reward(speaker, listener);
        metrics.attempted += 1;
        metrics.solved += 1;
        metrics.grammar_successes += 1;
    }

    fn numeric_reconciliation(
        &mut self,
        agents: &mut [Agent],
        lexicon: &mut CommunityLexicon,
        rng: &mut Rng,
        metrics: &mut TaskMetrics,
    ) {
        let trial = if !self.numeric_memory.is_empty() && rng.unit() < 0.65 {
            self.numeric_memory[rng.index(self.numeric_memory.len())].clone()
        } else {
            let (speaker, listener) = rng.pair(agents.len());
            let base = agents[speaker]
                .numeric
                .base
                .min(agents[listener].numeric.base);
            NumericTrial {
                speaker,
                listener,
                value: rng.index(base.max(1) as usize) as u32,
            }
        };
        let (speaker, listener) = two_agents(agents, trial.speaker, trial.listener);
        let signal = speaker.numeric.speak_number(trial.value, lexicon, rng);
        let understood = listener.numeric.decode_phrase(&signal, lexicon) == Some(trial.value);
        speaker.observe(std::slice::from_ref(&signal), 0.10, rng);
        listener.observe(std::slice::from_ref(&signal), 0.10, rng);
        metrics.attempted += 1;
        if understood {
            lexicon.observe_numeric_success(trial.value, &signal);
            lexicon.observe_base_success(speaker.numeric.base);
            reward(speaker, listener);
            metrics.solved += 1;
            metrics.numeric_successes += 1;
        } else {
            listener
                .numeric
                .learn_digit_mapping(&signal, trial.value, rng);
            failure(speaker, listener);
            metrics.failed += 1;
        }
        remember(&mut self.numeric_memory, trial, 240);
    }

    fn referential_signal(
        &mut self,
        agents: &mut [Agent],
        lexicon: &mut CommunityLexicon,
        rng: &mut Rng,
        metrics: &mut TaskMetrics,
    ) {
        let trial = if !self.referential_memory.is_empty() && rng.unit() < 0.65 {
            self.referential_memory[rng.index(self.referential_memory.len())].clone()
        } else {
            let (speaker, listener) = rng.pair(agents.len());
            SignalTrial {
                speaker,
                listener,
                meaning: REFERENTS[rng.index(REFERENTS.len())].to_string(),
            }
        };
        let (speaker, listener) = two_agents(agents, trial.speaker, trial.listener);
        let signal = speaker.referent_signal(&trial.meaning);
        let understood = listener.referent_for(&signal) == Some(trial.meaning.as_str());
        speaker.observe(std::slice::from_ref(&signal), 0.10, rng);
        listener.observe(std::slice::from_ref(&signal), 0.10, rng);
        metrics.attempted += 1;
        if understood {
            lexicon.observe_referential_success(&trial.meaning, &signal);
            reward(speaker, listener);
            metrics.solved += 1;
            metrics.referential_successes += 1;
        } else {
            listener.learn_referent(&trial.meaning, &signal);
            failure(speaker, listener);
            metrics.failed += 1;
        }
        remember(&mut self.referential_memory, trial, 240);
    }

    fn action_signal(
        &mut self,
        agents: &mut [Agent],
        lexicon: &mut CommunityLexicon,
        rng: &mut Rng,
        metrics: &mut TaskMetrics,
    ) {
        let trial = if !self.action_memory.is_empty() && rng.unit() < 0.65 {
            self.action_memory[rng.index(self.action_memory.len())].clone()
        } else {
            let (speaker, listener) = rng.pair(agents.len());
            SignalTrial {
                speaker,
                listener,
                meaning: ACTIONS[rng.index(ACTIONS.len())].to_string(),
            }
        };
        let (speaker, listener) = two_agents(agents, trial.speaker, trial.listener);
        let signal = speaker.action_signal(&trial.meaning);
        let understood = listener.action_for(&signal) == Some(trial.meaning.as_str());
        speaker.observe(std::slice::from_ref(&signal), 0.10, rng);
        listener.observe(std::slice::from_ref(&signal), 0.10, rng);
        metrics.attempted += 1;
        if understood {
            lexicon.observe_action_success(&trial.meaning, &signal);
            reward(speaker, listener);
            metrics.solved += 1;
            metrics.action_successes += 1;
        } else {
            listener.learn_action(&trial.meaning, &signal);
            failure(speaker, listener);
            metrics.failed += 1;
        }
        remember(&mut self.action_memory, trial, 240);
    }

    fn compositional_action(
        &mut self,
        agents: &mut [Agent],
        lexicon: &mut CommunityLexicon,
        rng: &mut Rng,
        metrics: &mut TaskMetrics,
    ) {
        let action = ACTIONS[rng.index(ACTIONS.len())];
        let referent = REFERENTS[rng.index(REFERENTS.len())];
        let value = rng.index(4) as u32;
        let Some(action_signal) = lexicon.action_signal(action).map(str::to_owned) else {
            metrics.failed += 1;
            return;
        };
        let Some(referent_signal) = lexicon.referential_signal(referent).map(str::to_owned) else {
            metrics.failed += 1;
            return;
        };
        let Some(number_signal) = lexicon.numeric_token(value).map(str::to_owned) else {
            metrics.failed += 1;
            return;
        };
        let order = [
            "referent".to_string(),
            "action".to_string(),
            "number".to_string(),
        ];
        let signal = format!("{referent_signal} {action_signal} {number_signal}");
        let (speaker_id, listener_id) = rng.pair(agents.len());
        let (speaker, listener) = two_agents(agents, speaker_id, listener_id);
        speaker.observe(std::slice::from_ref(&signal), 0.12, rng);
        listener.observe(&[signal], 0.12, rng);
        lexicon.observe_grammar_success("referent_action_number", &order);
        reward(speaker, listener);
        metrics.attempted += 1;
        metrics.solved += 1;
        metrics.grammar_successes += 1;
    }
}

fn practice_prompt(kind: StructuredPractice, trial: &ConceptTrial) -> Vec<String> {
    match kind {
        // A short chain is an ordering pressure: the middle term must remain
        // compatible with both its predecessor and proposed continuation.
        StructuredPractice::Narrative => vec![
            trial.anchor.clone(),
            trial.support.clone(),
            trial.distractor.clone(),
        ],
        // Prediction exposes a prefix and alternative continuation.
        StructuredPractice::Prediction => vec![trial.anchor.clone(), trial.support.clone()],
        // Proto role assignment keeps an event-like anchor with a candidate
        // participant and alternative receiver.
        StructuredPractice::Role => vec![
            trial.support.clone(),
            trial.anchor.clone(),
            trial.distractor.clone(),
        ],
        // Reconstruction asks which candidate best bridges an observed pair.
        StructuredPractice::Reconstruction => vec![
            trial.anchor.clone(),
            trial.distractor.clone(),
            trial.support.clone(),
        ],
        StructuredPractice::Property => vec![trial.anchor.clone(), trial.support.clone()],
        StructuredPractice::Compatibility => vec![trial.support.clone(), trial.anchor.clone()],
        // The listener is asked to preserve a quoted anchor/support relation
        // rather than merely echo an unfamiliar final token.
        StructuredPractice::Misunderstanding => vec![
            trial.anchor.clone(),
            trial.support.clone(),
            trial.distractor.clone(),
        ],
    }
}

fn practice_score(
    agent: &Agent,
    kind: StructuredPractice,
    trial: &ConceptTrial,
    support: bool,
) -> f32 {
    let candidate = if support {
        &trial.support
    } else {
        &trial.distractor
    };
    let anchor_affinity = agent
        .semantics
        .cosine_similarity(&trial.anchor, candidate)
        .unwrap_or(-1.0);
    let support_affinity = agent
        .semantics
        .cosine_similarity(&trial.support, candidate)
        .unwrap_or(-1.0);
    match kind {
        StructuredPractice::Narrative => 0.65 * anchor_affinity + 0.35 * support_affinity,
        StructuredPractice::Prediction => anchor_affinity,
        StructuredPractice::Role => 0.45 * anchor_affinity + 0.55 * support_affinity,
        StructuredPractice::Reconstruction => 0.35 * anchor_affinity + 0.65 * support_affinity,
        StructuredPractice::Property => 0.80 * anchor_affinity + 0.20 * support_affinity,
        StructuredPractice::Compatibility => 0.25 * anchor_affinity + 0.75 * support_affinity,
        StructuredPractice::Misunderstanding => 0.70 * anchor_affinity + 0.30 * support_affinity,
    }
}

fn record_structured_success(metrics: &mut TaskMetrics, kind: StructuredPractice) {
    metrics.concept_successes += 1;
    match kind {
        StructuredPractice::Narrative => metrics.narrative_successes += 1,
        StructuredPractice::Prediction => metrics.prediction_successes += 1,
        StructuredPractice::Role => metrics.role_successes += 1,
        StructuredPractice::Reconstruction => metrics.reconstruction_successes += 1,
        StructuredPractice::Property => metrics.property_successes += 1,
        StructuredPractice::Compatibility => metrics.compatibility_successes += 1,
        StructuredPractice::Misunderstanding => metrics.misunderstanding_repairs += 1,
    }
}

fn sample_indices(size: usize, count: usize, rng: &mut Rng) -> Vec<usize> {
    let mut pool: Vec<_> = (0..size).collect();
    let count = count.min(pool.len());
    for index in 0..count {
        let replacement = index + rng.index(pool.len() - index);
        pool.swap(index, replacement);
    }
    pool.truncate(count);
    pool
}

fn reward(speaker: &mut Agent, listener: &mut Agent) {
    speaker.fitness += 0.20;
    listener.fitness += 0.20;
    speaker.energy = (speaker.energy + 0.15).min(100.0);
    listener.energy = (listener.energy + 0.15).min(100.0);
    speaker.tasks_solved += 1;
    listener.tasks_solved += 1;
}

fn failure(speaker: &mut Agent, listener: &mut Agent) {
    speaker.frustration = (speaker.frustration + 0.03).min(1.0);
    listener.frustration = (listener.frustration + 0.05).min(1.0);
    listener.tasks_failed += 1;
}

fn remember<T>(memory: &mut VecDeque<T>, item: T, capacity: usize) {
    memory.push_back(item);
    while memory.len() > capacity {
        memory.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::TaskEngine;
    use crate::lexicon::CommunityLexicon;
    use crate::model::{Agent, Rng};

    #[test]
    fn repeated_grounding_creates_some_public_conventions() {
        let mut rng = Rng::new(42);
        let mut agents = (0..12)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        let mut lexicon = CommunityLexicon::default();
        let mut tasks = TaskEngine::default();
        let mut solved = 0;
        for _ in 0..160 {
            solved += tasks
                .run_generation(&mut agents, &mut lexicon, &mut rng)
                .solved;
        }
        assert!(solved > 0);
        assert!(
            !lexicon.numeric_conventions().is_empty()
                || !lexicon.referential_conventions().is_empty()
                || !lexicon.action_conventions().is_empty()
        );
    }

    #[test]
    fn mature_shared_tokens_unlock_rehearsable_concept_tasks() {
        let mut rng = Rng::new(43);
        let mut agents = (0..8)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        let context = [
            "river".to_string(),
            "water".to_string(),
            "flow".to_string(),
            "bank".to_string(),
        ];
        for agent in &mut agents {
            for _ in 0..5 {
                agent.observe(&context, 0.20, &mut rng);
            }
        }
        let mut lexicon = CommunityLexicon::default();
        let mut tasks = TaskEngine::default();
        let mut concept_attempts = 0;
        for _ in 0..6 {
            concept_attempts += tasks
                .run_generation(&mut agents, &mut lexicon, &mut rng)
                .concept_attempts;
        }
        assert!(concept_attempts > 0);
    }

    #[test]
    fn structured_practice_rotates_after_semantic_maturity() {
        let mut rng = Rng::new(44);
        let mut agents = (0..8)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        let context = [
            "river".to_string(),
            "water".to_string(),
            "flow".to_string(),
            "bank".to_string(),
        ];
        for agent in &mut agents {
            for _ in 0..6 {
                agent.observe(&context, 0.22, &mut rng);
            }
        }
        let mut lexicon = CommunityLexicon::default();
        let mut tasks = TaskEngine::default();
        let mut structured_successes = 0;
        for _ in 0..28 {
            let metrics = tasks.run_generation(&mut agents, &mut lexicon, &mut rng);
            structured_successes += metrics.narrative_successes
                + metrics.prediction_successes
                + metrics.role_successes
                + metrics.reconstruction_successes
                + metrics.property_successes
                + metrics.compatibility_successes
                + metrics.misunderstanding_repairs;
        }
        assert!(structured_successes > 0);
    }
}

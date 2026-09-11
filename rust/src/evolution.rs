//! Gentle population selection, tri-parent inheritance, and mutation.
//!
//! The Python coordinator replaces one low-performing agent per generation,
//! then creates a child from a socially/trait-compatible tri-parent group.
//! Keeping that gentle turnover is important: a language convention needs
//! many generations of stable carriers before it can become cultural memory.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::cognition::Traits;
use crate::language::LanguageProfile;
use crate::lexicon::CommunityLexicon;
use crate::model::{Agent, Rng};
use crate::phase3::{mutate_program, run_program};
use crate::semantics::SemanticStore;

#[derive(Clone, Debug, Default)]
pub struct EvolutionMetrics {
    pub culled_id: Option<usize>,
    pub child_id: Option<usize>,
    pub mean_fitness: f64,
    pub best_fitness: f64,
    pub child_vocabulary: usize,
    pub child_semantic_links: usize,
    pub child_semantic_families: usize,
    pub child_language_preferences: usize,
    pub child_program_len: usize,
    pub child_parent_lineages: Option<[u64; 3]>,
    pub program_mutated: bool,
    pub trait_diversity: f64,
    pub program_diversity: f64,
    pub base_diversity: usize,
    pub mean_social_degree: f64,
    pub lineage_rewards: usize,
}

pub fn evolve(
    agents: &mut Vec<Agent>,
    lexicon: &CommunityLexicon,
    generation: u64,
    rng: &mut Rng,
) -> EvolutionMetrics {
    let mut metrics = EvolutionMetrics::default();
    if agents.len() < 3 {
        return metrics;
    }
    evaluate_fitness(agents);
    metrics.lineage_rewards = reward_lineages(agents);
    metrics.mean_fitness =
        agents.iter().map(|agent| agent.total_fitness).sum::<f64>() / agents.len() as f64;
    metrics.best_fitness = agents
        .iter()
        .map(|agent| agent.total_fitness)
        .fold(0.0, f64::max);
    metrics.trait_diversity = trait_diversity(agents);
    metrics.program_diversity = program_diversity(agents);
    metrics.base_diversity = agents
        .iter()
        .map(|agent| agent.numeric.base)
        .collect::<BTreeSet<_>>()
        .len();
    metrics.mean_social_degree = agents
        .iter()
        .map(|agent| agent.social_memory.len() as f64)
        .sum::<f64>()
        / agents.len() as f64;

    // Cull from the lower-performing third, with a small exploration chance
    // for each candidate. This prevents one unlucky generation from deleting
    // the only agent carrying a useful but not-yet-rewarded experiment.
    let mut ranked: Vec<_> = agents.iter().enumerate().collect();
    ranked.sort_by(|left, right| {
        left.1
            .total_fitness
            .partial_cmp(&right.1.total_fitness)
            .unwrap_or(Ordering::Equal)
    });
    let pool = (ranked.len() / 3).max(1);
    let cull_index = ranked[rng.index(pool)].0;
    let id = agents[cull_index].id;
    agents.remove(cull_index);
    let parent_indices = select_parents(agents, rng);
    let parents = [
        &agents[parent_indices[0]],
        &agents[parent_indices[1]],
        &agents[parent_indices[2]],
    ];
    let (child, program_mutated) = make_child(id, parents, lexicon, generation, rng);
    metrics.child_vocabulary = child.vocabulary.len();
    metrics.child_semantic_links = child.semantics.link_count();
    metrics.child_semantic_families = child.semantics.family_count();
    metrics.child_language_preferences = child.language.cultural_counts().1;
    metrics.child_program_len = child.program.len();
    metrics.child_parent_lineages = child.parent_lineages;
    metrics.program_mutated = program_mutated;
    agents.push(child);
    for agent in agents {
        agent.update_motivations();
        agent.decay_social_memory(generation);
        // Task outcomes are generation-local selection evidence.  Retaining a
        // small trace rewards reliable agents without allowing runaway scores.
        agent.fitness *= 0.15;
        agent.tasks_solved = 0;
        agent.tasks_failed = 0;
        agent.frustration *= 0.85;
        agent.lineage_score *= 0.92;
    }
    metrics.culled_id = Some(id);
    metrics.child_id = Some(id);
    metrics
}

fn evaluate_fitness(agents: &mut [Agent]) {
    let mean_raw = agents.iter().map(|agent| agent.fitness).sum::<f64>() / agents.len() as f64;
    let distances: Vec<_> = agents
        .iter()
        .map(|agent| (agent.fitness - mean_raw).abs())
        .collect();
    let mut sorted_distances = distances.clone();
    sorted_distances.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let novelty_cutoff =
        sorted_distances[(sorted_distances.len() * 7 / 10).min(sorted_distances.len() - 1)];
    for agent in agents.iter_mut() {
        // Phase-3 genomes are a deliberately tiny, data-only arithmetic DSL.
        // Empty genomes remain neutral, matching the Python reference's
        // default agent; a populated genome adds bounded selection evidence.
        let program_score = (run_program(&agent.program) / 5_000.0).tanh() * 10.0;
        agent.fitness = (agent.fitness + program_score).clamp(-50.0, 50.0);
        agent.cooperation_bonus =
            (0.05 * agent.traits.cooperation_weight * (agent.fitness - mean_raw)).clamp(-5.0, 5.0);
        let novelty = (agent.fitness - mean_raw).abs().min(novelty_cutoff);
        agent.novelty_bonus = (novelty * 0.005).tanh() * agent.traits.novelty_weight * 0.5;
        let social_factor = (agent.social_memory.len() as f64 + 1.0).ln().min(1.5);
        let openness = 1.0 - agent.traits.trust_threshold;
        let trust_bonus = openness * agent.traits.cooperation_weight * social_factor * 3.0;
        let lineage_bonus = (agent.lineage_score * 0.20).clamp(-1.0, 1.0);
        agent.total_fitness = (agent.fitness
            + agent.cooperation_bonus
            + agent.novelty_bonus
            + trust_bonus
            + lineage_bonus)
            .clamp(0.0, 300.0);
    }
}

/// Feed a small, decaying outcome signal from a child back to its living
/// parents. This is intentionally weak: selection still follows each
/// agent's current conduct, while reliable cultural inheritance becomes
/// visible over several generations.
fn reward_lineages(agents: &mut [Agent]) -> usize {
    let child_scores: Vec<_> = agents
        .iter()
        .filter_map(|child| {
            child
                .parent_lineages
                .map(|parents| (parents, (child.total_fitness / 20.0).clamp(0.0, 1.0)))
        })
        .collect();
    let mut rewards = 0;
    for (parents, score) in child_scores {
        for parent_lineage in parents {
            if let Some(parent) = agents
                .iter_mut()
                .find(|agent| agent.lineage_id == parent_lineage)
            {
                parent.lineage_score = (0.85 * parent.lineage_score + 0.15 * score).clamp(0.0, 1.0);
                rewards += 1;
            }
        }
    }
    rewards
}

fn select_parents(agents: &[Agent], rng: &mut Rng) -> [usize; 3] {
    let mut ranked: Vec<_> = (0..agents.len()).collect();
    ranked.sort_by(|left, right| {
        agents[*right]
            .total_fitness
            .partial_cmp(&agents[*left].total_fitness)
            .unwrap_or(Ordering::Equal)
    });
    let elite_len = ((ranked.len() as f64 * 0.40).ceil() as usize)
        .max(3)
        .min(ranked.len());
    let elite = &ranked[..elite_len];
    let chooser = if rng.unit() < 0.60 {
        elite[rng
            .index((elite.len() as f64 * 0.20).ceil() as usize)
            .min(elite.len() - 1)]
    } else {
        elite[rng.index(elite.len())]
    };
    let second = choose_partner(agents, chooser, None, rng);
    let third = choose_partner(agents, chooser, Some(second), rng);
    [chooser, second, third]
}

fn choose_partner(
    agents: &[Agent],
    chooser: usize,
    excluded: Option<usize>,
    rng: &mut Rng,
) -> usize {
    let candidates: Vec<_> = (0..agents.len())
        .filter(|index| *index != chooser && Some(*index) != excluded)
        .collect();
    if candidates.is_empty() {
        return chooser;
    }
    let scores: Vec<_> = candidates
        .iter()
        .map(|candidate| {
            let peer = &agents[*candidate];
            let social = agents[chooser]
                .social_memory
                .get(&peer.id)
                .map(|record| record.mean_outcome + 0.05 * record.trust_delta)
                .unwrap_or(0.0);
            // Shift into positive territory so weighted sampling remains
            // exploratory instead of always choosing the top score.
            (1.0 + peer.total_fitness + 0.5 * social
                - 0.5 * agents[chooser].traits.compatibility_distance(&peer.traits))
            .max(0.01)
        })
        .collect();
    candidates[weighted_index(&scores, rng)]
}

fn make_child(
    id: usize,
    parents: [&Agent; 3],
    lexicon: &CommunityLexicon,
    generation: u64,
    rng: &mut Rng,
) -> (Agent, bool) {
    let template = parents[rng.index(parents.len())];
    let mut child = template.clone();
    child.id = id;
    child.lineage_id = (generation << 32) | id as u64;
    child.identity_token = format!("agent_{id}");
    child.birth_generation = generation;
    child.parent_lineages = Some([
        parents[0].lineage_id,
        parents[1].lineage_id,
        parents[2].lineage_id,
    ]);
    child.language = LanguageProfile::inherit_from(
        [
            &parents[0].language,
            &parents[1].language,
            &parents[2].language,
        ],
        rng,
    );
    child.language.reset_identity();
    child.traits = Traits::child_from(
        [&parents[0].traits, &parents[1].traits, &parents[2].traits],
        rng,
    );
    child.energy = 75.0;
    child.fitness = 0.0;
    child.total_fitness = 0.0;
    child.cooperation_bonus = 0.0;
    child.novelty_bonus = 0.0;
    child.lineage_score = 0.0;
    child.tasks_solved = 0;
    child.tasks_failed = 0;
    child.frustration = 0.0;
    child.social_memory.clear();
    child.trust.clear();
    child.referent_lexicon.clear();
    child.action_lexicon.clear();
    for parent in parents {
        for (meaning, signal) in &parent.referent_lexicon {
            if rng.unit() < 0.65 {
                child
                    .referent_lexicon
                    .insert(meaning.clone(), signal.clone());
            }
        }
        for (meaning, signal) in &parent.action_lexicon {
            if rng.unit() < 0.65 {
                child.action_lexicon.insert(meaning.clone(), signal.clone());
            }
        }
    }
    child.numeric = parents[rng.index(parents.len())].numeric.clone();
    if rng.unit() < child.traits.mutation_rate {
        child.numeric.mutate(rng);
    }
    let mut vocabulary = BTreeSet::new();
    for parent in parents {
        vocabulary.extend(parent.vocabulary.iter().cloned());
    }
    let mut public_vocabulary = BTreeSet::new();
    public_vocabulary.extend(lexicon.numeric_conventions().values().cloned());
    public_vocabulary.extend(lexicon.referential_conventions().values().cloned());
    public_vocabulary.extend(lexicon.action_conventions().values().cloned());
    vocabulary.extend(public_vocabulary.iter().cloned());
    let mut ranked_vocabulary: Vec<_> = vocabulary.into_iter().collect();
    ranked_vocabulary.sort_by(|left, right| {
        public_vocabulary
            .contains(right)
            .cmp(&public_vocabulary.contains(left))
            .then_with(|| {
                child
                    .language
                    .salience(right)
                    .partial_cmp(&child.language.salience(left))
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.cmp(right))
    });
    child.vocabulary = ranked_vocabulary.into_iter().take(80).collect();
    child.semantics = SemanticStore::inherit_from(
        [
            &parents[0].semantics,
            &parents[1].semantics,
            &parents[2].semantics,
        ],
        &child.vocabulary,
        rng,
    );
    child.ensure_numeric_semantics(rng);
    // Programs are inert data, never executable host code. Most founders are
    // empty, but rare bounded mutations allow the population to discover and
    // retain a useful program without pre-seeding a winning genome.
    child.program = parents[rng.index(parents.len())].program.clone();
    let inherited_program = child.program.clone();
    if rng.unit() < child.traits.mutation_rate {
        child.program = mutate_program(&child.program, rng);
    }
    let program_mutated = child.program != inherited_program;
    (child, program_mutated)
}

fn trait_diversity(agents: &[Agent]) -> f64 {
    if agents.len() < 2 {
        return 0.0;
    }
    let values: Vec<_> = agents
        .iter()
        .map(|agent| {
            [
                agent.traits.mutation_rate,
                agent.traits.cooperation_weight,
                agent.traits.novelty_weight,
                agent.traits.stability_weight,
                agent.traits.trust_threshold,
                agent.traits.chattiness,
                agent.traits.patience,
                agent.traits.teaching_drive,
                agent.traits.curiosity,
                agent.traits.expressiveness,
            ]
        })
        .collect();
    let dimensions = values[0].len();
    let variance = (0..dimensions)
        .map(|dimension| {
            let mean =
                values.iter().map(|traits| traits[dimension]).sum::<f64>() / values.len() as f64;
            values
                .iter()
                .map(|traits| (traits[dimension] - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64
        })
        .sum::<f64>()
        / dimensions as f64;
    variance.sqrt()
}

fn program_diversity(agents: &[Agent]) -> f64 {
    if agents.is_empty() {
        return 0.0;
    }
    let unique = agents
        .iter()
        .map(|agent| format!("{:?}", agent.program))
        .collect::<BTreeSet<_>>()
        .len();
    unique as f64 / agents.len() as f64
}

fn weighted_index(weights: &[f64], rng: &mut Rng) -> usize {
    let total: f64 = weights.iter().sum();
    let mut target = rng.unit() * total;
    for (index, weight) in weights.iter().enumerate() {
        target -= weight;
        if target <= 0.0 {
            return index;
        }
    }
    weights.len() - 1
}

#[cfg(test)]
mod tests {
    use super::evolve;
    use crate::lexicon::CommunityLexicon;
    use crate::model::{Agent, Rng};

    #[test]
    fn evolution_keeps_population_size_and_reuses_the_culled_identity() {
        let mut rng = Rng::new(61);
        let mut agents = (0..12)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        for (index, agent) in agents.iter_mut().enumerate() {
            agent.fitness = index as f64;
        }
        let metrics = evolve(&mut agents, &CommunityLexicon::default(), 1, &mut rng);
        assert_eq!(agents.len(), 12);
        assert_eq!(metrics.culled_id, metrics.child_id);
        assert!(
            agents
                .iter()
                .any(|agent| Some(agent.id) == metrics.child_id)
        );
        assert!(metrics.best_fitness > 0.0);
        let child = agents
            .iter()
            .find(|agent| Some(agent.id) == metrics.child_id)
            .expect("culled slot was refilled");
        assert!(child.lineage_id >= (1_u64 << 32));
        assert!(child.parent_lineages.is_some());
    }
}

//! A bounded, independently testable inquiry loop.
//!
//! Stories supply candidate concepts. This lab supplies a separate world to
//! test claims about them, so a repeated sentence cannot become knowledge
//! merely by being repeated.

use crate::model::{Agent, Rng};
use crate::reading::ReadingBridge;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

const FACTS: [(&str, &str); 5] = [
    ("rain", "wet"),
    ("fire", "hot"),
    ("seed", "grow"),
    ("night", "dark"),
    ("stone", "heavy"),
];

#[derive(Clone, Debug, Default)]
struct ClaimEvidence {
    support_lineages: BTreeSet<u64>,
    contradiction_lineages: BTreeSet<u64>,
}

impl ClaimEvidence {
    fn boring(&self) -> bool {
        self.support_lineages.len() >= 3
            && self.contradiction_lineages.len() * 4 <= self.support_lineages.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct InquiryLab {
    observations: usize,
    proposals: usize,
    verified: usize,
    rejected: usize,
    unresolved: usize,
    claims: BTreeMap<(String, String), ClaimEvidence>,
}

impl InquiryLab {
    pub fn tick(
        &mut self,
        agents: &mut [Agent],
        generation: usize,
        rng: &mut Rng,
        reading: &ReadingBridge,
        root: &Path,
    ) -> io::Result<()> {
        if agents.len() < 2 {
            return Ok(());
        }
        // Independent observation is the spark: each agent receives a small
        // sensory episode before any social claim is considered.
        for agent in agents.iter_mut() {
            let (subject, property) = FACTS[rng.index(FACTS.len())];
            agent.observe(&[subject.into(), property.into()], 0.08, rng);
            self.observations += 1;
        }

        let teacher_index = rng.index(agents.len());
        let mut learner_index = rng.index(agents.len() - 1);
        if learner_index >= teacher_index {
            learner_index += 1;
        }
        let story_words: Vec<_> = reading
            .exploration_candidates()
            .into_iter()
            .filter(|word| agents[teacher_index].vocabulary.contains(word))
            .collect();
        let story_inquiry = story_words.len() >= 2 && rng.unit() < 0.60;
        let (subject, true_property) = if story_inquiry {
            let subject = story_words[rng.index(story_words.len())].clone();
            (subject, String::new())
        } else {
            let (subject, property) = FACTS[rng.index(FACTS.len())];
            (subject.to_string(), property.to_string())
        };
        // The claim is selected from the teacher's own signed semantic map,
        // with a small exploration rate. It is not handed the true property.
        let properties: Vec<_> = if story_inquiry {
            let neighbours = reading.inquiry_neighbours(&subject);
            if neighbours.is_empty() {
                story_words.clone()
            } else {
                neighbours
            }
        } else {
            FACTS
                .iter()
                .map(|(_, property)| property.to_string())
                .collect()
        };
        let property = if rng.unit() < 0.15 {
            properties[rng.index(properties.len())].clone()
        } else {
            let best = properties
                .iter()
                .map(|property| self.priority(&agents[teacher_index], &subject, property))
                .fold(f32::NEG_INFINITY, f32::max);
            let candidates: Vec<_> = properties
                .into_iter()
                .filter(|property| {
                    (self.priority(&agents[teacher_index], &subject, property) - best).abs()
                        < f32::EPSILON
                })
                .collect();
            candidates[rng.index(candidates.len())].clone()
        };
        let claim = (subject, property);
        self.proposals += 1;

        // The learner checks the claim against a fresh observation. There is
        // no reward for a claim until this check has happened.
        let observed = if story_inquiry {
            format!(
                "{} shared passages",
                reading.inquiry_evidence(&claim.0, &claim.1)
            )
        } else {
            true_property.clone()
        };
        let verified = if story_inquiry {
            reading.inquiry_evidence(&claim.0, &claim.1) >= 2
        } else {
            observed == claim.1
        };
        let unresolved = story_inquiry && !verified;
        let (teacher, learner) = two_agents(agents, teacher_index, learner_index);
        let learner_lineage = learner.lineage_id;
        let utterance = vec![
            teacher.referent_signal("r0"),
            claim.0.clone(),
            claim.1.clone(),
        ];
        teacher.observe(&utterance, 0.04, rng);
        learner.observe(&utterance, 0.04, rng);
        if verified {
            teacher.semantics.link(&claim.0, &claim.1, 0.16, rng);
            learner.semantics.link(&claim.0, &claim.1, 0.18, rng);
            teacher.fitness += 0.04;
            learner.fitness += 0.05;
            self.verified += 1;
            self.record_evidence(&claim, learner_lineage, true);
        } else if !unresolved {
            // A counterexample weakens this proposed relation, rather than
            // declaring either word globally false or unrelated.
            teacher.semantics.link(&claim.0, &claim.1, -0.04, rng);
            learner.semantics.link(&claim.0, &claim.1, -0.08, rng);
            self.rejected += 1;
            self.record_evidence(&claim, learner_lineage, false);
        } else {
            self.unresolved += 1;
        }
        let path = root.join("inquiry_log.csv");
        let fresh = !path.exists();
        let mut log = OpenOptions::new().create(true).append(true).open(path)?;
        if fresh {
            writeln!(log, "generation,teacher,learner,claim,evidence,status")?;
        }
        writeln!(
            log,
            "{generation},{},{},{},{},{}",
            teacher.lineage_id,
            learner.lineage_id,
            format!("{}={}", claim.0, claim.1),
            observed,
            if verified {
                "verified"
            } else if unresolved {
                "unresolved"
            } else {
                "counterexample"
            }
        )?;
        fs::write(
            root.join("inquiry_monitor.html"),
            format!(
                "<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=5><title>World inquiry</title><style>body{{font:18px system-ui;max-width:850px;margin:40px auto;background:#101925;color:#e5edf6}}strong{{color:#61cfff}}</style><h1>Testing claims about the world</h1><p>Independent observations: <strong>{}</strong>; peer proposals: <strong>{}</strong>; independently verified: <strong>{}</strong>; counterexamples: <strong>{}</strong>; language hypotheses awaiting evidence: <strong>{}</strong>; settled claims now low priority: <strong>{}</strong>.</p><p>Teachers propose relations from their own signed semantic associations. Stable-world claims receive independent observations. Story-language claims are checked against separate recurring passages: corroboration strengthens them, while absence remains unresolved rather than becoming a false claim. A claim becomes low priority after three independent supporting lineages and few contradictions, leaving it available but making contested or unexplored language more likely to be investigated.</p><p>Details: inquiry_log.csv</p>",
                self.observations,
                self.proposals,
                self.verified,
                self.rejected,
                self.unresolved,
                self.claims.values().filter(|claim| claim.boring()).count()
            ),
        )?;
        Ok(())
    }

    fn priority(&self, agent: &Agent, subject: &str, property: &str) -> f32 {
        let association = teacher_association(agent, subject, property);
        let boring_penalty = self
            .claims
            .get(&(subject.to_string(), property.to_string()))
            .is_some_and(ClaimEvidence::boring)
            .then_some(0.30)
            .unwrap_or(0.0);
        association - boring_penalty
    }

    fn record_evidence(&mut self, claim: &(String, String), lineage: u64, supporting: bool) {
        let evidence = self.claims.entry(claim.clone()).or_default();
        if supporting {
            evidence.support_lineages.insert(lineage);
        } else {
            evidence.contradiction_lineages.insert(lineage);
        }
    }
}

fn two_agents(agents: &mut [Agent], first: usize, second: usize) -> (&mut Agent, &mut Agent) {
    if first < second {
        let (left, right) = agents.split_at_mut(second);
        (&mut left[first], &mut right[0])
    } else {
        let (left, right) = agents.split_at_mut(first);
        (&mut right[0], &mut left[second])
    }
}

fn teacher_association(agent: &Agent, subject: &str, property: &str) -> f32 {
    agent
        .semantics
        .association(subject, property)
        .unwrap_or(0.0)
}

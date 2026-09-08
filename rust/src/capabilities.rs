//! Non-linguistic, model-based learning in a bounded key/door/power world.
//! The learner receives only (state, action, next_state); environment rules
//! and evaluation answers are never passed to its planner.
use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

use crate::model::{Agent, Rng};

const ACTIONS: usize = 5;
const NAMES: [&str; ACTIONS] = [
    "take_key",
    "unlock",
    "power_on",
    "extend_bridge",
    "retrieve",
];
// Visible state bits: key held, door unlocked, power on, bridge extended, object held.
fn environment(state: u8, action: usize) -> u8 {
    let prerequisites = [0, 1, 0, 4, 2 | 8];
    if state & prerequisites[action] == prerequisites[action] {
        state | (1 << action)
    } else {
        state
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CapabilityMemory {
    // At most 32 states x 5 actions; duplicate testimony adds no evidence.
    observations: BTreeMap<(u8, usize), u8>,
}

impl CapabilityMemory {
    fn observe(&mut self, state: u8, action: usize, next: u8) {
        self.observations.insert((state, action), next);
    }

    fn merge(&mut self, other: &Self) {
        self.observations
            .extend(other.observations.iter().map(|(k, v)| (*k, *v)));
    }

    // Infer an additive effect and the smallest conjunction of prerequisites
    // consistent with ALL observed successes and failures. This hypothesis
    // class is supplied by us; the particular action rules are learned.
    fn rule(&self, action: usize) -> Option<(u8, u8)> {
        let samples: Vec<_> = self
            .observations
            .iter()
            .filter(|((_, a), _)| *a == action)
            .collect();
        let effect = samples
            .iter()
            .fold(0, |effect, ((state, _), next)| effect | (**next & !*state));
        if effect == 0 {
            return None;
        }
        (0u8..32)
            .filter(|required| {
                samples.iter().all(|((state, _), next)| {
                    let predicted = if state & required == *required {
                        state | effect
                    } else {
                        *state
                    };
                    predicted == **next
                })
            })
            .min_by_key(|required| (required.count_ones(), *required))
            .map(|required| (required, effect))
    }

    fn predict(&self, state: u8, action: usize) -> Option<u8> {
        self.rule(action).map(|(required, effect)| {
            if state & required == required {
                state | effect
            } else {
                state
            }
        })
    }

    fn plan(&self, start: u8, goal: u8) -> Option<Vec<usize>> {
        let mut queue = VecDeque::from([(start, Vec::new())]);
        let mut seen = [false; 32];
        seen[start as usize] = true;
        while let Some((state, path)) = queue.pop_front() {
            if state & goal == goal {
                return Some(path);
            }
            for action in 0..ACTIONS {
                let Some(next) = self.predict(state, action) else {
                    continue;
                };
                if !seen[next as usize] {
                    seen[next as usize] = true;
                    let mut extended = path.clone();
                    extended.push(action);
                    queue.push_back((next, extended));
                }
            }
        }
        None
    }
}

#[derive(Clone, Debug, Default)]
struct Score {
    solved: usize,
    trials: usize,
    wasted: usize,
    steps: usize,
    predicted: usize,
    correct: usize,
}

// Fixed evaluation tasks, different initial states from training's empty
// world. Evaluation never updates memory, including after a failed action.
fn evaluate(memory: &CapabilityMemory, goal: u8) -> Score {
    let mut score = Score::default();
    for start in [5, 6, 9, 10] {
        if start & goal == goal {
            continue;
        }
        score.trials += 1;
        let mut state = start;
        if let Some(plan) = memory.plan(start, goal) {
            for action in plan {
                let next = environment(state, action);
                score.steps += 1;
                score.wasted += usize::from(next == state);
                state = next;
            }
        }
        score.solved += usize::from(state & goal == goal);
    }
    // Exhaustive transition probe, including unseen state/action pairs.
    // Unknown predictions count against coverage, not as correct no-ops.
    for state in 0..32 {
        for action in 0..ACTIONS {
            if let Some(next) = memory.predict(state, action) {
                score.predicted += 1;
                score.correct += usize::from(next == environment(state, action));
            }
        }
    }
    score
}

#[derive(Default)]
pub struct CapabilityLab {
    rng: Option<Rng>,
    history: Vec<(usize, f64, f64)>,
    pub summary: String,
}

impl CapabilityLab {
    pub fn tick(
        &mut self,
        agents: &mut [Agent],
        generation: usize,
        seed: u64,
        root: &Path,
    ) -> io::Result<()> {
        let rng = self.rng.get_or_insert_with(|| Rng::new(seed ^ 0xa631_02fb));
        let stage = if generation <= 20 {
            1
        } else if generation <= 40 {
            2
        } else {
            3
        };
        let goal = [2, 8, 16][stage - 1];
        let available = if stage == 1 {
            2
        } else if stage == 2 {
            4
        } else {
            5
        };
        let trace_path = root.join("capability_episodes.jsonl");
        let mut trace = OpenOptions::new()
            .create(true)
            .append(true)
            .open(trace_path)?;
        // Three learners per generation. Others retain their own experience;
        // offspring inherit their selected parent's memory through Agent clone.
        for _ in 0..agents.len().min(3) {
            let index = rng.index(agents.len());
            let agent = &mut agents[index];
            let mut state = 0u8;
            let mut transitions = Vec::new();
            for _ in 0..12 {
                if state & goal == goal {
                    break;
                }
                let planned = agent
                    .capabilities
                    .plan(state, goal)
                    .and_then(|plan| plan.first().copied());
                let action = if rng.unit() < 0.25 {
                    rng.index(available)
                } else {
                    planned.unwrap_or_else(|| rng.index(available))
                };
                let predicted = agent.capabilities.predict(state, action);
                let next = environment(state, action);
                agent.capabilities.observe(state, action, next);
                transitions.push(serde_json::json!({"before": state, "action": NAMES[action], "predicted": predicted, "after": next}));
                state = next;
            }
            let solved = state & goal == goal;
            // Bounded reward for actual goal attainment, never agreement.
            if solved {
                agent.fitness += 0.15;
            }
            writeln!(
                trace,
                "{}",
                serde_json::json!({"generation": generation, "lineage": agent.lineage_id,
                "stage": stage, "goal": goal, "solved": solved, "transitions": transitions})
            )?;
        }
        if generation != 1 && !generation.is_multiple_of(5) {
            return Ok(());
        }
        let mut pooled = CapabilityMemory::default();
        for agent in agents.iter() {
            pooled.merge(&agent.capabilities);
        }
        let agent_path = root.join("capability_agents.csv");
        let agent_new = !agent_path.exists();
        let mut agent_csv = OpenOptions::new()
            .create(true)
            .append(true)
            .open(agent_path)?;
        if agent_new {
            writeln!(
                agent_csv,
                "generation,lineage,birth_generation,observations,learned_actions,retrieval_solved,retrieval_trials,correct_predictions,prediction_slots"
            )?;
        }
        for agent in agents.iter() {
            let score = evaluate(&agent.capabilities, 16);
            let learned = (0..ACTIONS)
                .filter(|action| agent.capabilities.rule(*action).is_some())
                .count();
            writeln!(
                agent_csv,
                "{generation},{},{},{},{learned},{},{},{},160",
                agent.lineage_id,
                agent.birth_generation,
                agent.capabilities.observations.len(),
                score.solved,
                score.trials,
                score.correct
            )?;
        }
        let path = root.join("capability_metrics.csv");
        let new = !path.exists();
        let mut csv = OpenOptions::new().create(true).append(true).open(path)?;
        if new {
            writeln!(
                csv,
                "generation,stage,mode,capability,solved,trials,success_rate,steps,wasted_actions,predictions,correct_predictions,prediction_slots"
            )?;
        }
        let mut final_rates = [0.0; 2];
        let mut rows = String::new();
        for (mode, memories) in [
            (
                "individual",
                agents
                    .iter()
                    .map(|a| a.capabilities.clone())
                    .collect::<Vec<_>>(),
            ),
            ("pooled", vec![pooled]),
            ("untrained", vec![CapabilityMemory::default()]),
        ] {
            for (label, target) in [("unlock", 2), ("bridge", 8), ("retrieve", 16)] {
                let mut total = Score::default();
                for memory in &memories {
                    let score = evaluate(memory, target);
                    total.solved += score.solved;
                    total.trials += score.trials;
                    total.steps += score.steps;
                    total.wasted += score.wasted;
                    total.predicted += score.predicted;
                    total.correct += score.correct;
                }
                let rate = total.solved as f64 / total.trials.max(1) as f64;
                writeln!(
                    csv,
                    "{generation},{stage},{mode},{label},{},{},{rate:.4},{},{},{},{},{}",
                    total.solved,
                    total.trials,
                    total.steps,
                    total.wasted,
                    total.predicted,
                    total.correct,
                    memories.len() * 160
                )?;
                rows.push_str(&format!("<tr><td>{mode}</td><td>{label}</td><td>{:.0}% ({}/{})</td><td>{}</td><td>{:.0}%</td></tr>",
                    rate*100.0, total.solved, total.trials, total.wasted, 100.0*total.correct as f64/(memories.len()*160) as f64));
                if label == "retrieve" {
                    if mode == "individual" {
                        final_rates[0] = rate;
                    }
                    if mode == "pooled" {
                        final_rates[1] = rate;
                    }
                }
            }
        }
        self.history
            .push((generation, final_rates[0], final_rates[1]));
        // Keep the live chart bounded; the CSV retains every measurement.
        if self.history.len() > 400 {
            self.history.remove(0);
        }
        self.summary = format!(
            "Capabilities (evaluated generation {generation}): stage={stage}; retrieval individual={:.0}%; pooled={:.0}%",
            final_rates[0] * 100.0,
            final_rates[1] * 100.0
        );
        let points = |pooled: bool| {
            self.history
                .iter()
                .map(|(g, solo, shared)| {
                    format!(
                        "{:.1},{:.1}",
                        40.0 + 700.0 * (*g as f64 / generation.max(1) as f64),
                        180.0 - 150.0 * if pooled { *shared } else { *solo }
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        fs::write(
            root.join("capability_monitor.html"),
            format!(
                r##"<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Capability learning</title>
<style>body{{font:17px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#101925;color:#e5edf6}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #405064}}svg{{width:100%;background:#182638}}p{{line-height:1.5}}.blue{{color:#61cfff}}.gold{{color:#ffce66}}</style>
<h1>Learning to act</h1><p>{}</p><p>Retrieval on fixed test problems: <span class="blue">individual</span> / <span class="gold">pooled evidence</span>.</p>
<svg viewBox="0 0 780 220" role="img" aria-label="Retrieval success from zero to one hundred percent over generations"><text x="2" y="32" fill="white">100%</text><text x="10" y="185" fill="white">0%</text><path d="M40 30V180H750" fill="none" stroke="#789"/><polyline points="{}" fill="none" stroke="#61cfff" stroke-width="3"/><polyline points="{}" fill="none" stroke="#ffce66" stroke-width="2"/><text x="300" y="210" fill="white">Generation (latest: {generation})</text></svg>
<table><tr><th>Knowledge</th><th>Goal</th><th>Test success</th><th>Wasted actions</th><th>Correct / all predictions</th></tr>{rows}</table>
<p>Training: generations 1–20 key/door; 21–40 power/bridge; 41 onward combined retrieval. Test episodes never update knowledge. Zero success before a capability is introduced is expected. Unknown predictions count as incorrect in this table.</p>
<p>Pooled evidence is an upper bound on sharing, not proof of learned communication. Untrained uses the same planner with empty memory and abstains. The supplied planner searches; agents learn action prerequisites and effects. This is a five-bit deterministic world, not general intelligence or spatial navigation. Fixed tests use four alternative starts; training always starts empty. These are held-out starting situations, not necessarily unseen intermediate states. State bits: 1 key, 2 unlocked, 4 power, 8 bridge, 16 object.</p>
<p>Refreshes every five seconds. Full history: capability_metrics.csv. Per-lineage learning: capability_agents.csv. Attempts and predictions: capability_episodes.jsonl. Knowledge is run-local; offspring inherit one parent's experience. Population scores include learning, inheritance and selection.</p></html>"##,
                self.summary,
                points(false),
                points(true)
            ),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_rules_from_outcomes_and_solves_unseen_combinations() {
        let mut memory = CapabilityMemory::default();
        assert_eq!(evaluate(&memory, 16).solved, 0);
        // Isolated experiments, never a demonstration of a retrieval plan.
        for (state, action) in [
            (0, 0),
            (0, 1),
            (1, 1),
            (0, 2),
            (0, 3),
            (4, 3),
            (0, 4),
            (2, 4),
            (8, 4),
            (10, 4),
        ] {
            memory.observe(state, action, environment(state, action));
        }
        let score = evaluate(&memory, 16);
        assert_eq!(score.solved, score.trials);
        assert_eq!(score.correct, 160);
        assert_eq!(score.wasted, 0);
    }

    #[test]
    fn failure_revises_an_overgeneral_rule() {
        let mut memory = CapabilityMemory::default();
        memory.observe(1, 1, 3);
        assert_eq!(memory.predict(0, 1), Some(2));
        memory.observe(0, 1, 0);
        assert_eq!(memory.predict(0, 1), Some(0));
        assert_eq!(memory.predict(5, 1), Some(7));
    }

    #[test]
    fn evaluation_is_read_only_and_pooling_combines_complementary_skills() {
        let mut key = CapabilityMemory::default();
        let mut bridge = CapabilityMemory::default();
        for state in 0..32 {
            for action in 0..5 {
                if action < 2 {
                    key.observe(state, action, environment(state, action));
                } else {
                    bridge.observe(state, action, environment(state, action));
                }
            }
        }
        assert_eq!(evaluate(&key, 16).solved, 0);
        let mut pooled = key.clone();
        pooled.merge(&bridge);
        let before = pooled.clone();
        let score = evaluate(&pooled, 16);
        assert_eq!(score.solved, score.trials);
        assert_eq!(before, pooled);
        assert_ne!(key, pooled);
    }
}

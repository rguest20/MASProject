use crate::model::{Agent, Rng};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

pub const ACTIONS: [&str; 5] = [
    "turn_wheel",
    "push_lever",
    "pull_lever",
    "push_button",
    "wait",
];

fn true_probability(previous: usize, next: usize) -> f64 {
    match (next + 5 - previous) % 5 {
        1 => 50.0,
        2 | 3 => 25.0,
        _ => 0.0,
    }
}

fn generate_process(rng: &mut Rng) -> Vec<usize> {
    let mut process = Vec::with_capacity(64);
    process.push(rng.index(ACTIONS.len()));
    while process.len() < 64 {
        let previous = *process.last().expect("machine always has a start action");
        let offset = if rng.unit() < 0.50 {
            1
        } else if rng.unit() < 0.50 {
            2
        } else {
            3
        };
        process.push((previous + offset) % ACTIONS.len());
    }
    process
}

#[derive(Clone, Debug, Default)]
pub struct MachineMemory {
    /// Direct, position-specific feedback for this particular machine.
    /// A sequence cannot be recovered reliably from the previous action alone:
    /// the same action may occur at several positions with different successors.
    pub positions: BTreeMap<(usize, usize), (usize, usize)>, // accepted, rejected
    /// Accepted transitions are evidence about the generator and survive a
    /// reshuffle. Rejections stay position-local because one sampled path can
    /// legitimately reject a 25% successor on that particular step.
    pub transitions: BTreeMap<(usize, usize), usize>,
}
impl MachineMemory {
    fn observe_position(&mut self, position: usize, action: usize, accepted: bool) {
        let entry = self.positions.entry((position, action)).or_default();
        if accepted {
            entry.0 += 1;
        } else {
            // A previously accepted action can only be rejected after the
            // machine changed. Forget that position's stale answer, while
            // retaining the rejection as a cue to explore another action.
            entry.0 = 0;
            entry.1 += 1;
        }
    }
    fn observe_transition(&mut self, previous: usize, next: usize) {
        *self.transitions.entry((previous, next)).or_default() += 1;
    }

    fn transition_probability(&self, previous: usize, next: usize) -> f64 {
        let total: usize = (0..ACTIONS.len())
            .map(|candidate| {
                self.transitions
                    .get(&(previous, candidate))
                    .copied()
                    .unwrap_or(0)
            })
            .sum();
        if total == 0 {
            0.0
        } else {
            self.transitions
                .get(&(previous, next))
                .copied()
                .unwrap_or(0) as f64
                / total as f64
        }
    }

    fn choose_for_position(
        &self,
        position: usize,
        previous: Option<usize>,
        rng: &mut Rng,
    ) -> usize {
        // Use a confirmed action immediately. Until then, exhaust actions with
        // the least rejection evidence before retrying a known failure.
        if let Some(action) = (0..5).find(|action| {
            self.positions
                .get(&(position, *action))
                .is_some_and(|(accepted, _)| *accepted > 0)
        }) {
            return action;
        }
        let lowest_rejections = (0..5)
            .map(|action| {
                self.positions
                    .get(&(position, action))
                    .map_or(0, |(_, rejected)| *rejected)
            })
            .min()
            .unwrap_or(0);
        let options: Vec<_> = (0..5)
            .filter(|action| {
                self.positions
                    .get(&(position, *action))
                    .map_or(0, |(_, rejected)| *rejected)
                    == lowest_rejections
            })
            .collect();
        let best_score = previous.map_or(0.0, |from| {
            options
                .iter()
                .map(|next| self.transition_probability(from, *next))
                .fold(0.0, f64::max)
        });
        let most_likely: Vec<_> = options
            .into_iter()
            .filter(|next| {
                previous.map_or(true, |from| {
                    (self.transition_probability(from, *next) - best_score).abs() < f64::EPSILON
                })
            })
            .collect();
        most_likely[rng.index(most_likely.len())]
    }
}

pub struct MachineLab {
    pub process: Vec<usize>,
    mastery: usize,
    streak: usize,
    rng: Rng,
    history: Vec<(usize, usize, f64, f64)>,
    pub won: bool,
    recent: std::collections::VecDeque<(usize, usize)>,
    reshuffles: usize,
}
impl MachineLab {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed ^ 0x4d41_4348);
        Self {
            process: generate_process(&mut rng),
            mastery: 0,
            streak: 0,
            rng,
            history: Vec::new(),
            won: false,
            recent: std::collections::VecDeque::new(),
            reshuffles: 0,
        }
    }
    pub fn tick(&mut self, agents: &mut [Agent], generation: usize, root: &Path) -> io::Result<()> {
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("machine_episodes.jsonl"))?;
        let mut solved = 0;
        let mut total_steps = 0;
        let mut first_attempts = 0;
        let mut first_attempts_right = 0;
        // Every agent gets one complete episode. Starting at step one lets the
        // feedback teach a path, rather than testing an ambiguous bigram.
        for agent in agents.iter_mut() {
            let mut position = 0;
            let mut steps = 0;
            let mut accepted = 0;
            let mut rejected = 0;
            let mut first_try_at_position = true;
            while position < self.process.len() && steps < self.process.len() * 5 {
                let expected = self.process[position];
                let previous = position.checked_sub(1).map(|index| self.process[index]);
                let action = agent
                    .machine
                    .choose_for_position(position, previous, &mut self.rng);
                let right = action == expected;
                if first_try_at_position {
                    first_attempts += 1;
                    first_attempts_right += usize::from(right);
                }
                agent.machine.observe_position(position, action, right);
                if right && position > 0 {
                    let previous = self.process[position - 1];
                    agent.machine.observe_transition(previous, action);
                    self.recent.push_back((previous, action));
                    if self.recent.len() > 100_000 {
                        self.recent.pop_front();
                    }
                }
                if right {
                    accepted += 1;
                } else {
                    rejected += 1;
                }
                if right {
                    position += 1;
                    first_try_at_position = true;
                } else {
                    first_try_at_position = false;
                }
                steps += 1;
            }
            let success = position == self.process.len();
            solved += usize::from(success);
            total_steps += steps;
            writeln!(
                log,
                "{}",
                serde_json::json!({"generation":generation,"lineage":agent.lineage_id,"length":self.process.len(),"success":success,"steps":steps,"accepted":accepted,"rejected":rejected})
            )?;
        }
        let completion_rate = solved as f64 / agents.len().max(1) as f64;
        let first_attempt_rate = first_attempts_right as f64 / first_attempts.max(1) as f64;
        if first_attempt_rate >= 0.9 {
            self.streak += 1
        } else {
            self.streak = 0
        };
        if self.streak >= 3 {
            self.streak = 0;
            if self.process.len() >= 64 {
                self.process = generate_process(&mut self.rng);
                self.reshuffles += 1;
                writeln!(
                    log,
                    "{}",
                    serde_json::json!({"event":"machine_reshuffled","generation":generation,"signal":"RESHUFFLE","reshuffles":self.reshuffles,"length":64})
                )?;
            }
        }
        let path = root.join("machine_metrics.csv");
        let fresh = !path.exists();
        let mut out = OpenOptions::new().create(true).append(true).open(path)?;
        if fresh {
            writeln!(
                out,
                "generation,length,completed,agents,completion_rate,first_attempt_accuracy,steps,mastery_streak"
            )?;
        }
        writeln!(
            out,
            "{generation},{},{solved},{},{completion_rate:.4},{first_attempt_rate:.4},{total_steps},{}",
            self.process.len(),
            agents.len(),
            self.streak
        )?;
        let mean_steps = total_steps as f64 / agents.len().max(1) as f64;
        self.history.push((
            generation,
            self.process.len(),
            first_attempt_rate,
            mean_steps,
        ));
        if self.history.len() > 500 {
            self.history.remove(0);
        }
        let latest = self.history.first().map(|p| p.0).unwrap_or(generation);
        let x = |g: usize| {
            45.0 + 700.0 * (g.saturating_sub(latest)) as f64
                / generation.saturating_sub(latest).max(1) as f64
        };
        let points = |index: usize| {
            self.history
                .iter()
                .map(|p| {
                    format!(
                        "{:.1},{:.1}",
                        x(p.0),
                        if index == 2 {
                            180.0 - 150.0 * p.2
                        } else {
                            180.0 - 150.0 * (p.3 / (p.1.max(1) * 5) as f64).min(1.0)
                        }
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        let sequence = self
            .process
            .iter()
            .map(|a| ACTIONS[*a])
            .collect::<Vec<_>>()
            .join(" → ");
        let mut transition_rows = String::new();
        for from in 0..5 {
            let total: usize = self.recent.iter().filter(|(prev, _)| *prev == from).count();
            transition_rows.push_str(&format!("<tr><td>{}</td>", ACTIONS[from]));
            for to in 0..5 {
                let count: usize = self
                    .recent
                    .iter()
                    .filter(|(prev, next)| *prev == from && *next == to)
                    .count();
                let pct = if total == 0 {
                    0.0
                } else {
                    100.0 * count as f64 / total as f64
                };
                let predictions: Vec<_> = agents
                    .iter()
                    .filter_map(|agent| {
                        let evidence: usize = (0..ACTIONS.len())
                            .map(|candidate| {
                                agent
                                    .machine
                                    .transitions
                                    .get(&(from, candidate))
                                    .copied()
                                    .unwrap_or(0)
                            })
                            .sum();
                        (evidence > 0).then(|| agent.machine.transition_probability(from, to))
                    })
                    .collect();
                let prediction = if predictions.is_empty() {
                    0.0
                } else {
                    100.0 * predictions.iter().sum::<f64>() / predictions.len() as f64
                };
                transition_rows.push_str(&format!(
                    "<td><span class='blue'>{prediction:.1}% learned</span><br>{pct:.1}% observed ({count})<br><span class='red'>{:.0}% target</span></td>",
                    true_probability(from, to)
                ));
            }
            transition_rows.push_str(&format!("<td>{total}</td></tr>"));
        }
        let transition_headers = ACTIONS
            .iter()
            .map(|a| format!("<th>{a}</th>"))
            .collect::<Vec<_>>()
            .join("");
        fs::write(
            root.join("machine_links.html"),
            format!(
                r##"<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Machine action links</title><style>body{{font:17px system-ui;max-width:1000px;margin:40px auto;background:#101925;color:#e5edf6}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #405064;text-align:left}}.red{{color:#ff6b6b}}.blue{{color:#61cfff}}</style><h1>Learned links between actions</h1><p>Each row is <strong>P(next accepted action | previous accepted action)</strong>. <span class="blue">Learned</span> is the agents’ mean probability estimate from accepted transition evidence, retained across reshuffles. Observed is the matching rate from the latest 100,000 accepted transitions. <span class="red">Target</span> is the generator rule: one successor is 50%, two are 25%, and the forbidden successor is 0%.</p><table><tr><th>After ↓ / next →</th>{transition_headers}<th>Observed total</th></tr>{transition_rows}</table><p>Rejected choices are deliberately excluded here: they say only that an action was wrong at one position in a sampled path, not that the general transition is impossible.</p>"##,
                transition_headers = transition_headers,
                transition_rows = transition_rows
            ),
        )?;
        fs::write(
            root.join("machine_monitor.html"),
            format!(
                r##"<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Machine process learning</title><style>body{{font:17px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;background:#101925;color:#e5edf6}}svg{{width:100%;background:#182638}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #405064;text-align:left}}.gold{{color:#ffce66}}.blue{{color:#61cfff}}.red{{color:#ff6b6b}}</style><h1>Learning the machine process</h1><p><strong>Current process ({length} actions):</strong> {sequence}</p><p>First-attempt accuracy: <strong>{first_accuracy:.0}%</strong>; eventual completion: {completion:.0}%; average steps per agent: {steps:.1}; mastery streak: {streak}/3; reshuffles: {mastery}.</p><svg viewBox="0 0 780 220"><text x="2" y="35" fill="white">100%</text><text x="10" y="185" fill="white">0%</text><path d="M45 30V180H750" stroke="#789" fill="none"/><polyline points="{success}" stroke="#61cfff" stroke-width="3" fill="none"/><polyline points="{effort}" stroke="#ffce66" stroke-width="2" fill="none"/><text x="245" y="210" fill="white">Generations {first}–{generation}</text></svg><p><span class="blue">Blue</span> is first-attempt accuracy: did they know the next action before feedback on that step? <span class="gold">Yellow</span> is effort remaining. Eventual completion can be 100% because agents may retry; it does not count as mastery. The process reshuffles only after first-attempt accuracy reaches 90% for three evaluations.</p><table><tr><th>Action</th><th>Meaning</th></tr>{actions}</table><p>Details: machine_metrics.csv, machine_episodes.jsonl, machine_summary.txt. Episode logs are compact summaries to keep generation time bounded.</p></html>"##,
                length = self.process.len(),
                sequence = sequence,
                first_accuracy = first_attempt_rate * 100.0,
                completion = completion_rate * 100.0,
                steps = mean_steps,
                streak = self.streak,
                mastery = self.reshuffles,
                success = points(2),
                effort = points(3),
                first = latest,
                generation = generation,
                actions = ACTIONS
                    .iter()
                    .enumerate()
                    .map(|(i, a)| format!("<tr><td>{a}</td><td>action {i}</td></tr>"))
                    .collect::<Vec<_>>()
                    .join("")
            ),
        )?;
        fs::write(
            root.join("machine_summary.txt"),
            format!(
                "Process length: {}\nReshuffles: {}\nFirst-attempt accuracy: {:.0}%\nEventual completion: {:.0}%\nAverage steps: {:.1}\n",
                self.process.len(),
                self.reshuffles,
                first_attempt_rate * 100.0,
                completion_rate * 100.0,
                mean_steps
            ),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{MachineMemory, generate_process};
    use crate::model::Rng;

    #[test]
    fn generated_process_uses_only_permitted_successors() {
        let mut rng = Rng::new(41);
        let process = generate_process(&mut rng);
        assert_eq!(process.len(), 64);
        for pair in process.windows(2) {
            let offset = (pair[1] + 5 - pair[0]) % 5;
            assert!(matches!(offset, 1..=3));
        }
    }

    #[test]
    fn position_feedback_learns_a_64_step_path_without_repeating_rejections() {
        let mut rng = Rng::new(42);
        let process = generate_process(&mut rng);
        let mut memory = MachineMemory::default();
        let mut attempts = 0;
        for (position, expected) in process.iter().copied().enumerate() {
            loop {
                let previous = position.checked_sub(1).map(|index| process[index]);
                let choice = memory.choose_for_position(position, previous, &mut rng);
                let accepted = choice == expected;
                memory.observe_position(position, choice, accepted);
                attempts += 1;
                if accepted {
                    break;
                }
            }
        }
        assert!(attempts <= 64 * 5);
        for (position, expected) in process.iter().copied().enumerate() {
            let previous = position.checked_sub(1).map(|index| process[index]);
            assert_eq!(
                memory.choose_for_position(position, previous, &mut rng),
                expected
            );
        }
    }

    #[test]
    fn rejection_reopens_a_stale_answer_after_a_reshuffle() {
        let mut rng = Rng::new(43);
        let mut memory = MachineMemory::default();
        memory.observe_position(0, 1, true);
        assert_eq!(memory.choose_for_position(0, None, &mut rng), 1);
        memory.observe_position(0, 1, false);
        assert_ne!(memory.choose_for_position(0, None, &mut rng), 1);
    }

    #[test]
    fn accepted_transitions_form_a_probability_estimate() {
        let mut memory = MachineMemory::default();
        for _ in 0..4 {
            memory.observe_transition(0, 1);
        }
        for _ in 0..2 {
            memory.observe_transition(0, 2);
            memory.observe_transition(0, 3);
        }
        assert_eq!(memory.transition_probability(0, 4), 0.0);
        assert!((memory.transition_probability(0, 1) - 0.5).abs() < f64::EPSILON);
        assert!((memory.transition_probability(0, 2) - 0.25).abs() < f64::EPSILON);
    }
}

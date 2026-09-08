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

#[derive(Clone, Debug, Default)]
pub struct MachineMemory {
    /// Direct, position-specific feedback for this particular machine.
    /// A sequence cannot be recovered reliably from the previous action alone:
    /// the same action may occur at several positions with different successors.
    pub positions: BTreeMap<(usize, usize), (usize, usize)>, // accepted, rejected
    pub transitions: BTreeMap<(usize, usize), usize>,
    pub signed: BTreeMap<(usize, usize), (usize, usize)>, // accepted, rejected
}
impl MachineMemory {
    fn observe_position(&mut self, position: usize, action: usize, accepted: bool) {
        let entry = self.positions.entry((position, action)).or_default();
        if accepted {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }
    fn observe_transition(&mut self, previous: usize, next: usize) {
        *self.transitions.entry((previous, next)).or_default() += 1;
    }
    fn observe_signed(&mut self, previous: usize, next: usize, accepted: bool) {
        let entry = self.signed.entry((previous, next)).or_default();
        if accepted {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }
    fn best_after(&self, previous: usize) -> Option<usize> {
        (0..5).max_by(|a, b| {
            let score = |next: usize| {
                let (p, n) = self
                    .signed
                    .get(&(previous, next))
                    .copied()
                    .unwrap_or((0, 0));
                (p as f64 + 1.0) / (p + n + 2) as f64
            };
            score(*a)
                .partial_cmp(&score(*b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
    fn choose_for_position(&self, position: usize, rng: &mut Rng) -> usize {
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
        options[rng.index(options.len())]
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
        Self {
            process: (0..64).map(|i| i % 5).collect(),
            mastery: 0,
            streak: 0,
            rng: Rng::new(seed ^ 0x4d41_4348),
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
        // Every agent gets one complete episode. Starting at step one lets the
        // feedback teach a path, rather than testing an ambiguous bigram.
        for agent in agents.iter_mut() {
            let mut position = 0;
            let mut steps = 0;
            let mut accepted = 0;
            let mut rejected = 0;
            while position < self.process.len() && steps < self.process.len() * 5 {
                let expected = self.process[position];
                let action = agent.machine.choose_for_position(position, &mut self.rng);
                let right = action == expected;
                agent.machine.observe_position(position, action, right);
                if position > 0 {
                    agent
                        .machine
                        .observe_transition(self.process[position - 1], action);
                    agent
                        .machine
                        .observe_signed(self.process[position - 1], action, right);
                    self.recent.push_back((self.process[position - 1], action));
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
        let rate = solved as f64 / agents.len().max(1) as f64;
        if rate >= 0.9 {
            self.streak += 1
        } else {
            self.streak = 0
        };
        if self.streak >= 3 {
            self.streak = 0;
            if self.process.len() >= 64 {
                for action in &mut self.process {
                    *action = self.rng.index(5);
                }
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
                "generation,length,solved,attempts,success_rate,steps,mastery_streak"
            )?;
        }
        writeln!(
            out,
            "{generation},{},{solved},{},{rate:.4},{total_steps},{}",
            self.process.len(),
            agents.len(),
            self.streak
        )?;
        let mean_steps = total_steps as f64 / agents.len().max(1) as f64;
        self.history
            .push((generation, self.process.len(), rate, mean_steps));
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
                let (positive, negative) = agents
                    .iter()
                    .map(|a| a.machine.signed.get(&(from, to)).copied().unwrap_or((0, 0)))
                    .fold((0usize, 0usize), |(p, n), (pp, nn)| (p + pp, n + nn));
                let valence =
                    (positive as f64 - negative as f64) / (positive + negative + 2) as f64;
                transition_rows.push_str(&format!(
                    "<td>{pct:.1}% ({count})<br><span class='red'>{:.0}% true</span><br><small>valence {valence:+.2} · +{positive}/−{negative}</small></td>",
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
                r##"<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Machine action links</title><style>body{{font:17px system-ui;max-width:900px;margin:40px auto;background:#101925;color:#e5edf6}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #405064;text-align:left}}</style><h1>Observed links between actions</h1><p>Read each row as <strong>P(next choice | previous action)</strong>. A value of 40% in the pull_lever column means agents chose pull_lever on 40% of observed attempts after the row action. Counts are raw attempts across the full population; low totals are weak evidence. Red is the machine's true probability: intended successor 50%, two alternatives 25%, forbidden successor 0%.</p><table><tr><th>After ↓ / choose →</th>{transition_headers}<th>Total</th></tr>{transition_rows}</table><p>This records behaviour, not a claim that a transition is correct. The main process chart is in machine_monitor.html.</p>"##,
                transition_headers = transition_headers,
                transition_rows = transition_rows
            ),
        )?;
        fs::write(
            root.join("machine_monitor.html"),
            format!(
                r##"<!doctype html><html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Machine process learning</title><style>body{{font:17px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;background:#101925;color:#e5edf6}}svg{{width:100%;background:#182638}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #405064;text-align:left}}.gold{{color:#ffce66}}.blue{{color:#61cfff}}.red{{color:#ff6b6b}}</style><h1>Learning the machine process</h1><p><strong>Current process ({length} actions):</strong> {sequence}</p><p>Latest run success: <strong>{rate:.0}%</strong>; average steps: {steps:.1}; mastery streak: {streak}/3; reshuffles: {mastery}.</p><svg viewBox="0 0 780 220"><text x="2" y="35" fill="white">100%</text><text x="10" y="185" fill="white">0%</text><path d="M45 30V180H750" stroke="#789" fill="none"/><polyline points="{success}" stroke="#61cfff" stroke-width="3" fill="none"/><polyline points="{effort}" stroke="#ffce66" stroke-width="2" fill="none"/><text x="245" y="210" fill="white">Generations {first}–{generation}</text></svg><p><span class="blue">Blue</span> is process completion; <span class="gold">yellow</span> is effort remaining (lower is better). The 64-step process reshuffles after three consecutive runs at ≥90% completion. Each step gives explicit feedback: correct actions advance the position; incorrect actions do not. Agents must discover the ordered sequence.</p><table><tr><th>Action</th><th>Meaning</th></tr>{actions}</table><p>Details: machine_metrics.csv, machine_episodes.jsonl, machine_summary.txt. Episodes begin at random positions and cover the whole population. The table is the learned previous-action link distribution; it is separate from the capability monitor.</p></html>"##,
                length = self.process.len(),
                sequence = sequence,
                rate = rate * 100.0,
                steps = mean_steps,
                streak = self.streak,
                mastery = self.mastery,
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
                "Process length: {}\nMastered expansions: {}\nLatest success: {:.0}%\nAverage steps: {:.1}\n",
                self.process.len(),
                self.mastery,
                rate,
                mean_steps
            ),
        )?;
        Ok(())
    }
}

//! Outcome-based learning with a mastery-gated, expanding action world.
//! Rules and goals belong to the environment; learners see only transitions.
use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

use crate::model::{Agent, Rng};

const INITIAL_ACTIONS: usize = 5;
pub const DEFAULT_ACTION_LIMIT: usize = 21;
pub const MAX_ACTION_LIMIT: usize = 31;
const EVIDENCE_PER_ACTION: usize = 64;
const MASTERY: f64 = 0.90;
const MASTERY_CHECKS: usize = 3;
const MIN_LEVEL_GENERATIONS: usize = 20;
const EVAL_CASES: usize = 12;
type State = u64;

#[derive(Clone, Debug)]
struct World {
    prerequisites: Vec<State>,
}

impl Default for World {
    fn default() -> Self {
        Self {
            prerequisites: vec![0, 1, 0, 4, 2 | 8],
        }
    }
}

impl World {
    fn actions(&self) -> usize {
        self.prerequisites.len()
    }
    fn mask(&self) -> State {
        (1 << self.actions()) - 1
    }
    fn goal(&self) -> State {
        1 << (self.actions() - 1)
    }
    fn step(&self, state: State, action: usize) -> State {
        if state & self.prerequisites[action] == self.prerequisites[action] {
            state | (1 << action)
        } else {
            state
        }
    }

    fn advance(&mut self, rng: &mut Rng, limit: usize) {
        if self.actions() >= limit {
            return;
        }
        // Add a new independent preparation skill and an assembly operation
        // that needs BOTH this skill and the previously mastered goal.
        // Edges only point backwards: every generated world is solvable.
        let previous_goal = self.goal();
        if self.actions() + 2 <= limit {
            self.prerequisites.push(0);
        }
        let preparation = self.goal();
        let extra = 1 << rng.index(self.actions());
        self.prerequisites.push(previous_goal | preparation | extra);
    }
}

fn action_name(action: usize) -> String {
    match action {
        0 => "take_key".into(),
        1 => "unlock".into(),
        2 => "power_on".into(),
        3 => "extend_bridge".into(),
        4 => "retrieve".into(),
        _ => format!("operation_{action}"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rule {
    required: State,
    effect: State,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CapabilityMemory {
    // Bounded, distinct transitions; observations are agent-owned and inherited.
    observations: BTreeMap<usize, VecDeque<(State, State)>>,
    rules: BTreeMap<usize, Rule>,
}

impl CapabilityMemory {
    fn observe(&mut self, state: State, action: usize, next: State) {
        let samples = self.observations.entry(action).or_default();
        samples.retain(|(before, _)| *before != state);
        samples.push_back((state, next));
        if samples.len() > EVIDENCE_PER_ACTION {
            samples.pop_front();
        }
        self.refresh(action);
    }

    fn merge(&mut self, other: &Self) {
        for (action, samples) in &other.observations {
            let own = self.observations.entry(*action).or_default();
            for sample in samples {
                if !own.contains(sample) {
                    own.push_back(*sample);
                }
            }
            while own.len() > EVIDENCE_PER_ACTION {
                own.pop_front();
            }
            self.refresh(*action);
        }
    }

    // Infer an additive effect, then cover every observed failure with at
    // least one missing prerequisite. Successful trials constrain candidates.
    // Greedy set cover avoids enumerating 2^N possible prerequisite masks.
    // This is a supplied hypothesis class, not learned general reasoning.
    fn refresh(&mut self, action: usize) {
        self.rules.remove(&action);
        let samples = &self.observations[&action];
        let effect = samples
            .iter()
            .fold(0, |bits, (state, next)| bits | (next & !state));
        if effect == 0 {
            return;
        }
        let candidates = samples
            .iter()
            .filter(|(state, next)| state != next)
            .fold(State::MAX, |bits, (state, _)| bits & state)
            & !effect;
        let mut failures: Vec<State> = samples
            .iter()
            .filter(|(state, next)| state == next && state & effect != effect)
            .map(|(state, _)| candidates & !state)
            .collect();
        if failures.contains(&0) {
            return;
        }
        let mut required = 0;
        while !failures.is_empty() {
            let bit = (0..MAX_ACTION_LIMIT)
                .map(|i| 1u64 << i)
                .max_by_key(|bit| {
                    (
                        failures.iter().filter(|mask| **mask & bit != 0).count(),
                        State::MAX - bit,
                    )
                })
                .expect("nonempty bit range");
            required |= bit;
            failures.retain(|mask| mask & bit == 0);
        }
        let rule = Rule { required, effect };
        if samples
            .iter()
            .all(|(state, next)| apply(rule, *state) == *next)
        {
            self.rules.insert(action, rule);
        }
    }

    fn predict(&self, state: State, action: usize) -> Option<State> {
        self.rules.get(&action).map(|rule| apply(*rule, state))
    }

    // Monotone dependency planning: no state-space enumeration. The supplied
    // planner finds prerequisite closure then executes applicable known rules.
    fn plan(&self, start: State, goal: State) -> Option<Vec<usize>> {
        let mut needed = goal & !start;
        loop {
            let old = needed;
            for rule in self.rules.values() {
                if rule.effect & needed != 0 {
                    needed |= rule.required & !start;
                }
            }
            if needed == old {
                break;
            }
        }
        let mut state = start;
        let mut path = Vec::new();
        for _ in 0..self.rules.len() {
            if state & goal == goal {
                return Some(path);
            }
            let (&action, rule) = self.rules.iter().find(|(_, rule)| {
                rule.effect & needed & !state != 0 && state & rule.required == rule.required
            })?;
            state = apply(*rule, state);
            path.push(action);
        }
        (state & goal == goal).then_some(path)
    }

    fn observations_count(&self) -> usize {
        self.observations.values().map(VecDeque::len).sum()
    }
}

fn apply(rule: Rule, state: State) -> State {
    if state & rule.required == rule.required {
        state | rule.effect
    } else {
        state
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
    slots: usize,
}
impl Score {
    fn rate(&self) -> f64 {
        self.solved as f64 / self.trials.max(1) as f64
    }
    fn add(&mut self, other: Self) {
        self.solved += other.solved;
        self.trials += other.trials;
        self.wasted += other.wasted;
        self.steps += other.steps;
        self.predicted += other.predicted;
        self.correct += other.correct;
        self.slots += other.slots;
    }
}

// A fixed diagnostic battery per goal/level, using its own reproducible RNG.
// Tests include no supplies, partial supplies, and individual missing skills.
// They are never training episodes; intermediate states can overlap training.
fn test_starts(_world: &World, goal: State) -> Vec<State> {
    let mut starts = vec![0];
    // Retention problems stay identical when later actions are introduced.
    let width = INITIAL_ACTIONS.max(goal.trailing_zeros() as usize + 1);
    let mask = (1u64 << width) - 1;
    let mut rng = Rng::new(0x91ab_331e ^ goal ^ width as u64);
    for i in 1..EVAL_CASES {
        let state = if i < EVAL_CASES / 2 {
            rng.next_u64() & mask
        } else {
            mask & !(1 << ((i - EVAL_CASES / 2) % width))
        };
        starts.push(state & !goal);
    }
    starts
}

fn evaluate(memory: &CapabilityMemory, world: &World, goal: State) -> Score {
    let mut score = Score::default();
    for start in test_starts(world, goal) {
        score.trials += 1;
        let mut state = start;
        if let Some(plan) = memory.plan(start, goal) {
            for action in plan {
                let next = world.step(state, action);
                score.steps += 1;
                score.wasted += usize::from(next == state);
                state = next;
            }
        }
        score.solved += usize::from(state & goal == goal);
    }
    // Bounded diagnostic transitions replace the old exhaustive 2^N probe.
    let mut rng = Rng::new(0x615f_7281 ^ world.actions() as u64);
    for i in 0..32 {
        let state = if i == 0 {
            0
        } else {
            rng.next_u64() & world.mask()
        };
        for action in 0..world.actions() {
            score.slots += 1;
            if let Some(next) = memory.predict(state, action) {
                score.predicted += 1;
                score.correct += usize::from(next == world.step(state, action));
            }
        }
    }
    score
}

/// One witnessed transition, not a copied rule or a pooled memory dump.
#[derive(Clone, Copy, Debug)]
struct Lesson {
    teacher: usize,
    state: State,
    action: usize,
    outcome: State,
}

fn propose_lesson(
    agents: &[Agent],
    learner: usize,
    available: usize,
    offset: usize,
) -> Option<Lesson> {
    let memory = &agents[learner].capabilities;
    let mut best = None;
    let mut best_rank = 0;
    // Rotate the first speaker so ties do not permanently favour agent zero.
    for step in 0..agents.len() {
        let teacher = (offset + step) % agents.len();
        if teacher == learner {
            continue;
        }
        for (&action, samples) in &agents[teacher].capabilities.observations {
            if action >= available {
                continue;
            }
            for &(state, outcome) in samples.iter().rev().take(8) {
                if memory
                    .observations
                    .get(&action)
                    .is_some_and(|own| own.contains(&(state, outcome)))
                {
                    continue;
                }
                let prediction = memory.predict(state, action);
                let rank = match prediction {
                    Some(next) if next == outcome => continue,
                    Some(_) => 3, // A concrete disagreement is worth repairing.
                    None => 1,
                } + usize::from(action == available - 1);
                if rank > best_rank {
                    best_rank = rank;
                    best = Some(Lesson {
                        teacher,
                        state,
                        action,
                        outcome,
                    });
                }
            }
        }
    }
    best
}

fn accept_lesson(memory: &mut CapabilityMemory, world: &World, lesson: Lesson) -> bool {
    // Physical replay checks testimony. Never let an asserted peer rule
    // overwrite the learner's model without an observed result.
    if world.step(lesson.state, lesson.action) != lesson.outcome {
        return false;
    }
    memory.observe(lesson.state, lesson.action, lesson.outcome);
    true
}

fn milestone_score(memory: &CapabilityMemory, world: &World, goals: &[State]) -> Score {
    let mut score = Score::default();
    for goal in goals {
        score.add(evaluate(memory, world, *goal));
    }
    score
}

#[derive(Default)]
struct TeachingStats {
    requests: usize,
    accepted: usize,
    improved: usize,
    harmed: usize,
    // Bounded recent paired diagnostic changes (lesson, random practice).
    recent: VecDeque<(f64, f64)>,
}

#[derive(Clone, Debug)]
struct Point {
    generation: usize,
    level: usize,
    individual: f64,
    pooled: f64,
}

/// A lightweight shared semantic scaffold. It stores neighbourhoods and
/// confidence for action prerequisites, but never stores an executable plan.
#[derive(Clone, Debug, Default)]
struct SemanticScaffold {
    edges: BTreeMap<(usize, usize), ScaffoldEdge>, // prerequisite bit -> action
}

#[derive(Clone, Debug, Default)]
struct ScaffoldEdge {
    lineages: BTreeMap<u64, usize>,
    support: usize,
    contradiction: usize,
}

impl SemanticScaffold {
    fn update(&mut self, agents: &[Agent], actions: usize) {
        for agent in agents {
            for action in 0..actions {
                if let Some(rule) = agent.capabilities.rules.get(&action) {
                    for bit in 0..MAX_ACTION_LIMIT {
                        if rule.required & (1 << bit) != 0 {
                            let edge = self.edges.entry((bit, action)).or_default();
                            edge.lineages.entry(agent.lineage_id).or_default();
                            edge.support += 1;
                        }
                    }
                }
            }
            for (&action, samples) in &agent.capabilities.observations {
                for &(state, next) in samples {
                    if state == next {
                        for bit in 0..MAX_ACTION_LIMIT {
                            if state & (1 << bit) != 0
                                && let Some(edge) = self.edges.get_mut(&(bit, action))
                            {
                                edge.contradiction += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    fn frontier(&self, memory: &CapabilityMemory, actions: usize) -> Vec<usize> {
        let mut candidates = self
            .edges
            .iter()
            .filter(|((_, action), edge)| {
                *action < actions
                    && edge.lineages.len() >= 2
                    && edge.support >= edge.contradiction.saturating_mul(2).max(1)
                    && !memory.rules.contains_key(action)
            })
            .map(|((_, action), _)| *action)
            .collect::<Vec<_>>();
        candidates.sort();
        candidates.dedup();
        candidates
    }
}

pub struct CapabilityLab {
    rng: Option<Rng>,
    teaching_rng: Option<Rng>,
    teaching_enabled: bool,
    teaching: TeachingStats,
    scaffold: SemanticScaffold,
    world: World,
    limit: usize,
    level: usize,
    level_started: usize,
    streak: usize,
    pending_advance: bool,
    history: VecDeque<Point>,
    milestones: Vec<State>,
    pub summary: String,
}

impl Default for CapabilityLab {
    fn default() -> Self {
        Self::new(DEFAULT_ACTION_LIMIT)
    }
}

impl CapabilityLab {
    pub fn new(limit: usize) -> Self {
        Self {
            rng: None,
            teaching_rng: None,
            teaching_enabled: true,
            teaching: TeachingStats::default(),
            scaffold: SemanticScaffold::default(),
            world: World::default(),
            limit,
            level: 0,
            level_started: 1,
            streak: 0,
            pending_advance: false,
            history: VecDeque::new(),
            milestones: vec![2, 8, 16],
            summary: String::new(),
        }
    }

    pub fn with_teaching(mut self, enabled: bool) -> Self {
        self.teaching_enabled = enabled;
        self
    }

    fn event(&self, root: &Path, value: serde_json::Value) -> io::Result<()> {
        writeln!(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(root.join("capability_events.jsonl"))?,
            "{value}"
        )
    }

    pub fn tick(
        &mut self,
        agents: &mut [Agent],
        generation: usize,
        seed: u64,
        root: &Path,
    ) -> io::Result<()> {
        if agents.is_empty() {
            return Ok(());
        }
        if self.rng.is_none() {
            self.rng = Some(Rng::new(seed ^ 0xa631_02fb));
        }
        self.scaffold.update(agents, self.world.actions());
        if self.pending_advance {
            self.world.advance(self.rng.as_mut().unwrap(), self.limit);
            self.level += 1;
            self.level_started = generation;
            self.streak = 0;
            self.pending_advance = false;
            self.milestones.push(self.world.goal());
            self.event(root, serde_json::json!({"event":"difficulty_increased", "generation":generation,
                "level":self.level, "actions":self.world.actions(), "prerequisites":self.world.prerequisites}))?;
            // Record the immediate transfer drop BEFORE any new training.
            self.report(agents, generation, root, "before_training")?;
        }
        if generation == 1 {
            self.event(root, serde_json::json!({"event":"started", "generation":1, "level":0,
                "action_limit":self.limit, "peer_teaching":self.teaching_enabled, "mastery_threshold":MASTERY, "consecutive_checks":MASTERY_CHECKS,
                "prerequisites":self.world.prerequisites}))?;
            self.report(agents, generation, root, "before_training")?;
        }
        self.train(agents, generation, root)?;
        if self.teaching_enabled {
            self.teach(agents, generation, seed, root)?;
        }
        if generation == 1 || generation.is_multiple_of(5) {
            self.report(agents, generation, root, "after_training")?;
        }
        Ok(())
    }

    fn train(&mut self, agents: &mut [Agent], generation: usize, root: &Path) -> io::Result<()> {
        let rng = self.rng.as_mut().unwrap();
        let stage = if generation <= 20 {
            1
        } else if generation <= 40 {
            2
        } else {
            3
        };
        let goal = match stage {
            1 => 2,
            2 => 8,
            _ => self.world.goal(),
        };
        let available = match stage {
            1 => 2,
            2 => 4,
            _ => self.world.actions(),
        };
        let budget = available * 3;
        let mut trace = OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("capability_episodes.jsonl"))?;
        for _ in 0..agents.len().min(3) {
            let agent = &mut agents[rng.index(agents.len())];
            let mut state = 0;
            let mut transitions = Vec::new();
            for _ in 0..budget {
                if state & goal == goal {
                    break;
                }
                let planned = agent
                    .capabilities
                    .plan(state, goal)
                    .and_then(|plan| plan.first().copied());
                let frontier = self.scaffold.frontier(&agent.capabilities, available);
                let action = if !frontier.is_empty() && rng.unit() < 0.55 {
                    frontier[rng.index(frontier.len())]
                } else if rng.unit() < 0.25 {
                    rng.index(available)
                } else {
                    planned.unwrap_or_else(|| rng.index(available))
                };
                let predicted = agent.capabilities.predict(state, action);
                let next = self.world.step(state, action);
                agent.capabilities.observe(state, action, next);
                transitions.push(serde_json::json!({"before":state,"action":action_name(action),"predicted":predicted,"after":next}));
                state = next;
            }
            let solved = state & goal == goal;
            if solved {
                agent.fitness += 0.15;
            }
            // One controlled intervention: observe an action in a sampled
            // supply state, to distinguish prerequisites from coincidences.
            // This is training only, no test answers or oracle rules supplied.
            let probe_state = rng.next_u64() & ((1 << available) - 1);
            let probe_action = rng.index(available);
            let predicted = agent.capabilities.predict(probe_state, probe_action);
            let outcome = self.world.step(probe_state, probe_action);
            agent
                .capabilities
                .observe(probe_state, probe_action, outcome);
            writeln!(
                trace,
                "{}",
                serde_json::json!({"generation":generation,"lineage":agent.lineage_id,
                "level":self.level,"actions":available,"goal":goal,"solved":solved,"budget":budget,
                "transitions":transitions,"intervention":{"before":probe_state,"action":action_name(probe_action),
                "predicted":predicted,"after":outcome}})
            )?;
        }
        Ok(())
    }

    fn teach(
        &mut self,
        agents: &mut [Agent],
        generation: usize,
        seed: u64,
        root: &Path,
    ) -> io::Result<()> {
        if agents.len() < 2 {
            return Ok(());
        }
        let rng = self
            .teaching_rng
            .get_or_insert_with(|| Rng::new(seed ^ 0x5eac_119b));
        let available = if generation <= 20 {
            2
        } else if generation <= 40 {
            4
        } else {
            self.world.actions()
        };
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("capability_teaching.jsonl"))?;
        let mut metrics = csv(
            root,
            "capability_teaching.csv",
            "generation,level,teacher_lineage,learner_lineage,action,verified,before_solved,after_solved,random_practice_solved,trials,current_before,current_after,current_random_practice",
        )?;
        let mut requests = 0;
        let mut delivered = 0;
        let mut improved = 0;
        let mut harmed = 0;
        // One short, directed exchange per selected learner, capped at three.
        for _ in 0..agents.len().min(3) {
            requests += 1;
            let learner = rng.index(agents.len());
            let offset = rng.index(agents.len());
            let Some(lesson) = propose_lesson(agents, learner, available, offset) else {
                writeln!(
                    log,
                    "{}",
                    serde_json::json!({"generation":generation,"level":self.level,
                    "learner_lineage":agents[learner].lineage_id,"status":"no_useful_example"})
                )?;
                continue;
            };
            let teacher_lineage = agents[lesson.teacher].lineage_id;
            let teacher_before = agents[lesson.teacher].capabilities.clone();
            let learner_before = agents[learner].capabilities.clone();
            let prediction_before = learner_before.predict(lesson.state, lesson.action);
            // Equal-cost local counterfactual: one random world action on an
            // otherwise identical copy. This copy never enters the population.
            let mut control = learner_before.clone();
            let control_action = rng.index(available);
            let control_state = rng.next_u64() & ((1 << available) - 1);
            let control_outcome = self.world.step(control_state, control_action);
            control.observe(control_state, control_action, control_outcome);
            let verified = accept_lesson(&mut agents[learner].capabilities, &self.world, lesson);
            // Evaluation happens AFTER selection and cannot choose the lesson,
            // reward either agent, undo a bad lesson, or teach any test outcome.
            let before = milestone_score(&learner_before, &self.world, &self.milestones);
            let after =
                milestone_score(&agents[learner].capabilities, &self.world, &self.milestones);
            let random = milestone_score(&control, &self.world, &self.milestones);
            let current_before = evaluate(&learner_before, &self.world, self.world.goal()).rate();
            let current_after = evaluate(
                &agents[learner].capabilities,
                &self.world,
                self.world.goal(),
            )
            .rate();
            let current_random = evaluate(&control, &self.world, self.world.goal()).rate();
            if verified {
                delivered += 1;
                improved += usize::from(after.solved > before.solved);
                harmed += usize::from(after.solved < before.solved);
                self.teaching
                    .recent
                    .push_back((after.rate() - before.rate(), random.rate() - before.rate()));
                if self.teaching.recent.len() > 100 {
                    self.teaching.recent.pop_front();
                }
            }
            debug_assert_eq!(agents[lesson.teacher].capabilities, teacher_before);
            writeln!(
                metrics,
                "{generation},{},{},{},{},{verified},{},{},{},{},{current_before:.4},{current_after:.4},{current_random:.4}",
                self.level,
                teacher_lineage,
                agents[learner].lineage_id,
                action_name(lesson.action),
                before.solved,
                after.solved,
                random.solved,
                before.trials
            )?;
            writeln!(
                log,
                "{}",
                serde_json::json!({"generation":generation,"level":self.level,
                "teacher_lineage":teacher_lineage,"learner_lineage":agents[learner].lineage_id,
                "status":if verified {"verified"}else{"rejected"},
                "lesson":{"state":lesson.state,"action":action_name(lesson.action),"claimed_outcome":lesson.outcome,
                    "prediction_before":prediction_before,"observed_outcome":self.world.step(lesson.state,lesson.action)},
                "random_practice":{"state":control_state,"action":action_name(control_action),"outcome":control_outcome},
                "test_before":before.solved,"test_after":after.solved,"test_random_practice":random.solved,"test_trials":before.trials})
            )?;
        }
        self.teaching.requests += requests;
        self.teaching.accepted += delivered;
        self.teaching.improved += improved;
        self.teaching.harmed += harmed;
        let mut totals = csv(
            root,
            "capability_teaching_totals.csv",
            "generation,level,requests,verified,improved,harmed",
        )?;
        writeln!(
            totals,
            "{generation},{},{requests},{delivered},{improved},{harmed}",
            self.level
        )?;
        Ok(())
    }

    fn report(
        &mut self,
        agents: &[Agent],
        generation: usize,
        root: &Path,
        phase: &str,
    ) -> io::Result<()> {
        let mut pooled = CapabilityMemory::default();
        for agent in agents {
            pooled.merge(&agent.capabilities);
        }
        let scaffold_path = root.join("capability_scaffold.jsonl");
        let mut scaffold_log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(scaffold_path)?;
        writeln!(
            scaffold_log,
            "{}",
            serde_json::json!({
                "generation": generation, "level": self.level,
                "edges": self.scaffold.edges.iter().map(|((bit, action), edge)|
                    serde_json::json!({"from_bit": bit, "to_action": action_name(*action), "support": edge.support,
                        "contradiction": edge.contradiction, "lineages": edge.lineages.len()})).collect::<Vec<_>>()
            })
        )?;
        let mut agent_csv = csv(
            root,
            "capability_agents.csv",
            "generation,level,phase,lineage,birth_generation,observations,learned_actions,current_solved,current_trials,correct_predictions,prediction_slots",
        )?;
        let mut retention_csv = csv(
            root,
            "capability_retention.csv",
            "generation,level,phase,lineage,goal,success_rate,steps,wasted_actions,observations,rule_known",
        )?;
        for agent in agents {
            let score = evaluate(&agent.capabilities, &self.world, self.world.goal());
            writeln!(
                agent_csv,
                "{generation},{},{phase},{},{},{},{},{},{},{},{}",
                self.level,
                agent.lineage_id,
                agent.birth_generation,
                agent.capabilities.observations_count(),
                agent.capabilities.rules.len(),
                score.solved,
                score.trials,
                score.correct,
                score.slots
            )?;
            for goal in &self.milestones {
                let score = evaluate(&agent.capabilities, &self.world, *goal);
                let action = goal.trailing_zeros() as usize;
                writeln!(
                    retention_csv,
                    "{generation},{},{phase},{},{},{:.4},{},{},{},{}",
                    self.level,
                    agent.lineage_id,
                    action_name(action),
                    score.rate(),
                    score.steps,
                    score.wasted,
                    agent.capabilities.observations_count(),
                    usize::from(agent.capabilities.rules.contains_key(&action))
                )?;
            }
        }
        let mut metrics = csv(
            root,
            "capability_metrics.csv",
            "generation,level,actions,phase,mode,capability,solved,trials,success_rate,steps,wasted_actions,predictions,correct_predictions,prediction_slots",
        )?;
        let mut current = [0.0; 2];
        let mut weakest = [1.0f64; 2];
        let mut table = String::new();
        let untrained = CapabilityMemory::default();
        for (mode, memories) in [
            (
                "individual",
                agents.iter().map(|a| &a.capabilities).collect::<Vec<_>>(),
            ),
            ("pooled", vec![&pooled]),
            ("untrained", vec![&untrained]),
        ] {
            for goal in &self.milestones {
                let mut score = Score::default();
                for memory in &memories {
                    score.add(evaluate(memory, &self.world, *goal));
                }
                let label = action_name(goal.trailing_zeros() as usize);
                let rate = score.rate();
                writeln!(
                    metrics,
                    "{generation},{},{},{phase},{mode},{label},{},{},{rate:.4},{},{},{},{},{}",
                    self.level,
                    self.world.actions(),
                    score.solved,
                    score.trials,
                    score.steps,
                    score.wasted,
                    score.predicted,
                    score.correct,
                    score.slots
                )?;
                table.push_str(&format!("<tr><td>{mode}</td><td>{label}</td><td>{:.0}% ({}/{})</td><td>{}</td><td>{:.0}%</td></tr>",rate*100.0,score.solved,score.trials,score.wasted,100.0*score.correct as f64/score.slots.max(1) as f64));
                if mode != "untrained" {
                    let index = usize::from(mode == "pooled");
                    weakest[index] = weakest[index].min(rate);
                    if *goal == self.world.goal() {
                        current[index] = rate;
                    }
                    if phase == "after_training" && *goal != self.world.goal() && rate < MASTERY {
                        self.event(root, serde_json::json!({"event":"retention_drop","generation":generation,
                            "level":self.level,"mode":mode,"goal":action_name(goal.trailing_zeros() as usize),
                            "success_rate":rate,"threshold":MASTERY}))?;
                    }
                }
            }
        }
        if phase == "after_training" && generation >= 40 {
            let mastered = weakest.iter().all(|rate| *rate + 1e-9 >= MASTERY);
            self.streak = if mastered { self.streak + 1 } else { 0 };
            if self.streak == MASTERY_CHECKS {
                self.event(
                    root,
                    serde_json::json!({"event":"mastery_reached","generation":generation,
                    "level":self.level,"generations_on_level":generation-self.level_started,
                    "individual_weakest":weakest[0],"pooled_weakest":weakest[1]}),
                )?;
            }
            self.pending_advance = self.streak >= MASTERY_CHECKS
                && generation.saturating_sub(self.level_started) >= MIN_LEVEL_GENERATIONS
                && self.world.actions() < self.limit;
        }
        let at_ceiling = self.world.actions() >= self.limit;
        self.summary = format!(
            "Capabilities: level={}; actions={}/{}; current individual={:.0}%; pooled={:.0}%; weakest retained={:.0}%; scaffold edges={}; mastery checks={}/{}{}",
            self.level,
            self.world.actions(),
            self.limit,
            current[0] * 100.0,
            current[1] * 100.0,
            weakest[0] * 100.0,
            self.scaffold.edges.len(),
            self.streak.min(MASTERY_CHECKS),
            MASTERY_CHECKS,
            if at_ceiling {
                "; resource ceiling reached"
            } else {
                ""
            }
        );
        self.history.push_back(Point {
            generation,
            level: self.level,
            individual: current[0],
            pooled: current[1],
        });
        if self.history.len() > 400 {
            self.history.pop_front();
        }
        self.write_monitor(root, generation, &table)?;
        let mut levels = csv(
            root,
            "capability_levels.csv",
            "generation,level,actions,phase,current_individual,current_pooled,weakest_individual,weakest_pooled,mastery_streak,generations_on_level,at_ceiling",
        )?;
        writeln!(
            levels,
            "{generation},{},{},{phase},{:.4},{:.4},{:.4},{:.4},{},{},{}",
            self.level,
            self.world.actions(),
            current[0],
            current[1],
            weakest[0],
            weakest[1],
            self.streak,
            generation - self.level_started,
            at_ceiling
        )?;
        Ok(())
    }

    fn write_monitor(&self, root: &Path, generation: usize, table: &str) -> io::Result<()> {
        let first_gen = self.history.front().map_or(0, |point| point.generation);
        let x = |g: usize| {
            45.0 + 700.0 * g.saturating_sub(first_gen) as f64
                / (generation - first_gen).max(1) as f64
        };
        let points = |pooled: bool| {
            self.history
                .iter()
                .map(|p| {
                    format!(
                        "{:.1},{:.1}",
                        x(p.generation),
                        180.0 - 150.0 * if pooled { p.pooled } else { p.individual }
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        let mut markers = String::new();
        let mut previous_level = self.history.front().map_or(0, |p| p.level);
        for point in &self.history {
            if point.level != previous_level {
                markers.push_str(&format!("<path d='M{0:.1} 25V180' stroke='#64748b' stroke-dasharray='4'/><text x='{0:.1}' y='20' fill='white'>L{1}</text>",x(point.generation),point.level));
                previous_level = point.level;
            }
        }
        let count = self.teaching.recent.len();
        let lesson_gain = self
            .teaching
            .recent
            .iter()
            .map(|(lesson, _)| lesson)
            .sum::<f64>()
            / count.max(1) as f64;
        let practice_gain = self
            .teaching
            .recent
            .iter()
            .map(|(_, practice)| practice)
            .sum::<f64>()
            / count.max(1) as f64;
        let teaching_panel = if self.teaching_enabled {
            format!(
                "<h2>Peer teaching</h2><p>{} requests · {} verified examples · {} improved / {} reduced milestone test scores.</p><p>Latest {} verified lessons: mean immediate gain <strong>{:+.2} percentage points</strong>; one random practice action on the same learner copy: <strong>{:+.2} pp</strong>. This measures local changes, not long-term causal benefit. Zero gain can still mean useful prerequisite learning.</p>",
                self.teaching.requests,
                self.teaching.accepted,
                self.teaching.improved,
                self.teaching.harmed,
                count,
                lesson_gain * 100.0,
                practice_gain * 100.0
            )
        } else {
            "<h2>Peer teaching disabled</h2><p>This run uses individual practice and inheritance only.</p>".to_string()
        };
        fs::write(
            root.join("capability_monitor.html"),
            format!(
                r##"<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="5"><title>Capability learning</title>
<style>body{{font:17px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;background:#101925;color:#e5edf6}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #405064}}svg{{width:100%;background:#182638}}p{{line-height:1.5}}.blue{{color:#61cfff}}.gold{{color:#ffce66}}.scroll{{overflow-x:auto}}</style>
<h1>Learning to act · Level {level}</h1><p>{summary}</p><p>Current goal: <strong>{goal}</strong>. <span class="blue">Individual</span> / <span class="gold">pooled evidence</span>. Vertical marks introduce harder goals; drops there are expected. Each level has its own fixed tests.</p>
<svg viewBox="0 0 790 220" role="img" aria-label="Current-goal success over generations with difficulty changes"><text x="2" y="40" fill="white">100%</text><text x="10" y="185" fill="white">0%</text><path d="M45 30V180H750" fill="none" stroke="#789"/>{markers}<polyline points="{individual}" fill="none" stroke="#61cfff" stroke-width="3"/><polyline points="{pooled}" fill="none" stroke="#ffce66" stroke-width="2"/><text x="250" y="210" fill="white">Generations {first_gen}–{generation}</text></svg>
<p>Advance when <strong>both individual and pooled success reach 90% on every milestone</strong>, sustained across three evaluations, with at least 20 generations on the level. Individual means population average across test cases, not every agent. New levels add preparation and assembly actions that depend on prior skills. Earlier milestones remain in the tests.</p>
{teaching_panel}
<div class="scroll"><table><tr><th>Knowledge</th><th>Goal</th><th>Test success</th><th>Wasted actions</th><th>Correct / all predictions</th></tr>{table}</table></div>
<p>Resource limit: {limit} actions. At the ceiling, training continues and difficulty stops; this is expandable, not infinite. Set --capability-limit N (5–31) at startup. No semantic-map settings are changed.</p>
<p>Diagnostics: capability_levels.csv records immediate before-training drops, mastery streaks and time on each level; capability_events.jsonl records introductions, mastery and retention drops. capability_metrics.csv keeps all goal results, capability_agents.csv follows lineages, capability_retention.csv records every agent/milestone score and evidence coverage, and capability_episodes.jsonl records attempts and interventions. capability_teaching.csv and capability_teaching.jsonl record directed lessons and paired diagnostics; capability_teaching_totals.csv records per-generation counts.</p>
<p>The supplied planner combines learned additive action rules. New primitive actions and the curriculum are supplied by the environment. Pooled evidence is a bounded aggregate comparison. Actual peer teaching sends one witnessed transition and the learner replays it; selection targets disagreements and unknown outcomes. This teaching policy is supplied, not learned. Untrained abstains. Tests never teach or reward; sampled training interventions can overlap test states. The monitor's prediction score includes unknowns as incorrect. Population results include selection and inherited experience.</p>
<p>Refreshes every five seconds. Knowledge and progression are run-local. Initial training: key/door to generation 20, bridge to 40, then the current goal. The chart keeps the latest 400 points; CSV files keep the full history.</p></html>"##,
                level = self.level,
                summary = self.summary,
                goal = action_name(self.world.actions() - 1),
                individual = points(false),
                pooled = points(true),
                limit = self.limit
            ),
        )
    }
}

fn csv(root: &Path, name: &str, header: &str) -> io::Result<fs::File> {
    let path = root.join(name);
    let new = !path.exists();
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if new {
        writeln!(file, "{header}")?;
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fully_learn(world: &World) -> CapabilityMemory {
        let mut memory = CapabilityMemory::default();
        for action in 0..world.actions() {
            let required = world.prerequisites[action];
            memory.observe(required, action, world.step(required, action));
            for bit in 0..world.actions() {
                let state = required & !(1 << bit);
                memory.observe(state, action, world.step(state, action));
            }
        }
        memory
    }
    #[test]
    fn outcomes_teach_rules_and_failure_revises_hypothesis() {
        let world = World::default();
        let mut memory = CapabilityMemory::default();
        memory.observe(1, 1, 3);
        assert_eq!(memory.predict(0, 1), Some(2));
        memory.observe(0, 1, 0);
        assert_eq!(memory.predict(0, 1), Some(0));
        let memory = fully_learn(&world);
        let score = evaluate(&memory, &world, 16);
        assert_eq!(score.solved, score.trials);
        assert_eq!(score.correct, score.slots);
        assert_eq!(score.wasted, 0);
    }
    #[test]
    fn evaluation_is_read_only_and_new_goal_needs_new_learning() {
        let mut world = World::default();
        let memory = fully_learn(&world);
        let before = memory.clone();
        world.advance(&mut Rng::new(3), 21);
        assert_eq!(evaluate(&memory, &world, world.goal()).solved, 0);
        assert_eq!(evaluate(&memory, &world, 16).rate(), 1.0);
        assert_eq!(memory, before);
        assert_eq!(
            evaluate(&fully_learn(&world), &world, world.goal()).rate(),
            1.0
        );
    }
    #[test]
    fn expanding_world_and_planner_remain_bounded_and_solvable() {
        let mut world = World::default();
        let mut rng = Rng::new(8);
        while world.actions() < MAX_ACTION_LIMIT {
            world.advance(&mut rng, MAX_ACTION_LIMIT);
        }
        assert_eq!(world.actions(), MAX_ACTION_LIMIT);
        world.advance(&mut rng, MAX_ACTION_LIMIT);
        let memory = fully_learn(&world);
        let plan = memory.plan(0, world.goal()).unwrap();
        assert_eq!(plan.len(), world.actions());
        let state = plan
            .iter()
            .fold(0, |state, action| world.step(state, *action));
        assert_eq!(state & world.goal(), world.goal());
        assert!(memory.observations_count() <= MAX_ACTION_LIMIT * EVIDENCE_PER_ACTION);
    }
    #[test]
    fn useful_peer_example_teaches_missing_action_without_copying_teacher() {
        let world = World::default();
        let mut rng = Rng::new(4);
        let mut agents = vec![Agent::new(0, &mut rng), Agent::new(1, &mut rng)];
        agents[0].capabilities = fully_learn(&world);
        agents[1].capabilities = fully_learn(&world);
        agents[1].capabilities.observations.remove(&2);
        agents[1].capabilities.rules.remove(&2);
        let teacher_before = agents[0].capabilities.clone();
        let before = evaluate(&agents[1].capabilities, &world, 16);
        let lesson = propose_lesson(&agents, 1, 5, 0).unwrap();
        assert_eq!(lesson.teacher, 0);
        assert_eq!(lesson.action, 2);
        assert!(accept_lesson(&mut agents[1].capabilities, &world, lesson));
        let after = evaluate(&agents[1].capabilities, &world, 16);
        assert!(after.solved > before.solved);
        assert_eq!(after.rate(), 1.0);
        assert_eq!(agents[0].capabilities, teacher_before);
        assert_eq!(agents[1].capabilities.observations[&2].len(), 1);
        assert!(propose_lesson(&agents, 1, 5, 0).is_none());
        assert!(propose_lesson(&agents[..1], 0, 5, 0).is_none());
    }

    #[test]
    fn inaccurate_testimony_is_rejected_without_overwriting_memory() {
        let world = World::default();
        let mut learner = fully_learn(&world);
        let before = learner.clone();
        let false_lesson = Lesson {
            teacher: 0,
            state: 0,
            action: 1,
            outcome: 2,
        };
        assert!(!accept_lesson(&mut learner, &world, false_lesson));
        assert_eq!(learner, before);
    }

    #[test]
    fn teaching_alone_spreads_a_capability_and_logs_paired_diagnostics() {
        let root = std::env::temp_dir().join(format!("mas-teaching-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut lab = CapabilityLab::new(5);
        let mut rng = Rng::new(5);
        let mut agents: Vec<_> = (0..6).map(|id| Agent::new(id, &mut rng)).collect();
        for agent in &mut agents {
            agent.capabilities = fully_learn(&lab.world);
            if agent.id != 0 {
                agent.capabilities.observations.remove(&2);
                agent.capabilities.rules.remove(&2);
            }
        }
        let before = agents.clone();
        for generation in 41..61 {
            lab.teach(&mut agents, generation, 88, &root).unwrap();
        }
        assert_eq!(lab.teaching.accepted, 5);
        assert_eq!(lab.teaching.improved, 5);
        for agent in &agents {
            assert_eq!(evaluate(&agent.capabilities, &lab.world, 16).rate(), 1.0);
            // Diagnostic results never award selection fitness.
            assert_eq!(agent.fitness, before[agent.id].fitness);
        }
        assert_eq!(agents[0].capabilities, before[0].capabilities);
        let log = fs::read_to_string(root.join("capability_teaching.csv")).unwrap();
        assert_eq!(log.lines().count(), 6);
        assert!(log.contains("random_practice_solved"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disabled_teaching_creates_no_exchange_artifacts() {
        let root =
            std::env::temp_dir().join(format!("mas-no-teaching-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut lab = CapabilityLab::new(5).with_teaching(false);
        let mut rng = Rng::new(12);
        let mut agents = vec![Agent::new(0, &mut rng), Agent::new(1, &mut rng)];
        lab.tick(&mut agents, 1, 123, &root).unwrap();
        assert_eq!(lab.teaching.requests, 0);
        assert!(!root.join("capability_teaching.csv").exists());
        assert!(
            fs::read_to_string(root.join("capability_monitor.html"))
                .unwrap()
                .contains("Peer teaching disabled")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pooling_combines_complementary_evidence() {
        let world = World::default();
        let all = fully_learn(&world);
        let mut a = CapabilityMemory::default();
        let mut b = CapabilityMemory::default();
        for (action, samples) in all.observations {
            for (state, next) in samples {
                if action < 2 {
                    a.observe(state, action, next);
                } else {
                    b.observe(state, action, next);
                }
            }
        }
        assert_eq!(evaluate(&a, &world, 16).solved, 0);
        a.merge(&b);
        assert_eq!(evaluate(&a, &world, 16).rate(), 1.0);
    }
    #[test]
    fn threshold_is_inclusive_streak_resets_and_ceiling_is_respected() {
        let root = std::env::temp_dir().join(format!("mas-threshold-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut lab = CapabilityLab::new(7);
        lab.level_started = 35;
        let mut rng = Rng::new(19);
        let mut agents: Vec<_> = (0..10).map(|id| Agent::new(id, &mut rng)).collect();
        for agent in agents.iter_mut().take(9) {
            agent.capabilities = fully_learn(&lab.world);
        }
        for generation in [40, 45, 50] {
            lab.report(&agents, generation, &root, "after_training")
                .unwrap();
        }
        assert_eq!(lab.streak, 3); // Exactly 90% qualifies, but dwell is only 15.
        assert!(!lab.pending_advance);
        lab.report(&agents, 55, &root, "after_training").unwrap();
        assert!(lab.pending_advance);
        agents[0].capabilities = CapabilityMemory::default();
        lab.report(&agents, 60, &root, "after_training").unwrap();
        assert_eq!(lab.streak, 0);
        assert!(!lab.pending_advance);
        lab.limit = 5;
        agents[0].capabilities = fully_learn(&lab.world);
        for generation in [65, 70, 75] {
            lab.report(&agents, generation, &root, "after_training")
                .unwrap();
        }
        assert_eq!(lab.streak, 3);
        assert!(!lab.pending_advance);
        assert!(lab.summary.contains("resource ceiling reached"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn retention_cases_are_stable_and_memory_is_bounded() {
        let mut world = World::default();
        let starts = test_starts(&world, 16);
        world.advance(&mut Rng::new(9), 8);
        world.advance(&mut Rng::new(9), 8);
        assert_eq!(world.actions(), 8); // Odd remaining capacity is safe.
        assert_eq!(test_starts(&world, 16), starts);
        let mut memory = CapabilityMemory::default();
        for state in 0..256 {
            memory.observe(state, 0, state | 1);
        }
        assert_eq!(memory.observations_count(), EVIDENCE_PER_ACTION);
        assert_eq!(memory.predict(0, 0), Some(1));
    }

    #[test]
    fn mastery_requires_retention_streak_and_dwell_then_advances_next_tick() {
        let root =
            std::env::temp_dir().join(format!("mas-progression-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut lab = CapabilityLab::new(7);
        let mut agents = vec![Agent::new(0, &mut Rng::new(1))];
        agents[0].capabilities = fully_learn(&lab.world);
        lab.report(&agents, 40, &root, "after_training").unwrap();
        lab.report(&agents, 45, &root, "after_training").unwrap();
        assert!(!lab.pending_advance);
        lab.report(&agents, 50, &root, "after_training").unwrap();
        assert!(lab.pending_advance);
        lab.tick(&mut agents, 51, 1, &root).unwrap();
        assert_eq!(lab.level, 1);
        assert_eq!(lab.world.actions(), 7);
        assert_eq!(lab.streak, 0);
        assert!(!lab.pending_advance);
        let events = fs::read_to_string(root.join("capability_events.jsonl")).unwrap();
        assert!(events.contains("difficulty_increased"));
        let csv = fs::read_to_string(root.join("capability_levels.csv")).unwrap();
        assert!(csv.contains("51,1,7,before_training,0.0000,0.0000"));
        fs::remove_dir_all(root).unwrap();
    }
}

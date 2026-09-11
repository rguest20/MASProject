//! Shared investigation agenda: topics come from observations and are selected
//! using novelty, disagreement, learning progress and cost. The controller is
//! supplied; it does not imply an independently invented motivation system.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

use crate::learning::{CONTEXT_LIMIT, EvidenceMemory, Observation, Prediction};
use crate::model::{Agent, Rng, two_agents};
use crate::reading::ReadingBridge;

mod memory;

const AGENDA_LIMIT: usize = 96;
const SOURCE_WINDOW: usize = 8192;
const HISTORY_LIMIT: usize = 100;
const PROJECTS_PER_GENERATION: usize = 3;
// A context needs several independently sourced outcomes before its loss or
// distribution tells us much.  This is a general evidence budget, not a
// language-specific rule.
const FOCUS_OBSERVATIONS: usize = 6;
const COMMUNITY_MEMORY_LIMIT: usize = SOURCE_WINDOW;
const COMMUNITY_CONTEXT_LIMIT: usize = 32;

#[derive(Clone, Debug, Default)]
struct Question {
    attempts: usize,
    observations: usize,
    loss: f64,
    progress: f64,
    last_visit: usize,
}

impl Question {
    fn score(&self, disagreement: f64, generation: usize) -> f64 {
        let novelty = 1.0 / (self.observations as f64 + 1.0).sqrt();
        let stalled = if self.observations >= 8 && self.progress < 0.01 {
            0.15
        } else {
            1.0
        };
        let success = (self.observations + 1) as f64 / (self.attempts + 1) as f64;
        let revisit = (generation.saturating_sub(self.last_visit) as f64 / 50.0).min(0.2);
        success * stalled * (novelty + disagreement + 3.0 * self.progress.max(0.0)) + revisit
    }

    fn record(&mut self, loss: f64) {
        if self.observations == 0 {
            self.loss = loss;
        }
        let gain = self.loss - loss;
        self.progress = 0.8 * self.progress + 0.2 * gain;
        self.loss = 0.8 * self.loss + 0.2 * loss;
        self.observations += 1;
    }
}

#[derive(Clone, Debug, Default)]
struct Metrics {
    observations: usize,
    reading_sources: usize,
    lessons: usize,
    peer_tests: usize,
    extended: usize,
    // Population slot participation is bounded; event logs retain lineage IDs.
    contributors: BTreeSet<usize>,
    training: VecDeque<(bool, f64)>,
    transfer: VecDeque<(f64, f64)>,
}

#[derive(Debug, Default)]
struct ProbeMetrics {
    generation: usize,
    predictions: usize,
    exact: usize,
    loss: f64,
    unigram_loss: f64,
    trust: [f64; CONTEXT_LIMIT + 1],
    consolidated: usize,
}

#[derive(Debug)]
pub struct InquiryLab {
    agenda: BTreeMap<Vec<String>, Question>,
    sources: VecDeque<u64>,
    source_set: BTreeSet<u64>,
    // Community-scale, source-backed predictive evidence. This is a prior,
    // not a replacement for a learner's own observations or a truth store.
    community: EvidenceMemory,
    role_families: BTreeMap<(u8, u8, u8), BTreeMap<String, u32>>,
    role_members: BTreeMap<(u8, u8, u8), BTreeSet<String>>,
    token_roles: BTreeMap<String, (u8, u8, u8)>,
    messages: VecDeque<String>,
    metrics: Metrics,
    evaluation_rng: Option<Rng>,
    commitments: BTreeMap<usize, (u64, Vec<String>, usize)>,
    probes: Vec<Observation>,
    probe_metrics: ProbeMetrics,
    elapsed_generations: usize,
    run_base: usize,
    corpus_fingerprint: u64,
    memory_status: String,
}

impl Default for InquiryLab {
    fn default() -> Self {
        Self {
            agenda: BTreeMap::new(),
            sources: VecDeque::new(),
            source_set: BTreeSet::new(),
            community: EvidenceMemory::with_limits(
                COMMUNITY_MEMORY_LIMIT,
                // The community source window is a total bound. Unlike an
                // agent, it does not need a second consolidated tier: its
                // retained source IDs already provide cross-run continuity.
                0,
                COMMUNITY_CONTEXT_LIMIT,
            ),
            role_families: BTreeMap::new(),
            role_members: BTreeMap::new(),
            token_roles: BTreeMap::new(),
            messages: VecDeque::new(),
            metrics: Metrics::default(),
            evaluation_rng: None,
            commitments: BTreeMap::new(),
            probes: Vec::new(),
            probe_metrics: ProbeMetrics::default(),
            elapsed_generations: 0,
            run_base: 0,
            corpus_fingerprint: 0,
            memory_status: String::new(),
        }
    }
}

impl InquiryLab {
    /// Shared story time gives a small rotating reading group direct evidence
    /// in the same model used for investigation and communication. Other agents
    /// can acquire these episodes later through source-preserving lessons.
    pub fn read(
        &mut self,
        agents: &mut [Agent],
        passage: &[Vec<String>],
        reading: &ReadingBridge,
        generation: usize,
        rng: &mut Rng,
        root: &Path,
    ) -> io::Result<()> {
        if agents.is_empty() {
            return Ok(());
        }
        let mut examples: Vec<_> = passage
            .iter()
            .flat_map(|words| ReadingBridge::inquiry_examples(words))
            .collect();
        let limit = examples.len().min(12);
        for i in 0..limit {
            let other = i + rng.index(examples.len() - i);
            examples.swap(i, other);
        }
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("inquiry_reading.jsonl"))?;
        for observation in examples.into_iter().take(limit) {
            if !self.source_set.contains(&observation.source) {
                self.remember_source(observation.source);
                self.metrics.reading_sources += 1;
            }
            self.observe_community(&observation, reading);
            let mut readers = Vec::new();
            for offset in 0..agents.len().min(3) {
                let index = (generation + offset) % agents.len();
                if Self::adopt(&mut agents[index], &observation, rng) {
                    readers.push(agents[index].lineage_id);
                }
            }
            // Reading is evidence for the agents, not an investigation in its
            // own right.  Do not put every token seen while reading on the
            // small inquiry agenda: that made untried candidates crowd out
            // questions the community had actually begun to investigate.
            writeln!(
                log,
                "{}",
                serde_json::json!({"generation": generation, "source": observation.source,
                "context": observation.context, "outcome": observation.outcome, "readers": readers})
            )?;
        }
        Ok(())
    }

    pub fn tick(
        &mut self,
        agents: &mut [Agent],
        generation: usize,
        rng: &mut Rng,
        reading: &mut ReadingBridge,
        root: &Path,
    ) -> io::Result<()> {
        if agents.len() < 2 {
            return Ok(());
        }
        self.elapsed_generations = self.run_base.saturating_add(generation);
        if self.probes.is_empty() {
            let evaluation_rng = self.evaluation_rng.get_or_insert_with(|| Rng::new(901_337));
            let mut sources = BTreeSet::new();
            for _ in 0..32 {
                if let Some(probe) =
                    reading.inquiry_observation(&[], &sources, true, evaluation_rng)
                {
                    sources.insert(probe.source);
                    self.probes.push(probe);
                }
            }
        }
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("inquiry_log.jsonl"))?;
        for turn in 0..PROJECTS_PER_GENERATION {
            // Rotate observation opportunities through all population slots.
            let scout_index = (generation * PROJECTS_PER_GENERATION + turn) % agents.len();
            if agents[scout_index].energy < 20.0 {
                continue;
            }
            let peer_index = rng.distinct_index(agents.len(), scout_index);
            let exploration_rate = (0.03
                + 0.08 * agents[scout_index].traits.curiosity
                + 0.04 * agents[scout_index].state.curiosity)
                .clamp(0.03, 0.15);
            // Keep discovery alive, while reserving most turns for gathering
            // enough independent evidence about contexts already encountered.
            // With a large vocabulary, treating every turn as a fresh topic
            // leaves every estimate based on a single example.
            let seek_new = self.agenda.is_empty()
                || (turn == 0 && (rng.unit() < 0.70 || rng.unit() < exploration_rate));
            let continuing = self
                .commitments
                .get(&agents[scout_index].id)
                .filter(|(lineage, context, turns)| {
                    *lineage == agents[scout_index].lineage_id
                        && *turns < 6
                        && self.agenda.get(context).is_some_and(|question| {
                            question.observations < FOCUS_OBSERVATIONS || question.progress > 0.01
                        })
                })
                .map(|(_, context, _)| context.clone());
            let mut reason;
            let context = if let Some(context) = continuing {
                reason = "continue current investigation";
                context
            } else if seek_new {
                reason = "new source";
                self.commitments.remove(&agents[scout_index].id);
                Vec::new()
            } else {
                reason = "uncertainty, disagreement and progress";
                self.commitments.remove(&agents[scout_index].id);
                self.choose_question(
                    &agents[scout_index],
                    &agents[peer_index],
                    self.elapsed_generations,
                    rng,
                )
            };
            if let Some(question) = self.agenda.get_mut(&context) {
                question.attempts += 1;
                question.last_visit = self.elapsed_generations;
            }
            let observation = reading.inquiry_observation(&context, &self.source_set, false, rng);
            let observation = if observation.is_none() && !context.is_empty() {
                reason = "topic exhausted; explore another source";
                reading.inquiry_observation(&[], &self.source_set, false, rng)
            } else {
                observation
            };
            let Some(observation) = observation else {
                continue;
            };
            self.remember_source(observation.source);
            self.observe_community(&observation, reading);
            let context = if !context.is_empty() && observation.context.ends_with(&context) {
                context
            } else {
                vec![
                    observation
                        .context
                        .last()
                        .expect("nonempty context")
                        .clone(),
                ]
            };
            let (scout, learner) = two_agents(agents, scout_index, peer_index);
            let project =
                self.commitments
                    .entry(scout.id)
                    .or_insert((scout.lineage_id, context.clone(), 0));
            if project.1 != context {
                *project = (scout.lineage_id, context.clone(), 0);
            }
            project.2 += 1;
            let before = self.predict(scout, &observation.context);
            let loss = before.loss(&observation.outcome);
            let correct = before.best() == Some(observation.outcome.as_str());
            let added = Self::adopt(scout, &observation, rng);
            if added {
                self.metrics.observations += 1;
                self.metrics.contributors.insert(scout.id);
                push_bounded(&mut self.metrics.training, (correct, loss), HISTORY_LIMIT);
                let question = self.agenda.entry(context.clone()).or_default();
                if question.attempts <= question.observations {
                    question.attempts += 1;
                }
                question.last_visit = self.elapsed_generations;
                question.record(loss);
                scout.state_event(if correct {
                    "learning_success"
                } else {
                    "curiosity_boost"
                });
                scout.energy = (scout.energy - 0.15).max(0.0);
                // Reward predictive usefulness before learning this example.
                scout.fitness += 0.02 * (1.0 - loss).clamp(0.0, 1.0);
            }
            self.extend_agenda(&observation, self.elapsed_generations);

            // Share a source observation, never the teacher's probability table.
            let lesson = scout.learning.lesson_for(&learner.learning, &context);
            let mut transfer = None;
            if let Some(lesson) = lesson {
                // Evaluate the same listener before/after one lesson on an
                // independent source. Test results never affect agent updates,
                // agenda scores, social rewards or evolution fitness.
                let evaluation_rng = self.evaluation_rng.get_or_insert_with(|| Rng::new(901_337));
                let test =
                    reading.inquiry_observation(&context, &BTreeSet::new(), true, evaluation_rng);
                let baseline = test
                    .as_ref()
                    .map(|test| self.predict(learner, &test.context));
                if Self::adopt(learner, &lesson, rng) {
                    self.metrics.lessons += 1;
                    learner.energy = (learner.energy - 0.10).max(0.0);
                    if let (Some(test), Some(baseline)) = (test, baseline) {
                        let after = self.predict(learner, &test.context);
                        let pair = (baseline.loss(&test.outcome), after.loss(&test.outcome));
                        push_bounded(&mut self.metrics.transfer, pair, HISTORY_LIMIT);
                        self.metrics.peer_tests += 1;
                        transfer = Some(
                            serde_json::json!({"source": test.source, "before_loss": pair.0, "after_loss": pair.1}),
                        );
                    }
                    self.message(format!(
                        "A{} → A{}: source {:x}: {} → {}",
                        scout.id,
                        learner.id,
                        lesson.source,
                        lesson.context.join(" "),
                        lesson.outcome
                    ));
                }
            }
            let proposal_words = learner.learning.compose(&context, 4);
            let proposed = proposal_words.join(" ");
            // The learning event remains one token. For display, recover a
            // short source continuation and award prefix credit to the phrase
            // only; this extra source text is never learned or rewarded.
            let generated = proposal_words
                .strip_prefix(context.as_slice())
                .unwrap_or_default();
            let expected = reading.inquiry_source_continuation(observation.source, 4);
            let checked = generated.len().min(expected.len());
            let prefix_match = generated
                .iter()
                .zip(&expected)
                .take_while(|(actual, target)| actual == target)
                .count();
            let proposal_first_match = checked > 0 && prefix_match > 0;
            let proposal_assessment = if checked == 0 {
                "no proposed next token; unscored".to_string()
            } else {
                format!("source-prefix credit {prefix_match}/{checked}")
            };
            self.message(format!(
                "A{} investigates source {:x}: {} → {}; asks [{} → ?] ({reason}); scout before evidence predicted {:?} ({}) · A{} proposes: {} [{}]",
                scout.id,
                observation.source,
                observation.context.join(" "),
                observation.outcome,
                context.join(" "),
                before.best(),
                if correct { "exact" } else { "not exact" },
                learner.id,
                if proposed.is_empty() { "—" } else { &proposed },
                proposal_assessment,
            ));
            writeln!(
                log,
                "{}",
                serde_json::json!({
                    "generation": generation, "scout": scout.lineage_id, "learner": learner.lineage_id,
                    "question": context, "reason": reason, "source": observation.source,
                    "context": observation.context, "outcome": observation.outcome,
                    "pre_observation_prediction": before.best(), "pre_observation_exact": correct,
                    "pre_observation_loss": loss, "new_to_scout": added, "peer_test": transfer,
                    "proposal": proposed, "proposal_generated_tokens": generated,
                    "proposal_first_token_matches_source": proposal_first_match,
                    "proposal_prefix_credit": {"matched": prefix_match, "checked": checked},
                    "proposal_prefix_display_only": true
                })
            )?;
        }
        if generation == 1 || generation.is_multiple_of(20) {
            self.evaluate(agents, generation);
        }
        self.write_monitor(generation, root)
    }

    fn evaluate(&mut self, agents: &[Agent], generation: usize) {
        let mut metrics = ProbeMetrics {
            generation,
            ..ProbeMetrics::default()
        };
        for agent in agents {
            for (total, weight) in metrics.trust.iter_mut().zip(agent.learning.trust_weights()) {
                *total += weight / agents.len().max(1) as f64;
            }
            metrics.consolidated += agent.learning.consolidated_len();
            let baseline = self.predict(agent, &[]);
            for probe in &self.probes {
                let prediction = self.predict(agent, &probe.context);
                metrics.predictions += 1;
                metrics.exact += usize::from(prediction.best() == Some(probe.outcome.as_str()));
                metrics.loss += prediction.loss(&probe.outcome);
                metrics.unigram_loss += baseline.loss(&probe.outcome);
            }
        }
        self.probe_metrics = metrics;
    }

    fn adopt(agent: &mut Agent, observation: &Observation, rng: &mut Rng) -> bool {
        if !agent.learning.observe(observation) {
            return false;
        }
        let mut words = observation.context.clone();
        words.push(observation.outcome.clone());
        agent.observe(&words, 0.025, rng);
        // Exposure supports a relation; a different sampled outcome does not
        // imply that another possible continuation is universally wrong.
        true
    }

    /// Blend private experience with shared, independently observed episodes.
    /// The shared estimate gains weight only when its matching suffix has
    /// repeated source evidence; no assertion or peer-generated text enters it.
    fn predict(&self, agent: &Agent, context: &[String]) -> Prediction {
        let private = agent.learning.predict(context);
        let shared = self.community.predict(context);
        let support = (1..=context.len().min(CONTEXT_LIMIT))
            .rev()
            .map(|length| {
                self.community
                    .context_evidence(&context[context.len() - length..])
            })
            .find(|evidence| *evidence > 0)
            .unwrap_or(0);
        let shared_weight = 0.65 * support as f64 / (support + 6) as f64;
        let blended = private.blend(&shared, shared_weight);
        let Some(signature) = context.last().and_then(|token| self.token_roles.get(token)) else {
            return blended;
        };
        let Some(outcomes) = self.role_families.get(signature) else {
            return blended;
        };
        let members = self.role_members.get(signature).map_or(0, BTreeSet::len);
        // A broad family is indistinguishable from a generic frequency prior
        // and expensive to materialise on every prediction. Keep transfer
        // local to compact recurring distributions.
        if outcomes.len() > 128 {
            return blended;
        }
        let total = outcomes.values().sum::<u32>() as usize;
        if members < 3 || total < 8 {
            return blended;
        }
        let family = Prediction {
            probabilities: outcomes
                .iter()
                .map(|(word, count)| (word.clone(), *count as f64 / total as f64))
                .collect(),
            evidence: total,
        };
        let weight = 0.35 * total as f64 / (total + 16) as f64;
        blended.blend(&family, weight)
    }

    fn observe_community(&mut self, observation: &Observation, reading: &ReadingBridge) {
        self.community.observe(observation);
        let Some(token) = observation.context.last() else {
            return;
        };
        let Some(signature) = reading.role_signature(token) else {
            return;
        };
        self.token_roles.insert(token.clone(), signature);
        self.role_members
            .entry(signature)
            .or_default()
            .insert(token.clone());
        *self
            .role_families
            .entry(signature)
            .or_default()
            .entry(observation.outcome.clone())
            .or_default() += 1;
    }

    fn choose_question(
        &self,
        scout: &Agent,
        peer: &Agent,
        generation: usize,
        rng: &mut Rng,
    ) -> Vec<String> {
        let mut selected = Vec::new();
        let mut best = f64::NEG_INFINITY;
        // Prefer an unfinished question with some evidence over a blank
        // candidate. This makes sampling effort accumulate until a context's
        // predicted outcome distribution can be estimated.
        let has_unfinished = self.agenda.values().any(|question| {
            question.observations > 0 && question.observations < FOCUS_OBSERVATIONS
        });
        for (context, question) in &self.agenda {
            if has_unfinished
                && (question.observations == 0 || question.observations >= FOCUS_OBSERVATIONS)
            {
                continue;
            }
            let a = self.predict(scout, context);
            let b = self.predict(peer, context);
            let remaining = (FOCUS_OBSERVATIONS.saturating_sub(question.observations) as f64
                / FOCUS_OBSERVATIONS as f64)
                .max(0.0);
            let score = question.score(a.disagreement(&b), generation)
                + 0.75 * remaining
                + rng.unit() * 0.05;
            if score > best {
                best = score;
                selected = context.clone();
            }
        }
        selected
    }

    pub fn propose(&self, agent: &Agent, rng: &mut Rng) -> Vec<String> {
        let contexts: Vec<_> = self
            .agenda
            .keys()
            .filter(|context| agent.learning.context_evidence(context) >= 2)
            .collect();
        let Some(context) = contexts.get(rng.index(contexts.len())) else {
            return Vec::new();
        };
        agent.learning.compose(context, 4)
    }

    fn extend_agenda(&mut self, observation: &Observation, generation: usize) {
        let base = vec![
            observation
                .context
                .last()
                .expect("nonempty context")
                .clone(),
        ];
        let base_question = self.agenda.entry(base).or_default();
        // Recurring unpredictable contexts suggest trying more preceding
        // tokens. All contexts are observed; no grammatical categories supplied.
        let extend = base_question.observations >= 3 && base_question.loss > 0.35;
        if extend {
            for length in 2..=observation.context.len().min(CONTEXT_LIMIT) {
                let context = observation.context[observation.context.len() - length..].to_vec();
                if !self.agenda.contains_key(&context) {
                    self.agenda.insert(
                        context,
                        Question {
                            last_visit: generation,
                            ..Question::default()
                        },
                    );
                    self.metrics.extended += 1;
                }
            }
        }
        while self.agenda.len() > AGENDA_LIMIT {
            let key = self
                .agenda
                .iter()
                .min_by(|a, b| {
                    // A blank candidate is cheap to rediscover.  An active
                    // question is expensive evidence already gathered, so do
                    // not evict it merely because novelty makes its priority
                    // lower than an untouched entry.
                    let retention = |question: &Question| {
                        if question.observations == 0 {
                            0_u8
                        } else if question.observations < FOCUS_OBSERVATIONS {
                            2
                        } else {
                            1
                        }
                    };
                    retention(a.1).cmp(&retention(b.1)).then_with(|| {
                        a.1.score(0.0, generation)
                            .total_cmp(&b.1.score(0.0, generation))
                    })
                })
                .map(|(context, _)| context.clone())
                .expect("nonempty agenda");
            self.agenda.remove(&key);
        }
    }

    fn remember_source(&mut self, source: u64) {
        self.sources.push_back(source);
        self.source_set.insert(source);
        if self.sources.len() > SOURCE_WINDOW {
            self.source_set
                .remove(&self.sources.pop_front().expect("over capacity"));
        }
    }

    fn message(&mut self, message: String) {
        push_bounded(&mut self.messages, message, 10);
    }

    fn write_monitor(&self, generation: usize, root: &Path) -> io::Result<()> {
        let mut html = String::from(
            "<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='5'><title>Community inquiry</title><style>body{background:#101923;color:#e8eef5;font:17px system-ui;margin:36px;max-width:1200px}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #34485e}th{color:#7dd3fc}li{margin:10px 0}</style><h1>Community inquiry</h1>",
        );
        let train_n = self.metrics.training.len();
        let transfer_n = self.metrics.transfer.len();
        let mean_loss =
            self.metrics.training.iter().map(|x| x.1).sum::<f64>() / train_n.max(1) as f64;
        let mean_gain =
            self.metrics.transfer.iter().map(|x| x.0 - x.1).sum::<f64>() / transfer_n.max(1) as f64;
        let probe_n = self.probe_metrics.predictions;
        let probe_loss = self.probe_metrics.loss / probe_n.max(1) as f64;
        let baseline_loss = self.probe_metrics.unigram_loss / probe_n.max(1) as f64;
        let probe_exact = self.probe_metrics.exact as f64 / probe_n.max(1) as f64;
        html.push_str(&format!("<p>Generation {generation} · {} source observations · {} delivered lessons · {} participating population slots · {} longer-context questions created.</p>", self.metrics.observations, self.metrics.lessons, self.metrics.contributors.len(), self.metrics.extended));
        html.push_str(&format!("<p>Ordinary reading supplied {} additional source episodes to rotating groups. The shared source-backed prior currently retains {} episodes; it is mixed into a private prediction only when the matching context has repeated shared evidence.</p>", self.metrics.reading_sources, self.community.observations().count()));
        html.push_str(&format!("<p><strong>Learned predictor trust (generation {}):</strong> frequencies {:.1}%; up to one context token {:.1}%; two {:.1}%; three {:.1}%. These are population-average mixture weights, updated from training predictions made before the answer arrived. Unsupported longer contexts fall back to shorter ones. {} source references are consolidated across the population; retelling a retained source does not strengthen it.</p>", self.probe_metrics.generation, self.probe_metrics.trust[0]*100.0,self.probe_metrics.trust[1]*100.0,self.probe_metrics.trust[2]*100.0,self.probe_metrics.trust[3]*100.0,self.probe_metrics.consolidated));
        html.push_str(&format!("<p>Inquiry memory: {}. Lifetime inquiry generations: {}. Checkpoints preserve each learner's sourced experience and active questions every 20 generations and on normal exit; an interrupted run can lose up to 19 completed generations. This is not a full simulation checkpoint.</p>", escape(&self.memory_status), self.elapsed_generations));
        if probe_n > 0 {
            html.push_str(&format!("<p><strong>Fixed inquiry transfer test, generation {}:</strong> {:.1}% exact; context loss {:.4}; ignoring-context loss {:.4}; context benefit {:+.4}. {} agent predictions across {} fixed corpus examples. All current agents participate; these examples never train inquiry memory. Positive context benefit means using context helps beyond learning word frequencies.</p>", self.probe_metrics.generation, probe_exact * 100.0, probe_loss, baseline_loss, baseline_loss - probe_loss, probe_n, self.probes.len()));
        } else {
            html.push_str("<p>Awaiting corpus examples for the fixed transfer test.</p>");
        }
        html.push_str(&format!("<p>Latest {train_n} observations, <strong>scout predictions made before any source lesson</strong>: <strong>{:.1}% exact</strong>; mean Brier loss <strong>{mean_loss:.3}</strong> (lower is better; unknown = 1). Brier loss gives partial credit to a non-top prediction that still assigns probability to the right next token. The later learner proposal is displayed separately with source-prefix credit: up to four source tokens are recovered solely to inspect the phrase, never added to learning, selection or rewards. Recent peer benefit: <strong>{}</strong>, across {transfer_n} paired tests ({} lifetime).</p>", 100.0 * self.metrics.training.iter().filter(|x| x.0).count() as f64 / train_n.max(1) as f64, if transfer_n == 0 { "awaiting test evidence".into() } else { format!("{mean_gain:+.4} Brier improvement") }, self.metrics.peer_tests));
        html.push_str("<p>Peer benefit compares the same listener before and after one lesson on a separate corpus example. Positive is improvement; negative is harm. Tests are withheld from this inquiry learner. This is a local teaching comparison, not a measure of general intelligence or long-term causal benefit.</p><h2>Shared questions</h2><p>Only contexts that an investigator has actually attempted appear below. Ordinary reading teaches the population but does not create a question for every token. Novel, disputed or improving questions attract work; repeatedly unproductive questions lose priority, while some turns always explore fresh sources.</p>");
        let mut ranked: Vec<_> = self
            .agenda
            .iter()
            .filter(|(_, question)| question.attempts > 0 || question.observations > 0)
            .collect();
        let pending = self.agenda.len().saturating_sub(ranked.len());
        html.push_str(&format!("<p>{} investigated contexts shown; {pending} pending longer-context candidates.</p><table><tr><th>Context → ?</th><th>Observations / attempts</th><th>Recent loss</th><th>Learning progress</th></tr>", ranked.len()));
        ranked.sort_by(|a, b| b.1.last_visit.cmp(&a.1.last_visit));
        let no_investigations = ranked.is_empty();
        for (context, question) in ranked.into_iter().take(24) {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{} / {}</td><td>{:.3}</td><td>{:.3}</td></tr>",
                escape(&context.join(" ")),
                question.observations,
                question.attempts,
                question.loss,
                question.progress
            ));
        }
        if no_investigations {
            html.push_str("<tr><td colspan='4'>No inquiry has completed yet.</td></tr>");
        }
        html.push_str("</table><h2>Recent exchanges</h2><p>Each exchange separately labels the scout's prediction before evidence and the learner's later proposal. Source-prefix credit compares the generated continuation with up to four tokens from its original source for display only; those extra tokens never teach, select or reward an agent. Proposals remain hypotheses.</p><ul>");
        for message in &self.messages {
            html.push_str(&format!("<li>{}</li>", escape(message)));
        }
        html.push_str("</ul><p>Limits: 96 active questions, 384 recent plus at most 384 consolidated source observations per agent (at most eight consolidated examples per final context token), 8,192 recent community source IDs, 3 investigations per generation, contexts of at most 3 tokens. Inquiry evidence retains original source IDs through teaching, inheritance and checkpoint restoration. Details: inquiry_log.jsonl; inquiry_restart.json records the transfer test before this run's first learning update.</p>");
        fs::write(root.join("inquiry_monitor.html"), html)?;
        let path = root.join("inquiry_metrics.csv");
        let fresh = !path.exists();
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        if fresh {
            writeln!(
                file,
                "generation,observations,lessons,questions,extended,training_n,mean_loss,peer_test_n,peer_gain,probe_generation,probe_predictions,probe_exact,probe_loss,unigram_loss"
            )?;
        }
        writeln!(
            file,
            "{generation},{},{},{},{},{train_n},{mean_loss:.6},{transfer_n},{mean_gain:.6},{},{probe_n},{probe_exact:.6},{probe_loss:.6},{baseline_loss:.6}",
            self.metrics.observations,
            self.metrics.lessons,
            self.agenda.len(),
            self.metrics.extended,
            self.probe_metrics.generation
        )
    }
}

fn push_bounded<T>(queue: &mut VecDeque<T>, value: T, limit: usize) {
    queue.push_back(value);
    if queue.len() > limit {
        queue.pop_front();
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn settled_or_unproductive_questions_become_boring() {
        let fresh = Question::default();
        let stale = Question {
            attempts: 40,
            observations: 40,
            loss: 0.8,
            ..Question::default()
        };
        assert!(fresh.score(0.0, 1) > stale.score(0.0, 1));
        let improving = Question {
            progress: 0.3,
            ..stale.clone()
        };
        assert!(improving.score(0.0, 1) > stale.score(0.0, 1));
    }

    #[test]
    fn ambiguity_opens_longer_questions_and_agenda_remains_bounded() {
        let mut lab = InquiryLab::default();
        lab.agenda.insert(
            vec!["cue".into()],
            Question {
                observations: 3,
                loss: 0.8,
                ..Question::default()
            },
        );
        let observation = Observation {
            source: 1,
            context: vec!["previous".into(), "cue".into()],
            outcome: "next".into(),
        };
        lab.extend_agenda(&observation, 1);
        assert!(lab.agenda.contains_key(&observation.context));
        for id in 0..200 {
            lab.extend_agenda(
                &Observation {
                    source: id,
                    context: vec![format!("token{id}")],
                    outcome: format!("next{id}"),
                },
                id as usize,
            );
        }
        assert!(lab.agenda.len() <= AGENDA_LIMIT);
    }

    #[test]
    fn observed_outcomes_do_not_become_unattempted_agenda_entries() {
        let mut lab = InquiryLab::default();
        let observation = Observation {
            source: 1,
            context: vec!["before".into(), "cue".into()],
            outcome: "outcome".into(),
        };
        lab.extend_agenda(&observation, 1);
        assert!(lab.agenda.contains_key(&vec!["cue".into()]));
        assert!(!lab.agenda.contains_key(&vec!["outcome".into()]));
    }

    #[test]
    fn agenda_eviction_keeps_an_active_question_over_blank_candidates() {
        let mut lab = InquiryLab::default();
        let active = vec!["active".into()];
        lab.agenda.insert(
            active.clone(),
            Question {
                attempts: 1,
                observations: 1,
                loss: 0.9,
                ..Question::default()
            },
        );
        for index in 0..AGENDA_LIMIT {
            lab.agenda
                .insert(vec![format!("candidate-{index}")], Question::default());
        }
        lab.extend_agenda(
            &Observation {
                source: 2,
                context: vec!["new".into()],
                outcome: "outcome".into(),
            },
            1,
        );
        assert!(lab.agenda.contains_key(&active));
        assert!(lab.agenda.len() <= AGENDA_LIMIT);
    }
}

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::alignment::{AlignmentMetrics, CommunitySemanticMap};
use crate::cognition::Action;
use crate::community_memory::{self, MemoryStatus};
use crate::config::{POPULATION, RunOptions};
use crate::conversation::Conversation;
use crate::dictionary::HumanDictionary;
use crate::evolution::{EvolutionMetrics, evolve};
use crate::language::{
    DialogueIntent, DialogueMeaning, answer_request, interpret_dialogue_act, produce_dialogue_act,
    repair_dialogue,
};
use crate::lexicon::CommunityLexicon;
use crate::model::{Agent, Rng, WorldModel};
use crate::phase3::{Phase3Runtime, SandboxSpec, Scope};
use crate::reading::ReadingBridge;
use crate::semantics::{configure_semantic_dimensions, semantic_dimensions};
use crate::tasks::{TaskEngine, TaskMetrics};

pub struct Coordinator {
    pub seed: u64,
    pub run_dir: PathBuf,
    pub report_path: PathBuf,
    pub metrics_path: PathBuf,
    pub dialogue_path: PathBuf,
    pub reading_path: PathBuf,
    pub summary_path: PathBuf,
    pub community_memory_path: PathBuf,
    pub community_memory_status: MemoryStatus,
    community_memory_enabled: bool,
    generation: usize,
    agents: Vec<Agent>,
    lexicon: CommunityLexicon,
    world: WorldModel,
    conversation: Conversation,
    reading: ReadingBridge,
    dictionary: HumanDictionary,
    tasks: TaskEngine,
    capability_lab: crate::capabilities::CapabilityLab,
    machine_lab: crate::machine::MachineLab,
    inquiry_lab: crate::inquiry::InquiryLab,
    community_semantics: CommunitySemanticMap,
    phase3: Phase3Runtime,
    rng: Rng,
    discomfort: f64,
    quiet_streak: usize,
    unresolved_concepts: BTreeMap<String, UnresolvedConcept>,
}

#[derive(Clone, Debug, Default)]
struct ActionMetrics {
    attempted: usize,
    succeeded: usize,
    social: usize,
    teaching: usize,
    explorers: Vec<usize>,
    curiosity_attempts: usize,
    curiosity_links: usize,
    curiosity_retries: usize,
    curiosity_questions: usize,
    curiosity_pending: usize,
}

#[derive(Clone, Debug, Default)]
struct ExplorerCuriosity {
    attempted: usize,
    linked: usize,
    retried: usize,
    question: Option<String>,
}

/// A word can stay uncertain without being forgotten or prematurely treated
/// as understood. These records are deliberately tiny and bounded by the
/// reading frontier, not a second unbounded world model.
#[derive(Clone, Debug, Default)]
struct UnresolvedConcept {
    attempts: usize,
    last_attempt_generation: usize,
    asked_ryan: bool,
}

#[derive(Clone, Debug, Default)]
struct DialogueMetrics {
    turns: usize,
    grounded_turns: usize,
    understood: usize,
    teaching_turns: usize,
    repair_turns: usize,
    private_turns: usize,
    failed_turns: usize,
    request_turns: usize,
    request_answers: usize,
    request_answers_understood: usize,
    repair_attempts: usize,
    repairs_resolved: usize,
    practice_turns: usize,
    assert_turns: usize,
    distinct_pairs: BTreeSet<(usize, usize)>,
}

#[derive(Clone, Debug, Default)]
struct SandboxMetrics {
    actions: usize,
    participating_agents: usize,
    energy_spent: f64,
}

impl Coordinator {
    pub fn machine_won(&self) -> bool {
        self.machine_lab.won
    }
    pub fn new(options: RunOptions) -> io::Result<Self> {
        if let Some(dimensions) = options.dimensions {
            configure_semantic_dimensions(dimensions)
                .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
        }
        let workspace_root = workspace_root();
        let seed = options.seed.unwrap_or_else(seed_from_clock);
        let run_dir = create_run_dir(&workspace_root.join("rust/runs"), seed)?;
        let report_path = run_dir.join("generation_report.txt");
        let metrics_path = run_dir.join("cultural_log.csv");
        let dialogue_path = run_dir.join("dialogue_log.txt");
        let reading_path = run_dir.join("reading_log.txt");
        let summary_path = run_dir.join("run_summary.txt");
        let community_memory_path = options.community_memory_path.clone().unwrap_or_else(|| {
            workspace_root.join(format!(
                "community_memory/rust-d{}.json",
                semantic_dimensions()
            ))
        });
        fs::write(
            &metrics_path,
            "generation,mean_energy,min_energy,max_energy,vocabulary,semantic_links,semantic_families,world_facts,discomfort,reading_bridge,conversation_positive,conversation_negative,dialogue_turns,dialogue_grounded,dialogue_understood,dialogue_requests,request_answers_understood,repairs_resolved,task_attempted,task_solved,task_failed,mean_fitness,best_fitness,trait_diversity,program_diversity,base_diversity,social_degree,alignment_tokens,alignment_repaired\n",
        )?;
        let mut lexicon = CommunityLexicon::default();
        let mut rng = Rng::new(seed);
        let mut agents = (0..POPULATION)
            .map(|id| {
                let mut agent = Agent::new(id, &mut rng);
                agent.ensure_numeric_semantics(&mut rng);
                agent
            })
            .collect::<Vec<_>>();
        let mut community_semantics = CommunitySemanticMap::default();
        let community_memory_status = if options.use_community_memory {
            community_memory::restore(
                &community_memory_path,
                &mut lexicon,
                &mut community_semantics,
                &mut agents,
                &mut rng,
            )
        } else {
            MemoryStatus::default()
        };
        fs::write(
            run_dir.join("metadata.json"),
            format!(
                "{{\n  \"seed\": {seed},\n  \"implementation\": \"rust\",\n  \"population\": {POPULATION},\n  \"semantic_dimensions\": {},\n  \"community_memory\": {{\n    \"path\": {},\n    \"loaded\": {},\n    \"tokens\": {},\n    \"relationships\": {}\n  }}\n}}\n",
                semantic_dimensions(),
                serde_json::to_string(&community_memory_path.to_string_lossy())
                    .expect("memory path is JSON"),
                community_memory_status.loaded,
                community_memory_status.tokens,
                community_memory_status.relationships,
            ),
        )?;
        let phase3 = Phase3Runtime::build(
            &agents,
            SandboxSpec {
                root: run_dir.join("sandbox"),
            },
        )?;
        Ok(Self {
            seed,
            run_dir,
            report_path,
            metrics_path,
            dialogue_path,
            reading_path,
            summary_path,
            community_memory_path,
            community_memory_status,
            community_memory_enabled: options.use_community_memory,
            generation: 0,
            agents,
            lexicon,
            world: WorldModel::default(),
            conversation: Conversation::new(options.converse_path),
            reading: ReadingBridge::load(&workspace_root),
            dictionary: HumanDictionary::discover(&workspace_root),
            tasks: TaskEngine::default(),
            capability_lab: crate::capabilities::CapabilityLab::new(options.capability_limit)
                .with_teaching(options.capability_teaching),
            machine_lab: crate::machine::MachineLab::new(seed),
            inquiry_lab: crate::inquiry::InquiryLab::default(),
            community_semantics,
            phase3,
            rng,
            discomfort: 0.0,
            quiet_streak: 0,
            unresolved_concepts: BTreeMap::new(),
        })
    }

    pub fn run_generation(&mut self) -> io::Result<()> {
        self.generation += 1;
        self.phase3.reset_generation();
        let dialogue_metrics = self.run_agent_dialogues()?;
        let task_metrics =
            self.tasks
                .run_generation(&mut self.agents, &mut self.lexicon, &mut self.rng);
        let mut action_metrics = self.run_agent_actions();
        let curiosity = self.run_explorer_curiosity(&action_metrics);
        action_metrics.curiosity_attempts = curiosity.attempted;
        action_metrics.curiosity_links = curiosity.linked;
        action_metrics.curiosity_retries = curiosity.retried;
        let sandbox_metrics = self.run_sandbox_steps();
        self.community_semantics.update(&self.agents);
        let alignment_metrics = self
            .community_semantics
            .repair_tasks(&mut self.agents, &mut self.rng);
        self.recover_agents();
        self.capability_lab
            .tick(&mut self.agents, self.generation, self.seed, &self.run_dir)?;
        self.machine_lab
            .tick(&mut self.agents, self.generation, &self.run_dir)?;
        self.inquiry_lab.tick(
            &mut self.agents,
            self.generation,
            &mut self.rng,
            &self.reading,
            &self.run_dir,
        )?;
        let evolution_metrics = evolve(
            &mut self.agents,
            &self.lexicon,
            self.generation as u64,
            &mut self.rng,
        );
        let reply = self.conversation.poll(
            self.generation,
            &mut self.world,
            &self.lexicon,
            &self.reading,
            &mut self.dictionary,
            &mut self.agents,
            &mut self.rng,
        )?;
        if let Some(topic) = self.conversation.take_resolved_question_topic() {
            self.unresolved_concepts.remove(&topic);
        }
        if reply.is_none()
            && let Some(topic) = curiosity.question.as_deref()
            && self.conversation.ask_ryan_about(topic, self.generation)?
        {
            if let Some(record) = self.unresolved_concepts.get_mut(topic) {
                record.asked_ryan = true;
            }
            action_metrics.curiosity_questions += 1;
        }
        action_metrics.curiosity_pending = self
            .unresolved_concepts
            .values()
            .filter(|record| !record.asked_ryan)
            .count();
        self.apply_dictionary_links();
        if reply.is_some() {
            self.quiet_streak = 0;
        } else {
            self.quiet_streak += 1;
        }
        if self.quiet_streak >= 3 && self.generation % self.reading_cadence() == 0 {
            self.run_reading_cycle()?;
        }
        self.update_discomfort(reply.is_some());
        for agent in &mut self.agents {
            agent.semantic_tick(&mut self.rng);
            agent.numeric.compact_against_community(&self.lexicon);
        }
        self.write_artifacts(
            reply.as_deref(),
            &task_metrics,
            &dialogue_metrics,
            &action_metrics,
            &sandbox_metrics,
            &evolution_metrics,
            &alignment_metrics,
        )?;
        self.write_run_summary(
            &task_metrics,
            &dialogue_metrics,
            &evolution_metrics,
            &alignment_metrics,
        )?;
        self.persist_community_memory()?;
        Ok(())
    }

    pub fn question_path(&self) -> &Path {
        self.conversation.question_path()
    }

    fn persist_community_memory(&self) -> io::Result<()> {
        if !self.community_memory_enabled {
            return Ok(());
        }
        community_memory::save(
            &self.community_memory_path,
            self.generation,
            &self.lexicon,
            &self.community_semantics,
        )
    }

    fn run_agent_dialogues(&mut self) -> io::Result<DialogueMetrics> {
        let mut metrics = DialogueMetrics::default();
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.dialogue_path)?;
        for _ in 0..10 {
            let speaker = self.rng.index(self.agents.len());
            let listener = select_dialogue_partner(&self.agents, speaker, &mut self.rng);
            let (speaker_agent, listener_agent) = two_agents(&mut self.agents, speaker, listener);
            let act = produce_dialogue_act(
                speaker_agent,
                listener_agent.id,
                &self.lexicon,
                &mut self.rng,
            );
            let mut understood = interpret_dialogue_act(listener_agent, &act, &self.lexicon);
            let learning_gain = match act.intent {
                DialogueIntent::Teach => 0.16,
                DialogueIntent::Ask => 0.10,
                DialogueIntent::Repair => 0.18,
                DialogueIntent::Practice => 0.08,
                DialogueIntent::Assert => 0.06,
            };
            listener_agent.observe(&act.tokens, learning_gain, &mut self.rng);
            speaker_agent.observe(&act.tokens, 0.06, &mut self.rng);
            listener_agent.language.record_heard(&act.tokens);
            metrics.turns += 1;
            metrics.distinct_pairs.insert((
                speaker_agent.id.min(listener_agent.id),
                speaker_agent.id.max(listener_agent.id),
            ));
            match act.intent {
                DialogueIntent::Practice => metrics.practice_turns += 1,
                DialogueIntent::Teach => metrics.teaching_turns += 1,
                DialogueIntent::Ask => metrics.request_turns += 1,
                DialogueIntent::Repair => metrics.repair_turns += 1,
                DialogueIntent::Assert => metrics.assert_turns += 1,
            }
            if !matches!(act.meaning, DialogueMeaning::Private) {
                metrics.grounded_turns += 1;
            } else {
                metrics.private_turns += 1;
            }

            // An unsuccessful utterance prompts an explicit, bounded repair.
            // The repairer either restates through public conventions or
            // echoes a private focus phrase without accidentally grounding it.
            if !understood {
                metrics.failed_turns += 1;
                metrics.repair_attempts += 1;
                let repair = repair_dialogue(
                    listener_agent,
                    speaker_agent.id,
                    &act,
                    &self.lexicon,
                    &mut self.rng,
                );
                let repair_understood =
                    interpret_dialogue_act(speaker_agent, &repair, &self.lexicon);
                listener_agent.observe(&act.tokens, 0.12, &mut self.rng);
                speaker_agent.observe(&repair.tokens, 0.08, &mut self.rng);
                listener_agent
                    .language
                    .record_turn(&repair, repair_understood);
                speaker_agent.language.record_heard(&repair.tokens);
                metrics.turns += 1;
                metrics.repair_turns += 1;
                if repair_understood {
                    metrics.repairs_resolved += 1;
                    understood = true;
                }
                writeln!(
                    log,
                    "Gen {} A{} -> A{} [Repair; {}]: {}",
                    self.generation,
                    listener_agent.id,
                    speaker_agent.id,
                    if repair_understood {
                        "understood"
                    } else {
                        "unresolved"
                    },
                    repair.render(),
                )?;
            }

            // A request that was understood earns a concise assertion in
            // response. This is the first deliberately multi-turn pragmatic
            // exchange, not a random second monologue.
            if act.intent == DialogueIntent::Ask
                && !matches!(act.meaning, DialogueMeaning::Private)
                && let Some(answer) = answer_request(
                    listener_agent,
                    speaker_agent.id,
                    &act,
                    &self.lexicon,
                    &mut self.rng,
                )
            {
                metrics.request_answers += 1;
                let answer_understood =
                    interpret_dialogue_act(speaker_agent, &answer, &self.lexicon);
                listener_agent
                    .language
                    .record_turn(&answer, answer_understood);
                speaker_agent.language.record_heard(&answer.tokens);
                listener_agent.observe(&answer.tokens, 0.06, &mut self.rng);
                speaker_agent.observe(&answer.tokens, 0.10, &mut self.rng);
                metrics.turns += 1;
                metrics.assert_turns += 1;
                metrics.grounded_turns += 1;
                if answer_understood {
                    metrics.request_answers_understood += 1;
                }
                writeln!(
                    log,
                    "Gen {} A{} -> A{} [Answer; {}]: {}",
                    self.generation,
                    listener_agent.id,
                    speaker_agent.id,
                    if answer_understood {
                        "understood"
                    } else {
                        "repair"
                    },
                    answer.render(),
                )?;
            }

            speaker_agent.language.record_turn(&act, understood);
            let outcome = if understood { 0.06 } else { -0.025 };
            speaker_agent.remember_interaction(listener_agent.id, outcome, self.generation as u64);
            listener_agent.remember_interaction(
                speaker_agent.id,
                outcome * 0.8,
                self.generation as u64,
            );
            speaker_agent.energy = (speaker_agent.energy - 0.12).max(0.0);
            listener_agent.energy = (listener_agent.energy - 0.08).max(0.0);
            if understood {
                speaker_agent.fitness += 0.04;
                listener_agent.fitness += 0.05;
                speaker_agent.state_event("successful_comm");
                listener_agent.state_event("successful_comm");
                metrics.understood += 1;
            } else {
                speaker_agent.state_event("failed_comm");
                listener_agent.state_event("learning_failure");
            }
            writeln!(
                log,
                "Gen {} A{} -> A{} [{:?}; {}]: {}",
                self.generation,
                speaker_agent.id,
                listener_agent.id,
                act.intent,
                if understood { "understood" } else { "repair" },
                act.render(),
            )?;
        }
        Ok(metrics)
    }

    fn apply_dictionary_links(&mut self) {
        let links = self.conversation.take_dictionary_links();
        for agent in &mut self.agents {
            for (left, right, gain) in &links {
                agent.vocabulary.insert(left.clone());
                agent.vocabulary.insert(right.clone());
                agent.semantics.link(left, right, *gain, &mut self.rng);
            }
        }
    }

    fn run_agent_actions(&mut self) -> ActionMetrics {
        let mut metrics = ActionMetrics::default();
        for agent_index in 0..self.agents.len() {
            let action = self.agents[agent_index].decide_action(&mut self.rng);
            metrics.attempted += 1;
            let success = match action {
                Action::Idle => true,
                Action::Social => {
                    metrics.social += 1;
                    self.run_social_action(agent_index)
                }
                Action::Teach => {
                    metrics.teaching += 1;
                    self.run_teaching_action(agent_index)
                }
                Action::Math => {
                    let value = self.rng.index(64) as u32;
                    let phrase = self.agents[agent_index].numeric.speak_number(
                        value,
                        &self.lexicon,
                        &mut self.rng,
                    );
                    self.agents[agent_index].observe(&[phrase], 0.08, &mut self.rng);
                    true
                }
                Action::Language => {
                    let word = self.agents[agent_index]
                        .vocabulary
                        .iter()
                        .nth(self.rng.index(self.agents[agent_index].vocabulary.len()))
                        .cloned()
                        .unwrap_or_else(|| "su".to_string());
                    self.agents[agent_index].observe(&[word], 0.06, &mut self.rng);
                    true
                }
                Action::Explore => {
                    metrics.explorers.push(agent_index);
                    self.agents[agent_index].semantic_tick(&mut self.rng);
                    true
                }
                Action::Learn | Action::Reorganise => {
                    self.agents[agent_index].semantic_tick(&mut self.rng);
                    true
                }
            };
            // Teaching owns its fine-grained reward, trust, and energy path;
            // other decisions receive the standard post-action update here.
            if action != Action::Teach {
                self.agents[agent_index].apply_action_result(action, success);
            }
            if success {
                metrics.succeeded += 1;
            }
        }
        metrics
    }

    /// An explorer distinguishes a verified bridge from mere shared exposure.
    /// A word carried by the same story into every agent is not therefore
    /// understood. Failed candidates remain in a small retry backlog and only
    /// reach Ryan after at least two explorer attempts.
    fn run_explorer_curiosity(&mut self, actions: &ActionMetrics) -> ExplorerCuriosity {
        let Some(&explorer_index) = actions
            .explorers
            .get(self.rng.index(actions.explorers.len().max(1)))
        else {
            return ExplorerCuriosity::default();
        };
        let (topic, retrying) = if let Some(topic) = self.due_unresolved_topic() {
            (topic, true)
        } else {
            let mut candidates = self.reading.exploration_candidates();
            candidates.retain(|topic| !self.unresolved_concepts.contains_key(topic));
            candidates
                .sort_by_key(|topic| self.agents[explorer_index].semantics.link_count_for(topic));
            // Keep exploration focused on the least-connected end of the
            // shared reading frontier. Raw story exposure is not a bridge.
            candidates.truncate((candidates.len() / 3).max(1));
            let Some(topic) = candidates
                .get(self.rng.index(candidates.len().max(1)))
                .cloned()
            else {
                return ExplorerCuriosity::default();
            };
            (topic, false)
        };
        let mut result = ExplorerCuriosity {
            attempted: 1,
            retried: retrying as usize,
            ..ExplorerCuriosity::default()
        };

        let known = &self.agents[explorer_index].vocabulary;
        let dictionary_anchor = self
            .dictionary
            .meaning_evidence(&topic)
            .into_iter()
            .flat_map(|evidence| {
                evidence
                    .category_tokens
                    .into_iter()
                    .chain(evidence.definition_tokens)
                    .chain(evidence.synonyms)
            })
            .find(|word| {
                word != &topic
                    && known.contains(word)
                    && self
                        .community_semantics
                        .token(word)
                        .is_some_and(|concept| concept.confidence >= 0.50)
            });
        if let Some(anchor) = dictionary_anchor {
            self.agents[explorer_index].observe(
                &[topic.clone(), anchor.clone()],
                0.10,
                &mut self.rng,
            );
            self.dictionary_links_from_exploration(&topic, &anchor);
            self.agents[explorer_index].state_event("learning_success");
            self.unresolved_concepts.remove(&topic);
            result.linked = 1;
        } else {
            self.agents[explorer_index].state_event("learning_failure");
            let record = self.unresolved_concepts.entry(topic.clone()).or_default();
            record.attempts += 1;
            record.last_attempt_generation = self.generation;
            if record.attempts >= 2 && !record.asked_ryan {
                result.question = Some(topic);
            }
        }
        result
    }

    fn due_unresolved_topic(&self) -> Option<String> {
        const RETRY_AFTER_GENERATIONS: usize = 6;
        self.unresolved_concepts
            .iter()
            .filter(|(_, record)| !record.asked_ryan)
            .filter(|(_, record)| {
                self.generation
                    .saturating_sub(record.last_attempt_generation)
                    >= RETRY_AFTER_GENERATIONS
            })
            .max_by(|left, right| {
                left.1.attempts.cmp(&right.1.attempts).then_with(|| {
                    right
                        .1
                        .last_attempt_generation
                        .cmp(&left.1.last_attempt_generation)
                })
            })
            .map(|(topic, _)| topic.clone())
    }

    fn dictionary_links_from_exploration(&mut self, topic: &str, anchor: &str) {
        // Reuse the existing community-wide semantic update path without
        // turning a tentative explorer attachment into a world fact.
        for agent in &mut self.agents {
            agent.semantics.link(topic, anchor, 0.04, &mut self.rng);
        }
    }

    fn recover_agents(&mut self) {
        // Python's coordinator applies a modest baseline recovery after
        // tasks/actions.  Without it, normal exploration costs eventually
        // dominate every other dynamic and evolution selects only agents that
        // happened not to act.
        for index in 0..self.agents.len() {
            self.agents[index].energy = (self.agents[index].energy + 1.0).min(100.0);
            if self.agents.len() > 1 {
                let peer_index = distinct_index(self.agents.len(), index, &mut self.rng);
                let peer_id = self.agents[peer_index].id;
                self.agents[index].adjust_trust(peer_id, 0.01);
            }
        }
    }

    fn run_sandbox_steps(&mut self) -> SandboxMetrics {
        let mut metrics = SandboxMetrics::default();
        for index in 0..self.agents.len() {
            // Draw all stochastic choices before taking the API's mutable
            // agent borrow.  This keeps the sandbox service isolated from
            // simulation scheduling and makes each operation independently
            // optional, like Python's background policy.
            let list_world = self.rng.unit() < 0.40;
            let read_notes = self.rng.unit() < 0.35;
            let scratch = self.rng.unit() < 0.50;
            let (agent_id, emit_note, token) = {
                let agent = &self.agents[index];
                let emit_note = self.rng.unit() < agent.traits.chattiness;
                let token = if agent.vocabulary.is_empty() {
                    "thought".to_string()
                } else {
                    agent
                        .vocabulary
                        .iter()
                        .nth(self.rng.index(agent.vocabulary.len()))
                        .cloned()
                        .unwrap_or_else(|| "thought".to_string())
                };
                (agent.id, emit_note, token)
            };
            if let Some(mut api) = self.phase3.api_for(&mut self.agents[index]) {
                if list_world {
                    if let Ok(paths) = api.list_paths("/", Scope::World) {
                        let _ = api.append_text(
                            "/log.txt",
                            &format!("seen {} world paths\n", paths.len()),
                            Scope::Home,
                        );
                    }
                }
                if read_notes {
                    if let Ok(notes) = api.read_text("/notes.txt", Scope::World) {
                        if !notes.is_empty() {
                            let _ = api.append_text(
                                "/log.txt",
                                &format!("read notes ({} chars)\n", notes.len()),
                                Scope::Home,
                            );
                        }
                    }
                }
                if emit_note {
                    let _ = api.append_text(
                        "/notes.txt",
                        &format!("A{agent_id}: {token}\n"),
                        Scope::World,
                    );
                }
                if scratch {
                    let _ = api.append_text("/scratch.txt", "note\n", Scope::Home);
                }
            }
            let stats = self.phase3.stats_for(agent_id);
            if stats.actions == 0 {
                // Match the reference's participation pressure without
                // letting background exploration dominate survival.
                self.agents[index].energy = (self.agents[index].energy - 1.0).max(0.0);
            } else {
                metrics.participating_agents += 1;
            }
            metrics.actions += stats.actions;
            metrics.energy_spent += stats.energy_spent;
        }
        metrics
    }

    fn run_social_action(&mut self, speaker_index: usize) -> bool {
        if self.agents.len() < 2 {
            return false;
        }
        let listener_index = distinct_index(self.agents.len(), speaker_index, &mut self.rng);
        let (speaker, listener) = two_agents(&mut self.agents, speaker_index, listener_index);
        let signal = speaker.referent_signal("r0");
        let utterance = vec![
            format!("a{}", speaker.id),
            signal,
            format!("a{}", listener.id),
        ];
        speaker.observe(&utterance, 0.10, &mut self.rng);
        listener.observe(&utterance, 0.10, &mut self.rng);
        speaker.remember_interaction(listener.id, 0.10, self.generation as u64);
        listener.remember_interaction(speaker.id, 0.10, self.generation as u64);
        true
    }

    fn run_teaching_action(&mut self, teacher_index: usize) -> bool {
        if self.agents.len() < 2 {
            return false;
        }
        let student_index = distinct_index(self.agents.len(), teacher_index, &mut self.rng);
        let (teacher, student) = two_agents(&mut self.agents, teacher_index, student_index);
        teacher.teach(student, self.generation as u64, &mut self.rng)
    }

    fn run_reading_cycle(&mut self) -> io::Result<()> {
        let passage = self.reading.read_passage();
        if passage.is_empty() {
            return Ok(());
        }
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.reading_path)?;
        writeln!(log, "Gen {} shared reading", self.generation)?;
        for words in &passage {
            for agent in &mut self.agents {
                agent.observe(words, 0.12, &mut self.rng);
            }
            writeln!(log, "  Read: {}", words.join(" "))?;
        }
        writeln!(
            log,
            "  Conversation bridge vocabulary={}",
            self.reading.promoted_count()
        )?;
        self.quiet_streak = 0;
        Ok(())
    }

    fn reading_cadence(&self) -> usize {
        if self.discomfort >= 0.5 { 2 } else { 4 }
    }

    fn update_discomfort(&mut self, answered: bool) {
        let quiet_pressure = (self.quiet_streak.saturating_sub(1) as f64 / 8.0).min(1.0);
        let target = if answered { 0.0 } else { quiet_pressure };
        self.discomfort = ((self.discomfort * 0.78) + (target * 0.22)).clamp(0.0, 1.0);
    }

    fn write_artifacts(
        &self,
        reply: Option<&str>,
        tasks: &TaskMetrics,
        dialogues: &DialogueMetrics,
        actions: &ActionMetrics,
        sandbox: &SandboxMetrics,
        evolution: &EvolutionMetrics,
        alignment: &AlignmentMetrics,
    ) -> io::Result<()> {
        let mean_energy =
            self.agents.iter().map(|agent| agent.energy).sum::<f64>() / self.agents.len() as f64;
        let min_energy = self
            .agents
            .iter()
            .map(|agent| agent.energy)
            .fold(f64::INFINITY, f64::min);
        let max_energy = self
            .agents
            .iter()
            .map(|agent| agent.energy)
            .fold(0.0, f64::max);
        let vocabulary: usize = self
            .agents
            .iter()
            .map(|agent| agent.vocabulary.len())
            .sum::<usize>()
            / self.agents.len();
        let semantic_links: usize = self
            .agents
            .iter()
            .map(|agent| agent.semantics.link_count())
            .sum::<usize>()
            / self.agents.len();
        let semantic_families: usize = self
            .agents
            .iter()
            .map(|agent| agent.semantics.family_count())
            .sum::<usize>()
            / self.agents.len();
        let structural_tokens: usize = self
            .agents
            .iter()
            .map(|agent| agent.semantics.structural_profile_metrics().0)
            .sum::<usize>()
            / self.agents.len();
        let structural_entropy: f32 = self
            .agents
            .iter()
            .map(|agent| agent.semantics.structural_profile_metrics().1)
            .sum::<f32>()
            / self.agents.len() as f32;
        let structural_coverage: f32 = self
            .agents
            .iter()
            .map(|agent| agent.semantics.structural_profile_metrics().2)
            .sum::<f32>()
            / self.agents.len() as f32;
        let structural_observations: u32 = self
            .agents
            .iter()
            .map(|agent| agent.semantics.structural_profile_metrics().3)
            .sum::<u32>()
            / self.agents.len() as u32;
        let structural_edges: usize = self
            .agents
            .iter()
            .map(|agent| agent.semantics.structural_edge_metrics().0)
            .sum::<usize>()
            / self.agents.len();
        let structural_edge_weight: f32 = self
            .agents
            .iter()
            .map(|agent| agent.semantics.structural_edge_metrics().1)
            .sum::<f32>()
            / self.agents.len() as f32;
        let structural_edge_evidence: u32 = self
            .agents
            .iter()
            .map(|agent| agent.semantics.structural_edge_metrics().2)
            .sum::<u32>()
            / self.agents.len() as u32;
        let structural_family_pairs: usize = self
            .agents
            .iter()
            .map(|agent| agent.semantics.structural_edge_metrics().3)
            .sum::<usize>()
            / self.agents.len();
        let (positive, negative) = self.conversation.feedback_counts();
        let (english_bigrams, promoted_bigrams, meaning_bigrams) =
            self.conversation.bigram_metrics();
        let (numeric_literals, numeric_literal_observations) =
            self.conversation.numeric_literal_metrics();
        let topic = self.conversation.topic().unwrap_or("-");
        let pending_topic = self
            .conversation
            .pending_topic_switch()
            .map(|(topic, confirmations)| format!("{topic} ({confirmations}/2)"))
            .unwrap_or_else(|| "-".to_string());
        let (path_points, path_predictions, path_resets) =
            self.conversation.semantic_path_metrics();
        let mut csv = OpenOptions::new().append(true).open(&self.metrics_path)?;
        writeln!(
            csv,
            "{generation},{mean_energy:.3},{min_energy:.3},{max_energy:.3},{vocabulary},{semantic_links},{semantic_families},{world_facts},{discomfort:.3},{reading_bridge},{positive},{negative},{dialogue_turns},{dialogue_grounded},{dialogue_understood},{dialogue_requests},{answer_understood},{repairs_resolved},{task_attempted},{task_solved},{task_failed},{mean_fitness:.3},{best_fitness:.3},{trait_diversity:.4},{program_diversity:.4},{base_diversity},{social_degree:.3},{alignment_tokens},{alignment_repaired}",
            generation = self.generation,
            world_facts = self.world.count(),
            discomfort = self.discomfort,
            reading_bridge = self.reading.promoted_count(),
            dialogue_turns = dialogues.turns,
            dialogue_grounded = dialogues.grounded_turns,
            dialogue_understood = dialogues.understood,
            dialogue_requests = dialogues.request_turns,
            answer_understood = dialogues.request_answers_understood,
            repairs_resolved = dialogues.repairs_resolved,
            task_attempted = tasks.attempted,
            task_solved = tasks.solved,
            task_failed = tasks.failed,
            mean_fitness = evolution.mean_fitness,
            best_fitness = evolution.best_fitness,
            trait_diversity = evolution.trait_diversity,
            program_diversity = evolution.program_diversity,
            base_diversity = evolution.base_diversity,
            social_degree = evolution.mean_social_degree,
            alignment_tokens = alignment.community_tokens,
            alignment_repaired = alignment.repaired,
        )?;
        let mut report = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.report_path)?;
        writeln!(report, "=== Generation {} ===", self.generation)?;
        writeln!(
            report,
            "Population: {}; energy mean/min/max={mean_energy:.2}/{min_energy:.2}/{max_energy:.2}; vocabulary: {vocabulary}",
            self.agents.len()
        )?;
        writeln!(
            report,
            "Conversation: task={}; topic={topic}; pending topic={pending_topic}; reply={}",
            self.conversation.dialogue_task(),
            reply.unwrap_or("idle")
        )?;
        writeln!(
            report,
            "Semantic path: waypoints={path_points}; predicted next concepts={path_predictions}; resets={path_resets}",
        )?;
        writeln!(
            report,
            "Conversation phrases: observed bigrams={english_bigrams}; promoted={promoted_bigrams}; meaning scaffolds={meaning_bigrams}",
        )?;
        writeln!(
            report,
            "Numeric literals: distinct={numeric_literals}; observations={numeric_literal_observations}; storage is bounded and separate from word semantics",
        )?;
        writeln!(
            report,
            "World facts: {}; reading bridge: {}; discomfort: {:.2}",
            self.world.count(),
            self.reading.promoted_count(),
            self.discomfort
        )?;
        writeln!(
            report,
            "Semantic links (mean): {semantic_links}; families (mean): {semantic_families}"
        )?;
        writeln!(
            report,
            "Structural grammar (mean): operators={structural_tokens}; entropy={structural_entropy:.2}; family coverage={structural_coverage:.2}; observations={structural_observations}; directed family edges={structural_edges}; family pairs={structural_family_pairs}; edge evidence={structural_edge_evidence}; edge weight={structural_edge_weight:.2}"
        )?;
        writeln!(
            report,
            "Tasks: attempted={}; solved={}; failed={}; numeric={}; referential={}; action={}; grammar={}",
            tasks.attempted,
            tasks.solved,
            tasks.failed,
            tasks.numeric_successes,
            tasks.referential_successes,
            tasks.action_successes,
            tasks.grammar_successes,
        )?;
        writeln!(
            report,
            "Task extensions: comparisons={}; quantity translations={}",
            tasks.comparison_successes, tasks.quantity_successes,
        )?;
        writeln!(
            report,
            "Pragmatic dialogue: turns={}; pairs={}; grounded={}; understood={}; failed={}; practice={}; teaching={}; asks={}; assertions={}; repair acts={}; repair outcomes={}/{}; answers={}/{}; private={}",
            dialogues.turns,
            dialogues.distinct_pairs.len(),
            dialogues.grounded_turns,
            dialogues.understood,
            dialogues.failed_turns,
            dialogues.practice_turns,
            dialogues.teaching_turns,
            dialogues.request_turns,
            dialogues.assert_turns,
            dialogues.repair_turns,
            dialogues.repairs_resolved,
            dialogues.repair_attempts,
            dialogues.request_answers_understood,
            dialogues.request_answers,
            dialogues.private_turns,
        )?;
        writeln!(
            report,
            "Concept tasks: attempts={}; successes={}; repairs={}; definition consensus={}",
            tasks.concept_attempts,
            tasks.concept_successes,
            tasks.concept_repairs,
            tasks.definition_consensus,
        )?;
        writeln!(
            report,
            "Structured concepts: narrative={}; prediction={}; role={}; reconstruction={}; property={}; compatibility={}; misunderstanding repairs={}",
            tasks.narrative_successes,
            tasks.prediction_successes,
            tasks.role_successes,
            tasks.reconstruction_successes,
            tasks.property_successes,
            tasks.compatibility_successes,
            tasks.misunderstanding_repairs,
        )?;
        writeln!(
            report,
            "Agent actions: attempted={}; succeeded={}; social={}; teaching={}; explorers={}; curiosity attempts={}; retries={}; verified links={}; unresolved backlog={}; Ryan questions={}",
            actions.attempted,
            actions.succeeded,
            actions.social,
            actions.teaching,
            actions.explorers.len(),
            actions.curiosity_attempts,
            actions.curiosity_retries,
            actions.curiosity_links,
            actions.curiosity_pending,
            actions.curiosity_questions,
        )?;
        writeln!(
            report,
            "Sandbox: actions={}; participants={}; energy spent={:.2}",
            sandbox.actions, sandbox.participating_agents, sandbox.energy_spent,
        )?;
        writeln!(
            report,
            "Evolution: culled={:?}; child={:?}; parent lineages={:?}; mean fitness={:.3}; best fitness={:.3}; lineage rewards={}",
            evolution.culled_id,
            evolution.child_id,
            evolution.child_parent_lineages,
            evolution.mean_fitness,
            evolution.best_fitness,
            evolution.lineage_rewards,
        )?;
        writeln!(
            report,
            "Child inheritance: vocabulary={}; semantic links={}; families={}; language preferences={}; program instructions={}; program mutated={}",
            evolution.child_vocabulary,
            evolution.child_semantic_links,
            evolution.child_semantic_families,
            evolution.child_language_preferences,
            evolution.child_program_len,
            evolution.program_mutated,
        )?;
        writeln!(
            report,
            "Diversity: traits={:.4}; programs={:.3}; numeric bases={}; mean social degree={:.2}",
            evolution.trait_diversity,
            evolution.program_diversity,
            evolution.base_diversity,
            evolution.mean_social_degree,
        )?;
        writeln!(
            report,
            "Semantic alignment: community tokens={}; public relationships={}; proposed={}; repaired={}; mean repair distance={:.3}",
            alignment.community_tokens,
            alignment.community_relationships,
            alignment.proposed,
            alignment.repaired,
            alignment.mean_distance,
        )?;
        writeln!(report)?;
        Ok(())
    }

    /// A compact, overwritten snapshot for people inspecting a completed
    /// experiment.  The full generation report remains append-only; this file
    /// answers the practical question: "what state did the run end in?"
    fn write_run_summary(
        &self,
        tasks: &TaskMetrics,
        dialogues: &DialogueMetrics,
        evolution: &EvolutionMetrics,
        alignment: &AlignmentMetrics,
    ) -> io::Result<()> {
        let mut summary = fs::File::create(&self.summary_path)?;
        let mut ranked: Vec<_> = self.agents.iter().collect();
        ranked.sort_by(|left, right| {
            right
                .total_fitness
                .partial_cmp(&left.total_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let public_numeric = self.lexicon.numeric_conventions();
        let public_referents = self.lexicon.referential_conventions();
        let public_actions = self.lexicon.action_conventions();
        writeln!(summary, "Rust MAS run summary")?;
        writeln!(summary, "====================")?;
        writeln!(summary, "Seed: {}", self.seed)?;
        writeln!(summary, "Completed generation: {}", self.generation)?;
        writeln!(summary, "Population: {}", self.agents.len())?;
        writeln!(summary, "{}", self.capability_lab.summary)?;
        writeln!(
            summary,
            "Capability monitor: capability_monitor.html; history: capability_metrics.csv"
        )?;
        writeln!(
            summary,
            "Fitness: mean={:.3}; best={:.3}; trait diversity={:.4}; program diversity={:.3}",
            evolution.mean_fitness,
            evolution.best_fitness,
            evolution.trait_diversity,
            evolution.program_diversity,
        )?;
        writeln!(
            summary,
            "Pragmatics: turns={}; pairs={}; grounded={}; understood={}; asks answered={}/{}; repairs resolved={}/{}",
            dialogues.turns,
            dialogues.distinct_pairs.len(),
            dialogues.grounded_turns,
            dialogues.understood,
            dialogues.request_answers_understood,
            dialogues.request_answers,
            dialogues.repairs_resolved,
            dialogues.repair_attempts,
        )?;
        writeln!(
            summary,
            "Tasks: solved={}/{}; concepts={}/{}; semantic repairs={}",
            tasks.solved,
            tasks.attempted,
            tasks.concept_successes,
            tasks.concept_attempts,
            alignment.repaired,
        )?;
        writeln!(
            summary,
            "Public conventions: numeric={:?}; referential={:?}; actions={:?}; base={:?}; grammar={:?}",
            public_numeric,
            public_referents,
            public_actions,
            self.lexicon.community_base(),
            self.lexicon.grammar_conventions(),
        )?;
        writeln!(summary, "\nTop agents:")?;
        for agent in ranked.into_iter().take(5) {
            let parents = agent
                .parent_lineages
                .map(|ids| format!("{:?}", ids))
                .unwrap_or_else(|| "founder".to_string());
            writeln!(
                summary,
                "  A{} fitness={:.3} lineage={:.3} energy={:.1} birth={} parents={} vocab={} links={} families={} program={}",
                agent.id,
                agent.total_fitness,
                agent.lineage_score,
                agent.energy,
                agent.birth_generation,
                parents,
                agent.vocabulary.len(),
                agent.semantics.link_count(),
                agent.semantics.family_count(),
                agent.program.len(),
            )?;
        }
        writeln!(
            summary,
            "\nInspect: generation_report.txt (full timeline), cultural_log.csv (analysis), dialogue_log.txt (turns), reading_log.txt (reading)."
        )?;
        Ok(())
    }
}

fn distinct_index(size: usize, excluded: usize, rng: &mut Rng) -> usize {
    let mut index = rng.index(size - 1);
    if index >= excluded {
        index += 1;
    }
    index
}

/// Prefer familiar, trustworthy partners while retaining a non-zero chance of
/// novel encounters.  This replaces purely random pairing so pragmatics can
/// form locally before a convention is carried through the population.
fn select_dialogue_partner(agents: &[Agent], speaker: usize, rng: &mut Rng) -> usize {
    let source = &agents[speaker];
    let candidates: Vec<_> = (0..agents.len())
        .filter(|index| *index != speaker)
        .collect();
    let weights: Vec<_> = candidates
        .iter()
        .map(|candidate| {
            let peer = &agents[*candidate];
            let trust = source
                .trust
                .get(&peer.id)
                .map(|profile| (profile.affinity + profile.reliability + profile.competence) / 3.0)
                .unwrap_or(0.0);
            let history = source
                .social_memory
                .get(&peer.id)
                .map(|record| record.mean_outcome)
                .unwrap_or(0.0);
            let shared_words = source.vocabulary.intersection(&peer.vocabulary).count() as f64;
            let lexical_affinity = shared_words / (source.vocabulary.len().max(1) as f64).sqrt();
            // Curiosity keeps unfamiliar partners viable even after social
            // clusters begin to form.
            (0.18
                + 0.42 * ((trust + 1.0) * 0.5)
                + 0.22 * ((history + 1.0) * 0.5)
                + 0.10 * lexical_affinity.min(1.0)
                + 0.08 * source.traits.curiosity)
                .max(0.01)
        })
        .collect();
    let total = weights.iter().sum::<f64>();
    let mut threshold = rng.unit() * total;
    for (index, weight) in weights.iter().enumerate() {
        threshold -= weight;
        if threshold <= 0.0 {
            return candidates[index];
        }
    }
    *candidates.last().expect("population has a partner")
}

fn two_agents(agents: &mut [Agent], left: usize, right: usize) -> (&mut Agent, &mut Agent) {
    assert_ne!(left, right);
    if left < right {
        let (head, tail) = agents.split_at_mut(right);
        (&mut head[left], &mut tail[0])
    } else {
        let (head, tail) = agents.split_at_mut(left);
        (&mut tail[0], &mut head[right])
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust project must have a workspace parent")
        .to_path_buf()
}

fn seed_from_clock() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

fn create_run_dir(root: &Path, seed: u64) -> io::Result<PathBuf> {
    fs::create_dir_all(root)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    for suffix in 0..1000 {
        let name = if suffix == 0 {
            format!("{stamp}_seed-{seed:016x}")
        } else {
            format!("{stamp}_seed-{seed:016x}_{suffix}")
        };
        let candidate = root.join(name);
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a Rust run directory",
    ))
}

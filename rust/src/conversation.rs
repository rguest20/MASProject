use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use crate::dictionary::HumanDictionary;
use crate::lexicon::CommunityLexicon;
use crate::model::{Agent, Rng, WorldModel};
use crate::reading::{ReadingBridge, tokenise};

#[derive(Debug)]
pub struct Conversation {
    path: PathBuf,
    question_path: PathBuf,
    pending: Option<PendingPrompt>,
    question_pending: Option<PendingPrompt>,
    processed_question_answer: Option<(usize, String)>,
    last_question_answer_generation: Option<usize>,
    human_pool: BTreeMap<String, f64>,
    transitions: BTreeMap<String, BTreeMap<String, usize>>,
    english_bigrams: BTreeMap<(String, String), f64>,
    meaning_bigrams: BTreeMap<(String, String), f64>,
    intent_modes: BTreeMap<String, IntentMode>,
    next_intent_mode: usize,
    working_frame: Option<WorkingFrame>,
    parked_frames: VecDeque<WorkingFrame>,
    reply_feedback: BTreeMap<String, f64>,
    bigram_feedback: BTreeMap<(String, String), f64>,
    dictionary_links: Vec<(String, String, f32)>,
    // A word that is new to the conversation can borrow a small number of
    // nearby, *community-selected* concepts.  These are semantic evidence,
    // not prewritten reply templates.
    horizon_links: BTreeMap<String, BTreeMap<String, f64>>,
    active_topic: Option<String>,
    // A plain new noun is evidence, not necessarily a subject change. It
    // must recur before it can displace the currently discussed topic.
    pending_topic_switch: Option<(String, usize)>,
    dialogue_task: DialogueTask,
    semantic_path: SemanticPath,
    last_prompt_generation: Option<usize>,
    open_question: Option<String>,
    resolved_question_topic: Option<String>,
    numeric_literals: NumericLiteralMemory,
    last_reply: Option<String>,
    last_claim: Option<(String, String, String)>,
    positive_feedback: usize,
    negative_feedback: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TranscriptPrompt {
    line_number: usize,
    body: String,
}

#[derive(Clone, Debug)]
struct PendingPrompt {
    prompt: TranscriptPrompt,
    transcript: String,
    first_seen_generation: usize,
}

#[derive(Clone, Debug, Default)]
struct IntentMode {
    tokens: BTreeMap<String, f64>,
    transitions: BTreeMap<String, BTreeMap<String, f64>>,
    shapes: BTreeMap<String, f64>,
    topics: BTreeMap<String, f64>,
    uses: usize,
}

#[derive(Clone, Debug)]
struct WorkingFrame {
    topic: Option<String>,
    mode: String,
    tokens: BTreeMap<String, f64>,
    transitions: BTreeMap<String, BTreeMap<String, f64>>,
    turns: usize,
}

/// The immediate job of a turn. This is intentionally separate from the
/// emergent semantic intent modes: it selects an existing reply capability
/// before a previously active semantic topic can take over the response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum DialogueTask {
    #[default]
    Conversation,
    Definition,
    Arithmetic,
    Number,
    Referent,
    Teaching,
    Reset,
    Stop,
}

impl DialogueTask {
    fn label(self) -> &'static str {
        match self {
            Self::Conversation => "conversation",
            Self::Definition => "definition",
            Self::Arithmetic => "arithmetic",
            Self::Number => "number",
            Self::Referent => "referent",
            Self::Teaching => "teaching",
            Self::Reset => "reset",
            Self::Stop => "stop",
        }
    }
}

/// A bounded semantic trajectory for the current conversation. It contains
/// content-topic waypoints only; the long-term maps remain with the agents.
/// Predictions are deliberately advisory, so a direct human question always
/// wins over an attractive but irrelevant continuation.
#[derive(Clone, Debug, Default)]
struct SemanticPath {
    points: VecDeque<String>,
    predictions: BTreeMap<String, f32>,
    resets: usize,
}

impl SemanticPath {
    const MAX_POINTS: usize = 8;
    const MAX_PREDICTIONS: usize = 6;
    const MIN_CONTINUITY: f32 = 0.10;
    const MIN_WAYPOINTS_FOR_PREDICTION: usize = 2;
    const MIN_PREDICTION_AGREEMENT: f32 = 0.45;
    const MIN_PREDICTION_SCORE: f32 = 0.20;

    fn observe(&mut self, topic: &str, agents: &[Agent]) {
        if self.points.back().is_some_and(|last| last == topic) {
            self.refresh_predictions(agents);
            return;
        }
        if !self.points.is_empty() && !self.is_likely_next(topic, agents) {
            self.points.clear();
            self.predictions.clear();
            self.resets += 1;
        }
        if self.points.len() == Self::MAX_POINTS {
            self.points.pop_front();
        }
        self.points.push_back(topic.to_string());
        self.refresh_predictions(agents);
    }

    fn clear(&mut self) {
        if !self.points.is_empty() || !self.predictions.is_empty() {
            self.resets += 1;
        }
        self.points.clear();
        self.predictions.clear();
    }

    fn predicted_words(&self, topic: &str) -> impl Iterator<Item = (&String, &f32)> {
        (self.points.len() >= Self::MIN_WAYPOINTS_FOR_PREDICTION
            && self.points.back().map(String::as_str) == Some(topic))
        .then_some(self.predictions.iter())
        .into_iter()
        .flatten()
    }

    fn prediction_score(&self, previous: &str, candidate: &str) -> f64 {
        self.predicted_words(previous)
            .find(|(word, _)| word.as_str() == candidate)
            .map(|(_, score)| *score as f64)
            .unwrap_or(0.0)
    }

    fn is_likely_next(&self, topic: &str, agents: &[Agent]) -> bool {
        let Some(last) = self.points.back() else {
            return true;
        };
        self.predictions
            .get(topic)
            .is_some_and(|score| *score >= Self::MIN_CONTINUITY)
            || mean_semantic_similarity(agents, last, topic)
                .is_some_and(|similarity| similarity >= Self::MIN_CONTINUITY)
    }

    fn refresh_predictions(&mut self, agents: &[Agent]) {
        if self.points.len() < Self::MIN_WAYPOINTS_FOR_PREDICTION {
            self.predictions.clear();
            return;
        }
        let Some(current) = self.points.back().cloned() else {
            return;
        };
        let previous = self.points.iter().rev().nth(1).cloned();
        let mut candidates = BTreeMap::<String, Vec<f32>>::new();
        for agent in agents {
            for candidate in agent.semantics.nearest(&current, 8) {
                if candidate != current && is_topic(&candidate) {
                    let similarity = agent
                        .semantics
                        .cosine_similarity(&current, &candidate)
                        .unwrap_or(0.0);
                    candidates.entry(candidate).or_default().push(similarity);
                }
            }
        }
        let mut ranked: Vec<_> = candidates
            .into_iter()
            .filter_map(|(candidate, similarities)| {
                let agreement = similarities.len() as f32 / agents.len().max(1) as f32;
                if agreement < Self::MIN_PREDICTION_AGREEMENT {
                    return None;
                }
                let local = similarities.iter().sum::<f32>() / similarities.len() as f32;
                let directional = previous
                    .as_deref()
                    .and_then(|previous| {
                        mean_directional_continuity(agents, previous, &current, &candidate)
                    })
                    .unwrap_or(0.0);
                // Nearby concepts are useful candidates. Once there are two
                // waypoints, preserving the direction of travel matters too.
                Some((
                    candidate,
                    (0.70 * local + 0.30 * directional.max(0.0)).max(0.0),
                ))
            })
            .filter(|(_, score)| *score >= Self::MIN_PREDICTION_SCORE)
            .collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(Self::MAX_PREDICTIONS);
        self.predictions = ranked.into_iter().collect();
    }
}

/// Bounded community evidence for decimal literals supplied by a human. This
/// is deliberately separate from the word map: a literal is recognised by
/// its value and decimal structure, never by open-ended text co-occurrence.
#[derive(Clone, Debug, Default)]
struct NumericLiteralMemory {
    observations: BTreeMap<u32, u32>,
}

impl NumericLiteralMemory {
    const MAX_LITERALS: usize = 256;

    fn observe(&mut self, value: u32) {
        if !self.observations.contains_key(&value) && self.observations.len() >= Self::MAX_LITERALS
        {
            if let Some(evicted) = self
                .observations
                .iter()
                .min_by_key(|(literal, observations)| (**observations, **literal))
                .map(|(literal, _)| *literal)
            {
                self.observations.remove(&evicted);
            }
        }
        *self.observations.entry(value).or_default() += 1;
    }

    fn decimal_digits(value: u32) -> Vec<u32> {
        crate::numeric::digits(value, 10)
    }

    fn counts(&self) -> (usize, u32) {
        (
            self.observations.len(),
            self.observations.values().copied().sum(),
        )
    }
}

impl Conversation {
    pub fn new(path: PathBuf) -> Self {
        let question_path = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(|parent| parent.join("question.txt"))
            .unwrap_or_else(|| PathBuf::from("question.txt"));
        Self {
            path,
            question_path,
            pending: None,
            question_pending: None,
            processed_question_answer: None,
            last_question_answer_generation: None,
            human_pool: BTreeMap::new(),
            transitions: BTreeMap::new(),
            english_bigrams: BTreeMap::new(),
            meaning_bigrams: BTreeMap::new(),
            intent_modes: BTreeMap::new(),
            next_intent_mode: 1,
            working_frame: None,
            parked_frames: VecDeque::with_capacity(4),
            reply_feedback: BTreeMap::new(),
            bigram_feedback: BTreeMap::new(),
            dictionary_links: Vec::new(),
            horizon_links: BTreeMap::new(),
            active_topic: None,
            pending_topic_switch: None,
            dialogue_task: DialogueTask::default(),
            semantic_path: SemanticPath::default(),
            last_prompt_generation: None,
            open_question: None,
            resolved_question_topic: None,
            numeric_literals: NumericLiteralMemory::default(),
            last_reply: None,
            last_claim: None,
            positive_feedback: 0,
            negative_feedback: 0,
        }
    }

    pub fn poll(
        &mut self,
        generation: usize,
        world: &mut WorldModel,
        lexicon: &CommunityLexicon,
        reading: &ReadingBridge,
        dictionary: &mut HumanDictionary,
        agents: &mut [Agent],
        rng: &mut Rng,
    ) -> std::io::Result<Option<String>> {
        self.poll_question_answer(generation, dictionary, agents, rng)?;
        let text = match fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let Some(prompt) = latest_unanswered(&text) else {
            self.pending = None;
            return Ok(None);
        };
        if self
            .pending
            .as_ref()
            .is_none_or(|pending| pending.prompt != prompt)
        {
            self.pending = Some(PendingPrompt {
                prompt,
                transcript: text,
                first_seen_generation: generation,
            });
            return Ok(None);
        }
        let pending = self
            .pending
            .as_ref()
            .expect("pending prompt was just set")
            .clone();
        // Match Python's small settling window: every member gets time to
        // observe the same human line before the community writes back.
        if generation.saturating_sub(pending.first_seen_generation) < 2 {
            return Ok(None);
        }
        // Do not answer stale text if Ryan edits while the community is
        // thinking. A later generation will register the edited line anew.
        if fs::read_to_string(&self.path)? != pending.transcript {
            self.pending = None;
            return Ok(None);
        }
        let prompt = pending.prompt.body;
        self.expire_stale_context(generation);
        let reply = self.answer(&prompt, world, lexicon, reading, dictionary, agents, rng);
        self.last_prompt_generation = Some(generation);
        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        if !pending.transcript.ends_with('\n') {
            writeln!(file)?;
        }
        writeln!(file, "Community: {reply}")?;
        writeln!(file, "Ryan: ")?;
        self.pending = None;
        if !is_feedback(&prompt) && !reply.is_empty() {
            self.last_reply = Some(reply.clone());
        }
        Ok(Some(reply))
    }

    pub fn topic(&self) -> Option<&str> {
        self.active_topic.as_deref()
    }

    pub fn pending_topic_switch(&self) -> Option<(&str, usize)> {
        self.pending_topic_switch
            .as_ref()
            .map(|(topic, confirmations)| (topic.as_str(), *confirmations))
    }

    pub fn dialogue_task(&self) -> &'static str {
        self.dialogue_task.label()
    }

    pub fn semantic_path_metrics(&self) -> (usize, usize, usize) {
        (
            self.semantic_path.points.len(),
            self.semantic_path.predictions.len(),
            self.semantic_path.resets,
        )
    }

    pub fn question_path(&self) -> &PathBuf {
        &self.question_path
    }

    pub fn feedback_counts(&self) -> (usize, usize) {
        (self.positive_feedback, self.negative_feedback)
    }

    pub fn bigram_metrics(&self) -> (usize, usize, usize) {
        let promoted = self
            .english_bigrams
            .keys()
            .chain(self.meaning_bigrams.keys())
            .filter(|pair| self.bigram_strength(pair) > 0.10)
            .count();
        (
            self.english_bigrams.len(),
            promoted,
            self.meaning_bigrams.len(),
        )
    }

    pub fn numeric_literal_metrics(&self) -> (usize, u32) {
        self.numeric_literals.counts()
    }

    pub fn take_dictionary_links(&mut self) -> Vec<(String, String, f32)> {
        std::mem::take(&mut self.dictionary_links)
    }

    pub fn take_resolved_question_topic(&mut self) -> Option<String> {
        self.resolved_question_topic.take()
    }

    /// Ask one human-facing curiosity question through the separate mailbox.
    /// The mailbox has one slot: until Ryan fills its answer line, explorers
    /// can investigate privately but cannot add another question.
    pub fn ask_ryan_about(&mut self, topic: &str, generation: usize) -> std::io::Result<bool> {
        let topic = topic.trim().to_ascii_lowercase();
        if !is_topic(&topic)
            || self.open_question.is_some()
            || self.last_question_answer_generation == Some(generation)
        {
            return Ok(false);
        }
        let text = match fs::read_to_string(&self.question_path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(error) => return Err(error),
        };
        let conversation_waiting = fs::read_to_string(&self.path)
            .ok()
            .is_some_and(|transcript| latest_unanswered(&transcript).is_some());
        if self.pending.is_some() || conversation_waiting || question_is_open(&text) {
            return Ok(false);
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.question_path)?;
        if !text.ends_with('\n') {
            writeln!(file)?;
        }
        // This is a mailbox protocol, rather than a reply template: it
        // exposes the explorer's selected topic and leaves the content of
        // Ryan's teaching completely open.
        writeln!(file, "Community: What is {topic}?")?;
        writeln!(file, "Ryan: ")?;
        self.open_question = Some(topic);
        self.last_prompt_generation = Some(generation);
        Ok(true)
    }

    fn answer(
        &mut self,
        prompt: &str,
        world: &mut WorldModel,
        lexicon: &CommunityLexicon,
        reading: &ReadingBridge,
        dictionary: &mut HumanDictionary,
        agents: &mut [Agent],
        rng: &mut Rng,
    ) -> String {
        let trimmed = prompt.trim();
        if matches!(trimmed, "+" | "feedback +") {
            return self.apply_feedback(1.0, world);
        }
        if matches!(trimmed, "-" | "feedback -") {
            return self.apply_feedback(-1.0, world);
        }
        let raw_words = tokenise(trimmed);
        let task = classify_dialogue_task(trimmed, &raw_words);
        let words = if task == DialogueTask::Reset {
            reset_content_words(&raw_words)
        } else {
            raw_words
        };
        self.dialogue_task = task;
        if task == DialogueTask::Stop {
            self.clear_working_context();
            // Silence is a valid conversational action. `poll` still writes
            // the Community protocol marker, so the prompt cannot reopen.
            return String::new();
        }
        if task == DialogueTask::Reset {
            self.clear_working_context();
        } else if matches!(
            task,
            DialogueTask::Arithmetic | DialogueTask::Number | DialogueTask::Referent
        ) {
            // Closed-form tasks must not inherit the noun currently being
            // discussed. Their answer route is complete in itself.
            self.park_working_context();
        }
        self.observe_numeric_literals(trimmed, agents);
        if let Some((left, operator, right)) = parse_arithmetic_expression(trimmed) {
            let result = match operator {
                '+' => left.checked_add(right),
                '-' => left.checked_sub(right),
                '*' => left.checked_mul(right),
                _ => None,
            };
            if let Some(result) = result {
                return format!("{left} {operator} {right} is {result}.");
            }
        }
        self.ingest(&words, dictionary, agents, rng);
        self.update_conversation_state(trimmed, &words);
        self.update_semantic_path(agents);

        if let Some(number) = parse_number_question(trimmed) {
            return self.describe_numeric_literal(number, lexicon);
        }
        if let Some(number) = parse_standalone_number(trimmed) {
            return self.describe_numeric_literal(number, lexicon);
        }
        if let Some(referent) = parse_show(trimmed) {
            if let Some(token) = lexicon.referential_signal(referent) {
                return format!("{referent} is '{token}' in our shared vocabulary.");
            }
        }
        if let Some(subject) = parse_world_question(&words) {
            if let Some(property) = world
                .relation(subject, "is")
                .or_else(|| world.fact(subject))
            {
                return format!("{subject} is {property}.");
            }
        }

        if let Some((relation, object)) = parse_relation_question(&words)
            && let Some(subject) = world.relation_subject(relation, object)
        {
            return format!("{subject} {relation} {object}.");
        }

        if let Some((subject, relation)) = parse_subject_relation_question(&words)
            && let Some(object) = world.relation(subject, relation)
        {
            return format!("{subject} {relation} {object}.");
        }

        if let Some((subject, property)) = parse_assertion(&words) {
            world.observe(&subject, &property, 0.5);
            self.last_claim = Some((subject.clone(), "is".to_string(), property.clone()));
            // The copula is retained as a directional relation. The leading
            // discourse token (for example `true,`) remains outside this
            // proposition and cannot become the subject of the claim.
            // Teaching is evidence, not an invitation to continue the last
            // free-association path. State the new grounded proposition and
            // leave the next turn open for correction or extension.
            return format!("{subject} is {property}.");
        }
        if let Some((subject, relation, object)) = parse_general_assertion(&words) {
            world.observe_relation(&subject, &relation, &object, 0.35);
            self.last_claim = Some((subject.clone(), relation.clone(), object.clone()));
            return format!("{subject} {relation} {object}.");
        }

        let topic = self.active_topic.clone();
        // Free conversation must pass through the community proposal/vote
        // path. In particular, a story word is evidence for a continuation,
        // not permission to emit the fixed "X and Y are related" sentence.
        self.compose_reply(&words, topic.as_deref(), reading, dictionary, agents, rng)
    }

    fn integrate_question_answer(&mut self, words: &[String], agents: &mut [Agent], rng: &mut Rng) {
        let Some(topic) = self.open_question.take() else {
            return;
        };
        for word in words.iter().filter(|word| is_topic(word)).take(4) {
            if word == &topic {
                continue;
            }
            for agent in agents.iter_mut() {
                agent.observe(&[topic.clone(), word.clone()], 0.12, rng);
            }
            self.dictionary_links
                .push((topic.clone(), word.clone(), 0.10));
        }
        self.resolved_question_topic = Some(topic);
    }

    fn observe_numeric_literals(&mut self, prompt: &str, agents: &mut [Agent]) {
        for literal in extract_numeric_literals(prompt) {
            self.numeric_literals.observe(literal);
            for agent in agents.iter_mut() {
                agent.observe_numeric_literal(literal);
            }
        }
    }

    fn describe_numeric_literal(&mut self, value: u32, lexicon: &CommunityLexicon) -> String {
        self.numeric_literals.observe(value);
        if let Some(phrase) = lexicon.numeric_phrase(value) {
            return format!("{value} is '{phrase}' in our shared number system.");
        }
        let digits = NumericLiteralMemory::decimal_digits(value)
            .into_iter()
            .map(|digit| digit.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        format!("{value} is a decimal number with digits {digits}.")
    }

    /// Quietly learn from an answer in `question.txt`.  It deliberately does
    /// not generate a second conversational turn: Ryan's answer itself is
    /// the useful evidence, and the mailbox is then free for a later topic.
    fn poll_question_answer(
        &mut self,
        generation: usize,
        dictionary: &mut HumanDictionary,
        agents: &mut [Agent],
        rng: &mut Rng,
    ) -> std::io::Result<()> {
        let text = match fs::read_to_string(&self.question_path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        let Some((topic, prompt)) = latest_question_answer(&text) else {
            self.question_pending = None;
            self.open_question = open_question_topic(&text);
            return Ok(());
        };
        let answer_key = (prompt.line_number, prompt.body.clone());
        if self.processed_question_answer.as_ref() == Some(&answer_key) {
            return Ok(());
        }
        if self
            .question_pending
            .as_ref()
            .is_none_or(|pending| pending.prompt != prompt)
        {
            self.question_pending = Some(PendingPrompt {
                prompt,
                transcript: text,
                first_seen_generation: generation,
            });
            return Ok(());
        }
        let pending = self
            .question_pending
            .as_ref()
            .expect("question pending was just set")
            .clone();
        if generation.saturating_sub(pending.first_seen_generation) < 2 {
            return Ok(());
        }
        if fs::read_to_string(&self.question_path)? != pending.transcript {
            self.question_pending = None;
            return Ok(());
        }
        self.open_question = Some(topic);
        self.observe_numeric_literals(&pending.prompt.body, agents);
        let words = tokenise(&pending.prompt.body);
        self.ingest(&words, dictionary, agents, rng);
        self.integrate_question_answer(&words, agents, rng);
        self.processed_question_answer = Some(answer_key);
        self.last_question_answer_generation = Some(generation);
        self.question_pending = None;
        Ok(())
    }

    fn ingest(
        &mut self,
        words: &[String],
        dictionary: &mut HumanDictionary,
        agents: &mut [Agent],
        rng: &mut Rng,
    ) {
        for word in words {
            *self.human_pool.entry(word.clone()).or_default() += 1.0;
        }
        for pair in words.windows(2) {
            *self
                .transitions
                .entry(pair[0].clone())
                .or_default()
                .entry(pair[1].clone())
                .or_default() += 1;
            *self
                .english_bigrams
                .entry((pair[0].clone(), pair[1].clone()))
                .or_default() += 1.0;
        }
        // Dictionary material is weak semantic scaffolding only.  It does
        // not assert a world fact and is not injected into response word
        // order, avoiding the synonym-attractor behaviour seen in Python's
        // early experiments.
        for word in words {
            let Some(evidence) = dictionary.meaning_evidence(word) else {
                continue;
            };
            for category in evidence.category_tokens {
                self.dictionary_links.push((word.clone(), category, 0.08));
            }
            for definition in evidence.definition_tokens {
                self.dictionary_links
                    .push((word.clone(), definition, 0.018));
            }
            for synonym in evidence.synonyms {
                self.dictionary_links.push((word.clone(), synonym, 0.12));
            }
            for antonym in evidence.antonyms {
                self.dictionary_links.push((word.clone(), antonym, -0.10));
            }
            for (left, right) in evidence.category_bigrams {
                // Preserve phrase-level evidence for semantic reporting, but
                // do not put its words in the human chat pool. A dictionary
                // gloss must not quietly become a response template.
                *self
                    .meaning_bigrams
                    .entry((left.clone(), right.clone()))
                    .or_default() += 0.35;
            }
        }
        // Human wording is the authoritative conversational evidence.  Each
        // member observes it before voting, but no word is assigned a fixed
        // meaning: local semantic links still emerge from co-occurrence.
        for agent in agents {
            agent.observe(words, 0.20, rng);
            agent.language.observe_human_tokens(words);
        }
    }

    fn update_conversation_state(&mut self, prompt: &str, words: &[String]) {
        self.decay_conversation_evidence();
        let shape = sentence_shape(prompt, words);
        let mode = self.choose_intent_mode(&shape, words);
        let proposed_topic = infer_topic(words).map(str::to_owned);
        let topic = self.resolve_topic(words, proposed_topic);
        self.active_topic = topic.clone();
        let profile = self.intent_modes.entry(mode.clone()).or_default();
        profile.uses += 1;
        *profile.shapes.entry(shape).or_default() += 1.0;
        for word in words {
            *profile.tokens.entry(word.clone()).or_default() += 1.0;
        }
        if let Some(topic) = topic.as_ref() {
            *profile.topics.entry(topic.clone()).or_default() += 1.0;
        }
        for pair in words.windows(2) {
            *profile
                .transitions
                .entry(pair[0].clone())
                .or_default()
                .entry(pair[1].clone())
                .or_default() += 1.0;
        }
        let frame_words: Vec<_> = self
            .pending_topic_switch
            .as_ref()
            .map(|(candidate, _)| {
                words
                    .iter()
                    .filter(|word| *word != candidate)
                    .cloned()
                    .collect()
            })
            .unwrap_or_else(|| words.to_vec());
        self.update_working_memory(mode, topic, &frame_words);
    }

    fn update_semantic_path(&mut self, agents: &[Agent]) {
        if let Some(topic) = self.active_topic.as_deref() {
            self.semantic_path.observe(topic, agents);
        }
    }

    fn resolve_topic(&mut self, words: &[String], proposed: Option<String>) -> Option<String> {
        let current = self.active_topic.clone();
        if releases_topic(words) {
            self.pending_topic_switch = None;
            return None;
        }
        let Some(proposed) = proposed else {
            self.pending_topic_switch = None;
            return current;
        };
        if current.as_deref() == Some(proposed.as_str()) || current.is_none() {
            self.pending_topic_switch = None;
            return Some(proposed);
        }
        if explicitly_introduces_topic(words) || parse_assertion(words).is_some() {
            self.pending_topic_switch = None;
            return Some(proposed);
        }
        let confirmations = self
            .pending_topic_switch
            .as_ref()
            .filter(|(candidate, _)| candidate == &proposed)
            .map(|(_, count)| *count + 1)
            .unwrap_or(1);
        if confirmations >= 2 {
            self.pending_topic_switch = None;
            Some(proposed)
        } else {
            self.pending_topic_switch = Some((proposed, confirmations));
            current
        }
    }

    fn choose_intent_mode(&mut self, shape: &str, words: &[String]) -> String {
        let word_set: BTreeSet<_> = words.iter().collect();
        let mut best: Option<(String, f64)> = None;
        for (id, mode) in &self.intent_modes {
            let lexical = word_set
                .iter()
                .map(|word| mode.tokens.get(*word).copied().unwrap_or(0.0).min(1.0))
                .sum::<f64>()
                / word_set.len().max(1) as f64;
            let sequential = words
                .windows(2)
                .map(|pair| {
                    mode.transitions
                        .get(&pair[0])
                        .and_then(|next| next.get(&pair[1]))
                        .copied()
                        .unwrap_or(0.0)
                        .min(1.0)
                })
                .sum::<f64>()
                / words.len().saturating_sub(1).max(1) as f64;
            let shape_match = mode.shapes.contains_key(shape) as u8 as f64;
            let score = 0.70 * lexical + 0.20 * sequential + 0.10 * shape_match;
            if best.as_ref().is_none_or(|(_, previous)| score > *previous) {
                best = Some((id.clone(), score));
            }
        }
        if let Some((mode, score)) = best
            && score >= 0.22
        {
            return mode;
        }
        let mode = format!("mode-{:03}", self.next_intent_mode);
        self.next_intent_mode += 1;
        self.intent_modes
            .insert(mode.clone(), IntentMode::default());
        mode
    }

    fn update_working_memory(&mut self, mode: String, topic: Option<String>, words: &[String]) {
        let switched = self
            .working_frame
            .as_ref()
            .is_some_and(|frame| frame.topic != topic);
        if switched {
            if let Some(frame) = self.working_frame.take()
                && !frame.tokens.is_empty()
            {
                self.parked_frames.retain(|old| old.topic != frame.topic);
                if self.parked_frames.len() == 4 {
                    self.parked_frames.pop_front();
                }
                self.parked_frames.push_back(frame);
            }
        }
        if self.working_frame.is_none() || switched {
            let restored = topic.as_ref().and_then(|topic| {
                let index = self
                    .parked_frames
                    .iter()
                    .position(|frame| frame.topic.as_ref() == Some(topic))?;
                self.parked_frames.remove(index)
            });
            self.working_frame = Some(restored.unwrap_or_else(|| WorkingFrame {
                topic: topic.clone(),
                mode: mode.clone(),
                tokens: BTreeMap::new(),
                transitions: BTreeMap::new(),
                turns: 0,
            }));
        }
        let frame = self
            .working_frame
            .as_mut()
            .expect("working frame was just created");
        frame.mode = mode;
        if topic.is_some() {
            frame.topic = topic;
        }
        frame.turns += 1;
        for word in words {
            *frame.tokens.entry(word.clone()).or_default() += 1.0;
        }
        for pair in words.windows(2) {
            *frame
                .transitions
                .entry(pair[0].clone())
                .or_default()
                .entry(pair[1].clone())
                .or_default() += 1.0;
        }
    }

    fn compose_reply(
        &mut self,
        prompt_words: &[String],
        topic: Option<&str>,
        reading: &ReadingBridge,
        dictionary: &mut HumanDictionary,
        agents: &mut [Agent],
        rng: &mut Rng,
    ) -> String {
        let first = topic
            .map(str::to_owned)
            .or_else(|| prompt_words.iter().find(|word| is_topic(word)).cloned())
            .or_else(|| prompt_words.last().cloned());
        let Some(first) = first else {
            return String::new();
        };
        let Some(mut allowed) = self
            .working_frame
            .as_ref()
            .map(|frame| frame.tokens.keys().cloned().collect::<BTreeSet<_>>())
        else {
            return String::new();
        };
        let pending_switch = self
            .pending_topic_switch
            .as_ref()
            .map(|(candidate, _)| candidate.as_str());
        // An unconfirmed noun can be relevant evidence, but cannot recruit a
        // reply into a different semantic neighbourhood on a single turn.
        allowed.extend(
            prompt_words
                .iter()
                .filter(|word| Some(word.as_str()) != pending_switch)
                .cloned(),
        );
        if let Some(candidate) = pending_switch {
            allowed.remove(candidate);
        }
        allowed.insert(first.clone());
        // A trajectory prediction is merely another low-confidence
        // continuation candidate. It is only available when this reply is
        // still anchored at the latest path waypoint.
        allowed.extend(
            self.semantic_path
                .predicted_words(&first)
                .map(|(word, _)| word.clone()),
        );
        let anchors = allowed.clone();
        let reading_words = reading.relevant_word_scores(&anchors);
        allowed.extend(reading_words.keys().cloned());

        // If the local conversation cannot continue from this word, widen
        // the agents' own semantic horizon. The dictionary may contribute
        // nearby candidates, but it never supplies a reply template.
        let has_local_continuation = self.working_frame.as_ref().is_some_and(|frame| {
            frame
                .transitions
                .get(&first)
                .is_some_and(|next| !next.is_empty())
        }) || self
            .transitions
            .get(&first)
            .is_some_and(|next| !next.is_empty())
            || !reading.relevant_continuations(&first, &allowed).is_empty();
        let stable_topic_frame = self
            .working_frame
            .as_ref()
            .is_some_and(|frame| frame.topic.as_deref() == topic && frame.turns >= 2);
        if !has_local_continuation && !stable_topic_frame {
            allowed.extend(self.expand_horizon(&first, dictionary, agents, rng));
        }
        let frame = self
            .working_frame
            .as_ref()
            .expect("working frame remains available while composing");

        // Every member proposes a short continuation from the same observed
        // sentence graph.  Private vocabularies make the proposals differ;
        // the final response is selected by a population vote below.
        let proposer_count = agents.len().max(1);
        let proposals: Vec<_> = (0..proposer_count)
            .filter_map(|index| {
                let agent = agents.get(index);
                let mut words = vec![first.clone()];
                let target_length = 3 + (index % 4);
                while words.len() < target_length {
                    let previous = words.last().expect("proposal is non-empty");
                    let mut candidates = self.promoted_continuations(previous, &allowed);
                    candidates.extend(self.horizon_continuations(previous, &allowed));
                    candidates.extend(
                        frame
                            .transitions
                            .get(previous)
                            .into_iter()
                            .flat_map(|next| next.keys())
                            .filter(|word| allowed.contains(*word))
                            .filter(|word| self.productive_human_pair(previous, word))
                            .cloned(),
                    );
                    candidates.extend(
                        self.transitions
                            .get(previous)
                            .into_iter()
                            .flat_map(|next| next.keys())
                            .filter(|word| allowed.contains(*word))
                            .filter(|word| self.productive_human_pair(previous, word))
                            .cloned(),
                    );
                    candidates.extend(reading.relevant_continuations(previous, &allowed));
                    candidates.sort();
                    candidates.dedup();
                    candidates.retain(|word| !words.contains(word));
                    candidates.retain(|word| !self.rejected_pair(previous, word));
                    if candidates.is_empty() {
                        candidates.extend(
                            self.human_pool
                                .keys()
                                .filter(|word| !words.contains(*word))
                                .filter(|word| allowed.contains(*word))
                                .filter(|word| !self.rejected_pair(previous, word))
                                .cloned(),
                        );
                    }
                    let Some(next) = choose_weighted_word(
                        &candidates,
                        |candidate| {
                            self.continuation_score(
                                agent,
                                previous,
                                candidate,
                                frame,
                                &reading_words,
                            )
                        },
                        rng,
                    ) else {
                        break;
                    };
                    words.push(next);
                }
                // A newly expanded horizon may have only one viable bridge
                // so far. It is still a community-authored attempt, and it
                // gives later turns something concrete to reinforce or
                // correct instead of declining to try.
                (words.len() >= 2).then(|| words.join(" "))
            })
            .collect();
        if proposals.is_empty() {
            // A turn must always be acknowledged once it has been accepted
            // as a prompt. Reusing its selected topic is intentionally a
            // minimal community-authored attempt, rather than emitting an
            // empty transcript line or a fixed refusal sentence.
            return format!("{first}.");
        }
        let mut ballot = BTreeMap::<String, usize>::new();
        for voter in agents {
            let chosen = proposals.iter().max_by(|left, right| {
                self.sentence_score(voter, left, prompt_words, topic, frame, &reading_words)
                    .partial_cmp(&self.sentence_score(
                        voter,
                        right,
                        prompt_words,
                        topic,
                        frame,
                        &reading_words,
                    ))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(chosen) = chosen {
                *ballot.entry(chosen.clone()).or_default() += 1;
            }
        }
        let phrase = ballot
            .into_iter()
            .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
            .map(|(proposal, _)| proposal)
            .or_else(|| proposals.first().cloned())
            .expect("proposals was checked non-empty");
        format!("{phrase}.")
    }

    /// Connect an unfamiliar word to concepts the population already has.
    /// Dictionary entries are one weak source of candidates alongside the
    /// agents' learned semantic neighbourhoods. The result is deliberately
    /// returned as a set of possible continuations, not as a sentence.
    fn expand_horizon(
        &mut self,
        word: &str,
        dictionary: &mut HumanDictionary,
        agents: &mut [Agent],
        rng: &mut Rng,
    ) -> Vec<String> {
        let word = word.trim().to_ascii_lowercase();
        if word.is_empty() {
            return Vec::new();
        }
        let mut candidates = BTreeMap::<String, f64>::new();
        if let Some(evidence) = dictionary.meaning_evidence(&word) {
            for candidate in evidence
                .category_tokens
                .into_iter()
                .chain(evidence.definition_tokens)
                .chain(evidence.synonyms)
            {
                if candidate != word && is_topic(&candidate) {
                    *candidates.entry(candidate).or_default() += 0.35;
                }
            }
        }
        for agent in agents.iter() {
            for candidate in agent.semantics.nearest(&word, 4) {
                if candidate != word && is_topic(&candidate) {
                    *candidates.entry(candidate).or_default() += 0.20;
                }
            }
        }
        // Even when neither source has an answer, an unfamiliar word should
        // be allowed to test itself against the community's current map.
        for candidate in self
            .human_pool
            .keys()
            .filter(|candidate| candidate.as_str() != word && is_topic(candidate))
        {
            *candidates.entry(candidate.clone()).or_default() += 0.05;
        }
        for agent in agents.iter() {
            for candidate in agent
                .vocabulary
                .iter()
                .filter(|candidate| candidate.as_str() != word && is_topic(candidate))
            {
                *candidates.entry(candidate.clone()).or_default() += 0.02;
            }
        }
        // If no recognisable content word exists yet, use any private token
        // as an exploratory neighbour. The population still chooses the
        // bridge; this merely prevents silence from being replaced by a
        // canned refusal sentence.
        if candidates.is_empty() {
            for agent in agents.iter() {
                for candidate in agent
                    .vocabulary
                    .iter()
                    .filter(|candidate| candidate.as_str() != word)
                {
                    *candidates.entry(candidate.clone()).or_default() += 0.01;
                }
            }
        }
        let mut weighted_candidates: Vec<_> = candidates.into_iter().collect();
        weighted_candidates.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        weighted_candidates.truncate(16);
        if weighted_candidates.is_empty() {
            return Vec::new();
        }
        let candidate_words: Vec<_> = weighted_candidates
            .iter()
            .map(|(candidate, _)| candidate.clone())
            .collect();

        // Give every agent a chance to choose its own bridge. These traces
        // are weak and decay; they do not become world facts or syntax.
        let mut selected = BTreeMap::<String, usize>::new();
        for agent in agents.iter_mut() {
            let chosen = choose_weighted_word(
                &candidate_words,
                |candidate| {
                    agent
                        .semantics
                        .cosine_similarity(&word, candidate)
                        .unwrap_or(0.0) as f64
                        + agent.language.salience(candidate).max(0.0)
                        + weighted_candidates
                            .iter()
                            .find(|(known, _)| known == candidate)
                            .map(|(_, evidence)| *evidence)
                            .unwrap_or(0.0)
                        + self
                            .human_pool
                            .get(candidate)
                            .copied()
                            .unwrap_or(0.0)
                            .ln_1p()
                            * 0.10
                },
                rng,
            )
            .expect("dictionary candidates were checked");
            agent.observe(&[word.clone(), chosen.clone()], 0.06, rng);
            self.dictionary_links
                .push((word.clone(), chosen.clone(), 0.05));
            *selected.entry(chosen).or_default() += 1;
        }
        let links = self.horizon_links.entry(word).or_default();
        for (candidate, votes) in &selected {
            *links.entry(candidate.clone()).or_default() += 0.05 * *votes as f64;
        }
        selected.into_keys().collect()
    }

    fn continuation_score(
        &self,
        agent: Option<&Agent>,
        previous: &str,
        candidate: &str,
        frame: &WorkingFrame,
        reading_words: &BTreeMap<String, f64>,
    ) -> f64 {
        let productive_pair = self.productive_human_pair(previous, candidate);
        let local = productive_pair
            .then(|| {
                frame
                    .transitions
                    .get(previous)
                    .and_then(|next| next.get(candidate))
                    .copied()
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0);
        let global = productive_pair
            .then(|| {
                self.transitions
                    .get(previous)
                    .and_then(|next| next.get(candidate))
                    .copied()
                    .unwrap_or_default() as f64
            })
            .unwrap_or(0.0);
        let familiar = agent
            .map(|agent| agent.vocabulary.contains(candidate) as u8 as f64)
            .unwrap_or(0.5);
        let human_repetition = self
            .human_pool
            .get(candidate)
            .copied()
            .unwrap_or(0.0)
            .ln_1p();
        let personal_salience = agent
            .map(|agent| agent.language.salience(candidate).max(0.0))
            .unwrap_or(0.0);
        let path_prediction = self.semantic_path.prediction_score(previous, candidate);
        (0.55 * self.bigram_strength(&(previous.to_string(), candidate.to_string())))
            + (0.45 * local)
            + (0.20 * global)
            + (0.20 * familiar)
            + (0.20 * human_repetition)
            + (0.12 * personal_salience)
            + (0.06 * path_prediction)
            + reading_words.get(candidate).copied().unwrap_or(0.0)
    }

    fn sentence_score(
        &self,
        voter: &Agent,
        sentence: &str,
        prompt_words: &[String],
        topic: Option<&str>,
        frame: &WorkingFrame,
        reading_words: &BTreeMap<String, f64>,
    ) -> f64 {
        let words = tokenise(sentence);
        if words.is_empty() {
            return f64::NEG_INFINITY;
        }
        let familiar = words
            .iter()
            .filter(|word| voter.vocabulary.contains(*word))
            .count() as f64
            / words.len() as f64;
        let local_pairs = words
            .windows(2)
            .map(|pair| {
                self.productive_human_pair(&pair[0], &pair[1])
                    .then(|| {
                        frame
                            .transitions
                            .get(&pair[0])
                            .and_then(|next| next.get(&pair[1]))
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .unwrap_or(0.0)
            })
            .sum::<f64>();
        let bigrams = words
            .windows(2)
            .map(|pair| self.bigram_strength(&(pair[0].clone(), pair[1].clone())))
            .sum::<f64>();
        let topical = topic
            .map(|topic| words.iter().filter(|word| word.as_str() == topic).count() as f64)
            .unwrap_or(0.0);
        let reading = words
            .iter()
            .map(|word| reading_words.get(word).copied().unwrap_or(0.0))
            .sum::<f64>();
        let human_repetition = words
            .iter()
            .map(|word| self.human_pool.get(word).copied().unwrap_or(0.0).ln_1p())
            .sum::<f64>()
            / words.len() as f64;
        let copied_prompt = (words == prompt_words) as u8 as f64;
        let reply_feedback = self
            .reply_feedback
            .get(sentence)
            .copied()
            .unwrap_or(0.0)
            .clamp(-3.0, 3.0);
        0.35 * familiar
            + 0.32 * local_pairs
            + 0.35 * bigrams
            + 0.55 * topical
            + 0.15 * reading
            + 0.16 * human_repetition
            - 0.25 * copied_prompt
            + 0.45 * reply_feedback
    }

    fn bigram_strength(&self, pair: &(String, String)) -> f64 {
        let observed = self.english_bigrams.get(pair).copied().unwrap_or(0.0);
        let feedback = self.bigram_feedback.get(pair).copied().unwrap_or(0.0);
        let meaning = self.meaning_bigrams.get(pair).copied().unwrap_or(0.0);
        let repeated = (observed - 1.0).max(0.0) / 2.0;
        (repeated + feedback + (0.15 * meaning).min(0.25)).clamp(-2.0, 2.0)
    }

    /// Human phrasing must recur before it can become productive syntax. A
    /// one-off question is evidence for understanding, not a reply template.
    fn productive_human_pair(&self, previous: &str, candidate: &str) -> bool {
        let pair = (previous.to_string(), candidate.to_string());
        self.bigram_strength(&pair) > -0.25
            && (self.english_bigrams.get(&pair).copied().unwrap_or(0.0) >= 2.0
                || self.bigram_feedback.get(&pair).copied().unwrap_or(0.0) > 0.0)
    }

    fn rejected_pair(&self, previous: &str, candidate: &str) -> bool {
        self.bigram_strength(&(previous.to_string(), candidate.to_string())) <= -0.25
    }

    fn promoted_continuations(&self, previous: &str, allowed: &BTreeSet<String>) -> Vec<String> {
        self.english_bigrams
            .keys()
            .filter(|pair| pair.0 == previous && allowed.contains(&pair.1))
            .filter(|pair| self.bigram_strength(pair) > 0.0)
            .map(|pair| pair.1.clone())
            .collect()
    }

    fn horizon_continuations(&self, previous: &str, allowed: &BTreeSet<String>) -> Vec<String> {
        self.horizon_links
            .get(previous)
            .into_iter()
            .flat_map(|links| links.iter())
            .filter(|(candidate, strength)| **strength >= 0.04 && allowed.contains(*candidate))
            .map(|(candidate, _)| candidate.clone())
            .collect()
    }

    fn apply_feedback(&mut self, reward: f64, world: &mut WorldModel) -> String {
        let Some(reply) = self.last_reply.clone() else {
            return "There is no recent community sentence to score.".to_string();
        };
        let reward = if reward > 0.0 { 1.0 } else { -1.0 };
        if reward < 0.0
            && let Some((subject, relation, object)) = self.last_claim.take()
        {
            world.contradict_relation(&subject, &relation, &object);
        }
        *self.reply_feedback.entry(reply.clone()).or_default() += reward;
        let words = tokenise(&reply);
        for pair in words.windows(2) {
            *self
                .bigram_feedback
                .entry((pair[0].clone(), pair[1].clone()))
                .or_default() += reward;
        }
        if reward > 0.0 {
            self.positive_feedback += 1;
            "Feedback accepted for the last community sentence.".to_string()
        } else {
            self.negative_feedback += 1;
            // A rejection is both learning evidence and a pragmatic cue to
            // stop riding the same topic/trajectory. The frame is parked so
            // its long-term links survive, but the next reply starts fresh.
            self.park_working_context();
            "Feedback rejected for the last community sentence.".to_string()
        }
    }

    fn decay_conversation_evidence(&mut self) {
        self.reply_feedback.retain(|_, score| {
            *score *= 0.995;
            score.abs() >= 0.025
        });
        self.bigram_feedback.retain(|_, score| {
            *score *= 0.995;
            score.abs() >= 0.025
        });
        self.english_bigrams.retain(|_, score| {
            *score *= 0.985;
            *score >= 0.025
        });
        self.meaning_bigrams.retain(|_, score| {
            *score *= 0.990;
            *score >= 0.025
        });
        for links in self.horizon_links.values_mut() {
            links.retain(|_, strength| {
                *strength *= 0.94;
                *strength >= 0.02
            });
        }
        self.horizon_links.retain(|_, links| !links.is_empty());
        self.human_pool.retain(|_, score| {
            // Conversation is working memory, not a permanent word bag.
            // Long-term semantic links live in agent/community semantics;
            // transcript salience should release a topic quickly.
            *score *= 0.90;
            *score >= 0.025
        });
        for mode in self.intent_modes.values_mut() {
            decay_counts(&mut mode.tokens, 0.96);
            decay_counts(&mut mode.shapes, 0.97);
            decay_counts(&mut mode.topics, 0.90);
            for next in mode.transitions.values_mut() {
                decay_counts(next, 0.86);
            }
        }
    }

    fn expire_stale_context(&mut self, generation: usize) {
        const CONTEXT_TTL_GENERATIONS: usize = 24;
        let Some(last_prompt) = self.last_prompt_generation else {
            return;
        };
        if generation.saturating_sub(last_prompt) <= CONTEXT_TTL_GENERATIONS {
            return;
        }
        self.park_working_context();
    }

    /// Park the ephemeral frame but preserve learned semantic and world maps.
    /// Bounded tasks such as arithmetic use this so an old topic cannot tint
    /// the answer, yet can be resumed if Ryan explicitly returns to it.
    fn park_working_context(&mut self) {
        if let Some(frame) = self.working_frame.take()
            && !frame.tokens.is_empty()
        {
            self.parked_frames.retain(|old| old.topic != frame.topic);
            if self.parked_frames.len() == 4 {
                self.parked_frames.pop_front();
            }
            self.parked_frames.push_back(frame);
        }
        self.active_topic = None;
        self.pending_topic_switch = None;
        self.semantic_path.clear();
    }

    /// A human-directed reset clears short-lived conversational attractors
    /// only. Long-term world facts, semantic maps, dictionary bridges and
    /// learned language modes are intentionally retained.
    fn clear_working_context(&mut self) {
        self.park_working_context();
        self.human_pool.clear();
        self.transitions.clear();
        self.english_bigrams.clear();
        self.horizon_links.clear();
        self.last_reply = None;
    }
}

fn decay_counts(counts: &mut BTreeMap<String, f64>, factor: f64) {
    counts.retain(|_, value| {
        *value *= factor;
        *value >= 0.025
    });
}

/// Each agent samples from its own scored continuation distribution. This is
/// deliberately not a global argmax: individual semantic histories must be
/// able to produce competing sentence proposals before the community votes.
fn choose_weighted_word(
    candidates: &[String],
    score: impl Fn(&str) -> f64,
    rng: &mut Rng,
) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }
    let weights: Vec<_> = candidates
        .iter()
        .map(|candidate| (0.05 + score(candidate).max(0.0)).min(20.0))
        .collect();
    let total = weights.iter().sum::<f64>();
    let mut threshold = rng.unit() * total;
    for (candidate, weight) in candidates.iter().zip(weights) {
        threshold -= weight;
        if threshold <= 0.0 {
            return Some(candidate.clone());
        }
    }
    candidates.last().cloned()
}

/// The community path is evaluated from the population's independent maps,
/// not from one privileged agent. Missing vectors are simply abstentions.
fn mean_semantic_similarity(agents: &[Agent], left: &str, right: &str) -> Option<f32> {
    let similarities: Vec<_> = agents
        .iter()
        .filter_map(|agent| agent.semantics.cosine_similarity(left, right))
        .collect();
    (!similarities.is_empty()).then(|| similarities.iter().sum::<f32>() / similarities.len() as f32)
}

/// Compare the direction `previous -> current` with `current -> candidate`
/// inside every agent that knows all three concepts, then average the votes.
fn mean_directional_continuity(
    agents: &[Agent],
    previous: &str,
    current: &str,
    candidate: &str,
) -> Option<f32> {
    let alignments: Vec<_> = agents
        .iter()
        .filter_map(|agent| {
            let previous = agent.semantics.vector(previous)?;
            let current = agent.semantics.vector(current)?;
            let candidate = agent.semantics.vector(candidate)?;
            let (mut dot, mut left_norm, mut right_norm) = (0.0_f32, 0.0_f32, 0.0_f32);
            for ((previous, current), candidate) in previous.iter().zip(current).zip(candidate) {
                let incoming = current - previous;
                let outgoing = candidate - current;
                dot += incoming * outgoing;
                left_norm += incoming * incoming;
                right_norm += outgoing * outgoing;
            }
            let denominator = left_norm.sqrt() * right_norm.sqrt();
            (denominator > f32::EPSILON).then_some(dot / denominator)
        })
        .collect();
    (!alignments.is_empty()).then(|| alignments.iter().sum::<f32>() / alignments.len() as f32)
}

fn sentence_shape(prompt: &str, words: &[String]) -> String {
    let terminal = prompt.trim().chars().last().unwrap_or(' ');
    let terminal = if matches!(terminal, '?' | '!' | '.') {
        terminal
    } else {
        '_'
    };
    format!("{terminal}:{}", words.len().min(6))
}

fn is_feedback(prompt: &str) -> bool {
    matches!(prompt.trim(), "+" | "-" | "feedback +" | "feedback -")
}

fn classify_dialogue_task(prompt: &str, words: &[String]) -> DialogueTask {
    if is_reset_request(words) {
        return DialogueTask::Reset;
    }
    if is_stop_request(words) {
        return DialogueTask::Stop;
    }
    if parse_arithmetic_expression(prompt).is_some() {
        return DialogueTask::Arithmetic;
    }
    if parse_number_question(prompt).is_some() || parse_standalone_number(prompt).is_some() {
        return DialogueTask::Number;
    }
    if parse_show(prompt).is_some() {
        return DialogueTask::Referent;
    }
    if parse_world_question(words).is_some() {
        return DialogueTask::Definition;
    }
    if parse_assertion(words).is_some() {
        return DialogueTask::Teaching;
    }
    DialogueTask::Conversation
}

/// Treat a deliberate reset as a control act even when it contains the small
/// typo that naturally occurred in the transcript ("conversaion"). This is
/// not semantic correction; it is a forgiving file-based UI command.
fn is_reset_request(words: &[String]) -> bool {
    let has_new = words.iter().any(|word| word == "new" || word == "fresh");
    let has_conversation = words.iter().any(|word| {
        matches!(
            word.as_str(),
            "conversation" | "conversaion" | "conversaton"
        )
    });
    (has_new && has_conversation)
        || words.windows(2).any(|pair| pair == ["new", "topic"])
        || words.windows(2).any(|pair| pair == ["start", "over"])
        || words.windows(2).any(|pair| pair == ["clear", "context"])
        || words.windows(2).any(|pair| pair == ["change", "topic"])
        || words.windows(2).any(|pair| pair == ["switch", "topic"])
}

fn is_stop_request(words: &[String]) -> bool {
    matches!(words, [word] if matches!(word.as_str(), "stop" | "quiet" | "enough" | "pause"))
        || words.windows(2).any(|pair| pair == ["stop", "talking"])
}

fn reset_content_words(words: &[String]) -> Vec<String> {
    words
        .iter()
        .filter(|word| {
            !matches!(
                word.as_str(),
                "new"
                    | "fresh"
                    | "conversation"
                    | "conversaion"
                    | "conversaton"
                    | "start"
                    | "over"
                    | "clear"
                    | "context"
                    | "change"
                    | "switch"
                    | "topic"
            )
        })
        .cloned()
        .collect()
}

fn latest_unanswered(text: &str) -> Option<TranscriptPrompt> {
    let mut pending = None;
    for (line_number, line) in text.lines().enumerate() {
        let Some((role, body)) = line.split_once(':') else {
            continue;
        };
        let body = body.trim();
        match role.trim().to_ascii_lowercase().as_str() {
            "ryan"
                if !body.is_empty()
                    && (is_feedback(body)
                        || !tokenise(body).is_empty()
                        || parse_arithmetic_expression(body).is_some()
                        || parse_standalone_number(body).is_some()) =>
            {
                pending = Some(TranscriptPrompt {
                    line_number: line_number + 1,
                    body: body.to_string(),
                });
            }
            // A Community line is a protocol acknowledgement even if an
            // older build left its text blank. This prevents a malformed
            // answer from repeatedly reopening the same Ryan prompt.
            "community" => pending = None,
            _ => {}
        }
    }
    pending
}

fn parse_arithmetic_expression(prompt: &str) -> Option<(u32, char, u32)> {
    let compact: String = prompt
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let (operator_index, operator) = compact
        .char_indices()
        .find(|(_, character)| matches!(character, '+' | '-' | '*'))?;
    let left = compact[..operator_index].parse().ok()?;
    let remainder = &compact[operator_index + operator.len_utf8()..];
    let right = remainder
        .split_once('=')
        .map(|(right, _)| right)
        .unwrap_or(remainder)
        .parse()
        .ok()?;
    Some((left, operator, right))
}

/// The explorer mailbox contains only questions written by the community in
/// the exact one-topic form below.  Keeping this parser deliberately narrow
/// means ordinary notes a human might keep in `question.txt` cannot be
/// mistaken for teaching evidence.
fn generated_question_topic(body: &str) -> Option<String> {
    let question = body.trim().strip_suffix('?')?.trim();
    let topic = question
        .strip_prefix("What is ")
        .or_else(|| question.strip_prefix("what is "))?
        .trim();
    let words = tokenise(topic);
    (words.len() == 1 && is_topic(&words[0])).then(|| words[0].clone())
}

fn question_entries(text: &str) -> Vec<(usize, String, String)> {
    text.lines()
        .enumerate()
        .filter_map(|(line_number, line)| {
            let (role, body) = line.split_once(':')?;
            Some((
                line_number + 1,
                role.trim().to_ascii_lowercase(),
                body.trim().to_string(),
            ))
        })
        .collect()
}

fn open_question_topic(text: &str) -> Option<String> {
    let entries = question_entries(text);
    let (index, (_, _, body)) = entries
        .iter()
        .enumerate()
        .rev()
        .find(|(_, (_, role, body))| {
            role == "community" && generated_question_topic(body).is_some()
        })?;
    let topic = generated_question_topic(body)?;
    let has_answer = entries
        .iter()
        .skip(index + 1)
        .find(|(_, role, _)| role == "ryan")
        .is_some_and(|(_, _, answer)| !answer.is_empty());
    (!has_answer).then_some(topic)
}

fn question_is_open(text: &str) -> bool {
    open_question_topic(text).is_some()
}

fn latest_question_answer(text: &str) -> Option<(String, TranscriptPrompt)> {
    let entries = question_entries(text);
    for (index, (line_number, role, body)) in entries.iter().enumerate().rev() {
        if role != "ryan" || body.is_empty() {
            continue;
        }
        let (_, previous_role, previous_body) = entries.get(index.checked_sub(1)?)?;
        if previous_role == "community"
            && let Some(topic) = generated_question_topic(previous_body)
        {
            return Some((
                topic,
                TranscriptPrompt {
                    line_number: *line_number,
                    body: body.clone(),
                },
            ));
        }
    }
    None
}

fn parse_number_question(prompt: &str) -> Option<u32> {
    let lower = prompt.to_ascii_lowercase();
    ["how many is ", "what is "].iter().find_map(|prefix| {
        lower
            .strip_suffix('?')?
            .strip_prefix(prefix)?
            .trim()
            .parse()
            .ok()
    })
}

fn parse_standalone_number(prompt: &str) -> Option<u32> {
    let cleaned = prompt.trim().trim_end_matches(['?', '.', '!']);
    (!cleaned.is_empty() && cleaned.chars().all(|character| character.is_ascii_digit()))
        .then(|| cleaned.parse().ok())
        .flatten()
}

/// Decimal literals are extracted as typed values rather than routed through
/// `tokenise`, which is intentionally English-word-only. The value cap is a
/// practical guard against turning arbitrary long digit strings into state.
fn extract_numeric_literals(text: &str) -> Vec<u32> {
    const MAX_LITERAL_VALUE: u32 = 999_999_999;
    text.split(|character: char| !character.is_ascii_digit())
        .filter(|part| !part.is_empty() && part.len() <= 9)
        .filter_map(|part| part.parse::<u32>().ok())
        .filter(|value| *value <= MAX_LITERAL_VALUE)
        .collect()
}

fn parse_show(prompt: &str) -> Option<&str> {
    let words: Vec<_> = prompt
        .trim_end_matches(['?', '.', '!'])
        .split_whitespace()
        .collect();
    if words.len() == 2 && matches!(words[0].to_ascii_lowercase().as_str(), "show" | "name") {
        Some(words[1])
    } else {
        None
    }
}

fn parse_world_question(words: &[String]) -> Option<&str> {
    if words.len() >= 3
        && matches!(words[0].as_str(), "what" | "who")
        && matches!(words[1].as_str(), "is" | "are")
    {
        // A definition has one noun phrase after the copula: "what is a
        // statue?". Open questions such as "what are you pushing towards?"
        // may contain several content words and must stay in free dialogue.
        let content: Vec<_> = words
            .iter()
            .skip(2)
            .filter(|word| !matches!(word.as_str(), "a" | "an" | "the"))
            .filter(|word| is_topic(word))
            .collect();
        // `then_some` evaluates its argument eagerly, so indexing there
        // panics for valid open questions with no topic words.
        if content.len() == 1 {
            Some(content[0].as_str())
        } else {
            None
        }
    } else {
        None
    }
}

fn explicitly_introduces_topic(words: &[String]) -> bool {
    words
        .iter()
        .any(|word| matches!(word.as_str(), "about" | "called" | "named"))
        || parse_world_question(words).is_some()
}

fn releases_topic(words: &[String]) -> bool {
    matches!(words.first().map(String::as_str), Some("who" | "what"))
        && matches!(words.get(1).map(String::as_str), Some("are"))
        && infer_topic(words).is_none()
}

/// Identify the subject of *this* turn before considering the prior frame.
/// Markers such as `about` and interrogative auxiliaries let a new topic beat
/// generic words like `talk`, while the simple statement subject remains the
/// preferred anchor for grounded assertions.
fn infer_topic(words: &[String]) -> Option<&str> {
    for marker in ["about", "called", "named"] {
        if let Some(index) = words.iter().position(|word| word == marker)
            && let Some(topic) = words.iter().skip(index + 1).find(|word| is_topic(word))
        {
            return Some(topic);
        }
    }
    if matches!(words.first().map(String::as_str), Some("what" | "who"))
        && matches!(words.get(1).map(String::as_str), Some("is" | "are"))
    {
        return parse_world_question(words);
    }
    if matches!(
        words.first().map(String::as_str),
        Some("how" | "what" | "who" | "did" | "can")
    ) {
        for marker in ["do", "does", "did", "can"] {
            if let Some(index) = words.iter().position(|word| word == marker)
                && let Some(topic) = words.iter().skip(index + 1).find(|word| is_topic(word))
            {
                return Some(topic);
            }
        }
    }
    statement_subject(words)
        .or_else(|| words.iter().find(|word| is_topic(word)).map(String::as_str))
}

fn statement_subject(words: &[String]) -> Option<&str> {
    let copula = words
        .iter()
        .position(|word| matches!(word.as_str(), "is" | "are" | "has" | "have"))?;
    words[..copula]
        .iter()
        .rev()
        .find(|word| is_topic(word))
        .map(String::as_str)
}

fn parse_assertion(words: &[String]) -> Option<(String, String)> {
    let subject = statement_subject(words)?.to_string();
    let copula = words
        .iter()
        .position(|word| matches!(word.as_str(), "is" | "are"))?;
    let property = words
        .iter()
        .skip(copula + 1)
        .take_while(|word| !matches!(word.as_str(), "because" | "but" | "however"))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    (!property.is_empty()).then_some((subject, property))
}

fn parse_general_assertion(words: &[String]) -> Option<(String, String, String)> {
    if words.len() < 3 || matches!(words.first()?.as_str(), "true" | "false") {
        return None;
    }
    let subject = words.first().filter(|word| is_topic(word))?.clone();
    let relation = words.get(1).filter(|word| is_topic(word))?.clone();
    let object = words[2..]
        .iter()
        .filter(|word| is_topic(word))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    (!object.is_empty() && relation != "is" && relation != "are")
        .then_some((subject, relation, object))
}

fn parse_relation_question(words: &[String]) -> Option<(&str, &str)> {
    if words.len() >= 3 && words.first().is_some_and(|word| word == "who") {
        let relation = words.get(1).filter(|word| is_topic(word))?;
        let object = words[2..].iter().find(|word| is_topic(word))?;
        return Some((relation, object));
    }
    None
}

fn parse_subject_relation_question(words: &[String]) -> Option<(&str, &str)> {
    if words.len() >= 4
        && words.first().is_some_and(|word| word == "what")
        && words.get(1).is_some_and(|word| word == "does")
    {
        let subject = words.get(2).filter(|word| is_topic(word))?;
        let relation = words.get(3).filter(|word| is_topic(word))?;
        return Some((subject, relation));
    }
    None
}

fn is_topic(word: &str) -> bool {
    word.len() > 2
        && !matches!(
            word,
            "a" | "an"
                | "about"
                | "and"
                | "are"
                | "be"
                | "can"
                | "do"
                | "does"
                | "for"
                | "have"
                | "how"
                | "i"
                | "im"
                | "is"
                | "it"
                | "me"
                | "my"
                | "of"
                | "on"
                | "tell"
                | "talk"
                | "think"
                | "say"
                | "try"
                | "the"
                | "to"
                | "we"
                | "what"
                | "who"
                | "many"
                | "you"
                | "your"
                | "this"
                | "that"
        )
}

#[cfg(test)]
mod tests {
    use super::{
        Conversation, SemanticPath, is_topic, latest_unanswered, parse_show, parse_world_question,
        question_is_open, tokenise,
    };
    use crate::dictionary::HumanDictionary;
    use crate::lexicon::CommunityLexicon;
    use crate::model::{Agent, Rng, WorldModel};
    use crate::reading::ReadingBridge;
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;

    #[test]
    fn working_memory_parks_and_restores_by_explicit_topic() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let apple = tokenise("Apple is red.");
        conversation.update_conversation_state("Apple is red.", &apple);
        let banana = tokenise("Let's talk about banana.");
        conversation.update_conversation_state("Let's talk about banana.", &banana);
        assert_eq!(
            conversation
                .working_frame
                .as_ref()
                .and_then(|frame| frame.topic.as_deref()),
            Some("banana")
        );
        assert!(
            conversation
                .parked_frames
                .iter()
                .any(|frame| frame.topic.as_deref() == Some("apple"))
        );
        let apple_topic = tokenise("Let's talk about apple.");
        conversation.update_conversation_state("Let's talk about apple.", &apple_topic);
        assert_eq!(
            conversation
                .working_frame
                .as_ref()
                .and_then(|frame| frame.topic.as_deref()),
            Some("apple")
        );
    }

    #[test]
    fn incomplete_show_prompt_is_not_a_question_and_never_panics() {
        assert_eq!(parse_show("show"), None);
        assert_eq!(parse_show("name"), None);
        assert_eq!(parse_show("show river"), Some("river"));
        assert_eq!(parse_show("name apple"), Some("apple"));
    }

    #[test]
    fn identical_new_ryan_lines_are_distinct_prompts() {
        let prompt = latest_unanswered("Ryan: hi\nCommunity: hello\nRyan: hi\n")
            .expect("the second Ryan line is unanswered");
        assert_eq!(prompt.line_number, 3);
        assert_eq!(prompt.body, "hi");
    }

    #[test]
    fn blank_community_marker_does_not_reopen_a_prompt() {
        assert!(latest_unanswered("Ryan: 1 + 1 = 2\nCommunity: \nRyan: \n").is_none());
    }

    #[test]
    fn standalone_decimal_literal_is_a_valid_prompt() {
        let prompt = latest_unanswered("Ryan: 123\n").expect("numeric literal prompt");
        assert_eq!(prompt.body, "123");
    }

    #[test]
    fn feedback_markers_remain_valid_prompts() {
        assert_eq!(
            latest_unanswered("Ryan: +\n").map(|prompt| prompt.body),
            Some("+".to_string())
        );
        assert_eq!(
            latest_unanswered("Ryan: -\n").map(|prompt| prompt.body),
            Some("-".to_string())
        );
    }

    #[test]
    fn arithmetic_prompt_has_a_nonempty_grounded_reply() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(89);
        let mut agents = (0..4)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        assert_eq!(
            conversation.answer(
                "1 + 1 = 2",
                &mut world,
                &lexicon,
                &reading,
                &mut dictionary,
                &mut agents,
                &mut rng,
            ),
            "1 + 1 is 2."
        );
    }

    #[test]
    fn arithmetic_parks_an_old_topic_before_answering() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(131);
        let mut agents = (0..4)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        conversation.answer(
            "What is a game?",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert_eq!(conversation.topic(), Some("game"));
        assert_eq!(
            conversation.answer(
                "1 + 2",
                &mut world,
                &lexicon,
                &reading,
                &mut dictionary,
                &mut agents,
                &mut rng,
            ),
            "1 + 2 is 3."
        );
        assert_eq!(conversation.dialogue_task(), "arithmetic");
        assert_eq!(conversation.topic(), None);
        assert!(
            conversation
                .parked_frames
                .iter()
                .any(|frame| frame.topic.as_deref() == Some("game"))
        );
    }

    #[test]
    fn teaching_retrieves_a_compact_fact_and_replaces_the_old_subject() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(137);
        let mut agents = (0..4)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        conversation.answer(
            "What is a future?",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert_eq!(
            conversation.answer(
                "Past is before now.",
                &mut world,
                &lexicon,
                &reading,
                &mut dictionary,
                &mut agents,
                &mut rng,
            ),
            "past is before now."
        );
        assert_eq!(conversation.dialogue_task(), "teaching");
        assert_eq!(conversation.topic(), Some("past"));
        assert_eq!(world.fact("past"), Some("before now"));
        assert_eq!(
            conversation.answer(
                "What is past?",
                &mut world,
                &lexicon,
                &reading,
                &mut dictionary,
                &mut agents,
                &mut rng,
            ),
            "past is before now."
        );
        assert_eq!(conversation.dialogue_task(), "definition");
    }

    #[test]
    fn reset_clears_short_term_attractors_but_retains_world_knowledge() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(139);
        let mut agents = (0..4)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        conversation.answer(
            "Statue is a rock likeness.",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert_eq!(conversation.topic(), Some("statue"));
        assert_eq!(
            conversation.answer(
                "New conversaion.",
                &mut world,
                &lexicon,
                &reading,
                &mut dictionary,
                &mut agents,
                &mut rng,
            ),
            ""
        );
        assert_eq!(conversation.dialogue_task(), "reset");
        assert_eq!(conversation.topic(), None);
        assert_eq!(world.fact("statue"), Some("a rock likeness"));
        assert!(conversation.human_pool.is_empty());
    }

    #[test]
    fn change_topic_control_can_start_a_fresh_named_frame() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(143);
        let mut agents = (0..4)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        conversation.answer(
            "Statue is a rock.",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        let reply = conversation.answer(
            "Change topic to games.",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert_eq!(conversation.dialogue_task(), "reset");
        assert_eq!(conversation.topic(), Some("games"));
        assert!(tokenise(&reply).first().is_some_and(|word| word == "games"));
        assert!(!conversation.human_pool.contains_key("statue"));
    }

    #[test]
    fn stop_is_a_silent_control_act_not_a_new_semantic_topic() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(149);
        let mut agents = (0..4)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        conversation.answer(
            "What is a game?",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert_eq!(
            conversation.answer(
                "stop",
                &mut world,
                &lexicon,
                &reading,
                &mut dictionary,
                &mut agents,
                &mut rng,
            ),
            ""
        );
        assert_eq!(conversation.dialogue_task(), "stop");
        assert_eq!(conversation.topic(), None);
    }

    #[test]
    fn decimal_literals_are_bounded_typed_evidence_not_word_vocabulary() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(103);
        let mut agents = (0..4)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        let reply = conversation.answer(
            "I have 1, 2, and 123 apples.",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert!(!reply.is_empty());
        assert_eq!(conversation.numeric_literal_metrics(), (3, 3));
        for agent in &agents {
            assert_eq!(agent.recognised_literals.get(&123), Some(&1));
            assert!(!agent.vocabulary.contains("123"));
        }
    }

    #[test]
    fn number_question_uses_composed_public_numerals_when_available() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let mut lexicon = CommunityLexicon::default();
        for (digit, token) in [(0, "na"), (1, "bel"), (2, "muk"), (3, "tor")] {
            lexicon.observe_numeric_success(digit, token);
            lexicon.observe_numeric_success(digit, token);
        }
        lexicon.observe_base_success(4);
        lexicon.observe_base_success(4);
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(107);
        let mut agents = (0..4)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        assert_eq!(
            conversation.answer(
                "What is 27?",
                &mut world,
                &lexicon,
                &reading,
                &mut dictionary,
                &mut agents,
                &mut rng,
            ),
            "27 is 'bel muk tor' in our shared number system."
        );
    }

    #[test]
    fn function_words_do_not_capture_the_discussion_topic() {
        assert!(!is_topic("can"));
        assert!(!is_topic("tell"));
        assert!(!is_topic("you"));
        assert!(is_topic("answer"));
    }

    #[test]
    fn explicit_new_topic_replaces_the_previous_working_frame() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let town = tokenise("The town is quiet.");
        conversation.update_conversation_state("The town is quiet.", &town);
        let game = tokenise("Can we talk about a game?");
        conversation.update_conversation_state("Can we talk about a game?", &game);
        assert_eq!(conversation.topic(), Some("game"));
        let frame = conversation
            .working_frame
            .as_ref()
            .expect("new topic frame");
        assert_eq!(frame.topic.as_deref(), Some("game"));
        assert!(!frame.tokens.contains_key("town"));
        assert!(
            conversation
                .parked_frames
                .iter()
                .any(|frame| frame.topic.as_deref() == Some("town"))
        );
    }

    #[test]
    fn incidental_noun_does_not_hijack_a_committed_topic_on_one_turn() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(113);
        let mut agents = (0..6)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        conversation.answer(
            "What is a game?",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        let reply = conversation.answer(
            "A crease folds badly.",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert_eq!(conversation.topic(), Some("game"));
        assert_eq!(
            conversation.pending_topic_switch.as_ref(),
            Some(&("crease".to_string(), 1))
        );
        assert_eq!(tokenise(&reply).first().map(String::as_str), Some("game"));
        assert!(!tokenise(&reply).contains(&"crease".to_string()));

        conversation.answer(
            "A crease folds badly.",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert_eq!(conversation.topic(), Some("crease"));
    }

    #[test]
    fn stale_context_is_parked_after_an_idle_generation_gap() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let town = tokenise("The town is quiet.");
        conversation.update_conversation_state("The town is quiet.", &town);
        conversation.last_prompt_generation = Some(4);
        conversation.expire_stale_context(29);
        assert_eq!(conversation.topic(), None);
        assert!(conversation.working_frame.is_none());
        assert!(
            conversation
                .parked_frames
                .iter()
                .any(|frame| frame.topic.as_deref() == Some("town"))
        );
    }

    #[test]
    fn an_unlikely_topic_jump_resets_the_short_semantic_path() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let apple = tokenise("Apple is round.");
        conversation.update_conversation_state("Apple is round.", &apple);
        // An empty population has no geometric evidence, so a different
        // human-selected topic is correctly treated as a fresh trajectory.
        conversation.update_semantic_path(&[]);
        let ocean = tokenise("Let's talk about ocean.");
        conversation.update_conversation_state("Let's talk about ocean.", &ocean);
        conversation.update_semantic_path(&[]);
        assert_eq!(conversation.semantic_path.points.len(), 1);
        assert_eq!(
            conversation.semantic_path.points.back().map(String::as_str),
            Some("ocean")
        );
        assert_eq!(conversation.semantic_path.resets, 1);
    }

    #[test]
    fn free_reply_never_uses_the_removed_related_template() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(91);
        let mut agents = (0..6)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        let reply = conversation.answer(
            "Can you answer?",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert!(!reply.contains(" are related"), "reply was {reply:?}");
        assert_eq!(conversation.topic(), Some("answer"));
    }

    #[test]
    fn unresolved_queries_are_composed_not_replaced_by_a_refusal_template() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(97);
        let mut agents = (0..6)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        let reply = conversation.answer(
            "What is florp?",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert!(
            !reply.is_empty(),
            "an unresolved prompt should still be attempted"
        );
        for template in [
            "We do not yet have a confirmed fact",
            "I am looking for a concept to connect",
            "We need a new example",
        ] {
            assert!(!reply.contains(template), "reply was {reply:?}");
        }
    }

    #[test]
    fn novel_word_expands_the_community_horizon_without_a_reply_template() {
        let path = std::env::temp_dir().join(format!(
            "mas-rust-conversation-lookup-{}.json",
            std::process::id()
        ));
        fs::write(
            &path,
            r#"{"QUOKKA":{"MEANINGS":[["Noun","a small marsupial animal",["Small animal"],[]]],"SYNONYMS":[],"ANTONYMS":[]}}"#,
        )
        .expect("write dictionary");
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::at(path.clone());
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(93);
        let mut agents = (0..6)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        let reply = conversation.answer(
            "Quokka?",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        assert!(reply.starts_with("quokka "), "reply was {reply:?}");
        assert!(!reply.contains("may be"), "reply was {reply:?}");
        assert!(reply.split_whitespace().count() >= 2, "reply was {reply:?}");
        assert!(
            conversation
                .horizon_links
                .get("quokka")
                .is_some_and(|links| !links.is_empty()),
            "dictionary evidence did not widen the semantic horizon"
        );
        assert!(!reply.contains("more context"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn identical_human_turns_are_settled_and_answered_twice() {
        let path = std::env::temp_dir().join(format!(
            "mas-rust-conversation-repeat-{}.txt",
            std::process::id()
        ));
        fs::write(&path, "Ryan: echo\n").expect("write transcript");
        let mut conversation = Conversation::new(path.clone());
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(92);
        let mut agents = (0..6)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        for generation in 1..=3 {
            let result = conversation
                .poll(
                    generation,
                    &mut world,
                    &lexicon,
                    &reading,
                    &mut dictionary,
                    &mut agents,
                    &mut rng,
                )
                .expect("poll transcript");
            assert_eq!(result.is_some(), generation == 3);
        }
        let transcript = fs::read_to_string(&path).expect("read first answer");
        fs::write(&path, transcript.replace("Ryan: \n", "Ryan: echo\n"))
            .expect("write repeated prompt");
        for generation in 4..=6 {
            let result = conversation
                .poll(
                    generation,
                    &mut world,
                    &lexicon,
                    &reading,
                    &mut dictionary,
                    &mut agents,
                    &mut rng,
                )
                .expect("poll repeated transcript");
            assert_eq!(result.is_some(), generation == 6);
        }
        let final_text = fs::read_to_string(&path).expect("read second answer");
        assert_eq!(final_text.matches("Community:").count(), 2);
        fs::remove_file(path).expect("remove temporary transcript");
    }

    #[test]
    fn explorer_mailbox_allows_only_one_question_until_ryan_answers() {
        let directory = std::env::temp_dir().join(format!(
            "mas-rust-question-mailbox-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos(),
        ));
        fs::create_dir_all(&directory).expect("create temporary directory");
        let converse = directory.join("converse.txt");
        fs::write(&converse, "").expect("create transcript");
        let mut conversation = Conversation::new(converse);
        assert!(
            conversation
                .ask_ryan_about("apple", 1)
                .expect("write first question")
        );
        assert!(
            !conversation
                .ask_ryan_about("river", 2)
                .expect("the open mailbox blocks a second question")
        );
        assert!(question_is_open(
            &fs::read_to_string(conversation.question_path()).expect("read mailbox")
        ));

        fs::write(
            conversation.question_path(),
            "Community: What is apple?\nRyan: apple is fruit\n",
        )
        .expect("answer question");
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut rng = Rng::new(79);
        let mut agents = (0..4)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        for generation in 3..=5 {
            conversation
                .poll_question_answer(generation, &mut dictionary, &mut agents, &mut rng)
                .expect("settle answer");
        }
        assert!(!question_is_open(
            &fs::read_to_string(conversation.question_path()).expect("read answered mailbox")
        ));
        assert!(
            conversation
                .take_dictionary_links()
                .iter()
                .any(|(left, right, _)| left == "apple" && right == "fruit")
        );
        assert_eq!(
            conversation.take_resolved_question_topic().as_deref(),
            Some("apple")
        );
        assert!(
            !conversation
                .ask_ryan_about("river", 5)
                .expect("answer generation cannot immediately enqueue another question")
        );
        assert!(
            conversation
                .ask_ryan_about("river", 6)
                .expect("a later generation can ask after the answer")
        );
        fs::remove_dir_all(directory).expect("remove temporary directory");
    }

    #[test]
    fn feedback_is_attached_to_the_last_real_response() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let apple = tokenise("Apple is red.");
        conversation.update_conversation_state("Apple is red.", &apple);
        conversation.last_reply = Some("apple is red".to_string());
        let mut world = WorldModel::default();
        assert!(
            conversation
                .apply_feedback(-1.0, &mut world)
                .contains("rejected")
        );
        assert!(conversation.reply_feedback["apple is red"] < 0.0);
        assert_eq!(conversation.last_reply.as_deref(), Some("apple is red"));
        assert_eq!(conversation.topic(), None);
    }

    #[test]
    fn an_open_question_is_not_misread_as_a_single_word_definition() {
        let words = tokenise("What are you pushing towards?");
        assert_eq!(parse_world_question(&words), None);
        let definition = tokenise("What is a statue?");
        assert_eq!(parse_world_question(&definition), Some("statue"));
    }

    #[test]
    fn a_single_waypoint_cannot_produce_path_predictions() {
        let mut path = SemanticPath::default();
        path.points.push_back("hello".to_string());
        path.predictions.insert("past".to_string(), 0.9);
        assert_eq!(path.predicted_words("hello").count(), 0);
        path.points.push_back("past".to_string());
        assert_eq!(path.predicted_words("past").count(), 1);
    }

    #[test]
    fn novel_surface_pattern_creates_a_new_opaque_mode() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let statement = tokenise("Apple is red.");
        conversation.update_conversation_state("Apple is red.", &statement);
        let question = tokenise("Can turtles dance?");
        conversation.update_conversation_state("Can turtles dance?", &question);
        assert!(conversation.intent_modes.len() >= 2);
    }

    #[test]
    fn human_bigrams_need_repetition_before_becoming_productive() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let words = tokenise("Apple is red.");
        let mut agents = Vec::new();
        let mut rng = Rng::new(83);
        conversation.ingest(&words, &mut dictionary, &mut agents, &mut rng);
        let pair = ("apple".to_string(), "is".to_string());
        assert!(conversation.bigram_strength(&pair) <= 0.0);
        assert!(!conversation.productive_human_pair("apple", "is"));
        conversation.ingest(&words, &mut dictionary, &mut agents, &mut rng);
        assert!(conversation.bigram_strength(&pair) > 0.0);
        assert!(conversation.productive_human_pair("apple", "is"));
    }

    #[test]
    fn community_vote_keeps_a_topic_bound_multiword_reply() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let mut dictionary = HumanDictionary::discover(Path::new("/definitely-not-a-dictionary"));
        let mut world = WorldModel::default();
        let lexicon = CommunityLexicon::default();
        let reading = ReadingBridge::default();
        let mut rng = Rng::new(81);
        let mut agents = (0..6)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        conversation.answer(
            "Apple is red.",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        conversation.answer(
            "Apple is round.",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        let reply = conversation.answer(
            "Apple thoughts?",
            &mut world,
            &lexicon,
            &reading,
            &mut dictionary,
            &mut agents,
            &mut rng,
        );
        let words = tokenise(&reply);
        assert!(words.len() >= 3, "reply was {reply:?}");
        assert_eq!(words.first().map(String::as_str), Some("apple"));
    }
}

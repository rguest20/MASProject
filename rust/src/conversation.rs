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
    pending: Option<PendingPrompt>,
    human_tokens: BTreeSet<String>,
    human_pool: BTreeMap<String, f64>,
    transitions: BTreeMap<String, BTreeMap<String, usize>>,
    english_bigrams: BTreeMap<(String, String), f64>,
    meaning_bigrams: BTreeMap<(String, String), f64>,
    intent_modes: BTreeMap<String, IntentMode>,
    next_intent_mode: usize,
    working_frame: Option<WorkingFrame>,
    parked_frames: VecDeque<WorkingFrame>,
    response_history: VecDeque<Vec<String>>,
    reply_feedback: BTreeMap<String, f64>,
    bigram_feedback: BTreeMap<(String, String), f64>,
    dictionary_links: Vec<(String, String, f32)>,
    active_topic: Option<String>,
    last_reply: Option<String>,
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

impl Conversation {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            pending: None,
            human_tokens: BTreeSet::new(),
            human_pool: BTreeMap::new(),
            transitions: BTreeMap::new(),
            english_bigrams: BTreeMap::new(),
            meaning_bigrams: BTreeMap::new(),
            intent_modes: BTreeMap::new(),
            next_intent_mode: 1,
            working_frame: None,
            parked_frames: VecDeque::with_capacity(4),
            response_history: VecDeque::with_capacity(8),
            reply_feedback: BTreeMap::new(),
            bigram_feedback: BTreeMap::new(),
            dictionary_links: Vec::new(),
            active_topic: None,
            last_reply: None,
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
        let pending = self.pending.as_ref().expect("pending prompt was just set");
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
        let prompt = pending.prompt.body.clone();
        let reply = self.answer(&prompt, world, lexicon, reading, dictionary, agents, rng);
        let mut file = OpenOptions::new().append(true).open(&self.path)?;
        if !pending.transcript.ends_with('\n') {
            writeln!(file)?;
        }
        writeln!(file, "Community: {reply}")?;
        writeln!(file, "Ryan: ")?;
        self.pending = None;
        if !is_feedback(&prompt) {
            self.record_response(&reply);
            self.last_reply = Some(reply.clone());
        }
        Ok(Some(reply))
    }

    pub fn topic(&self) -> Option<&str> {
        self.active_topic.as_deref()
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

    pub fn take_dictionary_links(&mut self) -> Vec<(String, String, f32)> {
        std::mem::take(&mut self.dictionary_links)
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
            return self.apply_feedback(1.0);
        }
        if matches!(trimmed, "-" | "feedback -") {
            return self.apply_feedback(-1.0);
        }
        let words = tokenise(trimmed);
        self.ingest(&words, dictionary, agents, rng);
        self.update_conversation_state(trimmed, &words);

        if let Some(number) = parse_number_question(trimmed) {
            return lexicon
                .numeric_token(number)
                .map(|token| format!("{number} is '{token}' in our shared number system."))
                .unwrap_or_else(|| {
                    "We do not yet have a community answer for that quantity.".to_string()
                });
        }
        if let Some(referent) = parse_show(trimmed) {
            return lexicon
                .referential_signal(referent)
                .map(|token| format!("{referent} is '{token}' in our shared vocabulary."))
                .unwrap_or_else(|| {
                    format!("We have not yet agreed a public word for {referent}.")
                });
        }
        if let Some(subject) = parse_world_question(&words) {
            return world
                .fact(subject)
                .map(|property| format!("{subject} is {property}."))
                .unwrap_or_else(|| {
                    format!("We do not yet have a confirmed fact about {subject}.")
                });
        }

        if let Some((subject, property)) = parse_assertion(&words) {
            world.observe(subject, property, 0.5);
        }

        let topic = self.active_topic.clone();
        // Free conversation must pass through the community proposal/vote
        // path. In particular, a story word is evidence for a continuation,
        // not permission to emit the fixed "X and Y are related" sentence.
        self.compose_reply(&words, topic.as_deref(), reading, agents, rng)
    }

    fn ingest(
        &mut self,
        words: &[String],
        dictionary: &mut HumanDictionary,
        agents: &mut [Agent],
        rng: &mut Rng,
    ) {
        self.human_tokens.extend(words.iter().cloned());
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
                // Dictionary phrases are semantic scaffolding.  They need
                // repeated human evidence or feedback before becoming a
                // productive continuation.
                *self
                    .meaning_bigrams
                    .entry((left.clone(), right.clone()))
                    .or_default() += 0.35;
                *self.human_pool.entry(left).or_default() += 0.12;
                *self.human_pool.entry(right).or_default() += 0.12;
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
        let topic = statement_subject(words)
            .map(str::to_owned)
            .or_else(|| words.iter().find(|word| is_topic(word)).cloned());
        if let Some(topic) = topic.as_ref() {
            self.active_topic = Some(topic.clone());
        }
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
        self.update_working_memory(mode, topic, words);
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
            .is_some_and(|frame| topic.is_some() && frame.topic != topic);
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
        &self,
        prompt_words: &[String],
        topic: Option<&str>,
        reading: &ReadingBridge,
        agents: &[Agent],
        rng: &mut Rng,
    ) -> String {
        let Some(frame) = self.working_frame.as_ref() else {
            return "We need more context before we can answer.".to_string();
        };
        let first = topic
            .map(str::to_owned)
            .or_else(|| prompt_words.iter().find(|word| is_topic(word)).cloned());
        let Some(first) = first else {
            return "We need more context before we can answer.".to_string();
        };
        let mut allowed: BTreeSet<_> = frame.tokens.keys().cloned().collect();
        allowed.extend(self.human_pool.keys().cloned());
        allowed.extend(prompt_words.iter().cloned());
        allowed.insert(first.clone());
        let anchors = allowed.clone();
        let reading_words = reading.relevant_word_scores(&anchors);
        allowed.extend(reading_words.keys().cloned());

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
                    candidates.extend(
                        frame
                            .transitions
                            .get(previous)
                            .into_iter()
                            .flat_map(|next| next.keys())
                            .filter(|word| allowed.contains(*word))
                            .cloned(),
                    );
                    candidates.extend(
                        self.transitions
                            .get(previous)
                            .into_iter()
                            .flat_map(|next| next.keys())
                            .filter(|word| allowed.contains(*word))
                            .cloned(),
                    );
                    candidates.extend(reading.relevant_continuations(previous, &allowed));
                    candidates.sort();
                    candidates.dedup();
                    candidates.retain(|word| !words.contains(word));
                    if candidates.is_empty() {
                        candidates.extend(
                            self.human_pool
                                .keys()
                                .filter(|word| !words.contains(*word))
                                .filter(|word| allowed.contains(*word))
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
                (words.len() >= 3).then(|| words.join(" "))
            })
            .collect();
        if proposals.is_empty() {
            return topic
                .map(|topic| format!("We need more context for {topic}."))
                .unwrap_or_else(|| "We need more context before we can answer.".to_string());
        }
        let mut ballot = BTreeMap::<String, usize>::new();
        for voter in agents {
            let chosen = proposals.iter().max_by(|left, right| {
                self.sentence_score(voter, left, topic, frame, &reading_words)
                    .partial_cmp(&self.sentence_score(voter, right, topic, frame, &reading_words))
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
        if self
            .response_history
            .iter()
            .any(|old| old.join(" ") == phrase)
            || self.reply_feedback.get(&phrase).copied().unwrap_or(0.0) < -0.5
        {
            return topic
                .map(|topic| format!("We need a new example about {topic}."))
                .unwrap_or_else(|| "We need a new example before we can answer.".to_string());
        }
        format!("{phrase}.")
    }

    fn continuation_score(
        &self,
        agent: Option<&Agent>,
        previous: &str,
        candidate: &str,
        frame: &WorkingFrame,
        reading_words: &BTreeMap<String, f64>,
    ) -> f64 {
        let local = frame
            .transitions
            .get(previous)
            .and_then(|next| next.get(candidate))
            .copied()
            .unwrap_or(0.0);
        let global = self
            .transitions
            .get(previous)
            .and_then(|next| next.get(candidate))
            .copied()
            .unwrap_or_default() as f64;
        let familiar = agent
            .map(|agent| agent.vocabulary.contains(candidate) as u8 as f64)
            .unwrap_or(0.5);
        let human_repetition = self.human_pool.get(candidate).copied().unwrap_or(0.0).ln_1p();
        let personal_salience = agent
            .map(|agent| agent.language.salience(candidate).max(0.0))
            .unwrap_or(0.0);
        (0.55 * self.bigram_strength(&(previous.to_string(), candidate.to_string())))
            + (0.45 * local)
            + (0.20 * global)
            + (0.20 * familiar)
            + (0.20 * human_repetition)
            + (0.12 * personal_salience)
            + reading_words.get(candidate).copied().unwrap_or(0.0)
    }

    fn sentence_score(
        &self,
        voter: &Agent,
        sentence: &str,
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
                frame
                    .transitions
                    .get(&pair[0])
                    .and_then(|next| next.get(&pair[1]))
                    .copied()
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
        0.35 * familiar
            + 0.32 * local_pairs
            + 0.35 * bigrams
            + 0.55 * topical
            + 0.15 * reading
            + 0.16 * human_repetition
            - 0.25 * copied_prompt
    }

    fn bigram_strength(&self, pair: &(String, String)) -> f64 {
        let observed = self.english_bigrams.get(pair).copied().unwrap_or(0.0);
        let feedback = self.bigram_feedback.get(pair).copied().unwrap_or(0.0);
        let meaning = self.meaning_bigrams.get(pair).copied().unwrap_or(0.0);
        let repeated = (observed - 1.0).max(0.0) / 2.0;
        (repeated + feedback + (0.15 * meaning).min(0.25)).clamp(-2.0, 2.0)
    }

    fn promoted_continuations(&self, previous: &str, allowed: &BTreeSet<String>) -> Vec<String> {
        self.english_bigrams
            .keys()
            .chain(self.meaning_bigrams.keys())
            .filter(|pair| pair.0 == previous && allowed.contains(&pair.1))
            .filter(|pair| self.bigram_strength(pair) > 0.0)
            .map(|pair| pair.1.clone())
            .collect()
    }

    fn apply_feedback(&mut self, reward: f64) -> String {
        let Some(reply) = self.last_reply.clone() else {
            return "There is no recent community sentence to score.".to_string();
        };
        let reward = if reward > 0.0 { 1.0 } else { -1.0 };
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
            "Feedback rejected for the last community sentence.".to_string()
        }
    }

    fn record_response(&mut self, reply: &str) {
        let words = tokenise(reply);
        if words.len() >= 2 {
            if self.response_history.len() == 8 {
                self.response_history.pop_front();
            }
            self.response_history.push_back(words);
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
        self.human_pool.retain(|_, score| {
            *score *= 0.97;
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

fn latest_unanswered(text: &str) -> Option<TranscriptPrompt> {
    let mut entries = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let Some((role, body)) = line.split_once(':') else {
            continue;
        };
        let body = body.trim();
        if !body.is_empty()
            && matches!(
                role.trim().to_ascii_lowercase().as_str(),
                "ryan" | "community"
            )
        {
            entries.push((
                line_number + 1,
                role.trim().to_ascii_lowercase(),
                body.to_string(),
            ));
        }
    }
    entries
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, (line_number, role, body))| {
            (role == "ryan"
                && entries
                    .get(index + 1)
                    .is_none_or(|next| next.1 != "community"))
            .then(|| TranscriptPrompt {
                line_number: *line_number,
                body: body.clone(),
            })
        })
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
        words.last().map(String::as_str)
    } else {
        None
    }
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

fn parse_assertion(words: &[String]) -> Option<(&str, &str)> {
    let subject = statement_subject(words)?;
    let copula = words
        .iter()
        .position(|word| matches!(word.as_str(), "is" | "are"))?;
    words
        .iter()
        .skip(copula + 1)
        .find(|word| is_topic(word))
        .map(|property| (subject, property.as_str()))
}

fn is_topic(word: &str) -> bool {
    word.len() > 2
        && !matches!(
            word,
            "a"
                | "an"
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
    use super::{Conversation, parse_show, tokenise};
    use crate::dictionary::HumanDictionary;
    use crate::lexicon::CommunityLexicon;
    use crate::model::{Agent, Rng, WorldModel};
    use crate::reading::ReadingBridge;
    use std::path::Path;
    use std::path::PathBuf;

    #[test]
    fn working_memory_parks_and_restores_by_explicit_topic() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        let apple = tokenise("Apple is red.");
        conversation.update_conversation_state("Apple is red.", &apple);
        let banana = tokenise("Banana is yellow.");
        conversation.update_conversation_state("Banana is yellow.", &banana);
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
        conversation.update_conversation_state("Apple is round.", &apple);
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
    fn feedback_is_attached_to_the_last_real_response() {
        let mut conversation = Conversation::new(PathBuf::from("/private/tmp/unused-converse.txt"));
        conversation.last_reply = Some("apple is red".to_string());
        conversation.record_response("apple is red");
        assert!(conversation.apply_feedback(-1.0).contains("rejected"));
        assert!(conversation.reply_feedback["apple is red"] < 0.0);
        assert_eq!(conversation.last_reply.as_deref(), Some("apple is red"));
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
        conversation.ingest(&words, &mut dictionary, &mut agents, &mut rng);
        assert!(conversation.bigram_strength(&pair) > 0.0);
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

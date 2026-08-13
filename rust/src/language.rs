//! Agent-owned proto-language production and pragmatic dialogue acts.
//!
//! Shared conventions remain in `CommunityLexicon`; this module owns the
//! intentionally private part of language: personal word invention, local
//! preferences, feedback, address forms, and a decision about whether an
//! utterance is exploratory practice, teaching, a request, or repair.

use std::collections::{BTreeMap, VecDeque};

use crate::lexicon::CommunityLexicon;
use crate::model::{Agent, Rng};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DialogueIntent {
    Practice,
    Teach,
    Ask,
    Repair,
    Assert,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DialogueMeaning {
    Number(u32),
    Referent(String),
    ReferentNumber {
        referent: String,
        value: u32,
    },
    ReferentActionNumber {
        referent: String,
        action: String,
        value: u32,
    },
    Private,
}

#[derive(Clone, Debug)]
pub struct DialogueAct {
    pub intent: DialogueIntent,
    pub meaning: DialogueMeaning,
    pub tokens: Vec<String>,
    pub address: Option<String>,
}

impl DialogueAct {
    pub fn render(&self) -> String {
        let mut words = self.address.clone().into_iter().collect::<Vec<_>>();
        words.extend(self.tokens.iter().cloned());
        words.join(" ")
    }
}

#[derive(Clone, Debug)]
pub struct LanguageProfile {
    concept_words: BTreeMap<String, String>,
    preferences: BTreeMap<String, f64>,
    recent: VecDeque<String>,
    utterance_feedback: BTreeMap<String, f64>,
    peer_names: BTreeMap<usize, String>,
    pub turns_spoken: u32,
    pub turns_heard: u32,
    length_bias: f64,
    repeat_bias: f64,
}

impl LanguageProfile {
    pub fn new(rng: &mut Rng) -> Self {
        Self {
            concept_words: BTreeMap::new(),
            preferences: BTreeMap::new(),
            recent: VecDeque::with_capacity(24),
            utterance_feedback: BTreeMap::new(),
            peer_names: BTreeMap::new(),
            turns_spoken: 0,
            turns_heard: 0,
            length_bias: rng.unit(),
            repeat_bias: rng.unit(),
        }
    }

    pub fn reset_identity(&mut self) {
        self.peer_names.clear();
        self.turns_spoken = 0;
        self.turns_heard = 0;
    }

    /// Merge cultural production tendencies without inheriting a parent's
    /// social address book.  Word forms and preferences are heritable;
    /// nicknames and dialogue counters belong to an individual lifetime.
    pub fn inherit_from(parents: [&LanguageProfile; 3], rng: &mut Rng) -> Self {
        let mut child = Self::new(rng);
        let concepts: std::collections::BTreeSet<_> = parents
            .iter()
            .flat_map(|parent| parent.concept_words.keys().cloned())
            .collect();
        for concept in concepts {
            let values: Vec<_> = parents
                .iter()
                .filter_map(|parent| parent.concept_words.get(&concept).cloned())
                .collect();
            if !values.is_empty() {
                child
                    .concept_words
                    .insert(concept, values[rng.index(values.len())].clone());
            }
        }
        let tokens: std::collections::BTreeSet<_> = parents
            .iter()
            .flat_map(|parent| parent.preferences.keys().cloned())
            .collect();
        for token in tokens {
            let values: Vec<_> = parents
                .iter()
                .filter_map(|parent| parent.preferences.get(&token).copied())
                .collect();
            if !values.is_empty() {
                let average = values.iter().sum::<f64>() / values.len() as f64;
                child.preferences.insert(
                    token,
                    (average + (rng.unit() - 0.5) * 0.03).clamp(-1.0, 1.0),
                );
            }
        }
        let utterances: std::collections::BTreeSet<_> = parents
            .iter()
            .flat_map(|parent| parent.utterance_feedback.keys().cloned())
            .collect();
        for utterance in utterances {
            let values: Vec<_> = parents
                .iter()
                .filter_map(|parent| parent.utterance_feedback.get(&utterance).copied())
                .collect();
            if !values.is_empty() {
                child.utterance_feedback.insert(
                    utterance,
                    (values.iter().sum::<f64>() / values.len() as f64 * 0.80).clamp(-1.0, 1.0),
                );
            }
        }
        let mut recent = VecDeque::with_capacity(24);
        for parent in parents {
            for token in parent.recent.iter().rev().take(8).rev() {
                if !recent.contains(token) {
                    recent.push_back(token.clone());
                }
            }
        }
        while recent.len() > 24 {
            recent.pop_front();
        }
        child.recent = recent;
        child.length_bias = (parents.iter().map(|parent| parent.length_bias).sum::<f64>()
            / parents.len() as f64
            + (rng.unit() - 0.5) * 0.04)
            .clamp(0.0, 1.0);
        child.repeat_bias = (parents.iter().map(|parent| parent.repeat_bias).sum::<f64>()
            / parents.len() as f64
            + (rng.unit() - 0.5) * 0.04)
            .clamp(0.0, 1.0);
        child
    }

    pub fn salience(&self, token: &str) -> f64 {
        let recent = self
            .recent
            .iter()
            .rev()
            .take(12)
            .any(|known| known == token) as u8 as f64;
        self.preference(token) + 0.25 * recent
    }

    pub fn cultural_counts(&self) -> (usize, usize) {
        (self.concept_words.len(), self.preferences.len())
    }

    fn nickname_for(&mut self, peer_id: usize, rng: &mut Rng) -> String {
        self.peer_names
            .entry(peer_id)
            .or_insert_with(|| format!("na{}", invented_word(rng)))
            .clone()
    }

    fn private_phrase(&mut self, candidates: Vec<String>, rng: &mut Rng) -> Vec<String> {
        let target = 2 + ((self.length_bias * 2.0) as usize).min(2);
        let mut tokens = Vec::new();
        if !candidates.is_empty() {
            let mut ranked = candidates;
            ranked.sort_by(|left, right| {
                self.preference(right)
                    .partial_cmp(&self.preference(left))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for token in ranked.into_iter().take(target) {
                tokens.push(token);
            }
        }
        while tokens.len() < target {
            let concept = format!("private-{}", tokens.len());
            let word = self
                .concept_words
                .entry(concept)
                .or_insert_with(|| invented_word(rng))
                .clone();
            tokens.push(word);
        }
        tokens
    }

    fn preference(&self, token: &str) -> f64 {
        self.preferences.get(token).copied().unwrap_or(0.0)
    }

    pub fn record_turn(&mut self, act: &DialogueAct, success: bool) {
        self.turns_spoken += 1;
        let utterance = act.tokens.join(" ");
        let reward = if success { 0.08 } else { -0.06 };
        let feedback = self.utterance_feedback.entry(utterance).or_default();
        *feedback = (*feedback + reward).clamp(-1.0, 1.0);
        for token in &act.tokens {
            let preference = self.preferences.entry(token.clone()).or_default();
            *preference = (*preference + reward * 0.35).clamp(-1.0, 1.0);
            self.recent.push_back(token.clone());
        }
        while self.recent.len() > 24 {
            self.recent.pop_front();
        }
        self.utterance_feedback.retain(|_, score| {
            *score *= 0.998;
            score.abs() >= 0.02
        });
        if !success {
            self.repeat_bias = (self.repeat_bias * 0.97).max(0.05);
        }
    }

    pub fn record_heard(&mut self, tokens: &[String]) {
        self.turns_heard += 1;
        for token in tokens {
            self.recent.push_back(token.clone());
        }
        while self.recent.len() > 24 {
            self.recent.pop_front();
        }
    }

    /// Human turns are privileged conversational evidence. Repetition raises
    /// a word's local production salience without declaring what it means;
    /// co-occurrence and later consensus still supply the semantics.
    pub fn observe_human_tokens(&mut self, tokens: &[String]) {
        for token in tokens {
            let preference = self.preferences.entry(token.clone()).or_insert(0.0);
            *preference = (*preference + 0.07).min(1.5);
            self.recent.push_back(token.clone());
        }
        while self.recent.len() > 24 {
            self.recent.pop_front();
        }
    }
}

pub fn produce_dialogue_act(
    speaker: &mut Agent,
    listener_id: usize,
    lexicon: &CommunityLexicon,
    rng: &mut Rng,
) -> DialogueAct {
    let trust = speaker
        .trust
        .get(&listener_id)
        .map(|profile| profile.affinity + profile.reliability)
        .unwrap_or(0.0)
        .clamp(-1.0, 1.0);
    let intent = choose_intent(speaker, trust, rng);
    let numeric = lexicon.numeric_conventions();
    let referents = lexicon.referential_conventions();
    let actions = lexicon.action_conventions();
    let (meaning, tokens) = if !numeric.is_empty()
        && !referents.is_empty()
        && !actions.is_empty()
        && lexicon.grammar_order("referent_action_number").is_some()
    {
        let (referent, referent_token) = choose_pair(&referents, rng);
        let (action, action_token) = choose_pair(&actions, rng);
        let (value, number_token) = choose_pair(&numeric, rng);
        let roles = BTreeMap::from([
            ("referent", referent_token),
            ("action", action_token),
            ("number", number_token),
        ]);
        let tokens = lexicon
            .grammar_order("referent_action_number")
            .expect("grammar was checked")
            .iter()
            .filter_map(|role| roles.get(role.as_str()).cloned())
            .collect();
        (
            DialogueMeaning::ReferentActionNumber {
                referent,
                action,
                value,
            },
            tokens,
        )
    } else if !numeric.is_empty()
        && !referents.is_empty()
        && lexicon.grammar_order("referent_quantity").is_some()
    {
        let (referent, referent_token) = choose_pair(&referents, rng);
        let (value, number_token) = choose_pair(&numeric, rng);
        let roles = BTreeMap::from([("referent", referent_token), ("number", number_token)]);
        let tokens = lexicon
            .grammar_order("referent_quantity")
            .expect("grammar was checked")
            .iter()
            .filter_map(|role| roles.get(role.as_str()).cloned())
            .collect();
        (DialogueMeaning::ReferentNumber { referent, value }, tokens)
    } else if !numeric.is_empty() {
        let (value, token) = choose_pair(&numeric, rng);
        (DialogueMeaning::Number(value), vec![token])
    } else if !referents.is_empty() {
        let (referent, token) = choose_pair(&referents, rng);
        (DialogueMeaning::Referent(referent), vec![token])
    } else {
        // No public convention yet: exploration is retained locally and can
        // become shareable only through repeated successful interaction.
        let candidates = speaker.semantics.reasoning_candidates(2);
        let tokens = speaker.language.private_phrase(candidates, rng);
        (DialogueMeaning::Private, tokens)
    };
    let address = (rng.unit() < 0.25 + 0.25 * speaker.traits.cooperation_weight)
        .then(|| speaker.language.nickname_for(listener_id, rng));
    DialogueAct {
        intent,
        meaning,
        tokens,
        address,
    }
}

pub fn interpret_dialogue_act(
    listener: &Agent,
    act: &DialogueAct,
    lexicon: &CommunityLexicon,
) -> bool {
    match &act.meaning {
        DialogueMeaning::Number(value) => listener
            .numeric
            .decode_phrase(&act.tokens.join(" "), lexicon)
            .is_some_and(|seen| seen == *value),
        DialogueMeaning::Referent(referent) => {
            act.tokens.first().and_then(|token| {
                listener
                    .referent_for(token)
                    .or_else(|| lexicon.referent_for_signal(token))
            }) == Some(referent.as_str())
        }
        DialogueMeaning::ReferentNumber { referent, value } => {
            interpret_composite(listener, &act.tokens, lexicon, "referent_quantity")
                == Some((Some(referent.as_str()), None, Some(*value)))
        }
        DialogueMeaning::ReferentActionNumber {
            referent,
            action,
            value,
        } => {
            interpret_composite(listener, &act.tokens, lexicon, "referent_action_number")
                == Some((Some(referent.as_str()), Some(action.as_str()), Some(*value)))
        }
        DialogueMeaning::Private => act
            .tokens
            .iter()
            .all(|token| listener.vocabulary.contains(token)),
    }
}

/// Form a grounded answer to a request when the responder can express the
/// requested meaning through the public convention.  This keeps `Ask` from
/// being just a label: it opens a compact second turn whose success can be
/// measured independently from the request.
pub fn answer_request(
    responder: &mut Agent,
    requester_id: usize,
    request: &DialogueAct,
    lexicon: &CommunityLexicon,
    rng: &mut Rng,
) -> Option<DialogueAct> {
    let tokens = public_tokens_for_meaning(&request.meaning, lexicon)?;
    Some(DialogueAct {
        intent: DialogueIntent::Assert,
        meaning: request.meaning.clone(),
        tokens,
        address: Some(responder.language.nickname_for(requester_id, rng)),
    })
}

/// Echo a failed utterance in the shortest usable public form where possible.
/// Private discoveries may only be echoed as a bounded focus phrase; they are
/// still learned locally, but are never falsely promoted as shared meaning.
pub fn repair_dialogue(
    repairer: &mut Agent,
    speaker_id: usize,
    failed: &DialogueAct,
    lexicon: &CommunityLexicon,
    rng: &mut Rng,
) -> DialogueAct {
    let tokens = public_tokens_for_meaning(&failed.meaning, lexicon)
        .unwrap_or_else(|| failed.tokens.iter().take(2).cloned().collect::<Vec<_>>());
    DialogueAct {
        intent: DialogueIntent::Repair,
        meaning: failed.meaning.clone(),
        tokens,
        address: Some(repairer.language.nickname_for(speaker_id, rng)),
    }
}

fn public_tokens_for_meaning(
    meaning: &DialogueMeaning,
    lexicon: &CommunityLexicon,
) -> Option<Vec<String>> {
    let numeric = lexicon.numeric_conventions();
    let referents = lexicon.referential_conventions();
    let actions = lexicon.action_conventions();
    match meaning {
        DialogueMeaning::Number(value) => numeric.get(value).cloned().map(|token| vec![token]),
        DialogueMeaning::Referent(referent) => {
            referents.get(referent).cloned().map(|token| vec![token])
        }
        DialogueMeaning::ReferentNumber { referent, value } => {
            let order = lexicon.grammar_order("referent_quantity")?;
            let roles = BTreeMap::from([
                ("referent", referents.get(referent)?.clone()),
                ("number", numeric.get(value)?.clone()),
            ]);
            Some(
                order
                    .iter()
                    .filter_map(|role| roles.get(role.as_str()).cloned())
                    .collect(),
            )
        }
        DialogueMeaning::ReferentActionNumber {
            referent,
            action,
            value,
        } => {
            let order = lexicon.grammar_order("referent_action_number")?;
            let roles = BTreeMap::from([
                ("referent", referents.get(referent)?.clone()),
                ("action", actions.get(action)?.clone()),
                ("number", numeric.get(value)?.clone()),
            ]);
            Some(
                order
                    .iter()
                    .filter_map(|role| roles.get(role.as_str()).cloned())
                    .collect(),
            )
        }
        DialogueMeaning::Private => None,
    }
}

fn choose_intent(speaker: &Agent, trust: f64, rng: &mut Rng) -> DialogueIntent {
    let teach = 0.15 + 0.45 * speaker.traits.teaching_drive + 0.20 * speaker.state.purpose;
    let ask = 0.10 + 0.45 * speaker.traits.curiosity + 0.25 * speaker.state.frustration;
    let repair = 0.05 + 0.35 * speaker.state.frustration + 0.20 * (-trust).max(0.0);
    let assert = 0.08 + 0.45 * speaker.traits.expressiveness + 0.15 * speaker.state.confidence;
    let practice = 0.30 + 0.30 * speaker.traits.cooperation_weight;
    let choices = [
        (DialogueIntent::Practice, practice),
        (DialogueIntent::Teach, teach),
        (DialogueIntent::Ask, ask),
        (DialogueIntent::Repair, repair),
        (DialogueIntent::Assert, assert),
    ];
    weighted_choice(&choices, rng)
}

fn choose_pair<K: Ord + Clone, V: Clone>(values: &BTreeMap<K, V>, rng: &mut Rng) -> (K, V) {
    let index = rng.index(values.len());
    values
        .iter()
        .nth(index)
        .map(|(key, value)| (key.clone(), value.clone()))
        .expect("dialogue only chooses from non-empty conventions")
}

fn weighted_choice(choices: &[(DialogueIntent, f64)], rng: &mut Rng) -> DialogueIntent {
    let total = choices
        .iter()
        .map(|(_, weight)| weight.max(0.01))
        .sum::<f64>();
    let mut target = rng.unit() * total;
    for (choice, weight) in choices {
        target -= weight.max(0.01);
        if target <= 0.0 {
            return *choice;
        }
    }
    choices[choices.len() - 1].0
}

fn interpret_composite<'a>(
    listener: &'a Agent,
    tokens: &'a [String],
    lexicon: &'a CommunityLexicon,
    grammar: &str,
) -> Option<(Option<&'a str>, Option<&'a str>, Option<u32>)> {
    let roles = lexicon.grammar_order(grammar)?;
    if roles.len() != tokens.len() {
        return None;
    }
    let mut referent = None;
    let mut action = None;
    let mut number = None;
    for (role, token) in roles.iter().zip(tokens) {
        match role.as_str() {
            "referent" => {
                referent = listener
                    .referent_for(token)
                    .or_else(|| lexicon.referent_for_signal(token))
            }
            "action" => {
                action = listener
                    .action_for(token)
                    .or_else(|| lexicon.action_for_signal(token))
            }
            "number" => number = listener.numeric.decode_phrase(token, lexicon),
            _ => return None,
        }
    }
    Some((referent, action, number))
}

fn invented_word(rng: &mut Rng) -> String {
    const VOWELS: &[u8] = b"aeiou";
    const CONSONANTS: &[u8] = b"bcdfghjklmnpqrstvwxyz";
    let syllables = 1 + (rng.unit() >= 0.70) as usize;
    (0..syllables)
        .map(|_| {
            let consonant = CONSONANTS[rng.index(CONSONANTS.len())] as char;
            let vowel = VOWELS[rng.index(VOWELS.len())] as char;
            format!("{consonant}{vowel}")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        DialogueAct, DialogueIntent, DialogueMeaning, LanguageProfile, answer_request,
        interpret_dialogue_act, produce_dialogue_act, weighted_choice,
    };
    use crate::lexicon::CommunityLexicon;
    use crate::model::{Agent, Rng};

    #[test]
    fn profile_keeps_private_feedback_and_identity_names_local() {
        let mut rng = Rng::new(701);
        let mut profile = LanguageProfile::new(&mut rng);
        let name = profile.nickname_for(4, &mut rng);
        assert!(name.starts_with("na"));
        assert_eq!(profile.nickname_for(4, &mut rng), name);
        profile.reset_identity();
        assert_ne!(profile.nickname_for(4, &mut rng), name);
    }

    #[test]
    fn pragmatic_choice_always_returns_a_valid_intent() {
        let mut rng = Rng::new(702);
        let choice = weighted_choice(
            &[(DialogueIntent::Teach, 0.0), (DialogueIntent::Repair, 1.0)],
            &mut rng,
        );
        assert!(matches!(
            choice,
            DialogueIntent::Teach | DialogueIntent::Repair
        ));
    }

    #[test]
    fn public_numeral_is_produced_and_interpreted_as_a_grounded_act() {
        let mut rng = Rng::new(703);
        let mut lexicon = CommunityLexicon::default();
        lexicon.observe_numeric_success(2, "bel");
        lexicon.observe_numeric_success(2, "bel");
        let mut speaker = Agent::new(1, &mut rng);
        let listener = Agent::new(2, &mut rng);
        let act = produce_dialogue_act(&mut speaker, listener.id, &lexicon, &mut rng);
        assert_eq!(act.meaning, DialogueMeaning::Number(2));
        assert_eq!(act.tokens, vec!["bel"]);
        assert!(interpret_dialogue_act(&listener, &act, &lexicon));
    }

    #[test]
    fn public_request_receives_an_interpretable_assertion() {
        let mut rng = Rng::new(705);
        let mut lexicon = CommunityLexicon::default();
        lexicon.observe_numeric_success(3, "mav");
        lexicon.observe_numeric_success(3, "mav");
        let mut responder = Agent::new(1, &mut rng);
        let requester = Agent::new(2, &mut rng);
        let request = DialogueAct {
            intent: DialogueIntent::Ask,
            meaning: DialogueMeaning::Number(3),
            tokens: vec!["mav".to_string()],
            address: None,
        };
        let answer = answer_request(&mut responder, requester.id, &request, &lexicon, &mut rng)
            .expect("a public numeral can be answered");
        assert_eq!(answer.intent, DialogueIntent::Assert);
        assert!(interpret_dialogue_act(&requester, &answer, &lexicon));
    }

    #[test]
    fn child_language_merges_preferences_but_not_peer_address_books() {
        let mut rng = Rng::new(704);
        let mut first = LanguageProfile::new(&mut rng);
        let mut second = LanguageProfile::new(&mut rng);
        let mut third = LanguageProfile::new(&mut rng);
        first.preferences.insert("river".to_string(), 0.8);
        second.preferences.insert("river".to_string(), 0.6);
        third.preferences.insert("river".to_string(), 0.4);
        first.nickname_for(9, &mut rng);
        let child = LanguageProfile::inherit_from([&first, &second, &third], &mut rng);
        assert!(child.salience("river") > 0.4);
        assert!(child.peer_names.is_empty());
    }
}

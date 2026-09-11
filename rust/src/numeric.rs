//! Agent-owned numeral systems.
//!
//! This module is the typed Rust counterpart to Python's `NumericSystem`.
//! Each agent begins with an injective private map.  Public conventions, when
//! present, override that map at utterance time; successful task interactions
//! are responsible for promoting those conventions in `CommunityLexicon`.

use std::collections::{BTreeMap, BTreeSet};

use crate::lexicon::CommunityLexicon;
use crate::model::Rng;

const POSSIBLE_BASES: [u32; 6] = [4, 6, 8, 10, 12, 16];
const VOWELS: &[u8] = b"aeiou";
const CONSONANTS: &[u8] = b"bcdfghjklmnpqrstvwxyz";

#[derive(Clone, Debug)]
pub struct NumericSystem {
    pub base: u32,
    symbols: BTreeMap<u32, String>,
    inverse_symbols: BTreeMap<String, u32>,
    overlays: BTreeMap<u32, String>,
}

impl NumericSystem {
    pub fn new(rng: &mut Rng) -> Self {
        let base = POSSIBLE_BASES[rng.index(POSSIBLE_BASES.len())];
        let mut system = Self {
            base,
            symbols: BTreeMap::new(),
            inverse_symbols: BTreeMap::new(),
            overlays: BTreeMap::new(),
        };
        for digit in 0..base {
            let token = system.fresh_symbol(&system.used_tokens(), rng);
            system.inverse_symbols.insert(token.clone(), digit);
            system.symbols.insert(digit, token);
        }
        system
    }

    pub fn symbols(&self) -> &BTreeMap<u32, String> {
        &self.symbols
    }

    pub fn digit_for(&self, token: &str, lexicon: &CommunityLexicon) -> Option<u32> {
        let token = clean_token(token)?;
        lexicon
            .numeric_digit(&token)
            .or_else(|| {
                self.overlays
                    .iter()
                    .find_map(|(digit, local)| (local == &token).then_some(*digit))
            })
            .or_else(|| self.inverse_symbols.get(&token).copied())
    }

    pub fn decode_phrase(&self, phrase: &str, lexicon: &CommunityLexicon) -> Option<u32> {
        let parts: Vec<_> = phrase
            .split([' ', '-', ','])
            .filter_map(clean_token)
            .collect();
        if parts.is_empty() {
            return None;
        }
        parts.into_iter().try_fold(0_u32, |value, part| {
            self.digit_for(&part, lexicon)
                .map(|digit| value.saturating_mul(self.base).saturating_add(digit))
        })
    }

    pub fn speak_number(
        &mut self,
        value: u32,
        lexicon: &CommunityLexicon,
        rng: &mut Rng,
    ) -> String {
        let base = lexicon.community_base().unwrap_or(self.base);
        digits(value, base)
            .into_iter()
            .map(|digit| {
                lexicon
                    .numeric_token(digit)
                    .map(str::to_owned)
                    .or_else(|| self.overlays.get(&digit).cloned())
                    .unwrap_or_else(|| self.symbol_for(digit, rng))
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn learn_digit_mapping(&mut self, token: &str, digit: u32, rng: &mut Rng) -> bool {
        if digit >= self.base {
            return false;
        }
        let Some(token) = clean_token(token) else {
            return false;
        };
        let old_token = self.overlays.get(&digit).cloned();
        if let Some(old_digit) = self
            .overlays
            .iter()
            .find_map(|(known_digit, known_token)| (known_token == &token).then_some(*known_digit))
            && old_digit != digit
        {
            self.overlays.insert(
                old_digit,
                old_token.unwrap_or_else(|| self.fresh_symbol(&self.used_tokens(), rng)),
            );
        }
        self.overlays.insert(digit, token);
        self.repair_injective_overlays(rng);
        true
    }

    pub fn compact_against_community(&mut self, lexicon: &CommunityLexicon) {
        self.overlays
            .retain(|digit, token| lexicon.numeric_token(*digit) != Some(token.as_str()));
    }

    pub fn mutate(&mut self, rng: &mut Rng) {
        if rng.unit() < 0.1 {
            self.base = POSSIBLE_BASES[rng.index(POSSIBLE_BASES.len())];
        }
        if rng.unit() < 0.3 && !self.symbols.is_empty() {
            let digit = *self
                .symbols
                .keys()
                .nth(rng.index(self.symbols.len()))
                .expect("symbols is non-empty");
            let old = self.symbols.get(&digit).cloned().unwrap_or_default();
            let suffix = VOWELS[rng.index(VOWELS.len())] as char;
            let token = format!("{old}{suffix}");
            self.symbols.insert(digit, token.clone());
            self.inverse_symbols.insert(token, digit);
        }
    }

    fn symbol_for(&mut self, digit: u32, rng: &mut Rng) -> String {
        if let Some(token) = self.symbols.get(&digit)
            && self.inverse_symbols.get(token) == Some(&digit)
        {
            return token.clone();
        }
        let token = self.fresh_symbol(&self.used_tokens(), rng);
        self.inverse_symbols.insert(token.clone(), digit);
        self.symbols.insert(digit, token.clone());
        token
    }

    fn used_tokens(&self) -> BTreeSet<String> {
        self.symbols
            .values()
            .chain(self.overlays.values())
            .cloned()
            .collect()
    }

    fn fresh_symbol(&self, reserved: &BTreeSet<String>, rng: &mut Rng) -> String {
        for _ in 0..256 {
            let token = format!(
                "{}{}",
                CONSONANTS[rng.index(CONSONANTS.len())] as char,
                VOWELS[rng.index(VOWELS.len())] as char,
            );
            if !reserved.contains(&token) && !self.inverse_symbols.contains_key(&token) {
                return token;
            }
        }
        loop {
            let token = format!(
                "{}{}{}{}",
                CONSONANTS[rng.index(CONSONANTS.len())] as char,
                VOWELS[rng.index(VOWELS.len())] as char,
                CONSONANTS[rng.index(CONSONANTS.len())] as char,
                VOWELS[rng.index(VOWELS.len())] as char,
            );
            if !reserved.contains(&token) && !self.inverse_symbols.contains_key(&token) {
                return token;
            }
        }
    }

    fn repair_injective_overlays(&mut self, rng: &mut Rng) {
        let mut used = BTreeSet::new();
        let digits: Vec<_> = self.overlays.keys().copied().collect();
        for digit in digits {
            let duplicate = self
                .overlays
                .get(&digit)
                .is_none_or(|token| !used.insert(token.clone()));
            if duplicate {
                let token = self.fresh_symbol(&used, rng);
                used.insert(token.clone());
                self.overlays.insert(digit, token);
            }
        }
    }
}

pub fn clean_token(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();
    (!value.is_empty()).then_some(value)
}

pub fn digits(value: u32, base: u32) -> Vec<u32> {
    if value == 0 {
        return vec![0];
    }
    let mut remaining = value;
    let mut result = Vec::new();
    while remaining > 0 {
        result.push(remaining % base);
        remaining /= base;
    }
    result.reverse();
    result
}

#[cfg(test)]
mod tests {
    use super::NumericSystem;
    use crate::lexicon::CommunityLexicon;
    use crate::model::Rng;

    #[test]
    fn seeded_system_has_a_distinct_word_for_each_digit() {
        let mut rng = Rng::new(4);
        let system = NumericSystem::new(&mut rng);
        let values: std::collections::BTreeSet<_> = system.symbols().values().collect();
        assert_eq!(values.len(), system.base as usize);
    }

    #[test]
    fn public_convention_overrides_private_speech() {
        let mut rng = Rng::new(7);
        let mut system = NumericSystem::new(&mut rng);
        let mut lexicon = CommunityLexicon::default();
        lexicon.observe_numeric_success(2, "bel");
        lexicon.observe_numeric_success(2, "bel");
        assert_eq!(system.speak_number(2, &lexicon, &mut rng), "bel");
    }

    #[test]
    fn learned_overlay_stays_injective() {
        let mut rng = Rng::new(9);
        let mut system = NumericSystem::new(&mut rng);
        assert!(system.learn_digit_mapping("bel", 1, &mut rng));
        assert!(system.learn_digit_mapping("bel", 2, &mut rng));
        let overlays: std::collections::BTreeSet<_> = system.overlays.values().collect();
        assert_eq!(overlays.len(), system.overlays.len());
    }
}

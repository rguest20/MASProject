//! Dense semantic vectors and a bounded association graph.
//!
//! Python stores a 32-dimensional NumPy vector per token and learns weighted
//! co-occurrence links between them.  This port keeps that same model but uses
//! contiguous `ndarray::Array1<f32>` values, limits graph degree, and runs the
//! stabilisation pass without Python object dispatch in the hot loop.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use ndarray::Array1;

use crate::model::Rng;

pub const DEFAULT_SEMANTIC_DIMS: usize = 32;
pub const MIN_SEMANTIC_DIMS: usize = 8;
pub const MAX_SEMANTIC_DIMS: usize = 256;
static SEMANTIC_DIMS: AtomicUsize = AtomicUsize::new(DEFAULT_SEMANTIC_DIMS);
const MAX_DEGREE: usize = 15;
const MAX_CONTEXT_TOKENS: usize = 15;

pub fn semantic_dimensions() -> usize {
    SEMANTIC_DIMS.load(AtomicOrdering::Relaxed)
}

pub fn configure_semantic_dimensions(dimensions: usize) -> Result<(), String> {
    if !(MIN_SEMANTIC_DIMS..=MAX_SEMANTIC_DIMS).contains(&dimensions) {
        return Err(format!(
            "semantic dimensions must be between {MIN_SEMANTIC_DIMS} and {MAX_SEMANTIC_DIMS}"
        ));
    }
    SEMANTIC_DIMS.store(dimensions, AtomicOrdering::Relaxed);
    Ok(())
}

#[derive(Clone, Debug)]
pub struct SemanticLink {
    pub weight: f32,
    usefulness: f32,
    last_seen: u64,
}

#[derive(Clone, Debug)]
pub struct SemanticStore {
    vectors: BTreeMap<String, Array1<f32>>,
    links: BTreeMap<String, BTreeMap<String, SemanticLink>>,
    numeric_tokens: BTreeSet<String>,
    usage: BTreeMap<String, u32>,
    cooldown: BTreeMap<String, u8>,
    tick: u64,
    families: Vec<SemanticFamily>,
    next_family_id: usize,
}

#[derive(Clone, Debug)]
pub struct SemanticFamily {
    pub id: String,
    pub centroid: Array1<f32>,
    pub members: BTreeSet<String>,
    pub confidence: f32,
    strength: f32,
}

impl Default for SemanticStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticStore {
    pub fn new() -> Self {
        Self {
            vectors: BTreeMap::new(),
            links: BTreeMap::new(),
            numeric_tokens: BTreeSet::new(),
            usage: BTreeMap::new(),
            cooldown: BTreeMap::new(),
            tick: 0,
            families: Vec::new(),
            next_family_id: 0,
        }
    }

    /// Build a child semantic store from three parent stores.
    ///
    /// Vectors are averaged only across parents that actually carry a token,
    /// then receive tiny bounded mutation noise. Associations are merged by
    /// strength under the normal degree cap. This preserves repeated cultural
    /// structure across gentle population turnover without copying a single
    /// parent's private semantic map wholesale.
    pub fn inherit_from(
        parents: [&SemanticStore; 3],
        retained_tokens: &BTreeSet<String>,
        rng: &mut Rng,
    ) -> Self {
        let mut child = Self::new();
        for token in retained_tokens {
            let vectors: Vec<_> = parents
                .iter()
                .filter_map(|parent| parent.vectors.get(token))
                .collect();
            if vectors.is_empty() {
                continue;
            }
            let mut vector = Array1::zeros(semantic_dimensions());
            for source in &vectors {
                vector += *source;
            }
            vector.mapv_inplace(|value| value / vectors.len() as f32);
            for value in &mut vector {
                *value += signed(rng, 0.018);
            }
            clip(&mut vector);
            child.vectors.insert(token.clone(), vector);
            let usage = parents
                .iter()
                .filter_map(|parent| parent.usage.get(token).copied())
                .sum::<u32>()
                / vectors.len() as u32;
            if usage > 0 {
                child.usage.insert(token.clone(), usage);
            }
        }

        // Collect undirected candidate associations. A parent that does not
        // know an edge contributes no evidence, so a cultural link needs to
        // be strong enough in at least one lineage to survive.
        let mut merged = BTreeMap::<(String, String), (f32, u32)>::new();
        for parent in parents {
            for (left, neighbours) in &parent.links {
                for (right, link) in neighbours {
                    if left >= right
                        || !child.vectors.contains_key(left)
                        || !child.vectors.contains_key(right)
                    {
                        continue;
                    }
                    let entry = merged.entry((left.clone(), right.clone())).or_default();
                    entry.0 += link.weight;
                    entry.1 += 1;
                }
            }
        }
        let mut candidates: Vec<_> = merged
            .into_iter()
            .filter_map(|((left, right), (total, count))| {
                let weight = 0.82 * total / count as f32;
                (weight.abs() >= 0.015).then_some((left, right, weight))
            })
            .collect();
        candidates.sort_by(|left, right| {
            right
                .2
                .abs()
                .partial_cmp(&left.2.abs())
                .unwrap_or(Ordering::Equal)
        });
        for (left, right, weight) in candidates {
            if child.links.get(&left).map_or(0, BTreeMap::len) >= MAX_DEGREE
                || child.links.get(&right).map_or(0, BTreeMap::len) >= MAX_DEGREE
            {
                continue;
            }
            let usefulness = child.usefulness(&left, &right);
            let reverse_usefulness = child.usefulness(&right, &left);
            child.links.entry(left.clone()).or_default().insert(
                right.clone(),
                SemanticLink {
                    weight,
                    usefulness,
                    last_seen: 0,
                },
            );
            child.links.entry(right).or_default().insert(
                left,
                SemanticLink {
                    weight,
                    usefulness: reverse_usefulness,
                    last_seen: 0,
                },
            );
        }

        // Families are optional high-confidence compressions of a graph. A
        // child receives only a weakened seed; later ticks must keep it
        // coherent, so an inherited family cannot become immortal baggage.
        let mut family_sets = BTreeSet::new();
        for parent in parents {
            for family in &parent.families {
                let members: BTreeSet<_> = family
                    .members
                    .iter()
                    .filter(|member| child.vectors.contains_key(*member))
                    .cloned()
                    .collect();
                if members.len() < 4 || !family_sets.insert(members.clone()) {
                    continue;
                }
                let vectors: Vec<_> = members
                    .iter()
                    .filter_map(|member| child.vectors.get(member))
                    .collect();
                let Some(centroid) = centroid(&vectors) else {
                    continue;
                };
                child.families.push(SemanticFamily {
                    id: format!("F{}", child.next_family_id),
                    centroid,
                    members,
                    confidence: (family.confidence * 0.85).min(0.90),
                    strength: family.strength * 0.70,
                });
                child.next_family_id += 1;
            }
        }
        child.families.truncate(64);
        child
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vectors.len()
    }

    pub fn link_count(&self) -> usize {
        self.links.values().map(BTreeMap::len).sum::<usize>() / 2
    }

    pub fn family_count(&self) -> usize {
        self.families.len()
    }

    pub fn families(&self) -> &[SemanticFamily] {
        &self.families
    }

    pub fn community_candidates(&self) -> Vec<(String, Array1<f32>, u32)> {
        self.vectors
            .iter()
            .filter_map(|(token, vector)| {
                let usage = self.usage.get(token).copied().unwrap_or(0);
                (usage >= 5 && !self.numeric_tokens.contains(token) && !identity_like(token))
                    .then_some((token.clone(), vector.clone(), usage))
            })
            .collect()
    }

    pub fn vector(&self, token: &str) -> Option<&Array1<f32>> {
        self.vectors.get(token)
    }

    pub fn token_names(&self) -> Vec<String> {
        self.vectors.keys().cloned().collect()
    }

    /// Tokens mature enough to be used in a deliberative concept task.  This
    /// keeps one-off noise, numeric symbols, and agent identity labels out of
    /// the reasoning curriculum.
    pub fn reasoning_candidates(&self, minimum_usage: u32) -> Vec<String> {
        self.vectors
            .keys()
            .filter(|token| {
                self.usage.get(*token).copied().unwrap_or(0) >= minimum_usage
                    && !self.numeric_tokens.contains(*token)
                    && !identity_like(token)
            })
            .cloned()
            .collect()
    }

    pub fn blend_from(&mut self, token: &str, teacher: &Array1<f32>, factor: f32, rng: &mut Rng) {
        self.ensure_vector(token, 0.25, rng);
        if let Some(student) = self.vectors.get_mut(token) {
            for (student_value, teacher_value) in student.iter_mut().zip(teacher.iter()) {
                *student_value = (1.0 - factor) * *student_value + factor * *teacher_value;
            }
            clip(student);
        }
    }

    pub fn ensure_numeric_token(&mut self, token: &str, digit: u32, base: u32, rng: &mut Rng) {
        let token = normalise(token);
        if token.is_empty() {
            return;
        }
        self.numeric_tokens.insert(token.clone());
        let denominator = base.saturating_sub(1).max(1) as f32;
        let mut anchor = Array1::zeros(semantic_dimensions());
        anchor[0] = digit as f32 / denominator * 0.2 + signed(rng, 0.02);
        for value in anchor.iter_mut().skip(1) {
            *value = signed(rng, 0.02);
        }
        self.vectors.insert(token, anchor);
    }

    pub fn observe(&mut self, words: &[String], gain: f32, rng: &mut Rng) {
        let mut cleaned: Vec<String> = words
            .iter()
            .map(|word| normalise(word))
            .filter(|word| !word.is_empty() && !self.numeric_tokens.contains(word))
            .collect();
        if cleaned.len() < 2 {
            return;
        }
        if cleaned.len() > MAX_CONTEXT_TOKENS {
            // Partial Fisher-Yates keeps the bounded work model from Python's
            // random sample while avoiding allocations proportional to a long
            // story sentence.
            for index in 0..MAX_CONTEXT_TOKENS {
                let replacement = index + rng.index(cleaned.len() - index);
                cleaned.swap(index, replacement);
            }
            cleaned.truncate(MAX_CONTEXT_TOKENS);
        }
        for word in &cleaned {
            self.ensure_vector(word, 0.25, rng);
            *self.usage.entry(word.clone()).or_default() += 1;
            if let Some(vector) = self.vectors.get_mut(word) {
                vector[0] += signed(rng, 0.02);
                for value in vector.iter_mut().skip(1) {
                    *value += signed(rng, 0.15);
                }
                clip(vector);
            }
        }
        for left in 0..cleaned.len() {
            for right in (left + 1)..cleaned.len() {
                self.link(&cleaned[left], &cleaned[right], gain, rng);
            }
        }
    }

    pub fn link(&mut self, left: &str, right: &str, weight: f32, rng: &mut Rng) {
        if left.is_empty()
            || right.is_empty()
            || left == right
            || self.numeric_tokens.contains(left)
            || self.numeric_tokens.contains(right)
        {
            return;
        }
        self.ensure_vector(left, 0.25, rng);
        self.ensure_vector(right, 0.25, rng);
        if !self.can_add_link(left, right) {
            return;
        }
        let weight = self.adjusted_weight(left, right, weight);
        let usefulness = self.usefulness(left, right);
        let reverse_usefulness = self.usefulness(right, left);
        let link = SemanticLink {
            weight,
            usefulness,
            last_seen: self.tick,
        };
        self.links
            .entry(left.to_string())
            .or_default()
            .entry(right.to_string())
            .and_modify(|entry| {
                entry.weight += weight;
                entry.usefulness = usefulness;
                entry.last_seen = self.tick;
            })
            .or_insert_with(|| link.clone());
        self.links
            .entry(right.to_string())
            .or_default()
            .entry(left.to_string())
            .and_modify(|entry| {
                entry.weight += weight;
                entry.usefulness = reverse_usefulness;
                entry.last_seen = self.tick;
            })
            .or_insert(link);
    }

    pub fn nearest(&self, token: &str, count: usize) -> Vec<String> {
        let Some(vector) = self.vectors.get(token) else {
            return Vec::new();
        };
        let mut candidates: Vec<_> = self
            .vectors
            .iter()
            .filter(|(other, _)| *other != token && !self.numeric_tokens.contains(*other))
            .map(|(other, other_vector)| {
                let distance = vector
                    .iter()
                    .skip(1)
                    .zip(other_vector.iter().skip(1))
                    .map(|(left, right)| (left - right).powi(2))
                    .sum::<f32>();
                (other.clone(), distance)
            })
            .collect();
        candidates.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal));
        candidates
            .into_iter()
            .take(count)
            .map(|(token, _)| token)
            .collect()
    }

    pub fn cosine_similarity(&self, left: &str, right: &str) -> Option<f32> {
        let left = self.vectors.get(left)?;
        let right = self.vectors.get(right)?;
        let dot = left.dot(right);
        let denominator = left.dot(left).sqrt() * right.dot(right).sqrt();
        (denominator > f32::EPSILON).then_some(dot / denominator)
    }

    pub fn tick(&mut self, rng: &mut Rng) {
        self.tick += 1;
        self.decay_cooldowns();
        self.frequency_guard();
        self.gravity_step();
        self.decay_links();
        self.inject_escape_noise(rng);
        if self.tick.is_multiple_of(5) {
            self.detect_families();
            self.reinforce_families();
            self.decay_families();
        }
    }

    /// Detect compact, frequently used linked components.  The graph gate is
    /// important: random nearby vectors must not become a "concept" merely
    /// because of accidental geometry.
    fn detect_families(&mut self) {
        let mut candidates: Vec<_> = self
            .links
            .iter()
            .filter_map(|(token, neighbours)| {
                (!self.numeric_tokens.contains(token)
                    && self.usage.get(token).copied().unwrap_or(0) >= 5
                    && neighbours.len() >= 3)
                    .then_some(token.clone())
            })
            .collect();
        candidates.sort();
        let mut made = 0;
        let mut visited = BTreeSet::new();
        for token in candidates {
            if visited.contains(&token) || made >= 2 {
                continue;
            }
            let mut members = BTreeSet::from([token.clone()]);
            if let Some(neighbours) = self.links.get(&token) {
                for (other, link) in neighbours {
                    if link.weight >= 0.20
                        && self.usage.get(other).copied().unwrap_or(0) >= 5
                        && !self.numeric_tokens.contains(other)
                    {
                        members.insert(other.clone());
                    }
                }
            }
            visited.extend(members.iter().cloned());
            if members.len() < 4 || self.families.iter().any(|family| family.members == members) {
                continue;
            }
            let vectors: Vec<_> = members
                .iter()
                .filter_map(|member| self.vectors.get(member))
                .collect();
            let Some(centroid) = centroid(&vectors) else {
                continue;
            };
            let mean_distance = vectors
                .iter()
                .map(|vector| euclidean_without_anchor(vector, &centroid))
                .sum::<f32>()
                / vectors.len() as f32;
            let cohesion = (1.0 - mean_distance).clamp(0.0, 1.0);
            if cohesion < 0.65 {
                continue;
            }
            let usage = members
                .iter()
                .map(|member| self.usage.get(member).copied().unwrap_or(0))
                .sum::<u32>() as f32
                / members.len() as f32;
            let confidence = (0.30 * (members.len() as f32 / 12.0).min(1.0)
                + 0.30 * cohesion
                + 0.20 * (usage / 50.0).min(1.0)
                + 0.20 * (1.0 - mean_distance).clamp(0.0, 1.0))
            .min(0.92);
            self.families.push(SemanticFamily {
                id: format!("F{}", self.next_family_id),
                centroid,
                members,
                confidence,
                strength: 0.40,
            });
            self.next_family_id += 1;
            made += 1;
        }
        self.families.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(Ordering::Equal)
        });
        self.families.truncate(250);
    }

    fn reinforce_families(&mut self) {
        for family in &self.families {
            for member in &family.members {
                if let Some(vector) = self.vectors.get_mut(member) {
                    for (value, centroid_value) in vector.iter_mut().zip(family.centroid.iter()) {
                        *value += 0.02 * family.strength * (centroid_value - *value);
                    }
                    clip(vector);
                }
            }
        }
    }

    fn decay_families(&mut self) {
        for family in &mut self.families {
            family.strength -= 0.003;
        }
        self.families
            .retain(|family| family.strength > 0.02 && family.members.len() >= 4);
    }

    fn ensure_vector(&mut self, token: &str, scale: f32, rng: &mut Rng) {
        self.vectors
            .entry(token.to_string())
            .or_insert_with(|| random_vector(scale, rng));
    }

    fn can_add_link(&mut self, left: &str, right: &str) -> bool {
        if self
            .links
            .get(left)
            .is_some_and(|neighbours| neighbours.contains_key(right))
        {
            return true;
        }
        let degree = self.links.get(left).map_or(0, BTreeMap::len);
        if degree < MAX_DEGREE {
            return true;
        }
        let new_usefulness = self.usefulness(left, right);
        let weakest = self.links.get(left).and_then(|neighbours| {
            neighbours
                .iter()
                .min_by(|left, right| {
                    left.1
                        .usefulness
                        .partial_cmp(&right.1.usefulness)
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(token, link)| (token.clone(), link.usefulness))
        });
        let Some((weakest_token, weakest_usefulness)) = weakest else {
            return false;
        };
        if new_usefulness <= weakest_usefulness {
            return false;
        }
        if let Some(neighbours) = self.links.get_mut(left) {
            neighbours.remove(&weakest_token);
        }
        if let Some(neighbours) = self.links.get_mut(&weakest_token) {
            neighbours.remove(left);
        }
        true
    }

    fn usefulness(&self, left: &str, right: &str) -> f32 {
        let similarity = self.cosine_similarity(left, right).unwrap_or(0.0);
        let degree_penalty = 1.0 / (1.0 + self.links.get(left).map_or(0, BTreeMap::len) as f32);
        let usage = self.usage.get(right).copied().unwrap_or(1) as f32;
        (0.50 * similarity + 0.20 * usage.ln_1p()) * degree_penalty
    }

    fn adjusted_weight(&self, left: &str, right: &str, weight: f32) -> f32 {
        let penalty = |token: &str| {
            self.cooldown
                .get(token)
                .is_some_and(|remaining| *remaining > 0)
                .then_some(0.4)
                .unwrap_or(1.0)
        };
        weight * penalty(left) * penalty(right)
    }

    fn gravity_step(&mut self) {
        let mut scored: Vec<_> = self
            .links
            .iter()
            .map(|(token, neighbours)| {
                (
                    token.clone(),
                    neighbours
                        .values()
                        .map(|link| link.weight.abs())
                        .sum::<f32>(),
                )
            })
            .collect();
        scored.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
        for (token, _) in scored.into_iter().take(12) {
            let Some(mut vector) = self.vectors.get(&token).cloned() else {
                continue;
            };
            let neighbours: Vec<_> = self
                .links
                .get(&token)
                .into_iter()
                .flat_map(|links| links.iter())
                .map(|(other, link)| (other.clone(), link.weight))
                .collect();
            for (other, weight) in neighbours {
                if let Some(other_vector) = self.vectors.get(&other) {
                    for (value, target) in vector.iter_mut().zip(other_vector.iter()) {
                        *value += 0.01 * weight * (target - *value);
                    }
                }
            }
            clip(&mut vector);
            self.vectors.insert(token, vector);
        }
    }

    fn decay_links(&mut self) {
        let tokens: Vec<_> = self.links.keys().cloned().collect();
        for token in tokens {
            let remove: Vec<_> = self
                .links
                .get_mut(&token)
                .map(|neighbours| {
                    neighbours.retain(|_, link| {
                        link.weight *= 0.995;
                        link.weight.abs() >= 1e-4
                    });
                    neighbours.is_empty().then_some(token.clone())
                })
                .flatten()
                .into_iter()
                .collect();
            for token in remove {
                self.links.remove(&token);
            }
        }
    }

    fn decay_cooldowns(&mut self) {
        self.cooldown.retain(|_, remaining| {
            *remaining = remaining.saturating_sub(1);
            *remaining > 0
        });
    }

    fn frequency_guard(&mut self) {
        if self.usage.len() < 2 {
            return;
        }
        let mut values: Vec<_> = self.usage.values().copied().collect();
        values.sort_unstable();
        let median = values[values.len() / 2];
        if median == 0 {
            return;
        }
        let mut frequent: Vec<_> = self.usage.iter().collect();
        frequent.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
        if frequent
            .first()
            .is_none_or(|(_, count)| **count < median * 3)
        {
            return;
        }
        for (token, _) in frequent.into_iter().take(5) {
            self.cooldown.insert(token.clone(), 5);
            if let Some(neighbours) = self.links.get_mut(token) {
                for link in neighbours.values_mut() {
                    link.weight *= 0.6;
                }
            }
        }
    }

    fn inject_escape_noise(&mut self, rng: &mut Rng) {
        let mut frequent: Vec<_> = self.usage.iter().collect();
        frequent.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
        for (token, _) in frequent.into_iter().take(5) {
            if let Some(vector) = self.vectors.get_mut(token) {
                for value in vector.iter_mut() {
                    *value += signed(rng, 0.01);
                }
                clip(vector);
            }
        }
    }
}

fn normalise(token: &str) -> String {
    token
        .chars()
        .filter(char::is_ascii_alphabetic)
        .collect::<String>()
        .to_ascii_lowercase()
}

fn random_vector(scale: f32, rng: &mut Rng) -> Array1<f32> {
    Array1::from_iter((0..semantic_dimensions()).map(|_| signed(rng, scale)))
}

fn signed(rng: &mut Rng, scale: f32) -> f32 {
    (rng.unit() as f32 * 2.0 - 1.0) * scale
}

fn clip(vector: &mut Array1<f32>) {
    for value in vector.iter_mut() {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
    let norm = vector.dot(vector).sqrt();
    if norm > 5.0 {
        vector.mapv_inplace(|value| value * 5.0 / norm);
    }
}

fn centroid(vectors: &[&Array1<f32>]) -> Option<Array1<f32>> {
    let first = vectors.first()?;
    let mut centroid = Array1::zeros(first.len());
    for vector in vectors {
        centroid += *vector;
    }
    centroid.mapv_inplace(|value| value / vectors.len() as f32);
    Some(centroid)
}

fn euclidean_without_anchor(left: &Array1<f32>, right: &Array1<f32>) -> f32 {
    left.iter()
        .skip(1)
        .zip(right.iter().skip(1))
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn identity_like(token: &str) -> bool {
    token
        .strip_prefix('a')
        .is_some_and(|suffix| suffix.chars().all(char::is_numeric))
        || token
            .strip_prefix("agent_")
            .is_some_and(|suffix| suffix.chars().all(char::is_numeric))
}

#[cfg(test)]
mod tests {
    use super::SemanticStore;
    use crate::model::Rng;

    #[test]
    fn observing_context_creates_vectors_and_bidirectional_links() {
        let mut rng = Rng::new(14);
        let mut store = SemanticStore::new();
        store.observe(
            &["red".into(), "apple".into(), "round".into()],
            0.1,
            &mut rng,
        );
        assert_eq!(store.vocabulary_size(), 3);
        assert_eq!(store.link_count(), 3);
        assert!(store.cosine_similarity("apple", "red").is_some());
    }

    #[test]
    fn numeric_tokens_are_isolated_from_general_cooccurrence() {
        let mut rng = Rng::new(15);
        let mut store = SemanticStore::new();
        store.ensure_numeric_token("bel", 2, 8, &mut rng);
        store.observe(
            &["bel".into(), "apple".into(), "round".into()],
            0.1,
            &mut rng,
        );
        assert_eq!(store.link_count(), 1);
        assert!(store.vector("bel").is_some());
    }

    #[test]
    fn repeated_coherent_context_can_form_a_semantic_family() {
        let mut rng = Rng::new(16);
        let mut store = SemanticStore::new();
        let words = ["red".into(), "apple".into(), "round".into(), "fruit".into()];
        for _ in 0..8 {
            store.observe(&words, 0.30, &mut rng);
        }
        let prototype = store.vector("red").unwrap().clone();
        for word in ["apple", "round", "fruit"] {
            store.blend_from(word, &prototype, 0.96, &mut rng);
        }
        for _ in 0..5 {
            store.tick(&mut rng);
        }
        assert!(store.family_count() > 0);
        assert!(
            store
                .families()
                .iter()
                .any(|family| family.members.len() >= 4)
        );
    }

    #[test]
    fn child_store_blends_parent_vectors_and_preserves_bounded_links() {
        let mut rng = Rng::new(17);
        let words = ["river".into(), "water".into(), "flow".into(), "bank".into()];
        let mut first = SemanticStore::new();
        let mut second = SemanticStore::new();
        let mut third = SemanticStore::new();
        for parent in [&mut first, &mut second, &mut third] {
            for _ in 0..5 {
                parent.observe(&words, 0.24, &mut rng);
            }
        }
        let retained = words.iter().cloned().collect();
        let child = SemanticStore::inherit_from([&first, &second, &third], &retained, &mut rng);
        assert_eq!(child.vocabulary_size(), 4);
        assert!(child.vector("river").is_some());
        assert!(child.link_count() > 0);
        assert!(child.reasoning_candidates(3).contains(&"river".to_string()));
    }
}

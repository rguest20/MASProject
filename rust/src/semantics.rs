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
const MIN_STRUCTURAL_OBSERVATIONS: u32 = 12;
const MIN_STRUCTURAL_FAMILIES: usize = 3;
const MIN_STRUCTURAL_ENTROPY: f32 = 0.80;

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
pub enum RelationEvidenceKind {
    Cooccurrence,
    Ordered,
    Predictive,
    Verified,
    Contradiction,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RelationEvidence {
    pub cooccurrence: u32,
    pub ordered: u32,
    pub predictive: u32,
    pub verified: u32,
    pub contradiction: u32,
}

impl RelationEvidence {
    fn observe(&mut self, kind: RelationEvidenceKind) {
        let counter = match kind {
            RelationEvidenceKind::Cooccurrence => &mut self.cooccurrence,
            RelationEvidenceKind::Ordered => &mut self.ordered,
            RelationEvidenceKind::Predictive => &mut self.predictive,
            RelationEvidenceKind::Verified => &mut self.verified,
            RelationEvidenceKind::Contradiction => &mut self.contradiction,
        };
        *counter = counter.saturating_add(1);
    }

    fn merge(&mut self, other: &Self) {
        self.cooccurrence = self.cooccurrence.saturating_add(other.cooccurrence);
        self.ordered = self.ordered.saturating_add(other.ordered);
        self.predictive = self.predictive.saturating_add(other.predictive);
        self.verified = self.verified.saturating_add(other.verified);
        self.contradiction = self.contradiction.saturating_add(other.contradiction);
    }
}

#[derive(Clone, Debug)]
pub struct SemanticLink {
    /// Net association: positive support, negative opposition. Not a probability.
    pub weight: f32,
    pub evidence: RelationEvidence,
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
    /// Family-context evidence used to distinguish widely distributed
    /// connective tokens from content that belongs to a compact region.
    family_distribution: BTreeMap<String, BTreeMap<String, u32>>,
    structural_tokens: BTreeSet<String>,
    structural_profiles: BTreeMap<String, StructuralProfile>,
    structural_edges: BTreeMap<(String, String, String), StructuralEdge>,
    relation_patterns: Vec<RelationPattern>,
}

#[derive(Clone, Debug)]
pub struct SemanticFamily {
    pub id: String,
    pub centroid: Array1<f32>,
    pub members: BTreeSet<String>,
    pub confidence: f32,
    strength: f32,
}

#[derive(Clone, Debug, Default)]
struct StructuralProfile {
    observations: u32,
    entropy: f32,
    coverage: f32,
}

/// A learned grammatical operation between two *semantic families*. The
/// operator itself is not a point in either family and therefore cannot pull
/// their content vectors together through ordinary co-occurrence.
#[derive(Clone, Debug)]
pub struct StructuralEdge {
    pub source_family_id: String,
    pub operator: String,
    pub target_family_id: String,
    pub grammatical_weight: f32,
    pub evidence: u32,
}

#[derive(Clone, Debug)]
pub struct RelationPattern {
    pub id: usize,
    pub members: BTreeSet<(String, String)>,
    pub signature: [f32; 5],
    pub confidence: f32,
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
            family_distribution: BTreeMap::new(),
            structural_tokens: BTreeSet::new(),
            structural_profiles: BTreeMap::new(),
            structural_edges: BTreeMap::new(),
            relation_patterns: Vec::new(),
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
        let mut merged = BTreeMap::<(String, String), (f32, u32, RelationEvidence)>::new();
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
                    entry.2.merge(&link.evidence);
                }
            }
        }
        let mut candidates: Vec<_> = merged
            .into_iter()
            .filter_map(|((left, right), (total, count, evidence))| {
                let weight = 0.82 * total / count as f32;
                (weight.abs() >= 0.015).then_some((left, right, weight, evidence))
            })
            .collect();
        candidates.sort_by(|left, right| {
            right
                .2
                .abs()
                .partial_cmp(&left.2.abs())
                .unwrap_or(Ordering::Equal)
        });
        for (left, right, weight, evidence) in candidates {
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
                    evidence: evidence.clone(),
                    usefulness,
                    last_seen: 0,
                },
            );
            child.links.entry(right).or_default().insert(
                left,
                SemanticLink {
                    weight,
                    evidence,
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

    pub fn structural_edges(&self) -> Vec<StructuralEdge> {
        self.structural_edges.values().cloned().collect()
    }

    pub fn relation_patterns(&self) -> &[RelationPattern] {
        &self.relation_patterns
    }

    pub fn pattern_for_pair(&self, left: &str, right: &str) -> Option<&RelationPattern> {
        self.relation_patterns.iter().find(|pattern| {
            pattern
                .members
                .contains(&(left.to_string(), right.to_string()))
                || pattern
                    .members
                    .contains(&(right.to_string(), left.to_string()))
        })
    }

    /// Mean entropy and family coverage for tokens that have crossed the
    /// structural threshold. These values are for inspection/reporting only;
    /// promotion still uses the stricter per-token threshold above.
    pub fn structural_profile_metrics(&self) -> (usize, f32, f32, u32) {
        let profiles: Vec<_> = self
            .structural_tokens
            .iter()
            .filter_map(|token| self.structural_profiles.get(token))
            .collect();
        let count = profiles.len();
        if count == 0 {
            return (0, 0.0, 0.0, 0);
        }
        (
            count,
            profiles.iter().map(|profile| profile.entropy).sum::<f32>() / count as f32,
            profiles.iter().map(|profile| profile.coverage).sum::<f32>() / count as f32,
            profiles.iter().map(|profile| profile.observations).sum(),
        )
    }

    pub fn structural_edge_metrics(&self) -> (usize, f32, u32, usize) {
        let distinct_family_pairs = self
            .structural_edges
            .values()
            .map(|edge| (edge.source_family_id.clone(), edge.target_family_id.clone()))
            .collect::<BTreeSet<_>>()
            .len();
        (
            self.structural_edges.len(),
            self.structural_edges
                .values()
                .map(|edge| edge.grammatical_weight)
                .sum(),
            self.structural_edges
                .values()
                .map(|edge| edge.evidence)
                .sum(),
            distinct_family_pairs,
        )
    }

    pub fn families(&self) -> &[SemanticFamily] {
        &self.families
    }

    pub fn community_candidates(&self) -> Vec<(String, Array1<f32>, u32)> {
        self.vectors
            .iter()
            .filter_map(|(token, vector)| {
                let usage = self.usage.get(token).copied().unwrap_or(0);
                (usage >= 5
                    && !self.numeric_tokens.contains(token)
                    && !self.structural_tokens.contains(token)
                    && !identity_like(token))
                .then_some((token.clone(), vector.clone(), usage))
            })
            .collect()
    }

    /// Strong, mature undirected associations eligible for community memory.
    /// This deliberately excludes private one-off links, numbers, and learned
    /// structural operators: public memory should seed a compact conceptual
    /// graph, not freeze every local co-occurrence into culture.
    pub fn community_link_candidates(&self) -> Vec<(String, String, f32)> {
        self.links
            .iter()
            .flat_map(|(left, neighbours)| {
                neighbours.iter().filter_map(move |(right, link)| {
                    (left < right
                        && self.usage.get(left).copied().unwrap_or(0) >= 5
                        && self.usage.get(right).copied().unwrap_or(0) >= 5
                        && !self.numeric_tokens.contains(left)
                        && !self.numeric_tokens.contains(right)
                        && !self.structural_tokens.contains(left)
                        && !self.structural_tokens.contains(right)
                        && !identity_like(left)
                        && !identity_like(right)
                        && link.weight.abs() >= 0.04)
                        .then_some((left.clone(), right.clone(), link.weight))
                })
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
                    && !self.structural_tokens.contains(*token)
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
        self.update_family_distribution(words);
        self.refresh_structural_roles();
        self.record_structural_edges(words, gain);
        let mut cleaned: Vec<String> = words
            .iter()
            .map(|word| normalise(word))
            .filter(|word| {
                !word.is_empty()
                    && !self.numeric_tokens.contains(word)
                    && !self.structural_tokens.contains(word)
            })
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

    /// Inspect a signed association independently of vector similarity.
    pub fn association(&self, left: &str, right: &str) -> Option<f32> {
        self.links.get(left)?.get(right).map(|link| link.weight)
    }

    /// Evidence dimensions let higher layers discover relation patterns from
    /// how a link was learned, without naming linguistic parts in advance.
    pub fn relation_evidence(&self, left: &str, right: &str) -> Option<RelationEvidence> {
        self.links
            .get(left)
            .and_then(|neighbours| neighbours.get(right))
            .map(|link| link.evidence.clone())
    }

    /// Add signed evidence. A negative update weakens support and can eventually
    /// reverse it; a single rejection is not a permanent logical prohibition.
    pub fn link(&mut self, left: &str, right: &str, weight: f32, rng: &mut Rng) {
        let kind = if weight < 0.0 {
            RelationEvidenceKind::Contradiction
        } else {
            RelationEvidenceKind::Cooccurrence
        };
        self.link_evidence(left, right, weight, kind, rng);
    }

    pub fn link_evidence(
        &mut self,
        left: &str,
        right: &str,
        weight: f32,
        kind: RelationEvidenceKind,
        rng: &mut Rng,
    ) {
        if !weight.is_finite()
            || weight == 0.0
            || left.is_empty()
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
            evidence: {
                let mut evidence = RelationEvidence::default();
                evidence.observe(kind.clone());
                evidence
            },
            usefulness,
            last_seen: self.tick,
        };
        self.links
            .entry(left.to_string())
            .or_default()
            .entry(right.to_string())
            .and_modify(|entry| {
                entry.weight += weight;
                entry.evidence.observe(kind.clone());
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
                entry.evidence.observe(kind);
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

    /// A small local familiarity signal for curiosity selection. Vectors are
    /// created on first exposure, so vector presence alone is not evidence
    /// that a word has acquired a useful conceptual neighbourhood.
    pub fn link_count_for(&self, token: &str) -> usize {
        self.links.get(token).map(BTreeMap::len).unwrap_or_default()
    }

    pub fn cosine_similarity(&self, left: &str, right: &str) -> Option<f32> {
        let left = self.vectors.get(left)?;
        let right = self.vectors.get(right)?;
        let dot = left.dot(right);
        let denominator = left.dot(left).sqrt() * right.dot(right).sqrt();
        (denominator > f32::EPSILON).then_some(dot / denominator)
    }

    fn family_for_token(&self, token: &str) -> Option<String> {
        self.families
            .iter()
            .filter(|family| family.members.contains(token))
            .max_by(|left, right| {
                left.confidence
                    .partial_cmp(&right.confidence)
                    .unwrap_or(Ordering::Equal)
            })
            .map(|family| family.id.clone())
    }

    /// Learn where each token occurs in family space. We count the family
    /// context around a token rather than assigning it to the family that its
    /// own vector happens to occupy: connective words earn broad support only
    /// by repeatedly appearing between varied content families.
    fn update_family_distribution(&mut self, words: &[String]) {
        if self.families.len() < MIN_STRUCTURAL_FAMILIES {
            return;
        }
        let tokens: Vec<_> = words
            .iter()
            .map(|word| normalise(word))
            .filter(|word| {
                !word.is_empty()
                    && !self.numeric_tokens.contains(word)
                    && !self.structural_tokens.contains(word)
            })
            .collect();
        for (index, token) in tokens.iter().enumerate() {
            let context_families: BTreeSet<_> = tokens
                .iter()
                .enumerate()
                .filter(|(other_index, _)| *other_index != index)
                .filter_map(|(_, other)| self.family_for_token(other))
                .collect();
            if context_families.is_empty() {
                continue;
            }
            let distribution = self.family_distribution.entry(token.clone()).or_default();
            for family_id in context_families {
                *distribution.entry(family_id).or_default() += 1;
            }
        }
    }

    fn refresh_structural_roles(&mut self) {
        let effective_family_count = self.families.len().clamp(1, 12) as f32;
        let candidates: Vec<_> = self
            .family_distribution
            .iter()
            .filter(|(token, _)| !self.structural_tokens.contains(*token))
            .filter_map(|(token, distribution)| {
                let observations: u32 = distribution.values().sum();
                let distinct = distribution.len();
                if observations < MIN_STRUCTURAL_OBSERVATIONS || distinct < MIN_STRUCTURAL_FAMILIES
                {
                    return None;
                }
                let entropy = normalised_entropy(distribution);
                let coverage = (distinct as f32 / effective_family_count).min(1.0);
                Some((token.clone(), observations, entropy, coverage))
            })
            .collect();
        for (token, observations, entropy, coverage) in candidates {
            self.structural_profiles.insert(
                token.clone(),
                StructuralProfile {
                    observations,
                    entropy,
                    coverage,
                },
            );
            if entropy >= MIN_STRUCTURAL_ENTROPY && coverage >= 0.25 {
                self.promote_structural_token(&token);
            }
        }
    }

    fn promote_structural_token(&mut self, token: &str) {
        if !self.structural_tokens.insert(token.to_string()) {
            return;
        }
        // Existing accidental co-occurrence edges would continue to pull
        // content families together, so remove them when the token becomes a
        // relation operator. Its vector may remain as historical evidence but
        // is excluded from family, graph, and community-candidate updates.
        if let Some(neighbours) = self.links.remove(token) {
            for neighbour in neighbours.keys() {
                if let Some(reverse) = self.links.get_mut(neighbour) {
                    reverse.remove(token);
                }
            }
        }
        for family in &mut self.families {
            family.members.remove(token);
        }
        self.families.retain(|family| family.members.len() >= 4);
    }

    fn record_structural_edges(&mut self, words: &[String], gain: f32) {
        let tokens: Vec<_> = words
            .iter()
            .map(|word| normalise(word))
            .filter(|word| !word.is_empty() && !self.numeric_tokens.contains(word))
            .collect();
        for (index, operator) in tokens.iter().enumerate() {
            if !self.structural_tokens.contains(operator) {
                continue;
            }
            let source = tokens[..index]
                .iter()
                .rev()
                .find_map(|token| self.family_for_token(token));
            let target = tokens
                .iter()
                .skip(index + 1)
                .find_map(|token| self.family_for_token(token));
            let (Some(source_family_id), Some(target_family_id)) = (source, target) else {
                continue;
            };
            let key = (
                source_family_id.clone(),
                operator.clone(),
                target_family_id.clone(),
            );
            let edge = self.structural_edges.entry(key).or_insert(StructuralEdge {
                source_family_id,
                operator: operator.clone(),
                target_family_id,
                grammatical_weight: 0.0,
                evidence: 0,
            });
            edge.grammatical_weight = (edge.grammatical_weight + gain).min(8.0);
            edge.evidence = edge.evidence.saturating_add(1);
        }
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
            self.detect_relation_patterns();
            self.reinforce_relation_patterns();
        }
    }

    /// Repeated evidence signatures become a compact retention scaffold.
    /// This boosts only existing pattern members; it does not invent links.
    fn reinforce_relation_patterns(&mut self) {
        let members: Vec<_> = self
            .relation_patterns
            .iter()
            .flat_map(|pattern| pattern.members.iter().cloned())
            .collect();
        for (left, right) in members {
            if let Some(link) = self
                .links
                .get_mut(&left)
                .and_then(|links| links.get_mut(&right))
            {
                link.weight = (link.weight * 1.015).clamp(-8.0, 8.0);
            }
            if let Some(link) = self
                .links
                .get_mut(&right)
                .and_then(|links| links.get_mut(&left))
            {
                link.weight = (link.weight * 1.015).clamp(-8.0, 8.0);
            }
        }
    }

    fn detect_relation_patterns(&mut self) {
        let mut buckets: BTreeMap<[u8; 5], BTreeSet<(String, String)>> = BTreeMap::new();
        for (left, neighbours) in &self.links {
            for (right, link) in neighbours {
                if left >= right || pattern_token(left) || pattern_token(right) {
                    continue;
                }
                let counts = [
                    link.evidence.cooccurrence,
                    link.evidence.ordered,
                    link.evidence.predictive,
                    link.evidence.verified,
                    link.evidence.contradiction,
                ];
                let total: u32 = counts.iter().sum();
                if total < 4 {
                    continue;
                }
                let key = counts.map(|count| ((count as f32 / total as f32) * 4.0).round() as u8);
                buckets
                    .entry(key)
                    .or_default()
                    .insert((left.clone(), right.clone()));
            }
        }
        self.relation_patterns = buckets
            .into_iter()
            .filter_map(|(key, members)| {
                let signature = key.map(|value| value as f32 / 4.0);
                (members.len() >= 3).then_some(RelationPattern {
                    id: 0,
                    signature,
                    confidence: (members.len() as f32 / 12.0).min(1.0)
                        * (0.25 + 0.75 * (signature[2] + signature[3])),
                    members,
                })
            })
            .take(64)
            .enumerate()
            .map(|(id, mut pattern)| {
                pattern.id = id;
                pattern
            })
            .collect();
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
                    && !self.structural_tokens.contains(token)
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
                        && !self.structural_tokens.contains(other)
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
                        // Opposition is a graph relationship, not evidence
                        // that concepts belong far apart in semantic space.
                        *value += 0.01 * weight.max(0.0) * (target - *value);
                    }
                }
            }
            clip(&mut vector);
            self.vectors.insert(token, vector);
        }
    }

    fn decay_links(&mut self) {
        self.links.retain(|_, neighbours| {
            neighbours.retain(|_, link| {
                link.weight *= 0.995;
                link.weight.abs() >= 1e-4
            });
            !neighbours.is_empty()
        });
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

fn normalised_entropy(distribution: &BTreeMap<String, u32>) -> f32 {
    let total: f32 = distribution.values().copied().sum::<u32>() as f32;
    if total <= f32::EPSILON || distribution.len() < 2 {
        return 0.0;
    }
    let entropy = distribution
        .values()
        .map(|count| {
            let probability = *count as f32 / total;
            -probability * probability.ln()
        })
        .sum::<f32>();
    (entropy / (distribution.len() as f32).ln()).clamp(0.0, 1.0)
}

fn identity_like(token: &str) -> bool {
    token
        .strip_prefix('a')
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(char::is_numeric))
        || token
            .strip_prefix("agent_")
            .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(char::is_numeric))
}

fn pattern_token(token: &str) -> bool {
    identity_like(token) || token.starts_with("pattern_") || token == "r0"
}

#[cfg(test)]
mod tests {
    use super::{RelationEvidenceKind, SemanticFamily, SemanticStore};
    use crate::model::Rng;
    use ndarray::Array1;
    use std::collections::BTreeSet;

    #[test]
    fn signed_associations_reverse_and_survive_inheritance() {
        let mut rng = Rng::new(144);
        let mut store = SemanticStore::new();
        store.link("hot", "cold", 0.2, &mut rng);
        store.link("hot", "cold", -0.8, &mut rng);
        assert!(store.association("hot", "cold").unwrap() < 0.0);
        assert_eq!(
            store.association("hot", "cold"),
            store.association("cold", "hot")
        );
        store.link("hot", "cold", f32::NAN, &mut rng);
        let tokens = ["hot".into(), "cold".into()].into_iter().collect();
        let child = SemanticStore::inherit_from([&store, &store, &store], &tokens, &mut rng);
        assert!(child.association("hot", "cold").unwrap() < 0.0);
    }

    #[test]
    fn links_retain_operational_evidence_kinds() {
        let mut rng = Rng::new(145);
        let mut store = SemanticStore::new();
        store.link("nightingale", "bird", 0.2, &mut rng);
        store.link_evidence(
            "nightingale",
            "bird",
            0.1,
            RelationEvidenceKind::Verified,
            &mut rng,
        );
        store.link_evidence(
            "nightingale",
            "tree",
            -0.1,
            RelationEvidenceKind::Contradiction,
            &mut rng,
        );
        let bird = store.relation_evidence("nightingale", "bird").unwrap();
        assert_eq!(bird.cooccurrence, 1);
        assert_eq!(bird.verified, 1);
        assert_eq!(store.relation_evidence("bird", "nightingale"), Some(bird));
        assert_eq!(
            store
                .relation_evidence("nightingale", "tree")
                .unwrap()
                .contradiction,
            1
        );
    }

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
    fn only_numbered_agent_tokens_are_identity_like() {
        assert!(!super::identity_like("a"));
        assert!(!super::identity_like("agent_"));
        assert!(super::identity_like("a12"));
        assert!(super::identity_like("agent_12"));
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

    #[test]
    fn high_entropy_connector_becomes_an_edge_operator_not_a_family_member() {
        let mut rng = Rng::new(21);
        let mut store = SemanticStore::new();
        let groups = [
            ["prince", "royal", "crown", "palace"],
            ["tower", "stone", "tall", "window"],
            ["bird", "wing", "nest", "fly"],
        ];
        for group in groups {
            let words = group.map(str::to_string);
            for _ in 0..3 {
                store.observe(&words, 0.30, &mut rng);
            }
        }
        for (index, group) in groups.iter().enumerate() {
            store.families.push(SemanticFamily {
                id: format!("F{index}"),
                centroid: Array1::zeros(super::semantic_dimensions()),
                members: group
                    .iter()
                    .map(|word| (*word).to_string())
                    .collect::<BTreeSet<_>>(),
                confidence: 0.9,
                strength: 0.5,
            });
        }

        for _ in 0..2 {
            for sentence in [
                ["prince", "have", "tower"],
                ["tower", "have", "bird"],
                ["bird", "have", "prince"],
            ] {
                store.observe(&sentence.map(str::to_string), 0.20, &mut rng);
            }
        }

        assert_eq!(store.structural_profile_metrics().0, 1);
        assert!(!store.reasoning_candidates(1).contains(&"have".to_string()));
        assert!(
            store
                .structural_edges()
                .iter()
                .any(|edge| edge.operator == "have" && edge.evidence > 0)
        );
        assert!(
            store
                .families()
                .iter()
                .all(|family| !family.members.contains("have"))
        );
        assert!(store.links.get("have").is_none());
    }
}

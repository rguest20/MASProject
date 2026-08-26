//! Confidence-gated community semantic centroids and repair tasks.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use ndarray::Array1;
use serde_json::{Value, json};

use crate::model::{Agent, Rng};
use crate::semantics::semantic_dimensions;

#[derive(Clone, Debug)]
pub struct CommunityToken {
    pub vector: Array1<f32>,
    pub coverage: f32,
    pub mean_distance: f32,
    pub confidence: f32,
    pub count: usize,
    age: u32,
}

#[derive(Clone, Debug, Default)]
pub struct CommunitySemanticMap {
    tokens: BTreeMap<String, CommunityToken>,
}

#[derive(Clone, Debug, Default)]
pub struct AlignmentMetrics {
    pub community_tokens: usize,
    pub proposed: usize,
    pub repaired: usize,
    pub mean_distance: f32,
}

impl CommunitySemanticMap {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn token(&self, token: &str) -> Option<&CommunityToken> {
        self.tokens.get(token)
    }

    pub fn update(&mut self, agents: &[Agent]) {
        let population = agents.len().max(1);
        let mut collected = BTreeMap::<String, Vec<Array1<f32>>>::new();
        for agent in agents {
            for (token, vector, _) in agent.semantics.community_candidates() {
                collected.entry(token).or_default().push(vector);
            }
        }
        for item in self.tokens.values_mut() {
            item.age += 1;
        }
        for (token, vectors) in collected {
            let coverage = vectors.len() as f32 / population as f32;
            if coverage < 0.15 {
                continue;
            }
            let mean = centroid(&vectors);
            let mean_distance = vectors
                .iter()
                .map(|vector| distance(vector, &mean))
                .sum::<f32>()
                / vectors.len() as f32;
            if mean_distance > 2.0 {
                continue;
            }
            let confidence =
                (0.6 * coverage + 0.4 * (1.0 / (1.0 + 0.5 * mean_distance))).clamp(0.0, 1.0);
            let alpha = 0.03 + 0.22 * confidence;
            self.tokens
                .entry(token)
                .and_modify(|current| {
                    for (value, target) in current.vector.iter_mut().zip(mean.iter()) {
                        *value = (1.0 - alpha) * *value + alpha * *target;
                    }
                    current.coverage = coverage;
                    current.mean_distance = mean_distance;
                    current.confidence = confidence;
                    current.count = vectors.len();
                    current.age = 0;
                })
                .or_insert(CommunityToken {
                    vector: mean,
                    coverage,
                    mean_distance,
                    confidence,
                    count: vectors.len(),
                    age: 0,
                });
        }
        self.tokens
            .retain(|_, item| item.age < 10 || item.confidence >= 0.25);
        if self.tokens.len() > 512 {
            let mut ranked: Vec<_> = self
                .tokens
                .iter()
                .map(|(token, item)| (token.clone(), item.confidence, item.age))
                .collect();
            ranked.sort_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| right.2.cmp(&left.2))
            });
            for (token, _, _) in ranked.into_iter().take(self.tokens.len() - 512) {
                self.tokens.remove(&token);
            }
        }
    }

    pub fn repair_tasks(&self, agents: &mut [Agent], rng: &mut Rng) -> AlignmentMetrics {
        let mut metrics = AlignmentMetrics {
            community_tokens: self.len(),
            ..AlignmentMetrics::default()
        };
        let mut candidates = Vec::new();
        for (agent_index, agent) in agents.iter().enumerate() {
            for (token, local, _) in agent.semantics.community_candidates() {
                let Some(community) = self.tokens.get(&token) else {
                    continue;
                };
                if community.count >= 3 && community.confidence >= 0.25 {
                    let distance = distance(&local, &community.vector);
                    if distance >= 0.60 {
                        candidates.push((agent_index, token, distance));
                    }
                }
            }
        }
        candidates.sort_by(|left, right| right.2.partial_cmp(&left.2).unwrap_or(Ordering::Equal));
        for (agent_index, token, before) in candidates.into_iter().take(4) {
            let Some(community) = self.tokens.get(&token) else {
                continue;
            };
            metrics.proposed += 1;
            let agent = &mut agents[agent_index];
            agent
                .semantics
                .blend_from(&token, &community.vector, 0.12, rng);
            let after = agent
                .semantics
                .vector(&token)
                .map(|local| distance(local, &community.vector))
                .unwrap_or(before);
            let improvement = (before - after).max(0.0);
            if improvement <= 0.0 {
                continue;
            }
            let similarity = cosine(
                agent
                    .semantics
                    .vector(&token)
                    .expect("repaired token exists"),
                &community.vector,
            )
            .max(0.0);
            let reward = (1.5 * (0.6 * similarity + 0.4 * improvement.min(1.0))).min(1.5) as f64;
            agent.fitness += reward;
            agent.energy = (agent.energy + 0.1 * reward).min(100.0);
            agent.state_event("learning_success");
            metrics.repaired += 1;
            metrics.mean_distance += after;
        }
        if metrics.repaired > 0 {
            metrics.mean_distance /= metrics.repaired as f32;
        }
        metrics
    }

    /// JSON-safe, bounded public centroids. Private agent maps and links are
    /// intentionally excluded: a later run starts with new individuals.
    pub fn memory_value(&self) -> Value {
        let tokens = self
            .tokens
            .iter()
            .map(|(token, item)| {
                (
                    token.clone(),
                    json!({
                        "vector": item.vector.to_vec(),
                        "coverage": item.coverage,
                        "mean_distance": item.mean_distance,
                        "confidence": item.confidence,
                        "count": item.count,
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        Value::Object(tokens)
    }

    pub fn restore_memory_value(&mut self, value: &Value) -> usize {
        let Some(tokens) = value.as_object() else {
            return 0;
        };
        let mut restored = BTreeMap::new();
        for (raw_token, item) in tokens {
            let token = raw_token.trim().to_ascii_lowercase();
            let Some(item) = item.as_object() else {
                continue;
            };
            let Some(vector) = item.get("vector").and_then(Value::as_array) else {
                continue;
            };
            if token.is_empty() || vector.len() != semantic_dimensions() {
                continue;
            }
            let values = vector.iter().map(Value::as_f64).collect::<Option<Vec<_>>>();
            let Some(values) = values else {
                continue;
            };
            if values.iter().any(|value| !value.is_finite()) {
                continue;
            }
            let count = item
                .get("count")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                .min(1_000_000) as usize;
            if count == 0 {
                continue;
            }
            let bounded = |name: &str| {
                item.get(name)
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .unwrap_or(0.0) as f32
            };
            restored.insert(
                token,
                CommunityToken {
                    vector: Array1::from(
                        values
                            .into_iter()
                            .map(|value| value as f32)
                            .collect::<Vec<_>>(),
                    ),
                    coverage: bounded("coverage").clamp(0.0, 1.0),
                    mean_distance: bounded("mean_distance").clamp(0.0, 100.0),
                    confidence: bounded("confidence").clamp(0.0, 1.0),
                    count,
                    age: 0,
                },
            );
        }
        if restored.len() > 512 {
            let mut ranked: Vec<_> = restored
                .iter()
                .map(|(token, item)| (token.clone(), item.confidence, item.count))
                .collect();
            ranked.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| right.2.cmp(&left.2))
            });
            ranked = ranked.into_iter().take(512).collect();
            let allowed: std::collections::BTreeSet<_> =
                ranked.into_iter().map(|(token, _, _)| token).collect();
            restored.retain(|token, _| allowed.contains(token));
        }
        self.tokens = restored;
        self.tokens.len()
    }

    /// Seed only confident public concepts into an otherwise fresh population.
    pub fn seed_agents(&self, agents: &mut [Agent], rng: &mut Rng) -> usize {
        let mut seeds: Vec<_> = self
            .tokens
            .iter()
            .filter(|(_, item)| item.confidence >= 0.35)
            .collect();
        seeds.sort_by(|left, right| {
            right
                .1
                .confidence
                .partial_cmp(&left.1.confidence)
                .unwrap_or(Ordering::Equal)
        });
        seeds.truncate(256);
        for agent in agents {
            for (token, item) in &seeds {
                let strength = 0.18 + 0.32 * item.confidence;
                agent
                    .semantics
                    .blend_from(token, &item.vector, strength, rng);
                agent.vocabulary.insert((*token).clone());
            }
        }
        seeds.len()
    }
}

fn centroid(vectors: &[Array1<f32>]) -> Array1<f32> {
    let mut mean = Array1::zeros(vectors[0].len());
    for vector in vectors {
        mean += vector;
    }
    mean.mapv(|value| value / vectors.len() as f32)
}

fn distance(left: &Array1<f32>, right: &Array1<f32>) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn cosine(left: &Array1<f32>, right: &Array1<f32>) -> f32 {
    let denominator = left.dot(left).sqrt() * right.dot(right).sqrt();
    if denominator > f32::EPSILON {
        left.dot(right) / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::CommunitySemanticMap;
    use crate::model::{Agent, Rng};

    #[test]
    fn shared_high_usage_vector_enters_the_community_map() {
        let mut rng = Rng::new(801);
        let words = ["red".into(), "apple".into(), "round".into()];
        let mut agents = (0..4)
            .map(|id| {
                let mut agent = Agent::new(id, &mut rng);
                for _ in 0..8 {
                    agent.observe(&words, 0.20, &mut rng);
                }
                agent
            })
            .collect::<Vec<_>>();
        let prototype = agents[0].semantics.vector("apple").unwrap().clone();
        for agent in agents.iter_mut().skip(1) {
            agent
                .semantics
                .blend_from("apple", &prototype, 0.98, &mut rng);
        }
        let mut map = CommunitySemanticMap::default();
        map.update(&agents);
        assert!(map.token("apple").is_some());
        assert!(map.token("apple").unwrap().confidence > 0.25);
    }
}

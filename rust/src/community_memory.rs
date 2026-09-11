//! Durable public memory between otherwise fresh simulation runs.
//!
//! It holds public semantic centroids, confidence-gated conceptual
//! relationships, and successful conventions. Agent-private vectors, links,
//! fitness, and identities are never serialised here.

use std::fs;
use std::io;
use std::path::Path;

use serde_json::{Value, json};

use crate::alignment::CommunitySemanticMap;
use crate::lexicon::CommunityLexicon;
use crate::model::{Agent, Rng};
use crate::semantics::semantic_dimensions;

const SCHEMA_VERSION: u64 = 1;

#[derive(Clone, Debug, Default)]
pub struct MemoryStatus {
    pub loaded: bool,
    pub tokens: usize,
    pub relationships: usize,
    pub seeded_tokens: usize,
}

pub fn restore(
    path: &Path,
    lexicon: &mut CommunityLexicon,
    semantics: &mut CommunitySemanticMap,
    agents: &mut [Agent],
    rng: &mut Rng,
) -> MemoryStatus {
    let Ok(text) = fs::read_to_string(path) else {
        return MemoryStatus::default();
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return MemoryStatus::default();
    };
    if value.get("schema_version").and_then(Value::as_u64) != Some(SCHEMA_VERSION)
        || value.get("implementation").and_then(Value::as_str) != Some("rust")
        || value.get("dimensions").and_then(Value::as_u64) != Some(semantic_dimensions() as u64)
    {
        return MemoryStatus::default();
    }
    lexicon.restore_memory_value(value.get("lexicon").unwrap_or(&Value::Null));
    let tokens = semantics.restore_memory_value(value.get("semantic").unwrap_or(&Value::Null));
    let relationships = semantics.restore_relationships_memory_value(
        value.get("semantic_relationships").unwrap_or(&Value::Null),
    );
    semantics.restore_patterns_memory_value(value.get("semantic_patterns").unwrap_or(&Value::Null));
    let seeded_tokens = semantics.seed_agents(agents, rng);
    MemoryStatus {
        loaded: true,
        tokens,
        relationships,
        seeded_tokens,
    }
}

pub fn save(
    path: &Path,
    generation: usize,
    lexicon: &CommunityLexicon,
    semantics: &CommunitySemanticMap,
) -> io::Result<()> {
    let value = json!({
        "schema_version": SCHEMA_VERSION,
        "implementation": "rust",
        "dimensions": semantic_dimensions(),
        "generation": generation,
        "semantic": semantics.memory_value(),
        "semantic_relationships": semantics.relationships_memory_value(),
        "semantic_patterns": semantics.patterns_memory_value(),
        "lexicon": lexicon.memory_value(),
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_string_pretty(&value).expect("JSON memory is serialisable"),
    )?;
    fs::rename(temporary, path)
}

#[cfg(test)]
mod tests {
    use super::{restore, save};
    use crate::alignment::CommunitySemanticMap;
    use crate::lexicon::CommunityLexicon;
    use crate::model::{Agent, Rng};

    #[test]
    fn restores_public_centroids_and_conventions_into_fresh_agents() {
        let path = std::env::temp_dir().join(format!(
            "mas-rust-community-memory-{}.json",
            std::process::id()
        ));
        let mut rng = Rng::new(71);
        let words = ["apple".to_string(), "red".to_string(), "round".to_string()];
        let mut old_agents = (0..4)
            .map(|id| {
                let mut agent = Agent::new(id, &mut rng);
                for _ in 0..8 {
                    agent.observe(&words, 0.20, &mut rng);
                }
                agent
            })
            .collect::<Vec<_>>();
        let prototype = old_agents[0]
            .semantics
            .vector("apple")
            .expect("apple vector")
            .clone();
        for agent in old_agents.iter_mut().skip(1) {
            agent
                .semantics
                .blend_from("apple", &prototype, 0.98, &mut rng);
        }
        let mut old_map = CommunitySemanticMap::default();
        old_map.update(&old_agents);
        assert!(old_map.relationship_count() > 0);
        let mut old_lexicon = CommunityLexicon::default();
        old_lexicon.observe_numeric_success(4, "bel");
        old_lexicon.observe_numeric_success(4, "bel");
        save(&path, 12, &old_lexicon, &old_map).expect("save memory");

        let mut new_agents = (0..4)
            .map(|id| Agent::new(id, &mut rng))
            .collect::<Vec<_>>();
        let mut new_lexicon = CommunityLexicon::default();
        let mut new_map = CommunitySemanticMap::default();
        let status = restore(
            &path,
            &mut new_lexicon,
            &mut new_map,
            &mut new_agents,
            &mut rng,
        );
        assert!(status.loaded);
        assert!(status.tokens > 0);
        assert!(status.relationships > 0);
        assert!(status.seeded_tokens > 0);
        assert_eq!(new_lexicon.numeric_token(4), Some("bel"));
        assert!(
            new_agents
                .iter()
                .all(|agent| agent.vocabulary.contains("apple"))
        );
        assert!(
            new_agents
                .iter()
                .all(|agent| agent.semantics.link_count() > 0)
        );
        let _ = std::fs::remove_file(path);
    }
}

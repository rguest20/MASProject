//! Inquiry checkpoints are separate from public vector memory. They contain
//! source IDs, not supplied answers; restoration resolves IDs against the corpus.
use super::*;
use crate::learning::{CONSOLIDATED_LIMIT, EvidenceMemory, MEMORY_LIMIT};
use serde_json::{Value, json};
use std::io::Read;

const VERSION: u64 = 2;
const MAX_BYTES: u64 = 4 * 1024 * 1024;

impl InquiryLab {
    pub fn start(
        &mut self,
        agents: &mut [Agent],
        reading: &mut ReadingBridge,
        path: Option<&Path>,
        root: &Path,
    ) -> io::Result<()> {
        self.corpus_fingerprint = reading.inquiry_fingerprint();
        let mut previous_transfer = None;
        self.memory_status = match path {
            None => "disabled (--fresh-community)".into(),
            Some(path) => match read_checkpoint(path) {
                Ok(value) => match self.restore_value(&value, agents, reading) {
                    Ok(()) => {
                        if value["version"].as_u64() == Some(VERSION) {
                            previous_transfer = Some(value["transfer_at_save"].clone());
                        }
                        format!(
                            "restored {} source references across {} learners, {} shared episodes and {} questions",
                            agents
                                .iter()
                                .map(|a| a.learning.observations().count())
                                .sum::<usize>(),
                            agents.len(),
                            self.community.observations().count(),
                            self.agenda.len()
                        )
                    }
                    Err(reason) => format!("not restored: {reason}"),
                },
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    "new inquiry memory".into()
                }
                Err(error) => format!("not restored: {error}"),
            },
        };
        if path.is_some()
            && self.memory_status.starts_with("restored")
            && previous_transfer.is_none()
        {
            self.memory_status.push_str("; migrated source evidence to learner v2, old prediction scores are not comparable");
        }
        // Start-up scoring happens before reading, inquiry, teaching or culling.
        // Keep the same probes across restarts; fresh runs use an independent RNG.
        if self.probes.is_empty() {
            let mut rng = Rng::new(901_337);
            let mut sources = BTreeSet::new();
            for _ in 0..32 {
                if let Some(probe) = reading.inquiry_observation(&[], &sources, true, &mut rng) {
                    sources.insert(probe.source);
                    self.probes.push(probe);
                }
            }
        }
        self.evaluate(agents, 0);
        let transfer = self.transfer_value(agents);
        let loss_change = previous_transfer
            .as_ref()
            .and_then(|previous| Some(transfer["loss"].as_f64()? - previous["loss"].as_f64()?));
        if let Some(change) = loss_change {
            self.memory_status.push_str(&format!(
                "; restart Brier change {change:+.6} (zero means retained)"
            ));
        }
        fs::write(
            root.join("inquiry_restart.json"),
            serde_json::to_vec_pretty(&json!({
                "status": self.memory_status, "lifetime_generations": self.elapsed_generations,
                "before_training": true, "transfer": transfer,
                "previous_checkpoint_transfer":previous_transfer, "restart_loss_change":loss_change,
                "source_references": agents.iter().map(|a| a.learning.observations().count()).sum::<usize>(),
                "questions": self.agenda.len(), "projects": self.commitments.len(),
            }))?,
        )?;
        self.write_monitor(0, root)
    }

    pub fn memory_status(&self) -> &str {
        &self.memory_status
    }

    fn transfer_value(&self, agents: &[Agent]) -> Value {
        let mut loss = 0.0;
        let mut baseline = 0.0;
        let mut exact = 0;
        for agent in agents {
            let unigram = self.predict(agent, &[]);
            for probe in &self.probes {
                let prediction = self.predict(agent, &probe.context);
                loss += prediction.loss(&probe.outcome);
                baseline += unigram.loss(&probe.outcome);
                exact += usize::from(prediction.best() == Some(probe.outcome.as_str()));
            }
        }
        let count = agents.len() * self.probes.len();
        json!({"predictions":count, "exact":exact, "loss":loss / count.max(1) as f64,
            "unigram_loss":baseline / count.max(1) as f64,
            "probe_sources":self.probes.iter().map(|p| p.source).collect::<Vec<_>>()})
    }

    fn memory_value(&self, agents: &[Agent]) -> Value {
        json!({
            "version":VERSION, "corpus":self.corpus_fingerprint,
            "context_limit":CONTEXT_LIMIT, "memory_limit":MEMORY_LIMIT,
            "generations":self.elapsed_generations,
            "agents":agents.iter().map(|agent| {
                let project = self.commitments.get(&agent.id)
                    .filter(|(lineage,context,_)| *lineage == agent.lineage_id && self.agenda.contains_key(context))
                    .map(|(_,context,turns)| json!({"context":context,"turns":turns}));
                json!({"id":agent.id, "sources":agent.learning.observations().map(|o| o.source).collect::<Vec<_>>(),
                    "project":project, "consolidated":agent.learning.consolidated_len(), "regret":agent.learning.trust_state()})
            }).collect::<Vec<_>>(),
            "agenda":self.agenda.iter().map(|(context,q)| json!({"context":context,
                "attempts":q.attempts,"observations":q.observations,"loss":q.loss,
                "progress":q.progress,"last_visit":q.last_visit})).collect::<Vec<_>>(),
            "sources":self.sources,
            "probes":self.probes.iter().map(|p| p.source).collect::<Vec<_>>(),
            "metrics":{"observations":self.metrics.observations,"reading_sources":self.metrics.reading_sources,
                "lessons":self.metrics.lessons,"peer_tests":self.metrics.peer_tests,"extended":self.metrics.extended,
                "contributors":self.metrics.contributors,"training":self.metrics.training,"transfer":self.metrics.transfer},
            "transfer_at_save":self.transfer_value(agents),
        })
    }

    pub fn save(&self, path: &Path, agents: &[Agent]) -> io::Result<()> {
        let bytes = serde_json::to_vec(&self.memory_value(agents))?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "inquiry checkpoint exceeds size limit",
            ));
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("json.tmp");
        let mut file = fs::File::create(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(temporary, path)
    }

    fn restore_value(
        &mut self,
        v: &Value,
        agents: &mut [Agent],
        reading: &ReadingBridge,
    ) -> Result<(), String> {
        if !matches!(v["version"].as_u64(), Some(1 | VERSION))
            || v["context_limit"].as_u64() != Some(CONTEXT_LIMIT as u64)
            || v["memory_limit"].as_u64() != Some(MEMORY_LIMIT as u64)
        {
            return Err("incompatible inquiry format".into());
        }
        if v["corpus"].as_u64() != Some(reading.inquiry_fingerprint()) {
            return Err("corpus changed; source memory was not mixed across corpora".into());
        }
        let saved_agents = array(&v["agents"], agents.len())?;
        if saved_agents.len() != agents.len() {
            return Err("population size changed".into());
        }
        let recent = source_ids(&v["sources"], SOURCE_WINDOW)?;
        let probes = source_ids(&v["probes"], 32)?;
        let mut all_sources: BTreeSet<_> = recent.iter().chain(&probes).copied().collect();
        let mut memories = BTreeMap::new();
        for item in saved_agents {
            let id = number(&item["id"])?;
            let sources = source_ids(&item["sources"], MEMORY_LIMIT + CONSOLIDATED_LIMIT)?;
            all_sources.extend(&sources);
            if !agents.iter().any(|a| a.id == id) || memories.insert(id, sources).is_some() {
                return Err("duplicate or unknown population slot".into());
            }
        }
        let resolved = reading.inquiry_sources(&all_sources);
        if resolved.len() != all_sources.len() {
            return Err("unknown corpus source ID".into());
        }
        let mut restored = Self {
            corpus_fingerprint: reading.inquiry_fingerprint(),
            elapsed_generations: number(&v["generations"])?,
            ..Self::default()
        };
        restored.run_base = restored.elapsed_generations;
        for source in recent {
            if resolved[&source].1 {
                return Err("evaluation source in training history".into());
            }
            restored.remember_source(source);
            restored.observe_community(&resolved[&source].0, reading);
        }
        for source in probes {
            if !resolved[&source].1 {
                return Err("training source in evaluation probes".into());
            }
            restored.probes.push(resolved[&source].0.clone());
        }
        for item in array(&v["agenda"], AGENDA_LIMIT)? {
            let context = context(&item["context"])?;
            let q = Question {
                attempts: number(&item["attempts"])?,
                observations: number(&item["observations"])?,
                loss: real(&item["loss"], 0.0, 2.0)?,
                progress: real(&item["progress"], -2.0, 2.0)?,
                last_visit: number(&item["last_visit"])?,
            };
            if q.observations > q.attempts
                || q.last_visit > restored.elapsed_generations
                || restored.agenda.insert(context, q).is_some()
            {
                return Err("invalid inquiry question history".into());
            }
        }
        let mut learned = BTreeMap::new();
        for item in saved_agents {
            let id = number(&item["id"])?;
            let mut observations = Vec::new();
            for source in &memories[&id] {
                let (observation, evaluation) = &resolved[source];
                if *evaluation {
                    return Err("evaluation source in learner memory".into());
                }
                observations.push(observation.clone());
            }
            let (consolidated, regret) = if v["version"].as_u64() == Some(1) {
                (0, [0.0; CONTEXT_LIMIT + 1])
            } else {
                let consolidated = number(&item["consolidated"])?;
                let values = array(&item["regret"], CONTEXT_LIMIT + 1)?;
                if values.len() != CONTEXT_LIMIT + 1 {
                    return Err("invalid trust state".into());
                }
                let mut regret = [0.0; CONTEXT_LIMIT + 1];
                for (value, saved) in regret.iter_mut().zip(values) {
                    *value = real(saved, -100.0, 100.0)?;
                }
                (consolidated, regret)
            };
            if consolidated > CONSOLIDATED_LIMIT
                || consolidated > observations.len()
                || observations.len() - consolidated > MEMORY_LIMIT
            {
                return Err("invalid consolidated memory size".into());
            }
            learned.insert(
                id,
                EvidenceMemory::restore(&observations, consolidated, regret),
            );
            let project = &item["project"];
            if !project.is_null() {
                let context = context(&project["context"])?;
                let turns = number(&project["turns"])?;
                if turns > 6 || !restored.agenda.contains_key(&context) {
                    return Err("invalid active project".into());
                }
                // Rebind responsibility to the new occupant, not an old identity.
                let lineage = agents.iter().find(|a| a.id == id).unwrap().lineage_id;
                restored.commitments.insert(id, (lineage, context, turns));
            }
        }
        let m = &v["metrics"];
        restored.metrics.observations = number(&m["observations"])?;
        restored.metrics.reading_sources = number(&m["reading_sources"])?;
        restored.metrics.lessons = number(&m["lessons"])?;
        restored.metrics.peer_tests = number(&m["peer_tests"])?;
        restored.metrics.extended = number(&m["extended"])?;
        for id in array(&m["contributors"], agents.len())? {
            let id = number(id)?;
            if !learned.contains_key(&id) {
                return Err("unknown contributor".into());
            }
            restored.metrics.contributors.insert(id);
        }
        for pair in array(&m["training"], HISTORY_LIMIT)? {
            let pair = array(pair, 2)?;
            if pair.len() != 2 {
                return Err("invalid training metric".into());
            }
            restored.metrics.training.push_back((
                pair[0].as_bool().ok_or("invalid exact score")?,
                real(&pair[1], 0.0, 2.0)?,
            ));
        }
        for pair in array(&m["transfer"], HISTORY_LIMIT)? {
            let pair = array(pair, 2)?;
            if pair.len() != 2 {
                return Err("invalid transfer metric".into());
            }
            restored
                .metrics
                .transfer
                .push_back((real(&pair[0], 0.0, 2.0)?, real(&pair[1], 0.0, 2.0)?));
        }
        // Commit only after the whole checkpoint passes validation.
        for agent in agents {
            agent.learning = learned.remove(&agent.id).unwrap();
        }
        *self = restored;
        Ok(())
    }
}

fn read_checkpoint(path: &Path) -> io::Result<Value> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "oversized inquiry checkpoint",
        ));
    }
    serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn array(v: &Value, max: usize) -> Result<&Vec<Value>, String> {
    let a = v.as_array().ok_or("missing or invalid checkpoint list")?;
    if a.len() > max {
        return Err("checkpoint list exceeds bound".into());
    }
    Ok(a)
}
fn number(v: &Value) -> Result<usize, String> {
    let n = v.as_u64().ok_or("missing or invalid counter")?;
    if n > 1_000_000_000_000 {
        return Err("counter exceeds limit".into());
    }
    usize::try_from(n).map_err(|_| "counter overflow".into())
}
fn real(v: &Value, low: f64, high: f64) -> Result<f64, String> {
    v.as_f64()
        .filter(|x| x.is_finite() && *x >= low && *x <= high)
        .ok_or("invalid score".into())
}
fn context(v: &Value) -> Result<Vec<String>, String> {
    let a = array(v, CONTEXT_LIMIT)?;
    if a.is_empty() {
        return Err("empty question context".into());
    }
    a.iter()
        .map(|v| {
            v.as_str()
                .filter(|s| !s.is_empty() && s.len() <= 256)
                .map(str::to_owned)
                .ok_or("invalid context token".into())
        })
        .collect()
}
fn source_ids(v: &Value, max: usize) -> Result<Vec<u64>, String> {
    let ids: Vec<_> = array(v, max)?
        .iter()
        .map(|v| v.as_u64().ok_or("invalid source ID".into()))
        .collect::<Result<_, String>>()?;
    if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err("duplicate source ID".into());
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (InquiryLab, Vec<Agent>, ReadingBridge) {
        let mut reading = ReadingBridge::inquiry_fixture(
            (0..180)
                .map(|i| {
                    vec![
                        format!("source{i}"),
                        "cue".into(),
                        if i % 3 == 0 { "left" } else { "right" }.into(),
                    ]
                })
                .collect(),
        );
        let mut rng = Rng::new(456);
        let mut agents: Vec<_> = (0..3).map(|id| Agent::new(id, &mut rng)).collect();
        let mut lab = InquiryLab {
            corpus_fingerprint: reading.inquiry_fingerprint(),
            elapsed_generations: 80,
            ..InquiryLab::default()
        };
        let context = vec!["cue".into()];
        for i in 0..90 {
            let observation = reading
                .inquiry_observation(&context, &lab.source_set, false, &mut rng)
                .unwrap();
            agents[i % 3].learning.observe(&observation);
            lab.remember_source(observation.source);
            lab.observe_community(&observation, &reading);
        }
        for _ in 0..10 {
            let used = lab.probes.iter().map(|p| p.source).collect();
            lab.probes.push(
                reading
                    .inquiry_observation(&context, &used, true, &mut rng)
                    .unwrap(),
            );
        }
        lab.agenda.insert(
            context.clone(),
            Question {
                attempts: 20,
                observations: 20,
                loss: 0.5,
                progress: 0.05,
                last_visit: 78,
            },
        );
        lab.commitments
            .insert(agents[1].id, (agents[1].lineage_id, context, 3));
        (lab, agents, reading)
    }

    #[test]
    fn restart_preserves_predictions_projects_and_source_identity() {
        let (lab, agents, reading) = fixture();
        let checkpoint = lab.memory_value(&agents);
        let mut rng = Rng::new(789);
        // Population order and fresh biological identities need not match.
        let mut restarted: Vec<_> = (0..3).rev().map(|id| Agent::new(id, &mut rng)).collect();
        let mut restored = InquiryLab::default();
        restored
            .restore_value(&checkpoint, &mut restarted, &reading)
            .unwrap();
        for agent in &mut restarted {
            let old = &agents[agent.id];
            assert_eq!(
                agent.learning.observations().collect::<Vec<_>>(),
                old.learning.observations().collect::<Vec<_>>()
            );
            for probe in &lab.probes {
                assert_eq!(
                    agent.learning.predict(&probe.context).probabilities,
                    old.learning.predict(&probe.context).probabilities
                );
                assert!(!agent.learning.contains(probe.source));
            }
            let lesson = old.learning.observations().next().unwrap();
            assert!(!agent.learning.observe(lesson));
        }
        assert_eq!(restored.sources, lab.sources);
        assert_eq!(restored.run_base, 80);
        let project = &restored.commitments[&1];
        assert_eq!(
            project.0,
            restarted.iter().find(|a| a.id == 1).unwrap().lineage_id
        );
        assert_eq!(project.2, 3);
        assert_eq!(restored.agenda[&project.1].last_visit, 78);
        // Compare identical population order for exact floating-point totals.
        restarted.sort_by_key(|a| a.id);
        assert_eq!(
            restored.transfer_value(&restarted),
            checkpoint["transfer_at_save"]
        );
    }

    #[test]
    fn invalid_checkpoint_never_partially_restores_learners_or_agenda() {
        let (lab, mut agents, reading) = fixture();
        let original = lab.memory_value(&agents);
        let mut variants = Vec::new();
        let mut value = original.clone();
        value["corpus"] = json!(0);
        variants.push(value);
        let mut value = original.clone();
        value["agents"][0]["sources"][0] = json!(lab.probes[0].source);
        variants.push(value);
        let mut value = original.clone();
        value["agents"][0]["sources"] = json!([0]);
        variants.push(value);
        let mut value = original.clone();
        value["agents"][0]["sources"] = json!(vec![1; MEMORY_LIMIT + 1]);
        variants.push(value);
        let mut value = original.clone();
        value["agents"][0]["sources"][1] = value["agents"][0]["sources"][0].clone();
        variants.push(value);
        let mut value = original.clone();
        value["metrics"]["transfer"] = json!([[2.5, 0.1]]);
        variants.push(value);
        for value in variants {
            let mut restored = InquiryLab::default();
            assert!(
                restored
                    .restore_value(&value, &mut agents, &reading)
                    .is_err()
            );
            assert!(restored.agenda.is_empty());
            assert_eq!(lab.memory_value(&agents), original);
        }
    }

    #[test]
    fn eviction_order_survives_restore_and_corpus_edit_invalidates_snapshot() {
        let (lab, mut agents, reading) = fixture();
        let checkpoint = lab.memory_value(&agents);
        let edited = ReadingBridge::inquiry_fixture(vec![vec!["different".into(), "text".into()]]);
        assert!(
            InquiryLab::default()
                .restore_value(&checkpoint, &mut agents, &edited)
                .is_err()
        );
        let mut restored = InquiryLab::default();
        let mut restarted = agents.clone();
        restored
            .restore_value(&checkpoint, &mut restarted, &reading)
            .unwrap();
        // Equal future evidence must evict equal old examples, including counts.
        for id in 0..MEMORY_LIMIT {
            let observation = Observation {
                source: id as u64,
                context: vec!["future".into()],
                outcome: "event".into(),
            };
            agents[0].learning.observe(&observation);
            restarted[0].learning.observe(&observation);
        }
        assert_eq!(
            agents[0].learning.observations().collect::<Vec<_>>(),
            restarted[0].learning.observations().collect::<Vec<_>>()
        );
        assert_eq!(
            agents[0].learning.predict(&[]).probabilities,
            restarted[0].learning.predict(&[]).probabilities
        );
    }
}

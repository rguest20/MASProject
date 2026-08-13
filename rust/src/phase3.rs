//! Per-run, per-agent sandbox services.
//!
//! This is the typed Rust counterpart to Python's `phase3` package.  It gives
//! agents a small private home, a shared world notebook, bounded mailbox
//! delivery, and a generation-local activity ledger.  Virtual paths are
//! always resolved beneath the run's `sandbox/` directory: neither an agent
//! program nor an accidental `..` can reach the workspace.

use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

use crate::model::{Agent, Rng};

pub const MAX_PROGRAM_LEN: usize = 12;
const MAX_MAILBOX: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramOp {
    Add,
    Subtract,
    SquareAdd,
    Damp,
    Invert,
}

impl ProgramOp {
    pub fn from_code(code: char) -> Option<Self> {
        match code {
            'w' => Some(Self::Add),
            'm' => Some(Self::Subtract),
            's' => Some(Self::SquareAdd),
            'd' => Some(Self::Damp),
            'x' => Some(Self::Invert),
            _ => None,
        }
    }

    pub fn code(self) -> char {
        match self {
            Self::Add => 'w',
            Self::Subtract => 'm',
            Self::SquareAdd => 's',
            Self::Damp => 'd',
            Self::Invert => 'x',
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProgramInstruction {
    pub op: ProgramOp,
    pub value: f64,
}

impl ProgramInstruction {
    pub fn new(op: ProgramOp, value: f64) -> Self {
        Self {
            op,
            value: safe(value),
        }
    }
}

/// Execute the Phase-3 arithmetic DSL without evaluating arbitrary code.
pub fn run_program(program: &[ProgramInstruction]) -> f64 {
    let mut accumulator = 0.0;
    for instruction in program.iter().take(MAX_PROGRAM_LEN) {
        let value = safe(instruction.value);
        accumulator = match instruction.op {
            ProgramOp::Add => accumulator + value,
            ProgramOp::Subtract => accumulator - value,
            ProgramOp::SquareAdd => accumulator + value * value * 0.001,
            ProgramOp::Damp => accumulator * 0.5,
            ProgramOp::Invert => -accumulator,
        };
        accumulator = safe(accumulator);
    }
    accumulator
}

pub fn mutate_program(program: &[ProgramInstruction], rng: &mut Rng) -> Vec<ProgramInstruction> {
    let mut result: Vec<_> = program
        .iter()
        .copied()
        .take(MAX_PROGRAM_LEN)
        .map(|instruction| {
            if rng.unit() < 0.20 {
                random_instruction(rng)
            } else {
                ProgramInstruction::new(instruction.op, instruction.value)
            }
        })
        .collect();
    if rng.unit() < 0.10 && result.len() < MAX_PROGRAM_LEN {
        result.push(random_instruction(rng));
    }
    if rng.unit() < 0.10 && result.len() > 1 {
        result.remove(rng.index(result.len()));
    }
    result
}

fn random_instruction(rng: &mut Rng) -> ProgramInstruction {
    let operation = match rng.index(5) {
        0 => ProgramOp::Add,
        1 => ProgramOp::Subtract,
        2 => ProgramOp::SquareAdd,
        3 => ProgramOp::Damp,
        _ => ProgramOp::Invert,
    };
    ProgramInstruction::new(operation, rng.unit() * 20.0 - 10.0)
}

pub fn safe(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            1_000_000.0
        } else {
            -1_000_000.0
        }
    } else {
        value.clamp(-1_000_000.0, 1_000_000.0)
    }
}

#[derive(Clone, Debug)]
pub struct PersistentFs {
    root: PathBuf,
}

impl PersistentFs {
    pub fn new(root: impl AsRef<Path>) -> io::Result<Self> {
        fs::create_dir_all(root.as_ref())?;
        Ok(Self {
            root: fs::canonicalize(root.as_ref())?,
        })
    }

    pub fn read_text(&self, path: &str) -> io::Result<String> {
        let path = self.resolve(path)?;
        match fs::read_to_string(path) {
            Ok(text) => Ok(text),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(String::new()),
            Err(error) => Err(error),
        }
    }

    pub fn write_text(&self, path: &str, text: impl AsRef<str>) -> io::Result<()> {
        let path = self.resolve(path)?;
        self.ensure_safe_parent(&path)?;
        fs::write(path, text.as_ref())
    }

    pub fn append_text(&self, path: &str, text: impl AsRef<str>) -> io::Result<()> {
        let path = self.resolve(path)?;
        self.ensure_safe_parent(&path)?;
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(text.as_ref().as_bytes())
    }

    pub fn list_paths(&self, prefix: &str) -> io::Result<Vec<String>> {
        let normalised = normalise_virtual_path(prefix)?;
        let mut paths = Vec::new();
        collect_paths(&self.root, &self.root, &normalised, &mut paths)?;
        paths.sort();
        Ok(paths)
    }

    fn resolve(&self, virtual_path: &str) -> io::Result<PathBuf> {
        let relative = normalise_virtual_path(virtual_path)?;
        Ok(self.root.join(relative))
    }

    fn ensure_safe_parent(&self, destination: &Path) -> io::Result<()> {
        let parent = destination.parent().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "sandbox destination has no parent",
            )
        })?;
        fs::create_dir_all(parent)?;
        let canonical_parent = fs::canonicalize(parent)?;
        if !canonical_parent.starts_with(&self.root) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "sandbox path escaped its root",
            ));
        }
        Ok(())
    }
}

fn normalise_virtual_path(path: &str) -> io::Result<PathBuf> {
    let mut output = PathBuf::new();
    for component in Path::new(path).components() {
        match component {
            Component::Normal(part) => output.push(part),
            Component::CurDir | Component::RootDir => {}
            Component::ParentDir | Component::Prefix(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "sandbox path cannot contain parent traversal",
                ));
            }
        }
    }
    Ok(output)
}

fn collect_paths(
    root: &Path,
    current: &Path,
    prefix: &Path,
    results: &mut Vec<String>,
) -> io::Result<()> {
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.file_type()?;
        if metadata.is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            collect_paths(root, &path, prefix, results)?;
        } else if metadata.is_file() {
            let relative = path.strip_prefix(root).expect("walk starts under root");
            if prefix.as_os_str().is_empty() || relative.starts_with(prefix) {
                results.push(format!("/{}", relative.display()));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct CommBus {
    inboxes: BTreeMap<usize, VecDeque<Message>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub from: usize,
    pub payload: BTreeMap<String, String>,
}

impl CommBus {
    pub fn send(&mut self, recipient: usize, message: Message) {
        let inbox = self.inboxes.entry(recipient).or_default();
        if inbox.len() >= MAX_MAILBOX {
            inbox.pop_front();
        }
        inbox.push_back(message);
    }

    pub fn recv_all(&mut self, agent_id: usize) -> Vec<Message> {
        self.inboxes
            .remove(&agent_id)
            .map(VecDeque::into_iter)
            .map(Iterator::collect)
            .unwrap_or_default()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ActivityStats {
    pub actions: usize,
    pub reads: usize,
    pub writes: usize,
    pub messages: usize,
    pub energy_spent: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivityKind {
    Read,
    Write,
    Message,
}

#[derive(Clone, Debug, Default)]
pub struct RewardLedger {
    stats: BTreeMap<usize, ActivityStats>,
}

impl RewardLedger {
    pub fn note(&mut self, agent_id: usize, kind: ActivityKind, energy_cost: f64) {
        let stats = self.stats.entry(agent_id).or_default();
        stats.actions += 1;
        match kind {
            ActivityKind::Read => stats.reads += 1,
            ActivityKind::Write => stats.writes += 1,
            ActivityKind::Message => stats.messages += 1,
        }
        stats.energy_spent += energy_cost.max(0.0);
    }

    pub fn stats_for(&self, agent_id: usize) -> ActivityStats {
        self.stats.get(&agent_id).copied().unwrap_or_default()
    }

    pub fn reset_generation(&mut self) {
        self.stats.clear();
    }

    pub fn total_actions(&self) -> usize {
        self.stats.values().map(|stats| stats.actions).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    World,
    Home,
}

pub struct AgentApi<'a> {
    agent: &'a mut Agent,
    world: &'a PersistentFs,
    home: &'a PersistentFs,
    bus: &'a mut CommBus,
    ledger: &'a mut RewardLedger,
}

impl AgentApi<'_> {
    pub fn list_paths(&mut self, prefix: &str, scope: Scope) -> io::Result<Vec<String>> {
        if !self.can_act() {
            return Ok(Vec::new());
        }
        self.charge(0.10, ActivityKind::Read);
        self.filesystem(scope).list_paths(prefix)
    }

    pub fn read_text(&mut self, path: &str, scope: Scope) -> io::Result<String> {
        if !self.can_act() {
            return Ok(String::new());
        }
        self.charge(0.10, ActivityKind::Read);
        self.filesystem(scope).read_text(path)
    }

    pub fn write_text(&mut self, path: &str, text: &str, scope: Scope) -> io::Result<()> {
        if !self.can_act() {
            return Ok(());
        }
        self.charge(0.20, ActivityKind::Write);
        self.filesystem(scope).write_text(path, text)
    }

    pub fn append_text(&mut self, path: &str, text: &str, scope: Scope) -> io::Result<()> {
        if !self.can_act() {
            return Ok(());
        }
        self.charge(0.20, ActivityKind::Write);
        self.filesystem(scope).append_text(path, text)
    }

    pub fn send(&mut self, recipient: usize, payload: BTreeMap<String, String>) {
        if !self.can_act() {
            return;
        }
        self.charge(0.10, ActivityKind::Message);
        self.bus.send(
            recipient,
            Message {
                from: self.agent.id,
                payload,
            },
        );
    }

    pub fn recv_all(&mut self) -> Vec<Message> {
        self.bus.recv_all(self.agent.id)
    }

    fn filesystem(&self, scope: Scope) -> &PersistentFs {
        match scope {
            Scope::World => self.world,
            Scope::Home => self.home,
        }
    }

    fn can_act(&self) -> bool {
        self.agent.energy > 0.0
    }

    fn charge(&mut self, base_cost: f64, kind: ActivityKind) {
        let enthusiasm =
            0.5 * self.agent.traits.novelty_weight + 0.5 * self.agent.traits.cooperation_weight;
        let cost = base_cost * (1.0 - 0.5 * enthusiasm).clamp(0.5, 1.0);
        self.agent.energy = (self.agent.energy - cost).max(0.0);
        self.ledger.note(self.agent.id, kind, cost);
    }
}

#[derive(Clone, Debug)]
pub struct SandboxSpec {
    pub root: PathBuf,
}

#[derive(Debug)]
pub struct Phase3Runtime {
    world: PersistentFs,
    homes: BTreeMap<usize, PersistentFs>,
    bus: CommBus,
    ledger: RewardLedger,
}

impl Phase3Runtime {
    pub fn build(agents: &[Agent], spec: SandboxSpec) -> io::Result<Self> {
        let world = PersistentFs::new(spec.root.join("world"))?;
        world.write_text("/notes.txt", "Shared notes begin here:\n")?;
        world.write_text("/dictionary.json", "{\"hello\":\"greeting\"}\n")?;
        world.write_text("/help_required.txt", "")?;
        world.write_text("/help_responses.txt", "")?;
        let mut homes = BTreeMap::new();
        for agent in agents {
            let home = PersistentFs::new(spec.root.join(format!("agent_{}", agent.id)))?;
            home.write_text("/log.txt", format!("agent {} home\n", agent.id))?;
            home.write_text("/scratch.txt", "")?;
            home.write_text(
                "/hints.txt",
                "Try reading /notes.txt (world)\nTry reading /dictionary.json (world)\nTry writing to /notes.txt (world)\nTry writing to /scratch.txt (home)\n",
            )?;
            homes.insert(agent.id, home);
        }
        Ok(Self {
            world,
            homes,
            bus: CommBus::default(),
            ledger: RewardLedger::default(),
        })
    }

    pub fn reset_generation(&mut self) {
        self.ledger.reset_generation();
    }

    pub fn api_for<'a>(&'a mut self, agent: &'a mut Agent) -> Option<AgentApi<'a>> {
        let Self {
            world,
            homes,
            bus,
            ledger,
        } = self;
        let home = homes.get(&agent.id)?;
        Some(AgentApi {
            agent,
            world,
            home,
            bus,
            ledger,
        })
    }

    pub fn stats_for(&self, agent_id: usize) -> ActivityStats {
        self.ledger.stats_for(agent_id)
    }

    pub fn total_actions(&self) -> usize {
        self.ledger.total_actions()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActivityKind, CommBus, Message, PersistentFs, ProgramInstruction, ProgramOp, RewardLedger,
        run_program,
    };
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_directory(name: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("mas-phase3-{name}-{nonce}"))
    }

    #[test]
    fn persistent_fs_rejects_parent_escape_and_keeps_paths_scoped() {
        let root = temporary_directory("paths");
        let sandbox = PersistentFs::new(&root).unwrap();
        sandbox.write_text("/notes/idea.txt", "safe").unwrap();
        assert_eq!(sandbox.read_text("/notes/idea.txt").unwrap(), "safe");
        assert!(sandbox.write_text("../outside.txt", "no").is_err());
        assert_eq!(
            sandbox.list_paths("/notes").unwrap(),
            vec!["/notes/idea.txt"]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mailbox_delivers_in_order_and_ledger_resets() {
        let mut bus = CommBus::default();
        bus.send(
            2,
            Message {
                from: 1,
                payload: BTreeMap::from([("word".to_string(), "belmuk".to_string())]),
            },
        );
        assert_eq!(bus.recv_all(2)[0].from, 1);
        assert!(bus.recv_all(2).is_empty());
        let mut ledger = RewardLedger::default();
        ledger.note(1, ActivityKind::Read, 0.1);
        assert_eq!(ledger.stats_for(1).reads, 1);
        ledger.reset_generation();
        assert_eq!(ledger.stats_for(1).actions, 0);
    }

    #[test]
    fn typed_program_dsl_is_bounded_and_deterministic() {
        let program = [
            ProgramInstruction::new(ProgramOp::Add, 8.0),
            ProgramInstruction::new(ProgramOp::SquareAdd, 10.0),
            ProgramInstruction::new(ProgramOp::Damp, 0.0),
            ProgramInstruction::new(ProgramOp::Invert, 0.0),
        ];
        assert!((run_program(&program) + 4.05).abs() < f64::EPSILON);
        assert_eq!(ProgramOp::from_code('w').unwrap().code(), 'w');
        assert!(ProgramOp::from_code('?').is_none());
    }
}

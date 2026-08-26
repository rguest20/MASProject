use std::path::PathBuf;

pub const POPULATION: usize = 30;
pub const DEFAULT_GENERATIONS: usize = 30;

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub generations: usize,
    pub watch: bool,
    pub delay_ms: u64,
    pub converse_path: PathBuf,
    pub seed: Option<u64>,
    pub dimensions: Option<usize>,
    pub community_memory_path: Option<PathBuf>,
    pub use_community_memory: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            generations: DEFAULT_GENERATIONS,
            watch: false,
            delay_ms: 250,
            // Commands in the root README run Cargo from the workspace root.
            converse_path: PathBuf::from("converse.txt"),
            seed: None,
            dimensions: None,
            community_memory_path: None,
            use_community_memory: true,
        }
    }
}

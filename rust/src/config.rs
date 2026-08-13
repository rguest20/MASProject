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
        }
    }
}

mod alignment;
mod cognition;
mod config;
mod conversation;
mod dictionary;
mod evolution;
mod language;
mod lexicon;
mod model;
mod numeric;
mod phase3;
mod reading;
mod semantics;
mod simulation;
mod tasks;

use std::env;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use config::RunOptions;
use simulation::Coordinator;

fn print_usage() {
    println!(
        "Usage: cargo run -- [--generations N] [--watch] [--delay SECONDS] [--converse PATH] [--seed N]"
    );
}

fn parse_options() -> Result<RunOptions, String> {
    let mut options = RunOptions::default();
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--generations" => {
                options.generations = args
                    .next()
                    .ok_or("--generations needs a value")?
                    .parse()
                    .map_err(|_| "--generations must be a non-negative integer")?;
            }
            "--watch" => options.watch = true,
            "--delay" => {
                options.delay_ms = (args
                    .next()
                    .ok_or("--delay needs a value")?
                    .parse::<f64>()
                    .map_err(|_| "--delay must be a number")?
                    .max(0.0)
                    * 1000.0) as u64;
            }
            "--converse" => {
                options.converse_path = PathBuf::from(args.next().ok_or("--converse needs a path")?)
            }
            "--seed" => {
                options.seed = Some(
                    args.next()
                        .ok_or("--seed needs a value")?
                        .parse()
                        .map_err(|_| "--seed must be an unsigned integer")?,
                );
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            unknown => return Err(format!("Unknown option: {unknown}")),
        }
    }
    Ok(options)
}

fn main() {
    let options = match parse_options() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            print_usage();
            std::process::exit(2);
        }
    };
    let mut coordinator = match Coordinator::new(options.clone()) {
        Ok(coordinator) => coordinator,
        Err(error) => {
            eprintln!("Could not create Rust simulation: {error}");
            std::process::exit(1);
        }
    };
    println!("Run directory: {}", coordinator.run_dir.display());
    println!("Seed: {}", coordinator.seed);
    println!("Conversation: {}", options.converse_path.display());

    if options.watch {
        println!("Watch mode is running. Edit converse.txt, then stop with Ctrl-C.");
        loop {
            if let Err(error) = coordinator.run_generation() {
                eprintln!("Generation failed: {error}");
                break;
            }
            if options.delay_ms > 0 {
                thread::sleep(Duration::from_millis(options.delay_ms));
            }
        }
    } else {
        for _ in 0..options.generations {
            if let Err(error) = coordinator.run_generation() {
                eprintln!("Generation failed: {error}");
                std::process::exit(1);
            }
        }
    }

    println!("Artifacts:");
    println!("  report: {}", coordinator.report_path.display());
    println!("  metrics: {}", coordinator.metrics_path.display());
    println!("  dialogues: {}", coordinator.dialogue_path.display());
    println!("  reading: {}", coordinator.reading_path.display());
    println!("  summary: {}", coordinator.summary_path.display());
}

from evolution.coordinator import Coordinator
from config import GENERATIONS


def main():
    coord = Coordinator()
    print(f"Run directory: {coord.run_dir}")
    print(f"Seed: {coord.random_seed}")

    for _ in range(GENERATIONS):
        coord.run_generation()

    print("Artifacts:")
    print(f"  report: {coord.report_path}")
    print(f"  metrics: {coord.csv_path}")
    print(f"  dialogues: {coord.dialogue_log_path}")
    print(f"  metadata: {coord.run_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

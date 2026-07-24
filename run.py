import argparse
import time

from evolution.coordinator import Coordinator
from config import GENERATIONS


def main():
    parser = argparse.ArgumentParser(description="Run the emergent-language simulation.")
    parser.add_argument("--generations", type=int, default=GENERATIONS, help="Generations to run outside watch mode.")
    parser.add_argument("--watch", action="store_true", help="Run continuously and poll converse.txt after each generation.")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between generations in watch mode.")
    parser.add_argument("--converse", default="converse.txt", help="Path to the Ryan/Community transcript.")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible random seed.")
    args = parser.parse_args()

    coord = Coordinator(seed=args.seed, conversation_path=args.converse)
    print(f"Run directory: {coord.run_dir}")
    print(f"Seed: {coord.random_seed}")
    print(f"Conversation: {coord.conversation_path}")

    try:
        if args.watch:
            print("Watch mode is running. Edit converse.txt, then stop with Ctrl-C.")
            while True:
                coord.run_generation()
                if args.delay > 0:
                    time.sleep(args.delay)
        else:
            for _ in range(max(0, args.generations)):
                coord.run_generation()
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("Artifacts:")
    print(f"  report: {coord.report_path}")
    print(f"  metrics: {coord.csv_path}")
    print(f"  dialogues: {coord.dialogue_log_path}")
    print(f"  metadata: {coord.run_dir / 'metadata.json'}")
    print(f"  conversation: {coord.conversation_path}")


if __name__ == "__main__":
    main()

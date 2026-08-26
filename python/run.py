import argparse
import os
from pathlib import Path
import sys
import time


def _configure_semantic_dimensions(argv):
    """Configure dimensions before importing agent modules that bind DIMS."""
    raw_value = None
    for index, argument in enumerate(argv):
        if argument == "--dimensions" and index + 1 < len(argv):
            raw_value = argv[index + 1]
            break
        if argument.startswith("--dimensions="):
            raw_value = argument.split("=", 1)[1]
            break
    if raw_value is None:
        return
    try:
        dimensions = int(raw_value)
    except ValueError:
        return
    if not 8 <= dimensions <= 256:
        raise SystemExit("--dimensions must be between 8 and 256")
    os.environ["MAS_SEMANTIC_DIMS"] = str(dimensions)


_configure_semantic_dimensions(sys.argv[1:])

from evolution.coordinator import Coordinator
from config import DIMS, GENERATIONS


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def main():
    parser = argparse.ArgumentParser(description="Run the emergent-language simulation.")
    parser.add_argument("--generations", type=int, default=GENERATIONS, help="Generations to run outside watch mode.")
    parser.add_argument("--watch", action="store_true", help="Run continuously and poll converse.txt after each generation.")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between generations in watch mode.")
    parser.add_argument(
        "--converse",
        type=Path,
        default=WORKSPACE_ROOT / "converse.txt",
        help="Path to the Ryan/Community transcript (defaults to the workspace converse.txt).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible random seed.")
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DIMS,
        choices=range(8, 257),
        metavar="N",
        help="Semantic-vector dimensions (8–256; default: 32).",
    )
    parser.add_argument(
        "--community-memory",
        type=Path,
        default=WORKSPACE_ROOT / "community_memory" / f"python-d{DIMS}.json",
        help="Persistent public community memory for this dimensionality.",
    )
    parser.add_argument(
        "--fresh-community",
        action="store_true",
        help="Start without loading or updating persistent community memory.",
    )
    args = parser.parse_args()

    coord = Coordinator(
        seed=args.seed,
        conversation_path=args.converse,
        community_memory_path=args.community_memory,
        use_community_memory=not args.fresh_community,
    )
    print(f"Run directory: {coord.run_dir}")
    print(f"Seed: {coord.random_seed}")
    print(f"Conversation: {coord.conversation_path}")
    print(f"Semantic dimensions: {DIMS}")
    if coord.community_memory is not None:
        print(
            f"Community memory: {coord.community_memory_path} "
            f"(loaded={coord.community_memory_status['loaded']}; "
            f"tokens={coord.community_memory_status['tokens']})"
        )

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

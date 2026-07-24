import argparse
from pathlib import Path

import pandas as pd


def _latest_log():
    logs = sorted(Path("runs").glob("*/cultural_log.csv"))
    if not logs:
        raise FileNotFoundError("No run logs found. Pass the path to cultural_log.csv explicitly.")
    return logs[-1]


def main():
    parser = argparse.ArgumentParser(description="Summarise a simulation cultural log.")
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        help="Path to cultural_log.csv (defaults to the newest run).",
    )
    args = parser.parse_args()
    log_path = args.log or _latest_log()
    df = pd.read_csv(log_path)

    print(df.describe())
    html_path = log_path.with_name("cultural_log_summary.html")
    df.to_html(html_path)
    print(f"HTML summary: {html_path}")


if __name__ == "__main__":
    main()

"""Render a co-occurrence heatmap for the archived dialogue sample."""

from pathlib import Path
import re
from itertools import combinations


def load_data():
    data_dir = Path(__file__).with_name("analysis_data")
    return "".join(
        path.read_text(encoding="utf-8")
        for path in sorted(data_dir.glob("*.txt"))
    )


def build_cooccurrence(data, markers):
    import pandas as pd

    counts = {marker: {other: 0 for other in markers} for marker in markers}
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, markers)) + r")\b")
    for line in data.splitlines():
        tokens = set(pattern.findall(line))
        for marker in tokens:
            counts[marker][marker] += 1
        for left, right in combinations(tokens, 2):
            counts[left][right] += 1
            counts[right][left] += 1
    return pd.DataFrame(counts).T.astype(int)


def main():
    import matplotlib.pyplot as plt
    import seaborn as sns

    markers = ["lo", "rin", "tar", "muk", "ka", "zev"]
    frame = build_cooccurrence(load_data(), markers)
    print(frame)
    plt.figure(figsize=(6, 5))
    sns.heatmap(frame, annot=True, fmt="d", cmap="YlGnBu", cbar=False, linewidths=0.5, square=True)
    plt.title("Co-occurrence Heatmap of Core Tokens")
    plt.xlabel("Co-occurring Token")
    plt.ylabel("Primary Token")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

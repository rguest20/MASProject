# Rust implementation

This is a native Rust counterpart to the Python MAS simulation. Its core is
standard-library Rust, with `ndarray` (Rayon-enabled) for dense semantic
vectors—the part of the workload where Python object overhead becomes most
costly. Cargo fetches that small dependency set on the first build.

It currently reproduces the first parity tranche of the project’s observable
experimentation loop:

- a population of agents with private, collision-safe numeral, referent, and
  action signals;
- evidence-gated shared numeric, referent, action, base, and grammar
  conventions (no pre-seeded public language);
- repeated grounded numeric, referential, action, and compositional-action
  tasks, including correction and rehearsal pressure;
- maturity-gated semantic similarity debates and definition exchanges, with
  peer consensus, gentle correction, and replayed disagreement trials;
- rotating maturity-gated narrative, prediction, proto-role, reconstruction,
  property, compatibility, and misunderstanding-repair practices;
- per-agent tunable `ndarray` semantic vectors (32 dimensions by default), bounded weighted
  association graphs, numerical-token isolation, decay, gravity, and
  anti-monopoly stabilisation, plus confidence-gated semantic families;
- an EM-style community semantic map with coverage/dispersion confidence and
  capped, reward-only alignment-repair tasks for genuinely misaligned vectors;
- typed affect, needs, action selection, trust/social memory, semantic
  teaching, and gentle one-agent-per-generation tri-parent evolution with
  blended semantic vectors/links/family seeds and language preferences,
  stable lineage IDs, bounded descendant-success feedback, and safe program
  mutation;
- agent-owned private word invention, preferences, utterance feedback, peer
  address forms, and pragmatic practice/teaching/request/repair/assertion
  dialogue acts, including grounded request→answer and failed-turn repair
  exchanges; public conventions remain the only grounded shared channel;
- `converse.txt` polling, controlled number/referent queries, simple
  subject-centred assertions and retrieval, feedback-sensitive replies,
  topic-safe working memory, and opaque learned interaction modes;
- lazy, bounded use of `python/filtered.json`: dictionary material becomes
  low-confidence agent semantic links, never automatic world facts or reply
  templates;
- quiet-generation reading, confidence-gated story vocabulary, and a
  content-topic relevance safeguard;
- human-observed English bigrams with repetition/feedback promotion, weak
  dictionary phrase scaffolds, context-gated reading continuations, and
  community-voted multiword reply proposals; human wording is observed by
  every agent before that vote;
- a typed Phase-3 sandbox: private agent homes plus a shared world notebook,
  path-traversal protection, bounded mailbox delivery, small energy costs, a
  per-generation activity ledger, and a safe arithmetic program DSL (never
  host-code execution);
- isolated `rust/runs/` artifacts: an append-only generation report, an
  analysis-ready CSV, dialogue and reading logs, a concise latest-state
  `run_summary.txt`, metadata, and a run-local `sandbox/` tree for inspection.

Run it from the workspace root:

```bash
cargo run --manifest-path rust/Cargo.toml -- --generations 30 --seed 12345
cargo run --manifest-path rust/Cargo.toml -- --watch --converse converse.txt
```

The Rust runner persists its public community centroids and evidence-backed
lexicon to `community_memory/rust-d32.json` after each completed generation. A
separate memory file is selected for each semantic dimensionality. New
runs have fresh agents but seed confident public concepts as a gentle prior.
Use `--community-memory PATH` to select a file or `--fresh-community` for an
isolated run that neither loads nor writes community memory.

Semantic dimensionality is tunable with `--dimensions N` (8–256, default 32).
Try 64 first for larger experiments; higher dimensions cost proportionally more
vector work and need more evidence before meaningful geometry emerges.

The Rust implementation is intentionally a clean reimplementation rather than
a line-for-line translation. Python remains the executable behavioural
reference while the following subsystems are still being ported:

- flavour channels and the remaining language-organ behaviour (richer role
  syntax and negotiation beyond the current request/answer and repair turns);
- detailed scoring and richer multi-turn forms for the task families;
- cross-implementation behavioural fixtures and any remaining report/analysis
  parity details.

The practical parity criterion is shared seeded behavioural contracts and
matching aggregate outcomes, rather than byte-identical trajectories: Python
and Rust use different random-number and floating-point implementations.

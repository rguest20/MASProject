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
- `converse.txt` polling with an executive turn router: arithmetic, numeric,
  referent, definition, and teaching turns use their dedicated capabilities
  before free conversation; concise retrieved/learned facts stop rather than
  trailing into an old topic;
- topic-safe working memory with parked frames. `New conversation` (including
  the common `New conversaion` typo), `start over`, `clear context`, or
  `change topic to games` clears only short-term conversational attractors;
  `stop` is a silent control act.
  These controls preserve world facts and the agents' long-term semantic maps;
- a bounded community semantic path: recent content topics become waypoints,
  each agent votes from its own vector geometry on nearby/directionally
  compatible next concepts. Predictions require at least two coherent
  waypoints and broad population agreement, then act only as weak free-chat
  continuations. An unlikely human-selected topic starts a new path;
- a human phrase needs repeated or positively reinforced evidence before it
  becomes productive reply syntax; negative feedback rejects its reply pairs
  and parks the current conversational frame;
- feedback-sensitive replies and opaque learned interaction modes;
- typed decimal literals in human text: values such as `1`, `2`, and `123`
  are bounded numeric records with decimal-digit structure, not ordinary word
  embeddings; once public digits and a base are agreed, larger values are
  rendered compositionally in the community's own numeral system;
- a learned structural-grammar layer: tokens repeatedly distributed across
  varied semantic families can become operators, are removed from ordinary
  vector/family attraction, and accumulate directed family-to-family edges;
- explorer curiosity: the least-connected, repeatedly-read story vocabulary
  is first linked through verified dictionary/community evidence, then (only if that
  fails twice) retained in a bounded unresolved-concept backlog and placed in
  the separate one-slot `question.txt` mailbox for Ryan;
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

`question.txt` lives alongside the chosen `converse.txt`. When an explorer
cannot attach a well-read word to anything it already knows, it may write:

```text
Community: What is willow?
Ryan:
```

Write your teaching response after `Ryan:` and save. The answer is absorbed as
semantic evidence after a short settling window; it does not generate a canned
chat reply. The mailbox accepts no further community question until that answer
has been processed.

The Rust runner persists public community centroids, confidence-gated
conceptual relationships, and the evidence-backed lexicon to
`community_memory/rust-d32.json` after each completed generation. A public
relationship is an undirected, mature positive association supported across
agents; memory is bounded to 2,048 relationships and 12 per concept. New
runs have fresh agents but seed confident public concepts and weak versions of
their strongest relationships as a gentle prior. Use `--community-memory PATH`
to select a file or `--fresh-community` for an isolated run that neither loads
nor writes community memory.

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

## Non-language capability learning

Every run now includes a bounded experimental world with five visible state
bits and five actions: take a key, unlock a door, activate power, extend a
bridge, and retrieve an object. Agents start without action rules. They learn
additive effects and prerequisite conjunctions from observed transitions,
including failures. A supplied breadth-first planner combines those learned
rules into action sequences. This learns how to use supplied actions; it does
not invent new primitive actions or learn the planning algorithm itself.

Three sampled agents train per generation, with at most 12 actions per episode.
The curriculum introduces key/door goals in generations 1–20, power/bridge in
21–40, and combined retrieval from 41 onward. Successful episodes add a small,
bounded fitness reward. Offspring inherit the selected parent's experience;
capability knowledge is currently run-local, separate from public semantic
memory. The experiment uses its own seeded RNG.

Open `capability_monitor.html` inside the printed run directory in a browser.
It refreshes every five seconds and compares individual knowledge, pooled
community evidence, and an untrained planner. The pooled result is an upper
bound on sharing, not a learned communication policy. The untrained planner
has no rules and abstains; it is not a random-action baseline.

- `capability_metrics.csv`: fixed test success, steps, wasted actions, prediction
  coverage and correct predictions, measured at generation 1 and every 5 thereafter.
- `capability_agents.csv`: per-lineage knowledge and evaluation history, allowing
  individual learning to be distinguished from population turnover/inheritance.
- `capability_episodes.jsonl`: training attempts with before/after states and
  predictions made before learning the outcome.
- `run_summary.txt`: latest capability evaluation and monitor location.

Evaluation never teaches or rewards agents. Test episodes begin in four fixed
alternative states; training always starts empty. Those test starts can occur
as intermediate training states, so this measures recombination and transfer
of learned rules, not wholly unseen worlds. All 160 state/action transitions
are also checked for prediction accuracy; the dashboard counts unknown
predictions as incorrect. Earlier goals remain in the evaluation after the
curriculum advances to expose forgetting. Population curves combine learning,
selection and inheritance; they are not isolated causal estimates of learning.

For an isolated 100-generation experiment, use a dedicated mailbox directory:

```bash
mkdir -p /tmp/mas-capability-demo
cargo run --manifest-path rust/Cargo.toml -- --generations 100 --seed 12345 --fresh-community --converse /tmp/mas-capability-demo/converse.txt
```

A 30-generation default run does not reach the final curriculum stage. Use at
least 60 generations to observe combined retrieval, or `--watch` for live use.

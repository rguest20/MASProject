# Rust implementation

The current community inquiry loop, monitor, evaluation limits, and next
architectural steps are documented in [COMMUNITY_MIND_REVIEW.md](COMMUNITY_MIND_REVIEW.md).
It replaces the fixed five-fact inquiry routine with source-grounded peer
investigations and a shared agenda. New runs print the inquiry monitor path.
Inquiry checkpoints now preserve each learner's sourced experience and active
questions across runs. The monitor reports a transfer test taken before any new
training, including the change from the saved score. See the review for scope,
checkpoint timing, and the isolated `check_inquiry_restart.py` test.

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
relationship is an undirected, mature signed association supported across
agents; memory is bounded to 2,048 relationships and 12 per concept. New
runs have fresh agents but seed confident public concepts and weak versions of
their strongest relationships as a gentle prior. Use `--community-memory PATH`
to select a file or `--fresh-community` for an isolated run that neither loads
nor writes community memory.

General semantic links accumulate signed evidence: positive weights support an
association, negative weights oppose it. Signs survive inheritance, community
aggregation, saving and seeding new agents. Conflicting evidence cancels rather
than becoming stronger consensus. Negative links do not repel concept vectors
and do not form positive semantic families. `SemanticStore::association` exposes
the net weight; it is neither a calibrated correlation nor a logical “never”.
Co-occurrence remains positive evidence; callers must supply explicit negative
feedback (as dictionary antonym links already do). This does not interpret every
sentence containing “not”, or turn a context-specific machine failure into a
universal conceptual prohibition.

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

Every run starts with five actions: take a key, unlock a door, activate power,
extend a bridge, and retrieve an object. Agents learn additive effects and
prerequisites from successes, failures, and one sampled intervention per
training episode. A supplied dependency planner combines learned rules into
plans without enumerating all possible states. No semantic-map settings are
changed: capability knowledge is separate from linguistic associations.

Difficulty now advances automatically. Both the population-average individual
score and pooled score must reach **90% on every milestone** for three
consecutive evaluations, with at least 20 generations at the current level.
Initial key/door training lasts to generation 20, bridge training to generation
40, and full-goal training follows. Evaluations run every five generations.

Each new level normally adds an independent preparation action and an assembly
action requiring both that preparation and the previous goal (plus a sampled
earlier prerequisite). This grows the number of skills and the length of plans.
Prerequisites only point backwards, so generated problems remain solvable.
Earlier action rules remain unchanged; this tests acquisition and composition
of new skills, not adaptation to contradictory rules. Primitive actions and
the planning algorithm are supplied, not invented by the agents.

The default resource ceiling is **21 actions** (eight expansions after the
initial world). Use `--capability-limit N` to select 5–31 actions. At an even
limit, the final expansion can add just one assembly action. Once the ceiling
is reached the monitor says so, and training continues without further
expansion. This is bounded progressive complexity, not an infinite curriculum.

Open the printed `capability_monitor.html` path in a browser. It refreshes every
five seconds, marks level changes on the chart, and shows current-goal success,
retention of earlier milestones, wasted actions, and prediction accuracy.
Scores on different levels are not directly comparable: the goal is harder.

- `capability_levels.csv`: current and weakest-milestone success, mastery streak,
  action count, time on level, and ceiling status. `before_training` rows record
  the immediate performance drop when a level opens.
- `capability_events.jsonl`: level introductions with environment prerequisites,
  and mastery timestamps for measuring generations to mastery.
- `capability_metrics.csv`: per-goal test success, steps, wasted actions and
  prediction coverage/accuracy, identified by level and evaluation phase.
- `capability_agents.csv`: per-lineage knowledge and current-goal scores.
- `capability_retention.csv`: every agent/milestone score, action evidence count,
  and whether the inferred rule is currently known. `capability_events.jsonl`
  emits `retention_drop` records when retained skills fall below 90%.
- `capability_episodes.jsonl`: bounded attempts and training interventions with
  predictions recorded before the resulting observations are learned.
- `run_summary.txt`: latest capability summary and monitor location.

Each goal has 12 deterministic test cases, including empty and partial supply
states. Earlier milestones retain the same test starts when levels change.
Evaluation never teaches or rewards agents; its RNG does not affect training.
Training interventions may overlap test states, so these are diagnostic tests
of recombination, not a guarantee of completely unseen experience. Prediction
checks use 32 fixed sampled states per current action (not exhaustive 2^N
states). Unknown predictions count as incorrect on the monitor.

Three sampled agents train per generation, at most three times the available
action count per episode, plus one intervention. Successful episodes get a
small bounded fitness reward. Per-agent memory retains at most 64 distinct
transitions per action. Offspring inherit a selected parent's experience.
Knowledge and difficulty progression remain **run-local**; restarting creates
a new curriculum. Public semantic memory does not restore capability learning.

Peer teaching is enabled by default. After individual practice, up to three
learners request one peer-witnessed transition each. A supplied selection
policy favours disputed outcomes or unknown actions; it scans at most eight
recent examples per action per potential teacher. Self-teaching, already held
examples, and examples the learner already predicts correctly are skipped.
The learner replays the proposed action in the sandbox and adopts the observed
example only if it matches the testimony. Peers exchange examples, not entire
models or evaluator answers. A verified lesson can still hurt generalisation;
those regressions are counted rather than rolled back using test results.

The monitor's **Peer teaching** panel reports requests, verified examples,
improvements/regressions, and the mean immediate change in milestone test
success over the latest 100 verified lessons. Each exchange is compared with
one random practice action on a copy of the exact same pre-lesson learner.
Both alternatives cost one sandbox action. The random-practice copy is then
discarded. These paired diagnostics do not select lessons, update fitness,
choose curriculum gates, or enter agent memory. They measure local effects,
not long-term causal benefit; individual and pooled evaluation after teaching
still controls normal curriculum advancement.

- `capability_teaching.csv`: sender/receiver lineages, action, verification,
  before/after/random-practice milestone results and current-goal rates.
- `capability_teaching.jsonl`: the actual examples, pre-lesson predictions,
  replay results, comparison interventions, and unanswered requests.
- `capability_teaching_totals.csv`: per-generation requests, verifications,
  improvements and regressions (including zero-delivery generations).

Use `--no-capability-teaching` for a run without exchanges. Teaching has its own
seeded RNG. Enabled/disabled runs can still diverge through selection and
curriculum timing, and disabled runs have fewer real practice actions; use the
paired single-action diagnostic for a comparison with equal practice cost.

Pooled evidence remains a bounded aggregate comparison; the blue individual
curve now includes actual peer learning. The teaching policy itself is supplied,
not evolved or learned.

The run also writes `capability_scaffold.jsonl`. This semantic-neighbourhood
layer connects prerequisite bits to actions using repeated community evidence.
It only biases agents toward uncertain boundary experiments; it contains no
ordered solution paths. Agents still verify every suggested connection in the
world, and the scaffold is rebuilt from current agents each generation.
The untrained planner has no rules and abstains, rather than acting randomly.
Population scores combine learning, selection and inheritance; they do not
mean that every individual is at 90% or isolate learning's causal contribution.

For live use:

```bash
cargo run --release --manifest-path rust/Cargo.toml -- --watch --converse converse.txt --capability-limit 21
```

For an isolated experiment with enough time to see multiple levels:

```bash
mkdir -p /tmp/mas-capability-demo
cargo run --release --manifest-path rust/Cargo.toml -- --generations 1000 --seed 12345 --fresh-community --converse /tmp/mas-capability-demo/converse.txt
```

Existing processes must be restarted to use the new code. The new reports use
level/current-goal columns rather than the earlier stage/retrieval-only schema;
old run artifacts remain unchanged.


The machine experiment now evaluates every agent and starts episodes at random
non-terminal positions, so completion cannot be achieved by always starting at
the beginning. `machine_links.html` reports the observed previous-action to
next-choice distribution with raw counts.

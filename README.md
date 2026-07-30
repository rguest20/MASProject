# Agent Sandbox Project

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Running a Conversation](#running-a-conversation)
- [Reading, Dictionary, and Learning Pressure](#reading-dictionary-and-learning-pressure)
- [Agent Criteria](#agent-criteria)
- [System Criteria](#system-criteria)
- [Risks & Failure Modes](#risks--failure-modes)
- [Architecture](#architecture)
  - [1. Agent Core](#1-agent-core)
  - [2. Communication System](#2-communication-system)
  - [3. Environment Layout](#3-environment-layout)
  - [4. Decision-Making Loop](#4-decision-making-loop)
  - [5. Fitness / Alignment Model](#5-fitness--alignment-model)
  - [6. Mutation and Drift](#6-mutation-and-drift)
  - [7. Tracking & Logging](#7-tracking--logging)
  - [8. Code Structure](#8-code-structure)
- [Timeline](#timeline)
  - [Phase 1 — Foundations](#phase-1--foundations-completed--in-progress)
  - [Phase 2 — Increasing Cognitive Complexity](#phase-2--increasing-cognitive-complexity)
  - [Phase 3 — Conflict--cooperation](#phase-3--conflict--cooperation)
  - [Phase 4 — Abstract Reasoning](#phase-4--abstract-reasoning-experimental)
  - [Phase 5 — Meta-Behaviour](#phase-5--meta-behaviour)
  - [Phase 6 — Research Questions](#phase-6--research-questions)
- [Upcoming Features](#upcoming-features)
- [Observations So Far](#observations-so-far)

The Agent Sandbox Project is a research-oriented environment for studying emergent behaviour in multi-agent systems. Each agent develops its own internal semantic map, communicates using evolving token structures, and adapts over generations through evolutionary pressure. By observing how concepts, cooperation, subcultures, numeracy, and proto-language evolve, this project aims to uncover how complex collective cognition can arise from simple individual components.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run.py --generations 30
```

Each invocation creates a timestamped directory under `runs/`; it does not
overwrite a previous experiment. The command prints the report, CSV, dialogue
log, and metadata paths when it finishes.

Useful options:

```bash
# Keep running and poll a conversation file after every generation.
python3 run.py --watch --converse converse.txt

# Use a reproducible run.
python3 run.py --generations 100 --seed 12345
```

## Running a Conversation

With `--watch`, the community monitors `converse.txt`. Add a completed line in
this form and save the file:

```text
Ryan: Hello there
```

After a short settling period, the community appends its answer and a blank
`Ryan:` prompt. It only answers the most recent Ryan line that does not yet
have a Community response, so it is safe to keep a readable transcript in the
same file.

The bridge supports both controlled simulation requests and exploratory prose:

- `How many is 12?`, `Show r3`, and `Do a2 to r3 with 4` query the community's
  learned number, referent, and action conventions.
- `Teach r3 is apple` and `Teach a2 means move` add explicit aliases.
- Ordinary sentences are treated as human-language evidence, not executable
  commands. The community builds a small topic-centred working frame and tries
  to compose a response from observed word order, grounded facts, and local
  semantic evidence.
- Send `+` or `-` on a Ryan line to approve or reject the preceding community
  answer. Feedback reinforces or suppresses its phrase and n-gram evidence.

The conversation system distinguishes short-lived working memory from durable
semantic links. A change of explicit subject starts a fresh frame; returning to
the same subject can restore its parked frame. This prevents one definition
from simply spilling into the next. The current state is held by the running
coordinator: the transcript remains on disk, but a new process does not replay
the entire transcript into a new conversation memory.

## Reading, Dictionary, and Learning Pressure

`clean_merged_fairy_tales_without_eos.txt` or
`cleaned_merged_fairy_tales_without_eos.txt`, when present, provides paced
adjacent passages during quiet generations. Reading supplies private
co-occurrence and sentence-order evidence; it does not create world facts.

Words may move from reading into a small conversation bridge only after they
have appeared in several distinct contexts with sufficient local evidence.
Even then, they are eligible only when their reading context overlaps a
content-word topic in the live conversation. Grammatical glue such as `are` or
`the` cannot activate a story word. This is deliberately conservative: it is
intended to prevent a character or phrase from a book appearing as an
unprompted community belief.

If `filtered.json` is available, its meanings, category labels, synonyms, and
antonyms provide weak semantic scaffolding. Dictionary material is evidence,
not an oracle: polysemous entries can still be ambiguous, so user context and
feedback are important when teaching a word.

The coordinator also tracks a bounded community discomfort signal. Long quiet
stretches and repeated failed tasks increase it; novel reading, successful
internal practice, and useful learning reduce it. The signal changes the
cadence of reading and exploration rather than prescribing a specific answer.

## Overview
This project is a sandbox for exploring how emergent behaviour arises within a multi-agent system, both at the individual agent level and at the collective level.  
To maintain interpretability, the system uses a limited number of agents with constrained but extensible abilities.

The aim is to observe how structure, language-like patterns, cooperation, and conceptual mapping evolve through interaction, selection, and drift.

---

## Agent Criteria
Agents should be able to:

- Hold internal concept representations  
- Link concepts across semantic distance  
- Categorise concepts into semantic families  
- Use concepts to perform assigned tasks  
- Communicate concepts with other agents  
- Expand or drift concepts as their understanding evolves  

---

## System Criteria
The system should:

- Apply pressure to keep the community adaptive and receptive  
- Monitor and prevent runaway collapse (e.g., monoculture)  
- Store a community-level concept map if concepts stabilise  
- Allow evolution via Darwinian selection and crossover  
- Increase task difficulty as the population improves  

---

## Risks & Failure Modes

- Collapse into a semantic monoculture  
- Weak or unstable semantic maps preventing task success  
- Tasks set at inappropriate difficulty levels  
- Over-emergence of coherence (agents converging too quickly)  

Mitigation currently relies on continuous monitoring and resetting, as each run starts from generation zero.

---

# Architecture

## 1. Agent Core
Each agent maintains:

- A **32-dimensional semantic map** storing vector embeddings (“tokens”) for concepts, numbers, behaviours, and linguistic fragments  
- A set of **traits** defining personality biases (talkativeness, curiosity, precision, etc.)  
- A **local memory** tracking token reinforcement, decay, and drift  

The semantic map acts as the agent’s internal world-model, enabling it to:

- Compare and cluster concepts  
- Drift or mutate conceptual positions  
- Integrate new tokens during interaction  
- Prune or decay unused tokens  

This internal space forms the substrate for emergent behaviour.

---

## 2. Communication System

Agents communicate in three modes:

### (a) Direct Pairwise Chat
Two agents exchange messages.  
Used for:

- semantic alignment  
- probing mutual understanding  
- exchanging conceptual innovations  

### (b) Community Hub
A shared “room” where all agents may speak.  
Used for:

- group consensus formation  
- stabilisation of vocabulary  
- collective emergence  

### (c) Task Assistance Requests
Agents may request help when performing tasks.  
This encourages:

- cooperation  
- reinforcement of shared knowledge  
- diffusion of concepts across the population  

All communication is logged for later analysis.

### (d) Human Conversation Bridge

The `CommunityConversation` bridge exposes the community through a plain text
Ryan/Community transcript. It keeps human word evidence separate from the
agents' private invented vocabulary, maintains bounded working frames, and
uses a conservative shared fact graph for simple subject-centred retrieval.
It can decline to produce a one-word activation trace when it lacks enough
evidence for a small phrase.

---

## 3. Environment Layout

### Agent Folder
Contains:

- private logs  
- personal semantic map snapshots  
- internal traits and state  

### World Folder
Contains:

- shared documents or reference tokens  
- communal logs  
- task instructions  
- full message history  

This directory-based environment keeps the emergent system interpretable and debuggable.

### Run Artifacts

Each run writes to `runs/<timestamp>_seed-<seed>/`:

- `generation_report.txt` — readable per-generation summaries.
- `cultural_log.csv` — numeric metrics for plotting or comparison.
- `dialogue_log.txt` — agent dialogue records.
- `reading_log.txt` — passages, bridge promotions, and internal reading
  practice, when reading is active.
- `metadata.json` — seed and configuration snapshot.

---

## 4. Decision-Making Loop

At each turn, an agent:

1. **Infers the partner’s semantic position**  
   Using a nearest-neighbour approach.

2. **Attempts to expand or refine the conceptual area**  
   Probes conceptual boundaries to test the partner’s understanding.

3. **Reinforces or drifts concepts**  
   - If alignment → reinforce tokens  
   - If misalignment → drift or reinterpret  

4. **Creates new tokens**  
   When encountering genuinely new concepts.

5. **Prunes unused space**  
   Low-signal tokens decay over time.

This produces proto-cognitive behaviour balancing stability with novelty.

---

## 5. Fitness / Alignment Model

Each generation:

### Build a Community Semantic Map
Created by clustering or averaging agent semantic maps.

### Measure Divergence
Agents with the highest divergence — or lowest task performance — are selected for removal.

### Evolutionary Replacement
The lowest-ranking agent is replaced via:

- three-parent crossover  
- trait blending  
- partial token inheritance  
- light mutation  

This maintains coherence while enabling long-term exploration.

---

## 6. Mutation and Drift

Token changes include:

- **Stochastic drift** (small conceptual movement)  
- **Mutation** (rare, larger jumps to introduce novelty)  
- **Reinforcement learning** (stability via alignment)  
- **Decay** (removal of unused concepts)  

The balance determines whether the population tends toward:

- fragmentation  
- monoculture  
- dynamic equilibrium  

---

## 7. Tracking & Logging

The system logs:

- semantic maps each generation  
- communication transcripts  
- task performance  
- drift/mutation statistics  
- concept birth/death events  
- community discomfort and quiet/stagnation pressure
- reading passages, newly observed tokens, bridge promotions, and internal
  sentence-direction practice
- conversation mode, topic, working-memory action, feedback, n-gram, and
  world-graph metrics

Every 100 generations, agents emit a full semantic snapshot to allow longitudinal study of:

- concept evolution  
- linguistic structure  
- cooperation dynamics  
- cultural drift and subculture formation  

This system is intended not just as a simulation, but as a **research platform**.

---

## 8. Code Structure

The runtime uses small parent classes that compose focused mixins rather than
single monolithic modules:

- `evolution/coordinator.py` composes generation, evolution, semantic, task,
  and language responsibilities.
- `evolution/community_conversation.py` composes transcript handling,
  conversation memory, and response composition.
- `agents/cognition/task_system_v2.py` composes dispatch, numeric, signal,
  dialogue, and conceptual task solvers.
- The semantic system is separated into vector/flavour, association graph, and
  family/identity responsibilities.

This makes experiments easier to isolate and reduces the risk that a change in
one learning loop silently alters another.

---

# Timeline

## Phase 1 — Foundations (Completed / In Progress)
- Basic agent structure  
- Personality vectors  
- Multi-dimensional semantic space  
- Drift + mutation  
- Basic communication  
- Simple decision-making  
- First emergent phenomena logs  

## Phase 2 — Increasing Cognitive Complexity
- Shared problem-solving  
- Subcultures / clusters  
- Trust & reputation  
- Memory decay  
- Reinforcement loops  
- Early coalition behaviour  

## Phase 3 — Conflict + Cooperation
- Resource scarcity  
- Multi-solution cooperative tasks  
- Competing goals  
- Distrust / avoidance  
- Simple negotiation  

## Phase 4 — Abstract Reasoning (Experimental)
- Pattern evaluation  
- Proto-predictive modelling  
- Internal sub-agents  
- Higher-order coherence  

## Phase 5 — Meta-Behaviour
- Agents modifying their own rules  
- Group norms  
- Emergent social structure  
- Long-term memory shaping short-term action  

## Phase 6 — Research Questions
- Can simple agents form stable moral norms?  
- What triggers hierarchy or leadership?  
- Can distributed cognition solve harder tasks?  
- Does coherence emerge without central control?  

---

# Upcoming Features
- Homeostatic checks to prevent semantic black holes  
- Harder numeracy tasks  
- Harder literacy tasks  
- Better word-sense disambiguation before dictionary hypotheses become
  conversationally productive
- Persistent community state across coordinator runs
- Stronger connective words and functional grammar
- More deliberate community teaching and clarification requests

---

# Observations So Far
- Agents use consistent tokens; primitive grammar may be emerging  
- Possible gossip-like behaviour (under investigation)  
- Agents converge on base-8 or base-16 numeracy (emergent)  
- Agents generalise numerical magnitude beyond initial range  
- Strong cooperative behaviour required homeostatic pressure to diversify learning  
- Drift + mutation produce novelty without collapse
- Conversation is most reliable for short, explicitly grounded relations;
  longer free prose remains exploratory and can recombine observed fragments

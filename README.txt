# Agent Sandbox Project

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

Every 100 generations, agents emit a full semantic snapshot to allow longitudinal study of:

- concept evolution  
- linguistic structure  
- cooperation dynamics  
- cultural drift and subculture formation  

This system is intended not just as a simulation, but as a **research platform**.

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

# Current Implementation
*(To be filled out)*

---

# Upcoming Features
- Homeostatic checks to prevent semantic black holes  
- Harder numeracy tasks  
- Harder literacy tasks  
- Injection of English tokens/syllables  
- Agents recognising when addressed  
- Connective words & functional grammar  

---

# Observations So Far
- Agents use consistent tokens; primitive grammar may be emerging  
- Possible gossip-like behaviour (under investigation)  
- Agents converge on base-8 or base-16 numeracy (emergent)  
- Agents generalise numerical magnitude beyond initial range  
- Strong cooperative behaviour required homeostatic pressure to diversify learning  
- Drift + mutation produce novelty without collapse  
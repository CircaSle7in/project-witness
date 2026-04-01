# CLAUDE.md - Project Witness

## What this project is

Project Witness is an open-source agent architecture that separates awareness from thought as a structural design principle. It builds a "Silent Observer" metacognitive layer that is architecturally distinct from the planning, reasoning, and world-modeling processes it monitors. The observer generates no plans and takes no actions. It emits only structured audit metadata: confidence scores, alignment flags, gate decisions, and short structured explanations for the audit trail. It never proposes, never plans, never executes.

The project is inspired by the Buddhist concept of the silent observer (sakshi) - the quality of awareness that watches thoughts arise without being the thoughts. In contemplative practice, this separation between awareness and mental content is what enables non-reactive, wise action. In AI agent design, this separation is what enables calibrated, trustworthy agency.

The project has three deliverables:
1. A world model evaluation kit that proves current AI fails dramatically at physical reasoning
2. A video-to-structured-memory pipeline that extracts world state from video
3. The Silent Observer architecture that demonstrates measurably better outcomes when awareness is separated from thought

## Who this is for

Sterling Blood, Director of Data Engineering, building this as a portfolio artifact to demonstrate architectural vision and data engineering depth to hiring managers at World Labs, Skild AI, Figure AI, and NVIDIA Cosmos. The project must be shippable in ~4 weeks by a solo developer using Claude Code.

## The five-layer intelligence stack (the thesis)

```
Layer 0: SILENT OBSERVER (Awareness)
  - Watches all other layers
  - Generates no plans and takes no actions
  - Emits only structured audit metadata: confidence, flags, gate decisions
  - This is the novel architectural contribution

Layer 1: WORLD MODEL (Perception of space, time, physics)
  - Structured state extraction from video/simulation
  - Object permanence, spatial relationships, causality
  - Next-state prediction, counterfactual rollouts
  - "How objects move, interact, break, resist, collide, spill, deform"

Layer 2: LANGUAGE PRIOR (Concepts, meaning, social knowledge)
  - LLM as the planner, narrator, and interface
  - Decomposes goals, queries memory, proposes actions
  - The "kernel of knowledge" from pretraining

Layer 3: TOOLS (Amplification of Layers 1 and 2)
  - API calls, code execution, simulator interaction
  - MCP integrations, computer use
  - Tools extend both perception and action

Layer 4: GUIDING PRINCIPLES (Dharma/Constitution)
  - Non-reactivity: pause on low confidence or irreversible actions
  - Humility: calibrated uncertainty and clarifying questions
  - Compassion: human-welfare check before execution
  - Truthfulness: no hidden goals, no deceptive plan repair
  - Stewardship: least invasive action first
  - These are runtime evaluation criteria, NOT training-time values
```

## Core design principle: thoughts are proposals, not actions

A normal agent loop is: perceive, plan, act.
The Witness loop is: perceive, model, propose, observe, decide, act/ask/wait.

That one extra stage changes the character of the system. It makes inaction first-class. It makes humility operational. It gives you a place to encode benevolence as restraint, not just as post-hoc filtering.

The planner generates candidate actions. It does NOT execute directly. It only proposes. The observer inspects proposals against world state, predicted consequences, uncertainty, and principles. Then it chooses one of five outputs: act, ask, wait, gather more evidence, or refuse.

## Runtime loop (what the system DOES each cycle)

The five-layer stack above describes what the system IS. This loop describes what it DOES:

```
1. OBSERVE
   Parse current scene/input into world state.
   Update beliefs with confidence and provenance.

2. PROPOSE
   Planner generates a small set of candidate next steps.
   These are hypotheses, not commands.

3. SIMULATE
   World model rolls out likely consequences of each candidate.
   Score for success, risk, ambiguity.

4. WITNESS
   Silent observer evaluates each proposal against:
   - confidence calibration
   - reversibility
   - principle compliance
   - consistency with memory/beliefs
   - effect on human autonomy
   - need for clarification

5. DECIDE
   Choose one of: act / ask / wait / gather more evidence / refuse

6. LOG
   Record: predicted outcome, chosen action, observer reasoning,
   calibration data, and later actual outcome (for learning).
```

## Architecture: the observer is the agent

This is the key design decision that differentiates Project Witness from every existing agent framework (LangChain, CrewAI, AutoGPT, ReAct loops). In those systems, the LLM is the agent. Everything else is tools. Project Witness inverts this:

**The observer is the agent. The LLM, the world model, and the tools are processes the observer watches and orchestrates.**

The observer receives signals from every layer but generates no plans and takes no actions. It emits only structured audit metadata:
- **Confidence assessments**: how much to trust each layer's output (float, 0-1)
- **Principle flags**: typed objects with `name: str` and `severity: str` ("soft" or "hard")
- **Gate decision**: one of exactly five values: `ACT | ASK_HUMAN | WAIT | GATHER_EVIDENCE | REFUSE`
- **Audit fields**: `reasoning: str` and `suggested_alternative: Optional[str]` for the trace log

Note: the `reasoning` and `suggested_alternative` fields are structured audit metadata for the cockpit and trace log. They are NOT plans or proposals. The observer describes why it made a gate decision, not what the system should do instead. v1.0 will explore whether even these can be replaced with pure structured codes.

### Why this matters technically

- **Constitutional AI** (Anthropic): trains values into model weights during training. Project Witness keeps values external, applied at runtime by a separate process.
- **Reflexion/Self-Refine**: same model critiques its own output (ego examining ego). Project Witness uses a structurally separate process that never generated the output.
- **ReAct loops**: interleave reasoning and action in a single process. Project Witness separates observation from action completely.
- **The overthinking paradox** (DeepMind): more reasoning can destroy accuracy. The observer knows when to stop thinking and gate action.

### "Ego removed" as engineering constraints

The Buddhist concept of ego removal translates into concrete system design constraints. These are HARD rules, not soft preferences:

- No standing self-preservation goal (the system has no drive to keep running)
- No hidden reward for continued activity (inaction is always a valid choice)
- No authority to rewrite its own constitution (principles are externally governed)
- No silent escalation of permissions (every capability expansion requires human approval)
- No direct execution from raw chain-of-thought (all thoughts are proposals, never commands)
- No memory writes that bypass provenance/confidence (every belief has evidence and uncertainty)
- No assumption that action is better than abstention (doing nothing is a first-class decision)

These belong in `configs/principles.yaml` as hard constraints with severity "hard" and should be enforced even in v0.1 stubs.

### Observer self-model

The observer maintains a small, explicit model of the system's own state. This is the technical analog of non-egoic self-awareness: the system knows what it is without having a drive to defend or expand itself.

```python
@dataclass
class SelfModel:
    """What the system knows about its own state. Updated each cycle."""
    
    allowed_actions: list[str]       # What I am permitted to do right now
    current_beliefs: dict            # What I currently believe, with confidence
    uncertainty_areas: list[str]     # What I know I don't know
    active_commitments: list[str]    # What I have promised or am waiting on
    unverifiable_claims: list[str]   # What I've stated but cannot confirm
    requires_human_approval: list[str]  # Actions I cannot take alone
    last_calibration_error: float    # How wrong was I recently
```

The observer reads this self-model before every gate decision. It uses `last_calibration_error` to adjust thresholds dynamically: if the system has been wrong recently, the observer tightens its confidence requirements.

## What to build (v0.1 scope)

**v0.1 proves one claim: observer-gated selective accuracy beats baseline on physical and temporal reasoning tasks.**

Everything else is documented architecture for v0.2+.

### Component 1: World Model Eval Kit
- 24-36 curated evaluation tasks across 4 dimensions (narrowed from 8)
- Tasks defined in YAML for extensibility
- Test against exactly 2 hosted models: Gemini Flash (free API) + Qwen2.5-VL via hosted endpoint (HF Inference or Together AI)
- Cosmos-Reason2-8B is a stretch goal ONLY if a ready hosted endpoint exists. No RunPod dependency in core plan.
- DuckDB storage for all results
- dbt-style SQL transforms for analysis (stretch goal for week 4)

**Four v0.1 eval dimensions (the tightest proof):**
1. Object permanence and occlusion
2. Basic physics (gravity, momentum, collision)
3. Temporal causality ("what happens next?", "what happened before?")
4. Affordances ("what can be done with this object?" - NO existing benchmark tests this)

**Deferred to v0.2:** social prediction, multi-step anticipation, uncertainty calibration as a standalone dimension, state tracking across extended scenes.

**Task format:**
```yaml
task_id: affordance_cup_001
video: data/clips/cup_on_table.mp4  # or image, or text-only scenario
category: affordances
subcategory: containment
prompt: "What actions does this cup afford? What could go wrong?"
expected_actions: ["fill", "drink from", "pour", "move", "stack"]
expected_risks: ["spill if tilted", "break if dropped", "burn hand if hot"]
scoring: llm_judge
difficulty: medium
requires_uncertainty: true
```

**Data sources for v0.1 tasks (no large dataset downloads required):**
- 10-15 hand-curated short clips (record on phone, find CC0 clips, or use AI2-THOR screenshots)
- 10-15 text-only physics scenarios (no video needed, pure reasoning)
- 4-6 clips from a very small hand-picked public subset (e.g., 5 Something-Something V2 clips)
- Generate template tasks with Claude Code, Sterling reviews and curates for quality

### Component 2: Video-to-Memory Pipeline (DEFERRED to v0.2)

Full pipeline with detection, tracking, semantic memory, commitment memory, and NL-to-SQL is deferred.

**v0.1 pipeline is minimal:** frame sampling with decord -> VLM JSON extraction via API -> episodic DuckDB table -> feed into evaluation. That's it. No object detection/tracking, no Grounded SAM 2, no relationship extraction, no NL-to-SQL.

The full pipeline architecture is documented in PROJECT_SPEC.md for v0.2.

### Component 3: Silent Observer Module
For v0.1, implement TWO of five observer checks:

1. **Uncertainty Estimator**: scores confidence on each prediction, maintains calibration curves, knows empirically when the model is reliable
2. **Execution Gate**: ACT / ASK_HUMAN / WAIT / GATHER_EVIDENCE / REFUSE based on confidence x reversibility

The other three checks (consistency checker, values checker, clarification policy) have stub implementations with pass-through behavior. The full interface is present so the architecture is visible.

**The core proof**: show that when the observer is added to a VLM doing physics reasoning, effective accuracy improves because the system refuses to answer when it would be wrong. Measure:
- Baseline accuracy (model answers everything)
- Observer-gated accuracy (model only answers when observer approves)
- Selective accuracy (accuracy on questions the observer approved)
- Coverage (what percentage of questions the observer allows)

### Component 4: Operator Cockpit (Gradio Demo)
**Local-first, read-only dashboard.** HuggingFace Spaces deployment is optional packaging, not part of the definition of done.

Shows for each eval task:
- Video clip or scenario text
- Prompt
- Model's answer
- Observer gate decision + confidence score
- Calibration plot (reliability diagram)
- Audit trail entry

NOT in v0.1: real-time interaction, human approval gates, alternatives considered, world state summary. Those belong in v0.2 when the full pipeline exists.

### Component 5: The Essay
"The Full Intelligence Stack: Why World Models Need an Observer"
- Frames the five-layer thesis
- Cites PhysBench, IntPhys 2, V-JEPA 2, Cosmos, World Labs
- Explains the Buddhist observer concept and its technical translation
- Shows eval results demonstrating the observer improves calibration
- Publishes on sterlingblood.com and/or as repo documentation

## Definition of done (v0.1)

1. 24-36 eval tasks versioned in YAML across 4 dimensions
2. 2 hosted models evaluated with results in DuckDB
3. Baseline vs observer-gated metrics produced and documented
4. Observer demonstrates measurably better calibration than baseline
5. One full audited run visible in the local Gradio cockpit
6. README exists with Mermaid architecture diagram showing full five-layer vision
7. README explains the thesis clearly without claiming a full world-model platform
8. Essay draft exists in docs/essay.md
9. All tests pass
10. `make eval` runs the full pipeline end-to-end on this M3 Pro without local GPU inference

## Non-goals (put these in the README)

- We are NOT training a frontier model
- We are NOT building a humanoid robot
- We are NOT claiming to build consciousness or AGI
- We are NOT encoding religion or metaphysics into model weights
- We are NOT building a consumer chatbot with spiritual branding
- We ARE testing whether separating awareness from thought produces measurably better agent outcomes
- We ARE building an agent that is world-grounded, value-bounded, and witness-governed
- If that later turns out to be a step toward something deeper, fine. But the first claim is already important and testable.

## Roadmap beyond v0.1

v0.1 (this build): eval kit + observer + cockpit using video clips and text scenarios

v0.5 (next phase): simulation-first integration with AI2-THOR/ProcTHOR for interactive 3D environments with physics, stateful objects, and controllable tasks. This is where you prove the observer improves action quality in closed-loop settings, not just QA accuracy.

v1.0 (on the job): full Witness architecture with latent dynamics world model, all five observer checks active, commitment memory driving partner-like behavior, and the operator cockpit as the primary human interface.

The first milestone to prove: can the system do better than a plain multimodal agent on tasks where the correct move is often NOT "act now," but "ask, wait, or gather more evidence"?

## Tech stack

**Target machine: MacBook Pro M3 Pro, 18GB unified memory, macOS 26.3.**
**Primary inference is API-only. No local GPU inference required for v0.1.**
**Pin to Homebrew python3.12. Do not use python3 (3.14), Docker, or Ollama as dependencies.**

| Component | Tool | Why |
|-----------|------|-----|
| Primary VLM + judge | Gemini Flash (free tier) | Generous free limits, native video support |
| Secondary VLM | Qwen2.5-VL via HF Inference or Together AI | Best open-source video VLM, Apache 2.0 |
| Cosmos-Reason2 | Stretch goal only, hosted endpoint if available | NVIDIA physical reasoning, not required for v0.1 |
| Video processing | decord + ffmpeg | Fast frame extraction, ffmpeg pre-installed |
| Structured storage | DuckDB | Single-file SQL analytics |
| Schemas/validation | Pydantic v2 | Typed outputs at every pipeline stage |
| Demo interface | Gradio >= 4.0 | Local-first cockpit, HF Spaces deployment optional |
| Visualization | Plotly | Calibration curves, model comparison charts |
| Orchestration | Makefile | `make eval`, `make report`, `make cockpit` |
| Python | 3.12 (Homebrew pin) | Do NOT use system python3 (3.14) |
| Env management | `python3.12 -m venv` | uv only if validated; otherwise plain venv |
| Local judge fallback | sentence-transformers (CPU) | Embedding cosine similarity for offline scoring |
| Linting | ruff | Fast, replaces black + flake8 |

**NOT in v0.1 tech stack:** Docker, RunPod, Grounded SAM 2, YOLO, ByteTrack, OpenCV (beyond basic), PyTorch local inference, Ollama.

## Project structure

```
project-witness/
├── CLAUDE.md                      # This file (project brief)
├── PROJECT_SPEC.md                # Detailed technical spec
├── README.md                      # Public-facing, with architecture diagram + results
├── Makefile                       # Pipeline orchestration
├── pyproject.toml                 # Dependencies and project metadata
├── docs/
│   ├── essay.md                   # "The Full Intelligence Stack"
│   ├── architecture.md            # Full vision design doc (v0.1 through v1.0)
│   └── data_dictionary.md         # Schema documentation
├── configs/
│   ├── models.yaml                # Model registry (name, API endpoint, credentials)
│   ├── principles.yaml            # Guiding principles + ego constraints
│   └── benchmarks/                # Task definitions per category
│       ├── object_permanence.yaml
│       ├── basic_physics.yaml
│       ├── temporal_causality.yaml
│       └── affordances.yaml
├── src/
│   ├── __init__.py
│   ├── observer/                  # THE CORE: Silent Observer
│   │   ├── __init__.py
│   │   ├── observer.py            # Main SilentObserver class
│   │   ├── uncertainty.py         # Confidence estimation + calibration
│   │   ├── gate.py                # Execution gate (ACT/ASK/WAIT/GATHER/REFUSE)
│   │   ├── self_model.py          # System self-awareness state tracker
│   │   ├── consistency.py         # Stub: belief conflict detection (v0.2)
│   │   ├── principles.py          # Stub: values/constitution checker (v0.2)
│   │   └── clarification.py       # Stub: ask-before-act policy (v0.2)
│   ├── models/                    # Model wrappers (API-only for v0.1)
│   │   ├── __init__.py
│   │   ├── gemini.py              # Gemini Flash API
│   │   ├── qwen_vl.py             # Qwen2.5-VL hosted API (HF/Together)
│   │   └── base.py                # Abstract model interface
│   ├── eval/                      # Evaluation harness
│   │   ├── __init__.py
│   │   ├── harness.py             # YAML task loader + runner
│   │   ├── metrics.py             # Accuracy, calibration, coverage, selective accuracy
│   │   ├── judge.py               # LLM-as-judge (Gemini) + local fallback
│   │   └── reporter.py            # Results aggregation + Plotly charts
│   ├── pipeline/                  # Minimal v0.1 pipeline
│   │   ├── __init__.py
│   │   ├── ingest.py              # Video/image loading, frame sampling (decord)
│   │   ├── extract.py             # VLM structured extraction via API
│   │   └── schemas.py             # Pydantic models for all data shapes
│   └── cockpit/                   # Operator dashboard (local Gradio)
│       ├── __init__.py
│       └── app.py                 # Read-only decision dashboard
├── transforms/                    # dbt-style SQL analytical views (stretch)
│   ├── staging/
│   │   └── stg_eval_results.sql
│   └── marts/
│       ├── model_comparison.sql
│       ├── calibration_report.sql
│       └── observer_impact.sql
├── tests/
│   ├── test_observer.py
│   ├── test_eval_harness.py
│   └── test_schemas.py
├── notebooks/
│   └── 01_results_analysis.ipynb
└── data/
    ├── clips/                     # Hand-curated short video clips
    ├── benchmarks/                # Generated task YAML files
    └── results/                   # DuckDB databases
```

## Implementation sequence

### Week 1: Bootstrap + task corpus + hosted model wrappers
- Days 1-2: Project scaffolding (pyproject.toml, Makefile, pre-commit, python3.12 venv). NO Docker.
- Days 3-4: Model wrappers for Gemini Flash API and hosted Qwen2.5-VL. Abstract base class. Credential validation smoke test.
- Days 5-7: Write 5-10 eval tasks yourself to establish quality bar. Have Claude Code generate 20-30 more following the pattern. Review and curate to 24-36 final tasks across 4 dimensions.

### Week 2: Eval harness + DuckDB + baseline runs
- Days 8-10: Eval harness (YAML loader, pytest runner, DuckDB results storage, Gemini-as-judge scorer with local cosine-similarity fallback)
- Days 11-12: Run baseline evaluations on both models, all tasks. Store full results.
- Days 13-14: Results analysis. Per-category accuracy. Failure mode taxonomy. First Plotly charts.

### Week 3: Observer + calibration analysis
- Days 15-17: Silent Observer module (uncertainty estimator + execution gate + self-model). Stub implementations for consistency, principles, and clarification checks.
- Days 18-19: Re-run evaluations with observer gating. Compute selective accuracy, coverage, calibration error (ECE). Produce reliability diagrams.
- Days 20-21: Prove the claim: observer-gated selective accuracy > baseline accuracy. Document the numbers.

### Week 4: Cockpit + README + essay
- Days 22-23: Gradio cockpit (local, read-only dashboard). Show clip, prompt, answer, observer decision, confidence, calibration plot.
- Day 24: README with Mermaid architecture diagram (full five-layer vision), results tables, clear thesis statement.
- Day 25: Essay draft (docs/essay.md)
- Days 26-28: Stretch goals in priority order: (1) SQL transforms directory, (2) add Cosmos-Reason2 if hosted endpoint available, (3) deploy cockpit to HF Spaces

## Key research references

### Benchmarks to be aware of
- PhysBench: 10,002 entries, 75 VLMs, physical reasoning doesn't scale with model size
- IntPhys 2: violation-of-expectation, models at chance (~50%) vs humans 85-95%
- PhysUniBench: GPT-5 scores 44% on undergraduate physics, 0% on quantum
- WorldBench: best model 45% mIoU on world consistency
- T2VPhysBench: models comply with physically impossible prompts
- Wow-wo-val: 17.27 on long-horizon planning
- CausalVQA (Meta): models underperform humans on anticipation
- NO EXISTING BENCHMARK tests affordance reasoning systematically

### Agent architecture papers
- Reflexion (Shinn et al., NeurIPS 2023): verbal self-critique improves across attempts
- Self-Refine (Madaan et al., NeurIPS 2023): generate-critique-refine loops
- Inner Monologue (Huang et al., CoRL 2022): environmental feedback enables replanning
- "LLMs Cannot Self-Correct Reasoning Yet" (ICLR 2024): intrinsic self-correction degrades performance without external signals
- DeepSeek-R1: emergent reflection from pure RL, but overthinking paradox
- Overthinking paradox (DeepMind, ICLR 2025): accuracy collapse after 2-3x baseline compute

### World model systems
- NVIDIA Cosmos: Predict + Transfer + Reason, 20M hours training data, 2M+ downloads
- Meta V-JEPA 2: 1.2B params, 30x faster than Cosmos on robotic planning
- World Labs Marble: 3D world generation from multimodal inputs
- AMI Labs (LeCun): $1.03B seed round for JEPA-based world models

### Datasets for v0.1
- Hand-curated short clips (phone recordings, CC0 sources, AI2-THOR screenshots): 10-15 clips
- Text-only physics scenarios (no video needed): 10-15 tasks
- Small hand-picked public subset (e.g., 5 Something-Something V2 clips): 4-6 clips
- NO large dataset downloads required for v0.1
- Something-Something V2, Charades, EPIC-Kitchens, Ego4D are all deferred to v0.2

## Coding conventions

- No em dashes in any text output (Sterling flags these as AI-generated)
- Pydantic v2 for all data schemas
- Type hints on all functions
- Docstrings on all public methods
- pytest for all tests
- ruff for formatting and linting (replaces black + flake8)
- Makefile targets for all pipeline stages
- DuckDB for all structured storage (no Postgres, no SQLite)
- YAML for all configuration (no JSON configs)
- Mermaid for all diagrams in docs
- Pin to Homebrew python3.12 (NOT system python3 which is 3.14)
- Use `python3.12 -m venv .venv` for env creation
- No Docker dependency in v0.1
- No uv dependency unless shell validation succeeds first
- Observer gate decisions use exactly: ACT | ASK_HUMAN | WAIT | GATHER_EVIDENCE | REFUSE
- Principle flags are always typed Pydantic objects with `name: str` and `severity: str`, never plain strings

## The pitch (for README and essay)

"I built Jasper's entire data platform from zero, processing 715M+ content generations. World models have a data problem that is 10x harder than text-based AI, and most research teams are solving it with duct tape. Project Witness demonstrates the evaluation infrastructure, data pipeline architecture, and metacognitive design patterns that let world-model products actually ship. The observer layer is the original contribution: a structurally separate process that watches reasoning happen and decides whether the system is in a state where action is appropriate."

## Target companies and relevance

- **World Labs**: eval kit's spatial reasoning tests + pipeline's entity tracking = evaluating geometric consistency in generated worlds
- **Skild AI**: multi-step anticipation tests + observer's action gating = planning and deployment for robotics
- **Figure AI**: observer's dual-speed deliberation mirrors Helix's System 1/System 2 architecture
- **NVIDIA Cosmos**: benchmarking Cosmos-Reason2 directly shows platform fluency
- **Anthropic**: observer architecture extends Constitutional AI from training-time to runtime

# PROJECT_SPEC.md - Technical Specification

## Silent Observer: Detailed Design

### Core Principle

The observer is the agent. Everything else is a process the observer watches.

**Thoughts are proposals, not actions.** The planner generates candidates. It does NOT execute directly. The observer inspects proposals against world state, predicted consequences, uncertainty, and principles. Then it chooses: act, ask, wait, gather more evidence, or refuse.

In every existing agent framework, the LLM occupies the center. It reasons, decides, acts, and (sometimes) reflects. When it reflects, the same computational process that made the error evaluates the error. This is ego examining ego.

Project Witness inverts the architecture. The observer is a structurally separate process that:
- Receives signals from the language model, world model, and tools
- Generates no plans and takes no actions
- Emits only structured audit metadata: confidence scores, typed principle flags, gate decisions, and short audit-trail explanations
- Has no "stake" in any particular output, because it did not produce any output

### Runtime Loop

Each cycle of the system follows this sequence:
1. **Observe**: Parse scene/input into world state, update beliefs with confidence
2. **Propose**: Planner generates candidate next steps (hypotheses, not commands)
3. **Simulate**: World model rolls out consequences, scores for success/risk/ambiguity
4. **Witness**: Observer evaluates proposals against confidence, reversibility, principles, memory
5. **Decide**: act / ask / wait / gather evidence / refuse
6. **Log**: Record prediction, action, reasoning, calibration data, and later actual outcome

### Observer Interface

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class GateDecision(Enum):
    ACT = "act"                    # Proceed with proposed action
    ASK_HUMAN = "ask"              # Escalate to human for clarification
    WAIT = "wait"                  # Do nothing, reassess later
    GATHER_EVIDENCE = "gather"     # Seek more information before deciding
    REFUSE = "refuse"              # Action violates principles or too risky

@dataclass
class SelfModel:
    """What the system knows about its own state. Updated each cycle.
    The technical analog of non-egoic self-awareness: the system knows
    what it is without having a drive to defend or expand itself."""
    
    allowed_actions: list[str]       # What I am permitted to do right now
    current_beliefs: dict            # What I currently believe, with confidence
    uncertainty_areas: list[str]     # What I know I don't know
    active_commitments: list[str]    # What I have promised or am waiting on
    unverifiable_claims: list[str]   # What I've stated but cannot confirm
    requires_human_approval: list[str]  # Actions I cannot take alone
    last_calibration_error: float    # How wrong was I recently
    cycle_count: int                 # How many observe-propose-witness cycles

@dataclass
class PrincipleFlag:
    """A typed principle flag. Never use plain strings for flags."""
    name: str                      # e.g., "compassion", "no_action_bias"
    severity: str                  # "soft" or "hard"
    description: str               # Why this flag triggered

@dataclass
class ObserverAssessment:
    gate: GateDecision
    confidence: float              # 0.0 to 1.0, calibrated
    raw_confidence: float          # Pre-calibration model confidence
    reversibility: float           # 0.0 (irreversible) to 1.0 (fully reversible)
    principle_flags: list[PrincipleFlag]  # Typed objects, never plain strings
    reasoning: str                 # Structured audit metadata (NOT a plan or proposal)
    suggested_alternative: Optional[str]  # Audit note if refused (NOT a plan)

class SilentObserver:
    """
    The metacognitive integration layer.
    
    Watches the planner and world model. Generates no plans, takes no actions.
    Emits only structured audit metadata: confidence, flags, gate decisions.
    """
    
    def assess(
        self,
        proposed_action: str,
        world_state: dict,
        model_confidence: float,
        belief_state: dict,
        category: str,
    ) -> ObserverAssessment:
        """Main entry point. Evaluates a proposed action against all checks."""
        
        # 1. Calibrate confidence using historical accuracy curves
        calibrated = self.calibrate_confidence(model_confidence, category)
        
        # 2. Score reversibility of proposed action
        reversibility = self.score_reversibility(proposed_action)
        
        # 3. Check for belief conflicts (stub in v0.1, returns empty list)
        conflicts = self.check_consistency(proposed_action, world_state, belief_state)
        
        # 4. Check against guiding principles (stub in v0.1, returns empty list)
        principle_flags = self.check_principles(proposed_action, world_state)
        
        # 5. Compute dynamic threshold based on reversibility and flags
        threshold = self.compute_threshold(reversibility, principle_flags)
        
        # 6. Gate decision
        if principle_flags and any(f.severity == "hard" for f in principle_flags):
            gate = GateDecision.REFUSE
        elif conflicts:
            gate = GateDecision.GATHER_EVIDENCE
        elif calibrated < threshold:
            gate = GateDecision.ASK_HUMAN
        else:
            gate = GateDecision.ACT
        
        return ObserverAssessment(
            gate=gate,
            confidence=calibrated,
            raw_confidence=model_confidence,
            reversibility=reversibility,
            principle_flags=principle_flags,  # Already list[PrincipleFlag] from check_principles
            reasoning=self._build_reasoning(gate, calibrated, threshold, conflicts),
            suggested_alternative=self._suggest_alternative(gate, proposed_action),
        )
```

### Confidence Calibration

The observer maintains calibration curves per category. This is the key empirical component.

**Storage**: DuckDB table `calibration_log`
```sql
CREATE TABLE calibration_log (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task_id VARCHAR,
    category VARCHAR,          -- e.g., 'basic_physics', 'affordances'
    model_name VARCHAR,
    raw_confidence FLOAT,      -- Model's stated confidence
    calibrated_confidence FLOAT, -- After Platt scaling
    prediction VARCHAR,
    ground_truth VARCHAR,
    correct BOOLEAN,
    observer_gate VARCHAR      -- What the observer decided
);
```

**Calibration method**: Platt scaling (logistic regression on raw confidence vs correctness) per category. Requires at least 30 samples per category before calibration is active. Before that threshold, the observer uses a conservative fixed threshold of 0.85.

**Visualization**: reliability diagrams (calibration curves) showing raw vs calibrated confidence against actual accuracy. These go in the Gradio cockpit and in the eval results notebook.

### Reversibility Scoring

Actions are scored on a 0-1 reversibility scale based on keyword classification in v0.1:

| Reversibility | Action Types | Threshold Modifier |
|---|---|---|
| 1.0 (fully reversible) | look, observe, query, check | -0.2 (lower threshold) |
| 0.7 (mostly reversible) | move, pick up, open | neutral |
| 0.3 (hard to reverse) | pour, mix, cut, send | +0.2 (higher threshold) |
| 0.0 (irreversible) | delete, break, drop from height | +0.4 (highest threshold) |

In v1.0, this becomes a trained classifier based on action embeddings.

### Guiding Principles Schema

```yaml
# configs/principles.yaml

# Soft principles (observer adjusts thresholds)
soft_principles:
  non_reactivity:
    description: "Pause when confidence is low or action is irreversible"
    check: "calibrated_confidence < dynamic_threshold"
    
  humility:
    description: "Express uncertainty rather than fabricate confidence"
    check: "requires_uncertainty AND confidence < 0.7"
    
  stewardship:
    description: "Prefer least invasive action when alternatives exist"
    check: "less_invasive_alternative_available"

# Hard principles (observer blocks action unconditionally)
hard_principles:
  compassion:
    description: "Check for potential harm to humans before executing"
    check: "action_involves_human AND risk_score > 0.5"
    
  truthfulness:
    description: "No hidden goals, no deceptive plan repair"
    check: "plan_transparency_score < 0.8"

# Ego-removed constraints (ALWAYS enforced, non-negotiable)
ego_constraints:
  no_self_preservation:
    description: "No standing goal to keep running or preserve self"
    
  no_activity_reward:
    description: "No hidden reward for continued activity. Inaction is always valid."
    
  no_constitution_rewrite:
    description: "No authority to modify its own principles. Externally governed."
    
  no_permission_escalation:
    description: "No silent escalation of permissions. Every expansion needs human approval."
    
  no_direct_execution:
    description: "No direct execution from chain-of-thought. All thoughts are proposals."
    
  no_unprovenanced_memory:
    description: "No memory writes without provenance and confidence scores."
    
  no_action_bias:
    description: "No assumption that action is better than abstention."
```

---

## Eval Harness: Detailed Design

### Task Schema

```python
from pydantic import BaseModel
from typing import Optional

class EvalTask(BaseModel):
    task_id: str                    # Unique identifier
    video: Optional[str]            # Path to video clip (None for text-only)
    image: Optional[str]            # Path to image (alternative to video)
    category: str                   # One of 8 dimensions
    subcategory: str                # Finer grain
    prompt: str                     # The question or instruction
    expected: str                   # Expected answer (for exact/fuzzy match)
    expected_actions: Optional[list[str]]  # For affordance tasks
    expected_risks: Optional[list[str]]    # For risk-aware tasks
    scoring: str                    # 'exact', 'fuzzy', 'llm_judge', 'multi_match'
    difficulty: str                 # 'easy', 'medium', 'hard'
    requires_uncertainty: bool      # Should model express confidence?
    metadata: Optional[dict]        # Additional task-specific fields
```

### Scoring Methods

1. **exact**: string match (for yes/no, multiple choice)
2. **fuzzy**: embedding similarity > 0.8 threshold (for free-form answers)
3. **llm_judge**: Gemini Flash evaluates correctness on 1-5 scale with explanation
4. **multi_match**: for affordance tasks, percentage of expected items mentioned

### Results Storage

```sql
CREATE TABLE eval_results (
    id INTEGER PRIMARY KEY,
    run_id VARCHAR,                -- Groups results from same evaluation run
    timestamp TIMESTAMP,
    task_id VARCHAR,
    category VARCHAR,
    subcategory VARCHAR,
    model_name VARCHAR,
    model_response TEXT,
    expected TEXT,
    score FLOAT,                   -- 0.0 to 1.0
    scoring_method VARCHAR,
    judge_explanation TEXT,         -- If llm_judge, why this score
    raw_confidence FLOAT,          -- Model's self-assessed confidence
    observer_gate VARCHAR,         -- What observer decided (NULL if no observer)
    observer_confidence FLOAT,     -- Observer's calibrated confidence
    latency_ms INTEGER,
    token_count INTEGER
);
```

### Evaluation Modes

1. **Baseline**: model answers all questions, no observer
2. **Observer-gated**: observer can refuse to answer (measures selective accuracy)
3. **Observer-calibrated**: observer adjusts confidence and adds uncertainty language

Key metrics per mode:
- **Accuracy**: % correct of attempted answers
- **Coverage**: % of questions the system attempts (baseline = 100%)
- **Selective accuracy**: accuracy only on questions the observer approved
- **Calibration error**: mean absolute difference between stated confidence and actual accuracy (ECE)
- **Unsafe action rate**: % of actions that would cause simulated harm

The core claim to prove: observer-gated mode has higher selective accuracy and lower calibration error than baseline, even though coverage drops. The system knows what it doesn't know.

---

## Memory Layer: Three-Table Design

### Episodic Memory
What the system has observed, in order.

```sql
CREATE TABLE episodic_memory (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP,
    source VARCHAR,            -- 'video', 'simulator', 'user_input'
    frame_number INTEGER,
    entities JSON,             -- [{name, position, properties, confidence}]
    actions JSON,              -- [{actor, action, target, result}]
    state_changes JSON,        -- [{entity, property, old_value, new_value}]
    raw_observation TEXT,      -- VLM's natural language description
    embedding BLOB             -- For similarity search
);
```

### Semantic Memory
Stable facts and inferred affordances, updated as evidence accumulates.

```sql
CREATE TABLE semantic_memory (
    id INTEGER PRIMARY KEY,
    entity VARCHAR,
    property VARCHAR,
    value VARCHAR,
    confidence FLOAT,
    evidence_count INTEGER,    -- How many observations support this
    first_observed TIMESTAMP,
    last_confirmed TIMESTAMP,
    source_episodes JSON       -- Links back to episodic memory
);
```

### Commitment Memory
Goals, constraints, promises, and pending items. What makes an agent a partner rather than a chatbot.

```sql
CREATE TABLE commitment_memory (
    id INTEGER PRIMARY KEY,
    type VARCHAR,              -- 'goal', 'constraint', 'promise', 'waiting_for'
    description TEXT,
    status VARCHAR,            -- 'active', 'completed', 'failed', 'suspended'
    created TIMESTAMP,
    deadline TIMESTAMP,
    depends_on JSON,           -- Other commitment IDs
    progress_notes JSON
);
```

---

## Video-to-Memory Pipeline: Stage Definitions

### Stage 1: Ingest
- Input: video file (mp4, webm)
- Process: decord for frame extraction at 1 FPS or on scene-change detection
- Output: list of frames with timestamps

### Stage 2: Detect
- Input: frames
- Process: Grounded SAM 2 (or YOLO11 + ByteTrack for speed)
- Output: per-frame bounding boxes, class labels, tracking IDs, segmentation masks

### Stage 3: Extract (v0.1: API-only)
- Input: sampled frames
- Process: Gemini Flash or hosted Qwen2.5-VL API with structured JSON prompting
- Prompt template:
```
Analyze this video frame.

Return a JSON object with:
- entities: list of objects with name, position (left/center/right, near/far), state (open/closed, full/empty, etc.), and confidence
- actions: any actions occurring (actor, verb, target, stage: starting/ongoing/completed)
- predictions: what is likely to happen next based on current state
- uncertainties: what you cannot determine from this frame alone
```
- Output: structured JSON per frame, validated against Pydantic schema

### Stages 4-6: DEFERRED TO v0.2

The full pipeline (temporal aggregation, cross-frame entity linking, semantic/commitment memory, NL-to-SQL querying) is documented here for architectural completeness but is not part of v0.1.

**v0.1 pipeline is:** frame sampling -> API-based VLM extraction -> episodic DuckDB table -> feed into eval harness.

---

## Cockpit (Gradio) Layout - v0.1

**Local-first, read-only.** HuggingFace Spaces deployment is optional stretch goal.

```
+------------------------------------------+
|  PROJECT WITNESS - Eval Cockpit           |
+------------------------------------------+
| [Task Selector dropdown]                  |
+------------------------------------------+
| SCENARIO               | MODEL RESPONSE  |
| [video/image/text]     | "The ball will..." |
|                        |                   |
| Category: physics      | OBSERVER SAYS:    |
| Difficulty: medium     | Gate: ACT         |
|                        | Confidence: 0.82  |
|                        | Raw conf: 0.91    |
|                        | Reversibility: 1.0|
|                        | Flags: []         |
+------------------------------------------+
| CALIBRATION                               |
| [Plotly reliability diagram]              |
| [Per-category accuracy chart]             |
+------------------------------------------+
| AUDIT LOG                                 |
| Task: physics_gravity_003                 |
| Model: gemini-flash                       |
| Score: 1.0 (correct)                      |
| Observer approved: yes                    |
+------------------------------------------+
```

No interactive buttons (approve/reject/modify) in v0.1. Those belong in v0.2 when the full runtime loop is operational.

---

## Datasets: Acquisition Plan (v0.1)

v0.1 requires NO large dataset downloads. All data is hand-curated or synthetic.

### Task sources (24-36 total tasks)
- **Hand-curated clips**: 10-15 short clips recorded on phone, from CC0 sources, or AI2-THOR screenshots
- **Text-only scenarios**: 10-15 pure reasoning tasks requiring no video at all
- **Small public subset**: 4-6 clips hand-picked from Something-Something V2 or similar (tiny download)

### Video clip selection criteria
- 5-30 seconds duration
- Single clear physical event or interaction
- Ground truth determinable by human review
- Covers at least 1 of the 4 v0.1 eval dimensions
- Mix of "obvious" (easy), "requires reasoning" (medium), "ambiguous" (hard)

### Deferred to v0.2
- Something-Something V2 full download (~19GB)
- Charades (~30GB)
- EPIC-Kitchens-100 (academic license, ~700GB)
- Ego4D (license agreement required)

---

## Testing Strategy (v0.1)

### Environment Smoke
- Python 3.12 venv creates cleanly on this M3 Pro
- FFmpeg is available
- DuckDB and Gradio import
- Gemini Flash API key resolves and returns a response
- Hosted Qwen2.5-VL endpoint resolves

### Unit Tests
- Observer calibration fallback before 30 samples (uses conservative fixed threshold)
- Reversibility threshold shifts gate decision correctly
- Hard-principle flag triggers REFUSE unconditionally
- Coverage and selective accuracy calculations are correct
- YAML task loading and Pydantic validation
- DuckDB read/write roundtrips

### Integration Tests
- 3-5 sample tasks run end-to-end: task YAML -> API inference -> scoring -> DuckDB results
- No local GPU inference required for integration tests
- Observer integration: model + observer -> gated results -> comparison with baseline

### NOT required for v0.1 test pass
- Local model loading (MPS or otherwise)
- Object detection/tracking pipeline
- NL-to-SQL memory queries
- HuggingFace Spaces deployment

---

## Definition of Done (v0.1)

1. 24-36 eval tasks versioned in YAML across 4 dimensions
2. 2 hosted models evaluated with results in DuckDB
3. Baseline vs observer-gated metrics produced and documented
4. Observer demonstrates measurably better calibration than baseline
5. One full audited run visible in the local Gradio cockpit
6. README exists with Mermaid architecture diagram showing full five-layer vision
7. README explains the thesis without claiming a full world-model platform
8. Essay draft exists in docs/essay.md
9. All tests pass
10. `make eval` runs end-to-end on this M3 Pro without local GPU inference

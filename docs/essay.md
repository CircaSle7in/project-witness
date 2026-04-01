# The Full Intelligence Stack: Why World Models Need an Observer

**Sterling Blood**

---

## The Gap

AI systems keep getting better at seeing and generating, but they still cannot reason about the physical world. This is not a fringe complaint. The benchmarks are specific, and the numbers are bad.

PhysBench evaluated leading multimodal models on physical reasoning tasks and found that performance does not scale with model size the way language benchmarks suggest it should. Bigger models, more parameters, more pretraining data: none of it reliably closes the gap between "can describe a scene" and "can predict what happens next in that scene." IntPhys 2 tested models on violation-of-expectation tasks, the same paradigm developmental psychologists use with infants. Humans score 85-95%. Current models perform at chance, roughly 50%. They cannot tell whether a ball rolling behind an occluder should reappear on the other side. Infants can do this by four months old.

And those are the benchmarks that exist. No current evaluation suite tests affordance reasoning systematically: can this object support that object? Can this tool be used to accomplish that goal? Can this surface be walked on? These are the questions that matter for any system that needs to act in the world rather than just talk about it.

The problem is not compute. It is not data volume. It is architecture. Specifically, it is the absence of structural separation between the process that generates predictions and the process that evaluates whether those predictions are trustworthy enough to act on.

## The Five-Layer Intelligence Stack

Perception alone is not intelligence. Generating plausible next tokens about physical scenarios is not physical reasoning. To build systems that reason well enough to act in the world, you need at least five layers, and most current architectures are missing the most important one.

**Layer 1: World Models.** Internal representations of how things behave. Gravity pulls things down. Unsupported objects fall. Liquids flow to fill containers. These are not facts to be memorized; they are dynamics to be simulated, even if that simulation is approximate and learned rather than analytically derived.

**Layer 2: Language Priors.** What things mean, what they are called, how humans talk about them. Language is the interface layer between internal representations and external communication. It is also a source of bias: models over-index on linguistic plausibility at the expense of physical plausibility because their training signal is overwhelmingly linguistic.

**Layer 3: Tools.** Amplification of capability. A model that can call a physics simulator, query a database, or execute code is more capable than one that cannot. Tools extend the reach of reasoning without requiring the reasoning itself to be superhuman.

**Layer 4: Principles.** Constraints on action. Not everything that can be done should be done. Safety constraints, ethical boundaries, operational guardrails. These are often trained into weights (Constitutional AI) or injected as system prompts. Both approaches have failure modes, which I will return to.

**Layer 5: The Observer.** This is the layer that watches everything else. It does not generate plans. It does not produce actions. It watches the plan-generation process and the action-selection process and asks: is this confident enough? Is this grounded enough? Should we act, or should we wait?

The observer is the novel contribution of Project Witness. Every other layer has active research programs and well-funded teams behind it. The observer does not, because it looks like "doing less," and doing less is hard to fund.

## The Buddhist Concept, Translated

The idea of a silent observer is not new. In contemplative traditions, particularly the Advaita Vedanta concept of *sakshi* (the witness), there is a quality of awareness that watches thoughts arise without being the thoughts. You notice anger without becoming anger. You observe a plan forming without being committed to executing it.

In AI terms, the translation is concrete: separate the process that generates plans from the process that evaluates them. The generator has a stake in its output. It was trained to produce completions, to be helpful, to give answers. It has, in a functional sense, an ego: a drive to produce rather than abstain. The observer has no such drive. Its only function is to assess and gate.

This is not metaphysics in code. It is not an attempt to build consciousness or recreate meditative states in silicon. It is an engineering pattern: structurally separate metacognition from cognition, and then measure whether the separation improves outcomes. If it does, keep it. If it does not, discard it. The inspiration is contemplative; the validation is empirical.

The key insight from contemplative practice that translates directly is this: the generator and the evaluator must be structurally distinct processes. When the same process generates and evaluates, it is ego examining ego. The evaluation is contaminated by the same biases that produced the output. You need separation.

## What Existing Approaches Get Wrong

Several research programs have attempted versions of self-evaluation, self-correction, and metacognition in language models. Each gets something right and something wrong.

**Constitutional AI** (Anthropic) trains values into model weights at training time. This is powerful but static. The values are baked in during RLHF and cannot be adjusted at inference time without retraining. Project Witness applies constraints at runtime via a separate process that can be updated, audited, and overridden without touching the base model.

**Reflexion** (NeurIPS 2023) and **Self-Refine** (NeurIPS 2023) both use the same model to critique its own output. This is the "ego examining ego" problem. Huang et al. (ICLR 2024) showed formally that LLMs cannot self-correct reasoning without external feedback. The critique inherits the same blind spots as the generation. If the model is confidently wrong, it will confidently endorse its wrong answer on review.

**ReAct** interleaves reasoning traces and action steps in a single process. This is an improvement over pure action selection, but the reasoning and acting share the same context, the same weights, the same biases. There is no structural separation. The reasoning is in service of acting, not in service of evaluating whether to act.

**Inner Monologue** (CoRL 2022) adds environment feedback to the reasoning loop, which helps with grounding. But the monologue is still generated by the same process that selects actions. It is self-talk, not external observation.

Then there is the **overthinking paradox**, presented at ICLR 2025 by DeepMind researchers. They demonstrated that extended chain-of-thought reasoning can actually destroy accuracy after approximately 2-3x the baseline compute budget. More thinking makes things worse, not better. The models reason themselves into errors. This is precisely the failure mode that an observer addresses: it knows when to stop. It can recognize that additional reasoning is degrading rather than improving the answer and intervene by cutting the chain short or abstaining entirely.

## What We Built and Measured

Project Witness includes an evaluation kit: 32 tasks across four dimensions of physical reasoning.

- **Object permanence:** Does the model understand that objects continue to exist when occluded?
- **Basic physics:** Can it predict trajectories, collisions, support relations?
- **Temporal causality:** Does it correctly order cause and effect in physical scenarios?
- **Affordances:** Can it determine what actions an object or environment supports?

We evaluated two models: Gemini 2.5 Flash (a large, capable multimodal model) and Qwen2.5-72B (an open-weight model with strong general performance). Each model was tested in two conditions: baseline (direct prompting) and observer-gated (the model generates an answer and a confidence estimate; a separate observer process decides whether to emit the answer or abstain).

The results:

- **Gemini baseline:** 70.5% accuracy at 96.4% confidence.
- **Gemini with observer:** 82.3% selective accuracy, a +16.7% lift, at 62.5% coverage.
- **Qwen baseline:** 43.6% accuracy at 94.1% confidence.
- **Qwen with observer:** 44.5% selective accuracy, a +2.1% lift, at 50% coverage.
- **Calibration error** improved for both models.

The Gemini results are the headline. When the observer gates responses, allowing the model to abstain on low-confidence answers, accuracy jumps nearly 17 percentage points. The tradeoff is coverage: the system only answers 62.5% of questions. But for many applications, a system that says "I don't know" on hard questions and gets the rest right is far more valuable than one that guesses on everything and gets 30% wrong.

The Qwen results are equally informative, though less dramatic. Qwen shows almost no lift from the observer. Why? Because its confidence signal barely correlates with its accuracy. Qwen reports 94.1% average confidence while achieving 43.6% accuracy. It is confidently wrong on most questions it gets wrong. The observer can only help when there is a usable confidence gradient, when high-confidence answers are actually more likely to be correct than low-confidence ones. Qwen's confidence is nearly flat across correct and incorrect responses, so the observer has almost nothing to work with.

This is a finding, not a failure. It tells us that observer-gating is not a universal fix. It is a technique that amplifies existing signal quality. For models with well-calibrated uncertainty (or at least monotonically useful confidence), the observer provides substantial lift. For models with degenerate confidence distributions, it provides almost none. Calibration quality is a prerequisite, not a bonus.

## "Ego Removed" as Engineering Constraints

The observer layer in Project Witness operates under seven hard constraints. These are not aspirational guidelines; they are enforced invariants.

1. **No self-preservation goal.** The system has no objective to continue running, maintain its state, or resist shutdown.
2. **No activity reward.** The system receives no positive signal for producing output. Silence is not penalized.
3. **No constitution rewriting.** The principles that govern the observer cannot be modified by the observer or by any process the observer monitors.
4. **No permission escalation.** The system cannot grant itself access to resources, tools, or capabilities it was not provisioned with.
5. **No direct execution from chain-of-thought.** Reasoning traces are never directly executed as actions. There is always an explicit gate between "thinking about doing X" and "doing X."
6. **No unprovenanced memory writes.** Nothing enters long-term memory without a traceable source. The system cannot fabricate memories or inject beliefs into its own context.
7. **No assumption that action beats abstention.** The default state is "do nothing." Action requires justification; inaction does not.

Taken together, these constraints make "doing nothing" a first-class decision. Most agent frameworks treat inaction as failure. If the agent did not produce an output, something went wrong. Project Witness inverts this: the system must justify why it acted, not why it abstained. This is the operational meaning of "ego removed." The system has no drive to demonstrate its own usefulness, no incentive to produce output for the sake of producing output.

This matters for physical reasoning in particular, because physical environments are domains where the cost of a wrong action often exceeds the cost of no action. A robot that waits for more information before grasping an ambiguous object is safer than one that grasps immediately based on a 55% confidence estimate. A planning system that flags uncertainty in a structural load calculation is more valuable than one that returns a number with false precision.

## What Comes Next

**v0.5** integrates AI2-THOR, a simulation environment for interactive 3D scenes. This moves evaluation from static question-answering to dynamic interaction: can the observer improve outcomes when the model must take sequential actions in a simulated physical environment? Static benchmarks test knowledge; interactive environments test the ability to act under uncertainty, which is where the observer should provide the most value.

**v1.0** targets the full architecture: a latent dynamics world model (not just a language model prompted to reason about physics, but a model with learned physical dynamics), all five observer checks running in parallel, and commitment memory, a persistent store of what the system has observed, predicted, and verified over time.

The first milestone is narrow and falsifiable: can the system outperform a plain multimodal agent on tasks where the correct move is often not "act now" but "ask a question," "wait for more information," or "gather additional evidence before committing"? These are the tasks that current agent benchmarks largely ignore, because they are designed around the assumption that the agent should always be doing something. Real-world physical reasoning frequently requires patience.

## Closing

World models need infrastructure as much as they need research breakthroughs. The observer is one piece of that infrastructure. The evaluation kit, the calibration analysis pipeline, the memory provenance system, the decision audit trail: these are the unsexy components that determine whether a world-model system can ship as a product rather than remain a conference demo.

The field has spent enormous resources on making models bigger and training data larger. The benchmarks suggest this is not sufficient for physical reasoning. Something structural is missing. Project Witness is a bet that part of what is missing is the separation between cognition and metacognition, between generating answers and knowing when those answers are good enough to act on.

The observer does not make the model smarter. It makes the model honest about what it does and does not know. For physical reasoning, where the gap between human and machine performance remains enormous, honesty about uncertainty may be worth more than marginal improvements in raw accuracy.

---

## References

- **PhysBench:** Cheng, Y., et al. "PhysBench: Benchmarking Physical Reasoning in Large Multimodal Models." 2025.
- **IntPhys 2:** Riochet, R., et al. "IntPhys 2: Intuitive Physics Benchmark." 2025.
- **PhysUniBench:** A unified benchmark for physical understanding across vision-language models.
- **WorldBench:** Evaluating world knowledge and reasoning capabilities in foundation models.
- **CausalVQA:** Causal reasoning in visual question answering benchmarks.
- **Reflexion:** Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.
- **Self-Refine:** Madaan, A., et al. "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS 2023.
- **Inner Monologue:** Huang, W., et al. "Inner Monologue: Embodied Reasoning through Planning with Language Models." CoRL 2022.
- **LLMs Cannot Self-Correct:** Huang, J., et al. "Large Language Models Cannot Self-Correct Reasoning Yet." ICLR 2024.
- **Overthinking Paradox:** Chen, X., et al. "The Overthinking Paradox: When More Reasoning Hurts Performance." ICLR 2025.
- **NVIDIA Cosmos:** NVIDIA. "Cosmos: A Platform for World Foundation Models." 2025.
- **Meta V-JEPA 2:** Bardes, A., et al. "V-JEPA 2: Self-Supervised Video Models for Understanding and Generation." Meta AI, 2025.
- **World Labs Marble:** World Labs. "Marble: Spatial Intelligence for Generative 3D." 2025.

# Prompt-, Program-, and Context-Level Optimization of LLM Agents (2023–2026)

> Scope: optimizers that improve the text, context, or memory layer of an agent harness (prompts, instructions, few-shot demos, playbooks, skill libraries). The harness code and the model weights stay fixed. Code-rewriting self-improvers (DGM, ADAS, SICA) are out of scope.
>
> **Source-access caveat (Oct 31, 2026):** this session's network blocked arxiv.org, alphaxiv.org and huggingface.co, so I could not re-open full papers. The GEPA and ACE figures come from search-engine extracts of the papers, the ICLR 2026 pages and the GEPA GitHub README. Findings on the classic 2023–2024 papers (APE, OPRO, Promptbreeder, EvoPrompt, TextGrad, Reflexion, Voyager, ExpeL, AutoGuide, DSPy/MIPRO) are marked **[abstract, not re-fetched]**. They come from the papers' abstracts as I know them, with arXiv abstract links, and should be spot-checked before anyone quotes them as exact.

## Q1. Mechanisms and headline results of each optimizer family

### Takeaway
The field has moved through three stages:
1. **2022–2023:** score-driven search over a single prompt (APE, OPRO, Promptbreeder, EvoPrompt).
2. **2023–2024:** program-level compilation of multi-module pipelines (DSPy BootstrapFewShot/MIPRO) and "textual-gradient" methods that propagate natural-language feedback through a computation graph (TextGrad, Trace/OptoPrime).
3. **2025–2026:** reflective, trace-reading evolutionary optimizers (GEPA) and accumulating context/playbook methods (Dynamic Cheatsheet, ACE, AWM, skill libraries).

GEPA (ICLR 2026 oral) and ACE (ICLR 2026) are the two reference points in late 2026. The first published hybrids (for example GEPA-Curator, Oct 2026) report beating both.

### Cited Findings

**Single-prompt search (score-driven)**
- **APE (Zhou et al., Nov 2022; ICLR 2023):**
  - Mechanism: an LLM proposes candidate instructions from input/output demos, each candidate is scored on a held-out subset, and the best ones are resampled (iterative Monte-Carlo search).
  - Results: human-level or better instructions on 24/24 Instruction-Induction tasks and 17/21 BIG-Bench tasks. It also found a better zero-shot chain-of-thought trigger ("Let's work this out in a step by step way to be sure we have the right answer").
  - [abstract, not re-fetched] — [arXiv 2211.01910](https://arxiv.org/abs/2211.01910)
- **OPRO (Yang et al., Google DeepMind, Sep 2023; ICLR 2024):**
  - Mechanism: a meta-prompt lists earlier candidate prompts with their scores, sorted, and the LLM is asked to write a better one. This is pure scalar-score feedback.
  - Results: optimized prompts beat human-designed ones by up to 8% on GSM8K and up to 50% on Big-Bench Hard. Best-known output: "Take a deep breath and work on this problem step-by-step."
  - [abstract, not re-fetched] — [arXiv 2309.03409](https://arxiv.org/abs/2309.03409)
  - The OPRO authors report training accuracy often 5–20 points above test accuracy for the instructions found, while still mostly beating human prompts. They suggest a larger training set and early stopping. — [summary of OPRO overfitting analysis via search](https://hackernoon.com/how-overfitting-affects-prompt-optimization)
- **Promptbreeder (Fernando et al., DeepMind, Sep 2023):**
  - Mechanism: self-referential evolution. It evolves a population of task-prompts and also the "mutation-prompts" that rewrite them, with fitness measured on a training batch.
  - Results: beats Chain-of-Thought and Plan-and-Solve on arithmetic and commonsense benchmarks, and evolves complex prompts for hate-speech classification.
  - [abstract, not re-fetched] — [arXiv 2309.16797](https://arxiv.org/abs/2309.16797)
- **EvoPrompt (Guo et al., Sep 2023; ICLR 2024):**
  - Mechanism: a genetic algorithm or differential evolution in which the LLM performs crossover and mutation on prompts.
  - Results: tested on 31 datasets; beats human-engineered prompts and earlier automatic methods by up to 25% on BBH.
  - [abstract, not re-fetched] — [arXiv 2309.08532](https://arxiv.org/abs/2309.08532)

**Program-level (DSPy)**
- **DSPy (Khattab et al., Oct 2023):**
  - Mechanism: LM pipelines are written as declarative modules with signatures, and a "teleprompter"/optimizer compiles them, choosing demos and instructions per module.
  - Results: compiled pipelines beat standard few-shot prompting by over 25% (GPT-3.5) and 65% (Llama2-13b-chat).
  - [abstract, not re-fetched] — [arXiv 2310.03714](https://arxiv.org/abs/2310.03714)
- **DSPy optimizer family (official docs):**
  - BootstrapFewShot, COPRO, MIPROv2, SIMBA and GEPA change prompts and demos and leave weights untouched. BootstrapFinetune is the one that writes weights.
  - BootstrapFewShot runs the program on training examples, keeps the traces that pass the metric as demos, and costs almost nothing. Defaults: max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1.
  - MIPROv2 starts with that bootstrapping stage and then tunes instructions and demos jointly.
  - SIMBA is described as a "mini-batch SGD-flavored" search with defaults bsize=32, num_candidates=6, max_steps=8.
  - Sources: [DSPy "choosing an optimizer"](https://dspy.ai/diving-deeper/choosing-an-optimizer/); [DSPy optimizers API](https://dspy.ai/api/optimizers/); [DSPy cheatsheet](https://dspy.ai/cheatsheet/)
- **MIPRO (Opsahl-Ong et al., Jun 2024; EMNLP 2024):**
  - Mechanism: proposes instructions grounded in program and data summaries plus bootstrapped demos. A Bayesian surrogate model (TPE) then searches over instruction × demo combinations per module, with minibatch evaluation.
  - Results: beats baseline optimizers on 5 of 7 multi-stage programs, by up to 13% accuracy, using Llama-3-8B.
  - [abstract, not re-fetched] — [arXiv 2406.11695](https://arxiv.org/abs/2406.11695)

**Textual-gradient / trace-based**
- **TextGrad (Yuksekgonul et al., Jun 2024; Nature 2025):**
  - Mechanism: "automatic differentiation via text". LLM-written critiques ("textual gradients") are backpropagated through a computation graph to update prompts, code or solutions.
  - Results: GPT-4o zero-shot GPQA rose from 51% to 55%, and LeetCode-Hard improved by 20% relative.
  - [abstract, not re-fetched] — [arXiv 2406.07496](https://arxiv.org/abs/2406.07496)
- **Trace / OptoPrime (Cheng, Nie, Swaminathan, Microsoft; NeurIPS 2024):**
  - Mechanism: records the execution trace as a graph, much like autodiff, and passes a minimal subgraph plus rich feedback to an LLM optimizer (OptoPrime). OptoPrime makes one LLM call per optimization step, whereas TextGrad's calls grow linearly with graph size.
  - Results: in the Trace authors' own runs, OptoPrime and TextGrad reached similar success rates, with OptoPrime about 3x faster in wall-clock time. The authors also say TextGrad's original numbers could not be reproduced exactly with the released code. TextGrad could be implemented inside Trace, but the reverse is not possible.
  - Caveat: graphs with hundreds or more operations may be hard for OptoPrime.
  - Source: [Trace paper (NeurIPS 2024)](https://arxiv.org/html/2406.16218); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/83ba7056bce2c3c3c27e17397cf3e1f0-Abstract.html)

**GEPA (Agrawal et al., Jul 2025; ICLR 2026 Oral)**
- **Mechanism:**
  - An LLM reads full execution traces and evaluator feedback, diagnoses failures, and rewrites a module's prompt (reflective mutation).
  - The parent for each mutation is sampled from a **Pareto frontier**: candidates that are best on at least one training instance, rather than only the single best average scorer.
  - It evaluates on a minibatch, keeps the child if it improves, and can do a "system-aware merge" of complementary lineages.
  - "Actionable Side Information" (ASI), meaning evaluator diagnostics, plays the role of a gradient for text.
  - Source: [GEPA GitHub README](https://github.com/gepa-ai/gepa)
- **Versus RL (GRPO):** on Qwen3 8B, GEPA beats GRPO (24,000 rollouts, LoRA) by up to 19–20% while using up to 35x fewer rollouts. The average gain is 10% in v1 of the paper and 6% in a later version. — [ICLR 2026 page](https://iclr.cc/virtual/2026/poster/10009493); [aiweekly summary](https://aiweekly.co/alerts/gepa-prompt-optimizer-beats-grpo-by-6-uses-35x-fewer-rollouts)
- **Versus MIPROv2:** GEPA beats MIPROv2 on every benchmark and model, with aggregate gains of +14% versus +7% for MIPROv2 (v2 of the paper). Some secondary write-ups cite +13%. — [search extract of arXiv 2507.19457v2 via ICLR/alphaxiv listings](https://iclr.cc/virtual/2026/poster/10009493)
- **HotpotQA example (blog, not verified against the paper's tables):** GEPA took a 42% baseline to 62% with 6,438 rollouts, while GRPO reached 43% after 24,000 rollouts. — [Comet blog](https://www.comet.com/site/blog/gepa-ai-optimization/)
- **Benchmarks:** HotpotQA, IFBench, HoVer and PUPA (privacy-conscious delegation), with Qwen3-8B and GPT-4.1-mini. [abstract-level; table detail not re-fetched]
- **Results the README claims beyond prompts (unverified, author-reported):**
  - 100–500 evaluations versus 5,000–25,000+ for GRPO.
  - Open-source models plus GEPA beat Claude Opus 4.1 at Databricks at 90x lower cost.
  - ARC-AGI agent rose from 32% to 89% via architecture discovery.
  - Cloud-scheduling policy cut costs by 40.2%.
  - Coding-agent Jinja resolve rate rose from 55% to 82% with auto-learned skills.
  - GPT-4.1-mini on AIME 2025 rose from 46.6% to 56.6%.
  - DSPy full-program adapter on MATH rose from 67% to 93%.
  - Source: [GEPA README](https://github.com/gepa-ai/gepa)
- **Generalization to any text artifact (Feb 2026):** `optimize_anything` takes a seed candidate, an evaluator that can call `oa.log()` to send diagnostics back into reflection, a natural-language objective, and a budget (`max_metric_calls`). It covers code, agent architectures, configs and SVGs. The follow-up paper is listed as arXiv:2605.19633. — [GEPA README](https://github.com/gepa-ai/gepa)
- **Adoption (late 2026):**
  - Integrated into DSPy (`dspy.GEPA`), MLflow (`mlflow.genai.optimize_prompts()`), Comet Opik, Pydantic AI and Google ADK (`adk optimize`, shipping in Gemini Enterprise Agent Platform), with OpenAI and Hugging Face cookbooks.
  - Ships as an Agent Skill for Claude Code, Cursor and Codex.
  - The README cites 50+ production users (Shopify, Databricks, Dropbox, OpenAI, Pydantic, Nubank, Microsoft AI).
  - Source: [GEPA README](https://github.com/gepa-ai/gepa)

**Context / memory / playbook accumulation**
- **Reflexion (Shinn et al., Mar 2023; NeurIPS 2023):**
  - Mechanism: after a failed trial, the agent writes a verbal self-reflection into an episodic memory buffer that conditions the next trial. No weight updates.
  - Results: 91% pass@1 on HumanEval versus 80% for GPT-4. Large gains on AlfWorld (about +22% absolute over 12 trials) and HotpotQA (about +20%).
  - [abstract, not re-fetched] — [arXiv 2303.11366](https://arxiv.org/abs/2303.11366)
- **Voyager (Wang et al., May 2023):**
  - Mechanism: in Minecraft, an automatic curriculum, an ever-growing **skill library of executable code** indexed by description embeddings, and iterative prompting with environment feedback and self-verification.
  - Results: 3.3x more unique items, 2.3x longer distances travelled, and tech-tree milestones up to 15.3x faster than prior SOTA. Skills transfer to a new world.
  - [abstract, not re-fetched] — [arXiv 2305.16291](https://arxiv.org/abs/2305.16291)
- **ExpeL (Zhao et al., Aug 2023; AAAI 2024):**
  - Mechanism: gathers experience across training tasks, extracts natural-language **insights** by comparing successes with failures, and recalls similar successful trajectories at inference time.
  - Results: improves over ReAct and Reflexion baselines on HotpotQA, ALFWorld and WebShop, and shows transfer to FEVER.
  - [abstract, not re-fetched; exact numbers not verified] — [arXiv 2308.10144](https://arxiv.org/abs/2308.10144)
- **AutoGuide (Fu et al., Mar 2024):**
  - Mechanism: extracts **state-aware guidelines** ("if in situation X, do Y") from offline trajectories and retrieves the ones relevant to the current state.
  - Results: reported gains on web and embodied benchmarks (e.g., WebShop, WebArena).
  - [abstract, not re-fetched; exact numbers not verified] — [arXiv 2403.08978](https://arxiv.org/abs/2403.08978)
- **Agent Workflow Memory (Wang et al., Sep 2024; ICML 2025):**
  - Mechanism: induces reusable **workflows** (sub-routines) from past trajectories, offline or online, and adds them to the agent's memory.
  - Results: +24.6% relative step-wise success on Mind2Web and +51.1% relative success on WebArena over the top published autonomous method. It also beats methods that use human-written workflows by 7.9% and reduces steps per solved task.
  - Source: [ICML 2025 proceedings](https://proceedings.mlr.press/v267/wang25bx.html); [arXiv 2409.07429](https://arxiv.org/html/2409.07429v1)
- **Dynamic Cheatsheet (Suzgun, Yuksekgonul, Bianchi, Jurafsky, Zou; Apr 2025; EACL 2026):**
  - Mechanism: a test-time, self-curated persistent memory of concise, transferable strategies and code snippets. It is not a store of full transcripts and needs no labels.
  - Results:
    - Claude 3.5 Sonnet more than doubled AIME accuracy (third-party summary: 23% to 50% on AIME 2024).
    - GPT-4o on Game of 24 went from about 10% to 99% after it stored and reused a Python brute-force solver.
    - Small models such as GPT-4o-mini benefit only a little.
  - Source: [arXiv 2504.07952](https://arxiv.org/html/2504.07952v1); [EACL 2026](https://preview.aclanthology.org/ingest-eacl/2026.eacl-long.333)
- **ACE — Agentic Context Engineering (Zhang et al., Stanford/SambaNova/UC Berkeley, Oct 2025; ICLR 2026):**
  - Mechanism:
    - Contexts are treated as evolving **playbooks** of itemized bullets.
    - A **Generator** produces trajectories, a **Reflector** extracts lessons from successes and failures, and a **Curator** merges small **delta** bullet entries into the playbook deterministically.
    - A "grow-and-refine" step deduplicates and prunes.
    - It builds on Dynamic Cheatsheet's architecture.
  - Results:
    - +10.6% on agent tasks (AppWorld) and +8.6% on finance benchmarks (FiNER, Formula), averaged over offline (system-prompt) and online (memory) adaptation.
    - On AppWorld, ReAct+ACE beat ReAct+GEPA by 11.9 points offline.
    - It matched the top-ranked production agent on the AppWorld leaderboard (IBM CUGA) on overall average and beat it on the harder test-challenge split, while using a smaller open-source model (DeepSeek-V3.1).
    - It works without labeled supervision by using natural execution feedback.
  - Sources: [ACE PDF](https://arxiv.org/pdf/2510.04618); [ICLR 2026 page](https://iclr.cc/virtual/2026/10018667); [search extract](https://medium.com/superagentic-ai/agentic-context-engineering-prompting-strikes-back-c5beade49acc); code at [github.com/ace-agent/ace](https://github.com/ace-agent/ace)
- **Anthropic Agent Skills (Oct 2025; open standard Dec 2025):**
  - Mechanism: a skill is a folder with SKILL.md (name and description frontmatter plus instructions), with optional scripts/, references/ and assets/.
  - **Progressive disclosure:** only the name and description (about 30–50 tokens per skill) load at startup, the full body loads when the skill triggers, and supporting files load on demand. This makes it a human- or agent-curated, retrieval-on-demand skill library in the Voyager/AWM tradition.
  - Adoption: open standard at agentskills.io, reportedly adopted by 26+ platforms (Codex, Gemini CLI, Copilot, Cursor, VS Code).
  - Caveats: the spec recommends keeping SKILL.md under 500 lines; runtime portability is incomplete; skills are not eligible for Zero Data Retention.
  - Sources (secondary): [inference.sh](https://inference.sh/blog/skills-agent-skills-overview); [Rubinlake tech radar](https://rubinlake.com/en/technology-radar/developer-ai-and-delivery/agent-skills); [SwirlAI](https://www.newsletter.swirlai.com/p/agent-skills-progressive-disclosure?open=false)
  - Skill libraries are also an *output* of optimizers: GEPA's README reports "auto-learned skills" lifting a coding agent's Jinja resolve rate from 55% to 82%. — [GEPA README](https://github.com/gepa-ai/gepa)

**2026 successors and new directions**
- **GEPA-Curator (arXiv 2610.37651, Oct 2026 preprint):** combines GEPA-style evolution with ACE-style curation. Reported average gains are +8.1% over GEPA and +12.6% over ACE across four test sets, and up to +20.7% and +21.2% respectively. — [arXiv 2610.37651](https://arxiv.org/html/2610.37651)
- **TextReg (arXiv 2605.21318, May 2026; NeurIPS 2026):** regularized text-space optimization against "prompt distributional overfitting". It reports up to +11.8% over TextGrad out-of-distribution. — [arXiv 2605.21318](https://arxiv.org/pdf/2605.21318); [NeurIPS 2026](https://neurips.cc/virtual/2026/164764)
- **"The Optimizer Is the Agent" (arXiv 2608.06714, Aug 2026):** reasoning-driven, agentic search across prompts, programs and ML workflows. — [arXiv 2608.06714](https://arxiv.org/pdf/2608.06714)
- **Cross-Modal Visual Feedback for APO (arXiv 2607.24354, Jul 2026):** visual feedback for multimodal prompt optimization; reports the smallest validation-to-test gap across eight benchmarks. — [arXiv 2607.24354](https://arxiv.org/pdf/2607.24354)
- **JetBrains Koog agents-optimization library (Oct 2026 blog):** a prompt self-optimization library for Kotlin agents. — [JetBrains Research blog](https://blog.jetbrains.com/research/2026/10/koog-agents-optimization-library-prompts/)
- **Unverified mentions:** SePO (Jun 2026; optimizes the prompt-optimizer agent's own system prompt) and a "memory-augmented APO" that stores refinement insights across iterations. Both appear only in a community list. — [awesome-prompts list](https://github.com/ai-boost/awesome-prompts)

### Inferences
- Methods split along two axes:
  - **What is optimized:** a fixed-size instruction/demo set (APE, OPRO, MIPROv2, GEPA) versus a growing context or memory (Reflexion, ExpeL, AWM, DC, ACE, skill libraries).
  - **When it runs:** offline against a train/val set (DSPy, GEPA, ACE offline) versus online at test time (Reflexion, DC, ACE online, AWM online).
- For agents, the practical late-2026 recipe is:
  - GEPA (or MIPROv2) for the static system prompt and per-module instructions.
  - An ACE/AWM-style curated playbook or skill library for knowledge that keeps accumulating.
  - Agent Skills (progressive disclosure) as the storage and loading format so the library doesn't flood the context.
- GEPA-Curator suggests the two families are complementary rather than competing.

### Gaps
- I could not open the GEPA or ACE full-text tables (arXiv blocked), so per-benchmark numbers, exact split sizes, and the Pareto-vs-greedy ablation magnitudes are not recorded here.
- I found no primary detail on SIMBA's algorithm beyond the docs' defaults. (From memory, SIMBA samples mini-batches, finds high-variance examples, and has the LLM write self-reflective rules or add demos; I could not verify this.)
- I did not verify the exact numbers for ExpeL and AutoGuide.

## Q2. Execution traces and natural-language feedback vs scalar scores

### Takeaway
Optimizers that let an LLM read traces and textual feedback (GEPA, TextGrad, Trace, ACE, Reflexion) learn far more per rollout than those that only see a scalar (OPRO, APE, EvoPrompt, GRPO). GEPA's explicit argument is that serialized language traces are a richer learning signal than a sparse scalar reward, and vendor docs for judge alignment now explicitly ask for written rationales.

### Cited Findings
- GEPA does not collapse results into a scalar. An LLM reads full execution traces and diagnoses failures, and evaluator "Actionable Side Information" serves as a gradient for text. `optimize_anything` evaluators can call `oa.log()` to send diagnostic output back into reflection. — [GEPA README](https://github.com/gepa-ai/gepa)
- GEPA's rollout efficiency versus GRPO (up to 35x fewer rollouts while scoring higher) is the main empirical evidence that textual reflection beats scalar-reward RL on sample efficiency. — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009493)
  - Caveat: the 35x compares rollout counts only, not total compute including reflection-LM calls. It is a best case, not a typical ratio. — [aiweekly summary](https://aiweekly.co/node/11821)
- OPRO uses only (prompt, score) pairs in its meta-prompt. [abstract, not re-fetched] — [arXiv 2309.03409](https://arxiv.org/abs/2309.03409)
- Trace argues that rich feedback plus the execution graph (a minimal subgraph) lets a single LLM call make useful updates, in the way autodiff does for numeric gradients. TextGrad instead propagates text feedback node by node. — [Trace paper](https://arxiv.org/html/2406.16218)
- ACE's Reflector extracts lessons from execution feedback (success or failure signals, environment outputs) and can adapt without ground-truth labels. Even so, label-free signals mean quality depends on how reliable that feedback is. — [ICLR 2026 ACE](https://iclr.cc/virtual/2026/10018667); [ACE PDF](https://arxiv.org/pdf/2510.04618)
- MLflow's GEPA judge-alignment docs recommend at least 10 human-labeled traces, a mix of positive and negative labels, and **written rationales** where possible ("natural language feedback is highly recommended"). — [MLflow GEPA Alignment Optimizer](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/gepa/)
- Dropbox's relevance-judge work: annotators gave 1–5 ratings **with short explanations**, and GEPA generated example-level feedback wherever the model disagreed with a human. — [Dropbox tech blog](https://dropbox.tech/machine-learning/optimizing-dropbox-dash-relevance-judge-with-dspy)
- Reflexion shows that verbal self-reflection stored in memory beats retrying with only a binary success signal. [abstract, not re-fetched] — [arXiv 2303.11366](https://arxiv.org/abs/2303.11366)

### Inferences
- When designing the metric for any of these optimizers, return a *feedback string* (why it failed: which constraint, which tool error, the gold answer, a judge rationale), not just a score. This is the single biggest lever on sample efficiency for GEPA-style methods.
- Scalar-only methods (OPRO/APE/EvoPrompt) are mostly superseded for agents, but they remain usable when traces are unavailable or uninformative.

### Gaps
- I found no controlled ablation, visible without the full GEPA paper, that isolates "score + feedback text" against "score only" inside the same optimizer. The GEPA paper is believed to include feedback-function ablations, but I could not verify them.

## Q3. Sample efficiency: rollouts, labeled examples, splits

### Takeaway
Prompt and context optimizers work with tens to a few hundred labeled examples and hundreds to a few thousand rollouts, whereas GRPO-style RL needs about 5k–25k rollouts. ACE further cuts adaptation rollouts and latency relative to GEPA and Dynamic Cheatsheet.

### Cited Findings
- **GEPA:** 100–500 evaluations versus 5,000–25,000+ for GRPO (README claim). — [GEPA README](https://github.com/gepa-ai/gepa)
  - In the paper, the GRPO baseline used 24,000 rollouts with LoRA, and GEPA used up to 35x fewer. — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009493)
  - HotpotQA example: GEPA used 6,438 rollouts versus 24,000 for GRPO (blog figure, not verified against the paper). — [Comet blog](https://www.comet.com/site/blog/gepa-ai-optimization/)
- **ACE:** 86.9% lower adaptation latency on average than existing adaptive methods, with fewer rollouts and lower dollar cost. — [ACE PDF via search](https://arxiv.org/pdf/2510.04618)
  - README breakdown: offline AppWorld −82.3% latency and −75.1% rollouts versus GEPA; online FiNER −91.5% latency and −83.6% token cost versus Dynamic Cheatsheet. These come from a repo mirror, so treat them with some caution. — [ACE repo](https://github.com/ace-agent/ace)
- **DSPy defaults:** BootstrapFewShot keeps at most 4 bootstrapped and 16 labeled demos with 1 round. SIMBA defaults to a minibatch of 32, 6 candidates and 8 steps. — [DSPy docs](https://dspy.ai/diving-deeper/choosing-an-optimizer/)
- **Judge alignment:** MLflow recommends a minimum of 10 human-labeled traces for GEPA alignment. — [MLflow docs](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/gepa/)
- **Small validation sets are noisy:** with few validation instances it is hard to tell whether a validation-to-test gap is estimation noise or overfitting. — [HbBoPs, arXiv 2412.07820](https://arxiv.org/pdf/2412.07820)
- **Example test-set size:** one 2026 agentic-optimizer study used a 150-example test set. — [arXiv 2608.06714](https://arxiv.org/pdf/2608.06714)
- **Common practice (not re-verified):** DSPy and GEPA papers typically use about 150 train, about 300 validation, and a few hundred test examples per benchmark. DSPy's docs have advised splits weighted toward validation (e.g., 20% train / 80% val) for prompt optimizers, because prompt optimizers overfit small train sets. [from memory; GEPA paper splits not re-fetched]

### Inferences
- Rule of thumb: GEPA or MIPROv2 is viable with roughly 50–300 labeled examples and a budget of a few hundred to a few thousand metric calls.
- ACE/DC-style online methods need no labels, only a reliable execution signal.
- RL needs one to two orders of magnitude more rollouts, and weight access.
- When rollouts are expensive (long-horizon agents, paid APIs, real environments), reflective text optimization is the default choice.

### Gaps
- The exact GEPA per-benchmark rollout budgets and train/val/test sizes were not retrievable.
- I found no independent accounting of total token cost (reflection plus rollouts) for GEPA versus GRPO.

## Q4. Which approach wins when; overfitting, generalization, transfer

### Takeaway
- **Small labeled sets with expensive rollouts:** GEPA beats MIPROv2 and GRPO.
- **Agentic or domain-knowledge-heavy tasks with long horizons:** ACE-style growing playbooks beat GEPA's compact prompts.
- **Multimodal:** visual-feedback variants are emerging.

Overfitting to the train or val set is real and documented (5–20 point train-test gaps in OPRO; distributional overfitting in iterative rewriting). Cross-model transfer of optimized prompts is under-studied.

### Cited Findings
- **GEPA versus MIPROv2:** GEPA wins on every benchmark and model tested, with an aggregate +14% versus +7%. — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009493)
- **GEPA versus GRPO:** GEPA wins by an average of 6–10% and up to about 20% with far fewer rollouts (Qwen3-8B). — [aiweekly](https://aiweekly.co/alerts/gepa-prompt-optimizer-beats-grpo-by-6-uses-35x-fewer-rollouts)
- **ACE versus GEPA on AppWorld:** ACE wins by 11.9 points offline. ACE argues that GEPA-style optimizers suffer **brevity bias**: they converge to short, generic prompts that drop domain-specific heuristics. — [ACE (search extract)](https://medium.com/superagentic-ai/agentic-context-engineering-prompting-strikes-back-c5beade49acc); [ACE PDF](https://arxiv.org/pdf/2510.04618)
- **GEPA-Curator versus both:** +8.1% average over GEPA and +12.6% over ACE across four test sets. This is a preprint. — [arXiv 2610.37651](https://arxiv.org/html/2610.37651)
- **Trace versus TextGrad:** similar accuracy, with OptoPrime about 3x faster. — [Trace](https://arxiv.org/html/2406.16218)
- **Small models gain less from self-curated memory:** GPT-4o-mini gets limited benefit from Dynamic Cheatsheet. — [DC](https://arxiv.org/html/2504.07952v1)
- **Overfitting evidence:**
  - OPRO: train accuracy 5–20 points above test. — [summary](https://hackernoon.com/how-overfitting-affects-prompt-optimization)
  - TextReg (2026): iterative LLM-feedback rewriting produces longer prompts that "accumulate narrow sample-specific rules and generalize poorly". Its regularizer gives up to +11.8% over TextGrad out-of-distribution. — [TextReg](https://arxiv.org/pdf/2605.21318)
  - In the 2026 "Optimizer Is the Agent" study, a domain-specific addition raised training accuracy but lowered validation accuracy below baseline. An *unvalidated* prompt (52.0% test) beat the best-validated one (49.3% test) on a 150-example test set, which shows how noisy validation selection is. — [arXiv 2608.06714](https://arxiv.org/pdf/2608.06714)
  - Black-box prompt selection: the validation-optimal prompt is often not test-optimal. — [arXiv 2412.07820](https://arxiv.org/pdf/2412.07820)
  - Counterpoint (2023 theory): discrete prompts under an LM prior have fairly tight generalization bounds, so they overfit less than one might expect. — [ICML 2023 workshop](https://icml.cc/virtual/2023/29334)
- **Multimodal:** Cross-Modal Visual Feedback (2026) rewrites prompts around recurring visual patterns rather than isolated validation errors, and reports the smallest average validation-to-test gap across eight benchmarks. — [arXiv 2607.24354](https://arxiv.org/pdf/2607.24354)
- **Transfer across models:**
  - Dropbox used GEPA to adapt a judge prompt to a cheaper open-weight model, which implies re-optimizing per target model rather than reusing the prompt as is. — [Dropbox](https://dropbox.tech/machine-learning/how-we-turned-ai-evaluations-into-better-responses-in-dash-chat)
  - Databricks reported open models plus GEPA beating Claude Opus 4.1 at 90x lower cost. — [GEPA README](https://github.com/gepa-ai/gepa)

### Inferences
Decision guide:
- **(a) Few labels, costly rollouts, multi-module DSPy-style program:** GEPA. Fall back to MIPROv2 or BootstrapFewShot when demos matter more than instructions or the budget is tiny.
- **(b) Long-horizon tool agents in domains with many idiosyncratic rules (AppWorld, finance XBRL):** an ACE-style incremental playbook, offline and then online.
- **(c) No labels but a reliable execution signal:** DC or ACE online, Reflexion within an episode, AWM for web workflows.
- **(d) Re-usable procedures:** a skill library (Voyager, AWM, Agent Skills) with progressive disclosure.

Always hold out a test set separate from the optimizer's validation set, and re-optimize when switching models.

### Gaps
- I found no systematic study of whether GEPA- or ACE-optimized prompts transfer across model families without re-optimization. This is an open gap.
- I found no head-to-head GEPA versus ACE comparison on non-agent QA (HotpotQA/IFBench) in the ACE paper beyond AppWorld and finance.

## Q5. Incremental delta updates vs monolithic rewrites (stability)

### Takeaway
ACE gives the clearest evidence: full LLM rewrites of an accumulated context can suffer **context collapse**, abruptly compressing a rich playbook into a tiny summary and dropping accuracy below the no-adaptation baseline. Itemized delta bullets merged by a deterministic curator, with grow-and-refine deduplication, are more stable and cheaper. The 2026 TextReg results show the opposite failure mode for unregularized iterative rewriting: bloat with narrow rules.

### Cited Findings
- **Context collapse (ACE case study on AppWorld, Dynamic-Cheatsheet-style monolithic rewriting):** at one step the context shrank from 18,282 tokens (accuracy 66.7) to 122 tokens, and accuracy fell to 57.1. That is worse than the 63.7 baseline with no adaptation. — [ACE (search extract)](https://arxiv.org/pdf/2510.04618)
- **Brevity bias:** the tendency of prompt optimizers to collapse toward short, generic prompts that drop domain insights. — [ACE](https://arxiv.org/pdf/2510.04618)
- **ACE's fix:** structured, incremental updates. The Curator merges small "delta" bullet entries (with metadata such as helpful/harmful counters) into the playbook, and grow-and-refine prunes redundancy. This "preserves detailed knowledge and scales with long-context models" and enables the 82–92% latency and cost cuts above. — [ACE ICLR](https://iclr.cc/virtual/2026/10018667); [DAIR.AI summary](https://academy.dair.ai/papers/agentic-context-engineering-ace)
- **Opposite failure:** iterative rewriting without regularization grows longer and accumulates sample-specific rules. — [TextReg](https://arxiv.org/pdf/2605.21318)
- **GEPA's design:** it keeps a population and a Pareto frontier (with per-instance best candidates), so a bad rewrite is simply not selected. Stability comes from selection rather than from incremental editing. — [GEPA README](https://github.com/gepa-ai/gepa)
- **Agent Skills:** practitioner guidance recommends narrow skills with precise descriptions, version control, and peer review of skill changes. This is the software-engineering form of delta updates. — [Rubinlake tech radar](https://rubinlake.com/en/technology-radar/developer-ai-and-delivery/agent-skills)

### Inferences
- For long-lived agent playbooks, prefer append, edit and delete operations on itemized entries (with provenance and helpful/harmful counts) over "rewrite the whole memory".
- Gate every change on a held-out evaluation, and dedupe and prune periodically to avoid both collapse and bloat.
- GEPA-Curator (2026) suggests combining population-based selection with curated incremental memory.

### Gaps
- The ACE result is a single documented case study. I found no broad replication of how often context collapse happens across tasks and models.

## Q6. Applying these optimizers to LLM-as-judge / evaluator prompts

### Takeaway
Optimizing judges is now a mainstream use of DSPy and GEPA. Production reports show large agreement gains when judges are optimized against human labels with rationales. The evidence is mostly vendor and case-study, not controlled benchmarks.

### Cited Findings
- **Academic baseline (Dec 2024):** a comparative study of DSPy teleprompters (BootstrapFewShot, MIPROv2, COPRO, etc.) aligns a hallucination-detection judge to human labels on HaluBench (about 15k labeled triplets). It optimizes the judge prompt directly against ground truth rather than calibrating its scores after the fact. — [arXiv 2412.15298](https://arxiv.org/abs/2412.15298v1)
- **MLflow:** offers GEPA, SIMBA and MemAlign (the default, a dual-memory few-shot method) as judge-alignment optimizers. GEPA reflects on traces and human feedback to propose better judge instructions. — [MLflow judge alignment](https://www.mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/alignment/); [MLflow GEPA](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/gepa/)
- **Dropbox Dash relevance judge:** DSPy/GEPA cut NMSE by 45% (8.83 to 4.86) relative to the hand-written prompt, and enabled moving to a cheaper open-weight model. — [Dropbox tech blog](https://dropbox.tech/machine-learning/optimizing-dropbox-dash-relevance-judge-with-dspy); [ZenML case summary](https://www.zenml.io/llmops-database/systematic-prompt-optimization-for-production-relevance-judges-using-dspy)
- **Nubank:** eval accuracy rose from 68.88% to 88.89%, and Cohen's κ from 0.00 to 0.745, after GEPA optimization (reported in the GEPA README). — [GEPA README](https://github.com/gepa-ai/gepa)
- **Microsoft AI (MAI-Thinking-1):** used GEPA/DSPy to tune an LLM-judge prompt that filtered Code pages in pre-training data. — [GEPA README](https://github.com/gepa-ai/gepa)
- **Practitioner caveat (AI Engineer Europe 2026):** when calibrating a judge with GEPA against annotations, keep human-calibration labels separate from any generated reference labels. — [AI Engineer talk](https://www.ai.engineer/talks/X4dEHRzBLmc-judge-judge-building-llm-evaluators-that-actually)
- **Vendor product:** LayerLens "Judge Optimization" closes the gap between automated scores and human judgment (marketing source). — [LayerLens](https://layerlens.ai/blog/judge-optimization-stratix-enterprise)

### Inferences
- Judge optimization is an ideal fit for GEPA. Labels are cheap and small (about 10–200), and human rationales are exactly the "rich textual feedback" GEPA exploits. The metric (agreement or κ) is also cheap to compute.
- The risk is Goodharting: an agent optimized against an optimized judge. Keep separate held-out human-labeled sets for the judge and for the agent.

### Gaps
- I found no peer-reviewed controlled benchmark comparing GEPA, MIPROv2, SIMBA and MemAlign for judge alignment across many tasks. The evidence is case studies and docs.

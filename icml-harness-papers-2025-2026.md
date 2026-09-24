# Harness-related papers at ICML, 2025 and 2026

Scope: ICML main conference only (the `icml` venue in CSrankings, i.e. DBLP `ICML`,
`ICML (1)`, `ICML (2)`, `ICML (3)`). Workshops, ICML-affiliated events and arXiv-only
preprints are excluded, as are other CSrankings venues.

"Harness-related" here means the paper treats a *harness* as a first-class object: the
evaluation/agent/measurement scaffolding that wraps a model or agent. Titles that merely
use "harness" as a verb ("Harnessing X for Y") are listed separately as non-matches,
because they dominate a keyword search and are not about harnesses.

## Method

- Venue definition taken from `util/csrankings.py` in `emeryberger/CSrankings`
  (branch `gh-pages`), which is the authoritative area-to-DBLP-venue mapping.
- ICML 2025: PMLR volume 267 (42nd ICML, Vancouver, 13–19 July 2025), 3,330 paper pages,
  plus the ICML virtual-site dump (3,459 records: 3,339 posters + 120 orals,
  3,339 distinct titles) which includes abstracts.
- ICML 2026: the ICML virtual-site dump (6,797 records, 6,646 distinct accepted papers,
  decisions `Accept (regular)` / `Accept (spotlight)`). Abstracts were not in the dump, so
  all 6,646 paper pages were fetched individually to recover them.
- Search was over title + abstract for `harness`, `harnesses`, `harnessed`, `harnessing`,
  then hand-classified into noun (harness-as-artifact) and verb senses. Full paper bodies
  were not searched.
- As of 2026-09-24 the ICML 2026 PMLR volume is not yet published, so 2026 entries cite the
  ICML virtual-site pages rather than PMLR.

## ICML 2025: no matches

Zero harness-as-artifact papers. All 15 occurrences of the exact words `harness`/`harnesses`
across the 3,339 titles and abstracts are verb usage ("we harness the discovered causal
model", "Federated learning harnesses the power of distributed optimization", ...).

The four "Harnessing" titles, all verb usage and therefore not matches:

| Title | PMLR |
| --- | --- |
| Navigating Conflicting Views: Harnessing Trust for Learning | [v267/lu25a](https://proceedings.mlr.press/v267/lu25a.html) |
| Heterogeneous Treatment Effect in Time-to-Event Outcomes: Harnessing Censored Data with Recursively Imputed Trees | [v267/meir25a](https://proceedings.mlr.press/v267/meir25a.html) |
| Harnessing Heterogeneous Statistical Strength for Personalized Federated Learning via Hierarchical Bayesian Inference | [v267/thapa25a](https://proceedings.mlr.press/v267/thapa25a.html) |
| TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization | [v267/zhu25c](https://proceedings.mlr.press/v267/zhu25c.html) |

## ICML 2026: 7 matches

### Harness is the subject of the paper

**VeRO: A Harness for Agents to Optimize Agents** — Varun Ursekar, Apaar Shanker,
Veronica Chatrath, Yuan Xue, Samuel Denton.
[poster/60518](https://icml.cc/virtual/2026/poster/60518)

The one paper at either year whose entire contribution is a harness. Targets "agent harness
optimization": a coding agent iteratively improves a target agent by editing its code. VeRO
(Versioning, Rewards, and Observations) is an *outer* harness giving versioned snapshots,
budget-controlled evaluation and structured execution traces over *target* harnesses, paired
with VeRO-Bench. Argues harness optimization differs from ordinary software engineering
because harnesses interleave deterministic code with stochastic LLM completions. Code at
`github.com/scaleapi/vero`.

**Position: Agent Evaluation Should Be Agentified for Openness, Standardization, and
Reproducibility** — Xiaoyuan Liu, Tianneng Shi, Wenbo Guo, Dawn Song.
[poster/67210](https://icml.cc/virtual/2026/poster/67210)

Position paper whose target is the harness itself: existing benchmarks rely on "fixed,
LLM-centric harnesses that require heavy integration, create test-production mismatch, and
limit fair comparison across diverse agent designs."

### Harness robustness or harness drift is the object of study

**RubricRobustness: Evaluating the Sensitivity of Rubrics-Based Benchmarks to Simple
Perturbations** — Manasi Sharma.
[poster/65214](https://icml.cc/virtual/2026/poster/65214)

Sensitivity analysis of the robustness of rubric-and-LLM-judge *evaluation harnesses*.

**Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking**
(spotlight) — Zhicheng Fang, Jingjie Zheng, Chenxu Fu, Wei Xu.
[poster/65657](https://icml.cc/virtual/2026/poster/65657)

Motivated by drift "in datasets, harnesses, and judging protocols"; contributes a multi-agent
pipeline that turns jailbreak papers into modules runnable inside one unified harness.

**Faults in Our Formal Benchmarking: Dataset Defects and Evaluation Failures in Lean Theorem
Proving** — Pawan Sasanka Ammanamanchi, Siddharth Bhat, Stella Biderman.
[poster/62980](https://icml.cc/virtual/2026/poster/62980)

Audits five Lean benchmarks; a stated failure mode is that evaluation harnesses are not
robust to trivial or adversarial solutions.

### Harness is a named contributed artifact or an experimental variable

**Implicit Intelligence — Evaluating Agents on What Users Don't Say** — Ved Sirdeshmukh,
Marc Wetter. [poster/64912](https://icml.cc/virtual/2026/poster/64912)

Contributes Agent-as-a-World (AaW), "a harness where interactive worlds are defined in
human-readable YAML files and simulated by language models."

**BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics** — Dionizije Fa,
Marko Culjak, Bruno Pandza, Mateo Cupic.
[poster/66549](https://icml.cc/virtual/2026/poster/66549)

Evaluates frontier models "across multiple agent harnesses", making the harness an explicit
experimental factor.

## ICML 2026: adjacent, harness released but not studied

These ship an evaluation or measurement harness as a side artifact. Include them only if you
want the loosest reading of "harness-related".

| Title | Authors | Link |
| --- | --- | --- |
| Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections (spotlight) | Borchmann, Van Landeghem, Turski, Padarha, Kearns, Mahdi et al. | [poster/62732](https://icml.cc/virtual/2026/poster/62732) |
| Benchmarking Reward Hack Detection in Code Environments via Contrastive Analysis | Deshpande, Kannappan, Qian | [poster/63139](https://icml.cc/virtual/2026/poster/63139) |
| BFCL Audio: An Audio Function Calling Evaluation for Large Language Models | Mao, Ghai, Dawoodani, Ginart, Patil, Emmons et al. | [poster/61489](https://icml.cc/virtual/2026/poster/61489) |
| Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines | Lambert, Luccioni | [poster/62967](https://icml.cc/virtual/2026/poster/62967) |

## ICML 2026: verb-usage titles, not matches

DIVA: Harnessing the Representation Divergence in Unified Multimodal Models for Mutual
Reinforcement ([63698](https://icml.cc/virtual/2026/poster/63698)) · EVOLVING ROLLOUTS:
Harnessing Historical Experience for Web Agent Evolution in Reinforcement Learning
([61544](https://icml.cc/virtual/2026/poster/61544)) · SE3Set: Harnessing Equivariant
Hypergraph Neural Networks for Molecular Representation Learning
([68832](https://icml.cc/virtual/2026/poster/68832)) · Harnessing Uncertainty:
Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents
([63273](https://icml.cc/virtual/2026/poster/63273)) · Harnessing Non-Adversarial Robustness
in Large Language Models ([66057](https://icml.cc/virtual/2026/poster/66057)) · Harnessing
Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation
Shaping ([62434](https://icml.cc/virtual/2026/poster/62434)) · Harnessing Spectrum Video for
Subject-Level Few-Shot and Cross-Montage EEG Generalization
([62262](https://icml.cc/virtual/2026/poster/62262)) · Better, Faster: Harnessing
Self-Improvement in Large Reasoning Models ([64514](https://icml.cc/virtual/2026/poster/64514))

## Takeaway

Harness work at ICML is brand new: nothing in 2025, seven papers in 2026, and every one of
them is about LLM-agent or LLM-evaluation harnesses rather than the software-testing sense.
The fuzzing-harness literature that dominates the term elsewhere sits in software engineering
and security venues (FSE, ISSTA, and similar), not ICML.

# Harness-related papers at ICML, 2025 and 2026

For the categorization of the full ICML corpus for these two years, see
`icml-2025-2026-category-breakdown.md`.

Scope: ICML main conference only (the `icml` venue in CSrankings, i.e. DBLP `ICML`,
`ICML (1)`, `ICML (2)`, `ICML (3)`). Workshops, ICML-affiliated events and arXiv-only
preprints are excluded, as are other CSrankings venues.

"Harness-related" here means the paper uses *harness* as a noun: the evaluation, agent or
measurement scaffolding that wraps a model or agent. Titles that use "harness" only as a verb
("Harnessing X for Y") are listed separately as non-matches, because they dominate a keyword
search and are not about harnesses.

**Headline: ICML 2025 has zero harness papers. ICML 2026 has 10, plus 4 that release a
harness as a side artifact. Every one is an LLM-agent or LLM-evaluation harness; none is a
harness in the software-testing or fuzzing sense.**

## Method

- Venue definition taken from `util/csrankings.py` in `emeryberger/CSrankings`
  (branch `gh-pages`), the authoritative area-to-DBLP-venue mapping.
- ICML 2025: PMLR volume 267 (42nd ICML, Vancouver, 13–19 July 2025), 3,330 paper pages,
  plus the ICML virtual-site dump (3,459 records: 3,339 posters + 120 orals,
  3,339 distinct titles), which carries abstracts.
- ICML 2026: the ICML virtual-site dump (6,797 records, 6,646 distinct accepted papers,
  decisions `Accept (regular)` / `Accept (spotlight)`). Abstracts are absent from that dump,
  so all 6,646 paper pages were fetched individually to recover them.
- Detection: every occurrence of `harness`, `harnesses`, `harnessed`, `harnessing` and any
  hyphenated compound (`APE-Harness`, `eval-harness`, …) in title + abstract was enumerated
  and hand-classified into noun and verb senses. 33 noun-form occurrences in 2026 across 24
  papers; 15 in 2025. Full paper bodies were not searched, so a paper that only mentions its
  harness in the body would be missed.
- Category labels below are mine; the ICML topic column is the conference's own label from the
  virtual-site metadata.
- As of 2026-09-24 the ICML 2026 PMLR volume is not published yet, so 2026 entries cite
  `icml.cc` virtual pages rather than PMLR.

## ICML 2025: no matches

Zero harness papers. All 15 noun-form occurrences of `harness`/`harnesses` across the 3,339
titles and abstracts are verb usage: "we harness the discovered causal model", "Federated
learning harnesses the power of distributed optimization", "an algorithm that harnesses
quantum signals", and so on. No hyphenated harness compounds appear at all.

The four "Harnessing" titles, all verb usage and therefore not matches:

| Title | PMLR |
| --- | --- |
| Navigating Conflicting Views: Harnessing Trust for Learning | [v267/lu25a](https://proceedings.mlr.press/v267/lu25a.html) |
| Heterogeneous Treatment Effect in Time-to-Event Outcomes: Harnessing Censored Data with Recursively Imputed Trees | [v267/meir25a](https://proceedings.mlr.press/v267/meir25a.html) |
| Harnessing Heterogeneous Statistical Strength for Personalized Federated Learning via Hierarchical Bayesian Inference | [v267/thapa25a](https://proceedings.mlr.press/v267/thapa25a.html) |
| TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization | [v267/zhu25c](https://proceedings.mlr.press/v267/zhu25c.html) |

## ICML 2026 by category

| # | Category | Papers |
| --- | --- | --- |
| 1 | Agent harness engineering and optimization | 2 |
| 2 | Harness for RL training | 1 |
| 3 | Agent-evaluation harness infrastructure | 3 |
| 4 | Harness as a source of measurement error | 2 |
| 5 | Safety and security evaluation harness | 1 |
| 6 | Formal-mathematics proof-engineering harness | 1 |
| — | Adjacent: harness released, not studied | 4 |

### 1. Agent harness engineering and optimization

The harness itself is the artifact being built or automated away.

**VeRO: A Harness for Agents to Optimize Agents** — Varun Ursekar, Apaar Shanker,
Veronica Chatrath, Yuan Xue, Samuel Denton. ICML topic: Deep Learning → Large Language Models.
[poster/60518](https://icml.cc/virtual/2026/poster/60518)

The purest harness paper at either year. Targets "agent harness optimization": a coding agent
iteratively improves a target agent by editing its code. VeRO (Versioning, Rewards,
Observations) is an *outer* harness supplying versioned snapshots, budget-controlled
evaluation and structured execution traces over *target* harnesses, paired with VeRO-Bench.
Argues harness optimization is unlike ordinary software engineering because harnesses
interleave deterministic code with stochastic LLM completions. Code at
`github.com/scaleapi/vero`.

**Meta Context Engineering via Agentic Skill Evolution** — Haoran Ye, Xuning He, Vincent Arak,
Haonan Dong, Guojie Song. ICML topic: Deep Learning → Large Language Models.
[poster/64296](https://icml.cc/virtual/2026/poster/64296)

Frames the harness as the bottleneck: "current CE methods rely on manually crafted harnesses,
such as rigid generation-reflection workflows and predefined context schemas. They impose
structural biases…" The contribution replaces the hand-built harness with evolved skills.

### 2. Harness for RL training

**Reinforcement Learning for Tool-Calling Agents in Fast Healthcare Interoperability
Resources (FHIR)** — Marius Knorr, Robert Müller, Jan Bremer, Nils Schweingruber.
ICML topic: Applications → Health / Medicine.
[poster/65318](https://icml.cc/virtual/2026/poster/65318)

The only paper at either year where the harness serves RL training rather than evaluation. A
multi-turn CodeAct agent is post-trained "using a custom harness and tools", with an LLM judge
supplying execution-grounded rewards, and the paper presents an end-to-end pipeline whose
stages are named as "environment building, **harness construction**, model training and custom
evaluation." So a harness-for-RL category exists at ICML 2026, but it has exactly one member.

### 3. Agent-evaluation harness infrastructure

**Position: Agent Evaluation Should Be Agentified for Openness, Standardization, and
Reproducibility** — Xiaoyuan Liu, Tianneng Shi, Wenbo Guo, Dawn Song.
ICML topic: General Machine Learning → Evaluation.
[poster/67210](https://icml.cc/virtual/2026/poster/67210)

Position paper aimed squarely at harness design: benchmarks rely on "fixed, LLM-centric
harnesses that require heavy integration, create test-production mismatch, and limit fair
comparison across diverse agent designs."

**Implicit Intelligence — Evaluating Agents on What Users Don't Say** — Ved Sirdeshmukh,
Marc Wetter. ICML topic: Deep Learning → Large Language Models.
[poster/64912](https://icml.cc/virtual/2026/poster/64912)

Contributes Agent-as-a-World (AaW), "a harness where interactive worlds are defined in
human-readable YAML files and simulated by language models."

**BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics** — Dionizije Fa,
Marko Culjak, Bruno Pandza, Mateo Cupic. ICML topic: Applications → Everything Else.
[poster/66549](https://icml.cc/virtual/2026/poster/66549)

Evaluates frontier models "across multiple agent harnesses", making the harness an explicit
experimental factor rather than fixed background.

### 4. Harness as a source of measurement error

Both papers ask whether the harness, not the model, is producing the score.

**RubricRobustness: Evaluating the Sensitivity of Rubrics-Based Benchmarks to Simple
Perturbations** — Manasi Sharma. ICML topic: not assigned.
[poster/65214](https://icml.cc/virtual/2026/poster/65214)

Sensitivity analysis of rubric-plus-LLM-judge *evaluation harnesses*, whose "intrinsic
robustness … remains critically under-investigated."

**Faults in Our Formal Benchmarking: Dataset Defects and Evaluation Failures in Lean Theorem
Proving** — Pawan Sasanka Ammanamanchi, Siddharth Bhat, Stella Biderman.
ICML topic: Applications → Everything Else.
[poster/62980](https://icml.cc/virtual/2026/poster/62980)

Audits five Lean benchmarks; a named failure mode is that evaluation harnesses are not robust
to trivial or adversarial solutions.

### 5. Safety and security evaluation harness

**Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking**
(spotlight) — Zhicheng Fang, Jingjie Zheng, Chenxu Fu, Wei Xu.
ICML topic: Social Aspects → Safety.
[poster/65657](https://icml.cc/virtual/2026/poster/65657)

Motivated by drift "in datasets, harnesses, and judging protocols"; a multi-agent pipeline
turns jailbreak papers into modules runnable inside one unified harness.

### 6. Formal-mathematics proof-engineering harness

**APE-Bench: Evaluating Automated Proof Engineering for Formal Math Libraries** — Huajian Xin,
Zheng Yuan, Jacques Fleuriot, Wenda Li. ICML topic: Deep Learning → Large Language Models.
[poster/64323](https://icml.cc/virtual/2026/poster/64323)

Contributes **APE-Harness**, "a unified execution framework based on task contract
abstraction", alongside the benchmark. One of only two named harness artifacts at ICML 2026,
the other being VeRO.

### Adjacent: harness released, not studied

These ship an evaluation or measurement harness as a side artifact. Include them only under
the loosest reading of "harness-related".

| Title | Authors | ICML topic | Link |
| --- | --- | --- | --- |
| Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections (spotlight) | Borchmann, Van Landeghem, Turski, Padarha, Kearns, Mahdi et al. | DL → LLMs | [62732](https://icml.cc/virtual/2026/poster/62732) |
| Benchmarking Reward Hack Detection in Code Environments via Contrastive Analysis | Deshpande, Kannappan, Qian | not assigned | [63139](https://icml.cc/virtual/2026/poster/63139) |
| BFCL Audio: An Audio Function Calling Evaluation for Large Language Models | Mao, Ghai, Dawoodani, Ginart, Patil, Emmons et al. | GML → Evaluation | [61489](https://icml.cc/virtual/2026/poster/61489) |
| Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines | Lambert, Luccioni | Applications → Energy | [62967](https://icml.cc/virtual/2026/poster/62967) |

Reward Hack Detection is the closest of these to an RL harness (it concerns reward hacking in
code environments), and Energy Accounting is the only systems-style *measurement* harness at
either year.

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

Harness work at ICML is one year old: nothing in 2025, ten papers in 2026. The distribution is
lopsided toward evaluation — six of the ten are about evaluating agents or about the
trustworthiness of evaluation harnesses themselves — while only one uses a harness for RL
training and only two contribute a named harness artifact (VeRO, APE-Harness). The
fuzzing-harness literature that dominates the word elsewhere lives in software engineering and
security venues (FSE, ISSTA and similar), not ICML.

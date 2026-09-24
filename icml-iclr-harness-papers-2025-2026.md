# Harness-related papers at ICML and ICLR, 2025 and 2026

For the categorization of the full ICML and ICLR corpora for these two years, see
`icml-iclr-2025-2026-category-breakdown.md`.

Scope: ICML and ICLR main conferences only (the `icml` and `iclr` venues in CSrankings, i.e.
DBLP `ICML`, `ICML (1..3)`, `ICLR`, `ICLR (Poster)`). Workshops, affiliated events and
arXiv-only preprints are excluded, as are other CSrankings venues.

"Harness-related" here means the paper uses *harness* as a noun: the evaluation, agent, test or
measurement scaffolding that wraps a model or agent. Titles that use "harness" only as a verb
("Harnessing X for Y") are listed separately as non-matches, because they dominate a keyword
search and are not about harnesses.

## Headline

| Conference-year | Papers searched | Harness papers | Adjacent |
| --- | --- | --- | --- |
| ICML 2025 | 3,339 | 0 | 0 |
| ICML 2026 | 6,646 | 10 | 4 |
| ICLR 2025 | 3,830 | 0 | 2 |
| ICLR 2026 | 5,468 | 6 | 4 |

Harness work at both venues is one year old: nothing in 2025 at either, 16 papers across the two
in 2026. Almost all concern LLM-agent or LLM-evaluation harnesses. The single exception is
ICLR 2026's *Agnostics*, which uses "test harnesses" in the software-testing sense — the sense
that dominates the term in software engineering and security venues.

## Method

- Venue definitions taken from `util/csrankings.py` in `emeryberger/CSrankings`
  (branch `gh-pages`), the authoritative area-to-DBLP-venue mapping.
- ICML 2025: PMLR volume 267 (42nd ICML, Vancouver, 13–19 July 2025), 3,330 paper pages,
  plus the ICML virtual-site dump (3,459 records, 3,339 distinct titles), which carries
  abstracts.
- ICML 2026: the ICML virtual-site dump (6,797 records, 6,646 distinct accepted papers).
  Abstracts are absent from that dump, so all 6,646 paper pages were fetched individually.
- ICLR 2025: the ICLR virtual-site dump (4,040 records, 3,830 distinct papers), which carries
  abstracts.
- ICLR 2026: the ICLR virtual-site dump (5,691 records, 5,468 distinct papers). Abstracts again
  absent, so all 5,468 paper pages were fetched individually; 5,466 of 5,468 abstracts recovered.
- Detection: every occurrence of `harness`, `harnesses`, `harnessed`, `harnessing` and any
  hyphenated compound (`APE-Harness`, `eval-harness`, …) in title + abstract was enumerated and
  hand-classified into noun and verb senses. Noun-form occurrences: 15 in ICML 2025, 33 in
  ICML 2026, 24 in ICLR 2025, 39 in ICLR 2026. Full paper bodies were not searched, so a paper
  that only mentions its harness in the body would be missed.
- Category labels are mine; topic labels quoted per paper are the conference's own from the
  virtual-site metadata.
- As of 2026-09-24 the ICML 2026 PMLR volume is not published, so 2026 entries cite virtual-site
  pages.

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

## ICLR 2025: no matches

Zero harness papers among 3,830 accepted papers. Of the 24 noun-form occurrences, 22 are verb
usage and two are incidental noun mentions where the harness is neither contribution nor object
of study:

- **Filtered not Mixed: Filtering-Based Online Gating for Mixture of Large Language Models** —
  Saqur, Kratsios, Krach, Limmer, Horvath, Rudzicz. Deep Learning → Large Language Models.
  [poster/28916](https://iclr.cc/virtual/2025/poster/28916). Describes its MoE-F algorithm as
  "deployable as a plug-and-play filtering harness", a passing characterization of the
  deliverable.
- **Combatting Dimensional Collapse in LLM Pre-Training Data via Submodular File Selection**
  (oral) — Fan, Du, Hu, Wang, Shen, Zhang et al. Unlabeled.
  [poster/28897](https://iclr.cc/virtual/2025/poster/28897). Evaluates "across nine tasks from
  the Harness framework", i.e. uses `lm-evaluation-harness` as a measuring tool.

## ICLR 2026 by category

| # | Category | Papers |
| --- | --- | --- |
| 1 | Agent-evaluation harness infrastructure | 2 |
| 2 | Harness synthesis | 1 |
| 3 | Harness as an experimental variable | 1 |
| 4 | Safety and security evaluation harness | 1 |
| 5 | Test harness in the software-testing sense | 1 |
| — | Adjacent: harness released or used, not studied | 4 |

### 1. Agent-evaluation harness infrastructure

**Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation** —
Sayash Kapoor, Benedikt Stroebl, Peter Kirgis, Nitya Nadgir, Zachary Siegel, Boyi Wei et al.
ICLR topic: Social Aspects → Accountability, Transparency and Interpretability.
[poster/10006806](https://iclr.cc/virtual/2026/poster/10006806)

The most substantial harness paper at either venue in engineering terms. Its first stated
contribution is "a standardized evaluation harness that orchestrates parallel evaluations across
hundreds of VMs, reducing evaluation time from weeks to hours while eliminating common
implementation bugs", and it validates that harness with 21,730 agent rollouts across 9 models
and 9 benchmarks at a cost of roughly $40,000. Its three-dimensional analysis explicitly
separates models, scaffolds and benchmarks.

**lmgame-Bench: How Good are LLMs at Playing Games?** — Lanxiang Hu, Mingjia Huo, Yuxuan Zhang,
Haoyang Yu, Eric P. Xing, Ion Stoica et al. ICLR topic: Computer Vision → Vision Models &
Multimodal. [poster/10007223](https://iclr.cc/virtual/2026/poster/10007223)

The harness is the instrument of the science: a "modular harness — including perception, memory,
and reasoning modules — that can be toggled to selectively probe distinct capabilities", as
opposed to prior game benchmarks that entangle skills.

### 2. Harness synthesis

**ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution** — Robert Lange,
Yuki Imajuku, Edoardo Cetin. Unlabeled.
[poster/10007692](https://iclr.cc/virtual/2026/poster/10007692)

ICLR's closest analogue to ICML's VeRO: the system "designs robust agentic harnesses for AIME
mathematical reasoning tasks", making harnesses an output of program evolution rather than a
fixed input.

### 3. Harness as an experimental variable

**From Reproduction to Replication: Evaluating Research Agents with Progressive Code Masking** —
Gyeongwon J. Kim, Alex Wilf, Louis-Philippe Morency, Daniel Fried.
ICLR topic: Applications → Everything Else.
[poster/10007273](https://iclr.cc/virtual/2026/poster/10007273)

Reports that "agents that can dynamically interact with the environment (e.g. to debug their
code) can outperform agents in fixed ``agentless'' harnesses", making harness architecture a
measured factor rather than background.

### 4. Safety and security evaluation harness

**MCP Security Bench (MSB): Benchmarking Attacks Against Model Context Protocol in LLM Agents** —
Dongsen Zhang, Zekun Li, Xu Luo, Xuannan Liu, Pei Li, Wenjun Xu.
ICLR topic: Social Aspects → Fairness, Equity, Justice and Safety.
[poster/10007929](https://iclr.cc/virtual/2026/poster/10007929)

The harness is contribution (2) of three: "an evaluation harness that executes attacks by running
real tools (both benign and malicious) via MCP rather than simulation." The
real-execution-versus-simulation distinction is a harness design claim.

### 5. Test harness in the software-testing sense

**Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal
Reinforcement Learning Environment** — Aleksander Boruch-Gruszecki, Yangtian Zi, Zixuan Wu,
Tejas Oberoi, Carolyn Anderson, Joydeep Biswas et al.
ICLR topic: Applications → Everything Else.
[poster/10007548](https://iclr.cc/virtual/2026/poster/10007548)

The only paper across all four conference-years that uses "harness" in the software-testing
sense that dominates the term in SE and security venues: "every new language seems to require new
datasets, test harnesses, and reinforcement learning (RL) infrastructure", and Agnostics is a
language-agnostic pipeline that eliminates that per-language engineering. It is also, with ICML's
FHIR paper, one of only two harness-and-RL-training papers at either venue.

### Adjacent: harness released or used, not studied

| Title | Authors | ICLR topic | Harness role | Link |
| --- | --- | --- | --- | --- |
| Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces | Merrill, Shaw, Carlini, Li, Raj, Bercovich et al. | unlabeled | publishes dataset and evaluation harness | [10008736](https://iclr.cc/virtual/2026/poster/10008736) |
| Bee: A High-Quality Corpus and Full-Stack Suite to Unlock Advanced Fully Open MLLMs | Zhang, Ni, Chen, Zhang, Rao, Peng et al. | CV → Vision Models & Multimodal | evaluation harness among released resources | [10010295](https://iclr.cc/virtual/2026/poster/10010295) |
| When LLMs get significantly worse: A statistical approach to detect model degradations | Kübler, Budhathoki, Kleindessner, Zhou, Yin, Khetan et al. | DL → Generative Models and Autoencoders | implemented on top of `lm-evaluation-harness` | [10008517](https://iclr.cc/virtual/2026/poster/10008517) |
| Celo2: Towards Learned Optimization Free Lunch | Moudgil, Knyazev, Belilovsky | Optimization → Learning for Optimization | compatibility with "modern optimization harness" (optimizer-recipe sense) | [10008009](https://iclr.cc/virtual/2026/poster/10008009) |

## Takeaway

Harness research appeared at both venues simultaneously and only in 2026: zero papers at ICML
2025 and ICLR 2025, then 10 at ICML 2026 and 6 at ICLR 2026 — 16 papers out of the 12,114 the
two conferences accepted that year, and out of 19,283 searched across all four
conference-years.

The two venues split by emphasis. **ICML 2026 leans conceptual and evaluative**: six of its ten
papers are about evaluating agents or about whether evaluation harnesses are themselves
trustworthy, including a position paper arguing the whole harness model is wrong. **ICLR 2026
leans infrastructural**: HAL contributes a harness that orchestrates evaluation across hundreds
of VMs and validates it with 21,730 rollouts, and lmgame-Bench builds a modular harness whose
components can be toggled to isolate capabilities.

Two themes appear at both: **harness synthesis**, where a system designs harnesses rather than
consuming them (VeRO at ICML, ShinkaEvolve at ICLR), and **harness-as-variable**, where the
harness is a measured experimental factor rather than fixed background (BioAgent Bench at ICML,
From Reproduction to Replication at ICLR).

Only two papers in 12,114 connect harnesses to RL training rather than evaluation (ICML's FHIR
tool-calling paper, ICLR's Agnostics), and Agnostics is the single paper at either venue using
"harness" in the software-testing sense. That sense still lives almost entirely in software
engineering and security venues — FSE, ISSTA and similar — where fuzzing-harness generation is an
established line of work. ML venues have converged on "harness" meaning the scaffolding around an
LLM agent, a distinct and much newer usage.

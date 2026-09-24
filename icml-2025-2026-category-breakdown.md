# ICML 2025 and 2026: all papers by category

A complete categorization of every accepted main-conference paper at ICML 2025 and ICML 2026,
using ICML's own topic taxonomy. Companion to `icml-harness-papers-2025-2026.md`, which drills
into one narrow topic; this file covers the whole corpus.

Totals: **3,339 papers in 2025, 6,646 in 2026** — the conference almost exactly doubled (1.99x).

## Method and caveats

- Categories are ICML's own `topic` labels from the conference virtual-site metadata
  (`Area->Subtopic`), not labels I invented. There are 92 distinct label strings across the two
  years, spanning 8 stable top-level areas plus a 2025-only `Position` area.
- Distinct accepted papers: 3,339 for 2025 (3,339 posters + 120 orals, orals being duplicates
  of poster records) and 6,646 for 2026.
- **Label coverage is not complete.** 42 of 3,339 papers in 2025 (1.3%) and 633 of 6,646 in
  2026 (9.5%) carry no topic label. All percentages below are computed over *labeled* papers
  (3,297 and 6,013), so they describe label shares, not corpus shares.
- I tried to fill the 633 unlabeled 2026 papers with a keyword classifier, but validating it
  against the 6,013 ground-truth labels gave only **49.9% accuracy** — it over-assigns
  "Deep Learning" because nearly every abstract now mentions LLMs or transformers. Rather than
  publish 633 coin-flip labels, I left them unlabeled. They are spread evenly across all poster
  sessions and across both `Accept (regular)` and `Accept (spotlight)` tiers, so they do not
  look like a systematic subgroup, but I cannot rule out that they skew the shares by a point
  or two.
- Two taxonomy changes between years make a few comparisons apples-to-oranges:
  - 2025 allowed **area-only labels** with no subtopic (e.g. plain `Reinforcement Learning`,
    66 papers). Those appear below as "(area only, no subtopic)". 2026 has none.
  - 2025 had a top-level **`Position`** area (66 papers across 5 subtopics). 2026 dropped it
    and filed position papers under their subject area instead. By title prefix there are
    75 position papers in 2025 and **215** in 2026, so position papers nearly tripled even
    as their dedicated category disappeared.

## Top-level areas

Share is of labeled papers; growth is the raw count ratio.

| Area | 2025 | share | 2026 | share | share shift | growth |
| --- | --- | --- | --- | --- | --- | --- |
| Deep Learning | 1122 | 34.0% | 2221 | 36.9% | +2.9 | 1.98x |
| Applications | 589 | 17.9% | 1361 | 22.6% | +4.8 | 2.31x |
| General Machine Learning | 468 | 14.2% | 674 | 11.2% | -3.0 | 1.44x |
| Social Aspects | 271 | 8.2% | 638 | 10.6% | +2.4 | 2.35x |
| Reinforcement Learning | 202 | 6.1% | 374 | 6.2% | +0.1 | 1.85x |
| Theory | 324 | 9.8% | 368 | 6.1% | -3.7 | 1.14x |
| Optimization | 157 | 4.8% | 212 | 3.5% | -1.2 | 1.35x |
| Probabilistic Methods | 98 | 3.0% | 165 | 2.7% | -0.2 | 1.68x |
| Position (2025 only) | 66 | 2.0% | 0 | 0.0% | -2.0 | — |
| **Total labeled** | **3297** | 100% | **6013** | 100% | | 1.82x |
| *(unlabeled)* | *42* | | *633* | | | |

### What moved

Against a conference that doubled in size, the interesting numbers are the ones that did *not*
double.

- **Theory barely grew: 324 to 368, just 1.14x.** Its share fell from 9.8% to 6.1%, the largest
  drop of any area. Optimization (1.35x) and General Machine Learning (1.44x) also badly lagged
  the 1.82x average for labeled papers.
- **Applications grew fastest (2.31x)** and Social Aspects nearly matched it (2.35x). Within
  Applications, Robotics went 36 to 149 (4.1x), Time Series 27 to 94 (3.5x), Health/Medicine
  70 to 171 (2.4x) and Computer Vision 174 to 463 (2.7x).
- **Large Language Models is now the single largest subtopic by a wide margin: 439 to 1101.**
  At 18.3% of labeled 2026 papers, it is nearly two and a half times the next subtopic
  (Computer Vision, 463). Add Foundation Models (131) and Attention Mechanisms (80) and the
  LLM-adjacent cluster reaches 1,312 papers — more than three times the whole Theory area.
- **Safety, Alignment and Evaluation are the standout risers.** Social Aspects → Alignment went
  14 to 76 (5.4x), Safety 43 to 148 (3.4x), Security 28 to 75 (2.7x), and General Machine
  Learning → Evaluation 39 to 119 (3.1x). Meanwhile Privacy grew only 63 to 80 (1.3x) and
  Fairness 23 to 36 (1.6x), so the safety-adjacent growth is concentrated in alignment and
  model safety rather than the older fairness-and-privacy agenda.
- **Reinforcement Learning held its share exactly** (6.1% to 6.2%), which is notable given how
  much RL-for-LLM work exists — much of that is presumably filed under Deep Learning → LLMs
  rather than RL.
- Two subtopics are new in 2026: General Machine Learning → **Data** (23) and → **Methodology**
  (20). Supervised Learning (41 to 27), Clustering (33 to 29), Online Learning and Bandits
  (55 to 45) and Sequential Models/Time series under Deep Learning (41 to 40) are the few
  subtopics that shrank outright.

## Full subtopic breakdown

### Deep Learning — 1122 in 2025 (34.0%), 2221 in 2026 (36.9%)

| Subtopic | 2025 | 2026 |
| --- | --- | --- |
| Large Language Models | 439 | 1101 |
| Generative Models and Autoencoders | 164 | 321 |
| Graph Neural Networks | 104 | 122 |
| Foundation Models | 49 | 131 |
| Theory | 49 | 96 |
| Algorithms | 55 | 79 |
| Robustness | 60 | 72 |
| Attention Mechanisms | 37 | 80 |
| Other Representation Learning | 30 | 74 |
| Everything Else | 34 | 67 |
| Sequential Models, Time series | 41 | 40 |
| Self-Supervised Learning | 15 | 38 |
| (area only, no subtopic) | 45 | 0 |

### Applications — 589 in 2025 (17.9%), 1361 in 2026 (22.6%)

| Subtopic | 2025 | 2026 |
| --- | --- | --- |
| Computer Vision | 174 | 463 |
| Chemistry, Physics, and Earth Sciences | 114 | 189 |
| Health / Medicine | 70 | 171 |
| Robotics | 36 | 149 |
| Everything Else | 50 | 98 |
| Neuroscience, Cognitive Science | 44 | 91 |
| Time Series | 27 | 94 |
| Language, Speech and Dialog | 34 | 83 |
| (area only, no subtopic) | 30 | 0 |
| Social Sciences | 8 | 19 |
| Energy | 2 | 4 |

### General Machine Learning — 468 in 2025 (14.2%), 674 in 2026 (11.2%)

| Subtopic | 2025 | 2026 |
| --- | --- | --- |
| Evaluation | 39 | 119 |
| Causality | 68 | 83 |
| Representation Learning | 51 | 91 |
| Transfer, Multitask and Meta-learning | 55 | 80 |
| Everything Else | 29 | 44 |
| Supervised Learning | 41 | 27 |
| Unsupervised and Semi-supervised Learning | 29 | 38 |
| Online Learning, Active Learning and Bandits | 31 | 33 |
| Clustering | 33 | 29 |
| (area only, no subtopic) | 47 | 0 |
| Sequential, Network, and Time Series Modeling | 17 | 25 |
| Scalable Algorithms | 13 | 20 |
| Hardware and Software | 8 | 24 |
| Kernel methods | 7 | 18 |
| Data | 0 | 23 |
| Methodology | 0 | 20 |

### Social Aspects — 271 in 2025 (8.2%), 638 in 2026 (10.6%)

| Subtopic | 2025 | 2026 |
| --- | --- | --- |
| Accountability, Transparency, and Interpretability | 75 | 166 |
| Safety | 43 | 148 |
| Privacy | 63 | 80 |
| Security | 28 | 75 |
| Alignment | 14 | 76 |
| Fairness | 23 | 36 |
| Robustness | 11 | 29 |
| Everything Else | 2 | 28 |
| (area only, no subtopic) | 12 | 0 |

### Reinforcement Learning — 202 in 2025 (6.1%), 374 in 2026 (6.2%)

| Subtopic | 2025 | 2026 |
| --- | --- | --- |
| Deep RL | 42 | 103 |
| Multi-agent | 29 | 66 |
| Batch/Offline | 24 | 59 |
| Everything Else | 9 | 63 |
| (area only, no subtopic) | 66 | 0 |
| Online | 8 | 34 |
| Planning | 15 | 23 |
| Inverse | 7 | 11 |
| Policy Search | 2 | 15 |

### Theory — 324 in 2025 (9.8%), 368 in 2026 (6.1%)

| Subtopic | 2025 | 2026 |
| --- | --- | --- |
| Learning Theory | 85 | 88 |
| Online Learning and Bandits | 55 | 45 |
| Deep Learning | 37 | 44 |
| Reinforcement Learning and Planning | 34 | 39 |
| Game Theory | 31 | 40 |
| Optimization | 21 | 37 |
| Everything Else | 17 | 34 |
| Probabilistic Methods | 15 | 26 |
| Domain Adaptation and Transfer Learning | 12 | 14 |
| (area only, no subtopic) | 13 | 0 |
| Active Learning and Interactive Learning | 4 | 1 |

### Optimization — 157 in 2025 (4.8%), 212 in 2026 (3.5%)

| Subtopic | 2025 | 2026 |
| --- | --- | --- |
| Large Scale, Parallel and Distributed | 31 | 43 |
| Discrete and Combinatorial Optimization | 26 | 35 |
| Non-Convex | 15 | 33 |
| Zero-order and Black-box Optimization | 12 | 35 |
| Stochastic | 14 | 30 |
| (area only, no subtopic) | 43 | 0 |
| Everything Else | 6 | 22 |
| Convex | 10 | 14 |

### Probabilistic Methods — 98 in 2025 (3.0%), 165 in 2026 (2.7%)

| Subtopic | 2025 | 2026 |
| --- | --- | --- |
| Monte Carlo and Sampling Methods | 21 | 44 |
| Bayesian Models and Methods | 15 | 48 |
| Everything Else | 8 | 35 |
| Variational Inference | 14 | 13 |
| Gaussian Processes | 11 | 11 |
| (area only, no subtopic) | 20 | 0 |
| Structure Learning | 4 | 6 |
| Graphical Models | 4 | 5 |
| Spectral Methods | 1 | 3 |

### Position — 66 in 2025 (2.0%), 0 in 2026 (0.0%)

| Subtopic | 2025 | 2026 |
| --- | --- | --- |
| Methodology | 36 | 0 |
| Risk_safety_policy | 12 | 0 |
| Social_ethical_env_impact | 9 | 0 |
| Understanding | 5 | 0 |
| Data | 4 | 0 |

## Where the harness papers land

For cross-reference with `icml-harness-papers-2025-2026.md`: the 10 harness papers of 2026 are
scattered rather than clustered, which is itself a sign the topic has no home area yet. Four
sit in Deep Learning → Large Language Models (VeRO, Meta Context Engineering, Implicit
Intelligence, APE-Bench), two in Applications → Everything Else (BioAgent Bench, Faults in Our
Formal Benchmarking), one in General Machine Learning → Evaluation (Position: Agent Evaluation
Should Be Agentified), one in Social Aspects → Safety (Jailbreak Foundry), one in Applications →
Health / Medicine (RL for Tool-Calling Agents in FHIR), and one is among the 633 unlabeled
(RubricRobustness).

# ICML and ICLR, 2025 and 2026: all papers by category

A complete categorization of every accepted main-conference paper at ICML and ICLR for 2025 and
2026, using each conference's own topic taxonomy. Companion to
`icml-iclr-harness-papers-2025-2026.md`, which drills into one narrow topic; this file covers
the whole corpus.

| Conference-year | Papers | Labeled | Unlabeled | Growth |
| --- | --- | --- | --- | --- |
| ICML 2025 | 3,339 | 3,297 | 42 (1.3%) | — |
| ICML 2026 | 6,646 | 6,013 | 633 (9.5%) | 1.99x |
| ICLR 2025 | 3,830 | 3,325 | 505 (13.2%) | — |
| ICLR 2026 | 5,468 | 4,632 | 836 (15.3%) | 1.43x |

ICML almost exactly doubled; ICLR grew far more slowly at 1.43x, and ICML overtook ICLR in size
in 2026 (6,646 versus 5,468) having been smaller in 2025 (3,339 versus 3,830).

## Method and caveats

- Categories are each conference's own `topic` labels from its virtual-site metadata
  (`Area->Subtopic`), not labels I invented: 92 distinct label strings for ICML across the two
  years, 72 for ICLR 2025 and 57 for ICLR 2026.
- Venue definitions come from `util/csrankings.py` in `emeryberger/CSrankings` (branch
  `gh-pages`): the `icml` and `iclr` areas, main conference only, no workshops.
- **Label coverage is incomplete**, and worse for ICLR: 13.2% of ICLR 2025 and 15.3% of ICLR
  2026 papers carry no topic label, against 1.3% and 9.5% for ICML. All percentages are computed
  over *labeled* papers, so they are label shares rather than corpus shares. A keyword
  classifier I tried on the unlabeled ICML 2026 papers validated at only 49.9% against the
  6,013 ground-truth labels, so I left unlabeled papers unlabeled everywhere rather than publish
  coin-flip categories.
- **ICLR 2026 revamped its taxonomy, so ICLR year-over-year area comparison is invalid.** The
  changes are large:
  - **Computer Vision was promoted to a top-level area** (923 papers, 19.9%), having previously
    been a single subtopic under Applications (356 in 2025). It now has its own subtopics:
    Vision Models & Multimodal (367), Image and Video Generation (280), 3D Rendering &
    Reconstruction (116), Classification and Understanding (50), Segmentation (25).
  - **The `Large Language Models` subtopic was deleted outright.** It was ICLR 2025's single
    biggest bucket at 660 papers; in 2026 there is no LLM label at all, and language work is
    spread across `Applications->Language, Speech and Dialog` (320) and elsewhere. The most
    plausible reading is that the label stopped discriminating once most submissions involved
    LLMs.
  - `Miscellaneous Aspects of Machine Learning` was renamed `General Machine Learning`, matching
    ICML's naming.
  - `Probabilistic Methods` was demoted from a top-level area to a subtopic of General Machine
    Learning (52 papers).
  - New in 2026: `Theory->Interpretability and Visualization` (71) and `Applications->Climate`
    (13). Dropped: Deep Learning's Self-Supervised Learning, Other Representation Learning and
    Sequential Models subtopics.
  - Consequence: **Deep Learning's apparent collapse at ICLR from 42.8% to 18.8% is an artifact
    of this reshuffle, not a real decline.** Vision and representation-learning papers simply
    moved out from under it.
- ICML's taxonomy was more stable, with two wrinkles: 2025 allowed area-only labels with no
  subtopic (shown as "(area only, no subtopic)"), and 2025 had a top-level `Position` area
  (66 papers) that 2026 dropped, filing position papers under their subject area instead. By
  title prefix, ICML position papers went from 75 to 215.
- As of 2026-09-24 the ICML 2026 PMLR volume is not published, so ICML 2026 rows derive from the
  conference virtual site rather than PMLR.

## Cross-venue area shares

Share of labeled papers. Dashes mark areas that do not exist in that conference-year's
taxonomy — which is most of the story, and the reason the columns cannot be read as a clean
four-way comparison.

| Area | ICML 2025 | ICML 2026 | ICLR 2025 | ICLR 2026 |
| --- | --- | --- | --- | --- |
| Deep Learning | 34.0% | 36.9% | 42.8% | 18.8% |
| Applications | 17.9% | 22.6% | 24.1% | 22.8% |
| Computer Vision | — | — | — | 19.9% |
| General Machine Learning | 14.2% | 11.2% | — | 9.1% |
| Miscellaneous Aspects of ML | — | — | 6.0% | — |
| Social Aspects | 8.2% | 10.6% | 8.5% | 9.5% |
| Theory | 9.8% | 6.1% | 5.7% | 7.0% |
| Reinforcement Learning | 6.1% | 6.2% | 6.7% | 8.7% |
| Optimization | 4.8% | 3.5% | 4.1% | 4.2% |
| Probabilistic Methods | 3.0% | 2.7% | 2.1% | — |
| Position | 2.0% | — | — | — |

Four areas keep stable definitions across all four conference-years and can be compared
directly:

- **Reinforcement Learning.** Flat at ICML (6.1% to 6.2%) but rising at ICLR (6.7% to 8.7%), so
  ICLR 2026 is meaningfully more RL-heavy than ICML 2026.
- **Theory.** The two venues crossed over. ICML fell from 9.8% to 6.1% while ICLR rose from 5.7%
  to 7.0%, so ICLR is now the more theory-heavy of the two, reversing 2025.
- **Optimization.** Stable and small everywhere, 3.5–4.8%.
- **Social Aspects.** Rising at both, 8.2% to 10.6% at ICML and 8.5% to 9.5% at ICLR.

## ICML: full subtopic breakdown

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

## ICLR: full subtopic breakdown

### ICLR 2025

**Deep Learning** — 1423 papers, 42.8% of labeled

| Subtopic | Papers |
| --- | --- |
| Large Language Models | 660 |
| Generative Models and Autoencoders | 282 |
| Graph Neural Networks | 97 |
| Robustness | 66 |
| Other Representation Learning | 66 |
| Theory | 51 |
| Everything Else | 51 |
| Algorithms | 44 |
| Self-Supervised Learning | 41 |
| Attention Mechanisms | 36 |
| Sequential Models, Time series | 29 |

**Applications** — 800 papers, 24.1% of labeled

| Subtopic | Papers |
| --- | --- |
| Computer Vision | 356 |
| Chemistry and Drug Discovery | 78 |
| Robotics | 65 |
| Language, Speech and Dialog | 60 |
| Neuroscience, Cognitive Science | 60 |
| Physics | 56 |
| Everything Else | 38 |
| Time Series | 33 |
| Genetics, Cell Biology, Health, etc | 27 |
| Health | 27 |

**Social Aspects** — 284 papers, 8.5% of labeled

| Subtopic | Papers |
| --- | --- |
| Trustworthy Machine Learning | 84 |
| Fairness, Equity, Justice and Safety | 76 |
| Accountability, Transparency and Interpretability | 75 |
| Privacy-preserving Statistics and Machine Learning | 39 |
| Everything Else | 10 |

**Reinforcement Learning** — 223 papers, 6.7% of labeled

| Subtopic | Papers |
| --- | --- |
| Deep RL | 72 |
| Everything Else | 43 |
| Multi-agent | 32 |
| Batch/Offline | 30 |
| Planning | 20 |
| Online | 15 |
| Inverse | 8 |
| Function Approximation | 2 |
| (area only, no subtopic) | 1 |

**Miscellaneous Aspects of Machine Learning** — 200 papers, 6.0% of labeled

| Subtopic | Papers |
| --- | --- |
| Transfer, Multitask and Meta-learning | 46 |
| Causality | 36 |
| Representation Learning | 29 |
| Unsupervised and Semi-supervised Learning | 29 |
| General Machine Learning Techniques | 17 |
| Everything Else | 11 |
| Online Learning, Active Learning and Bandits | 10 |
| Sequential, Network, and Time Series Modeling | 9 |
| Scalable Algorithms | 5 |
| Supervised Learning | 5 |
| Kernel methods | 3 |

**Theory** — 189 papers, 5.7% of labeled

| Subtopic | Papers |
| --- | --- |
| Learning Theory | 49 |
| Optimization | 31 |
| Deep Learning | 21 |
| Online Learning and Bandits | 20 |
| Game Theory | 15 |
| Reinforcement Learning and Planning | 14 |
| Domain Adaptation and Transfer Learning | 14 |
| Everything Else | 13 |
| Probabilistic Methods | 11 |
| Active Learning and Interactive Learning | 1 |

**Optimization** — 136 papers, 4.1% of labeled

| Subtopic | Papers |
| --- | --- |
| Non-Convex | 34 |
| Large Scale, Parallel and Distributed | 25 |
| Learning for Optimization | 20 |
| Everything Else | 17 |
| Sampling and Optimization | 13 |
| Zero-order and Black-box Optimization | 11 |
| Optimization and Learning under Uncertainty | 10 |
| Global Optimization | 6 |

**Probabilistic Methods** — 70 papers, 2.1% of labeled

| Subtopic | Papers |
| --- | --- |
| Everything Else | 19 |
| Bayesian Models and Methods | 17 |
| Monte Carlo and Sampling Methods | 16 |
| Variational Inference | 7 |
| Gaussian Processes | 5 |
| Structure Learning | 3 |
| Graphical Models | 2 |
| Spectral Methods | 1 |

### ICLR 2026

**Applications** — 1057 papers, 22.8% of labeled

| Subtopic | Papers |
| --- | --- |
| Language, Speech and Dialog | 320 |
| Robotics | 146 |
| Everything Else | 126 |
| Neuroscience, Cognitive Science | 104 |
| Time Series | 96 |
| Chemistry and Drug Discovery | 94 |
| Physics | 62 |
| Health | 49 |
| Genetics, Cell Biology, Health, etc | 47 |
| Climate | 13 |

**Computer Vision** — 923 papers, 19.9% of labeled

| Subtopic | Papers |
| --- | --- |
| Vision Models & Multimodal | 367 |
| Image and Video Generation | 280 |
| 3D Rendering & Reconstruction | 116 |
| Everything Else | 85 |
| Classification and Understanding | 50 |
| Segmentation | 25 |

**Deep Learning** — 869 papers, 18.8% of labeled

| Subtopic | Papers |
| --- | --- |
| Generative Models and Autoencoders | 288 |
| Everything Else | 151 |
| Algorithms | 150 |
| Graph Neural Networks | 86 |
| Attention Mechanisms | 80 |
| Robustness | 59 |
| Theory | 55 |

**Social Aspects** — 442 papers, 9.5% of labeled

| Subtopic | Papers |
| --- | --- |
| Trustworthy Machine Learning | 139 |
| Fairness, Equity, Justice and Safety | 133 |
| Accountability, Transparency and Interpretability | 96 |
| Privacy-preserving Statistics and Machine Learning | 49 |
| Everything Else | 25 |

**General Machine Learning** — 420 papers, 9.1% of labeled

| Subtopic | Papers |
| --- | --- |
| Representation Learning | 165 |
| Transfer, Multitask and Meta-learning | 78 |
| Everything Else | 78 |
| Probabilistic Methods | 52 |
| Causality | 47 |

**Reinforcement Learning** — 404 papers, 8.7% of labeled

| Subtopic | Papers |
| --- | --- |
| Deep RL | 145 |
| Everything Else | 77 |
| Multi-agent | 56 |
| Batch/Offline | 48 |
| Online | 36 |
| Planning | 30 |
| Function Approximation | 6 |
| Inverse | 6 |

**Theory** — 322 papers, 7.0% of labeled

| Subtopic | Papers |
| --- | --- |
| Learning Theory | 102 |
| Interpretability and Visualization | 71 |
| Optimization | 32 |
| Reinforcement Learning and Planning | 31 |
| Domain Adaptation and Transfer Learning | 28 |
| Probabilistic Methods | 26 |
| Game Theory | 17 |
| Everything Else | 15 |

**Optimization** — 195 papers, 4.2% of labeled

| Subtopic | Papers |
| --- | --- |
| Everything Else | 43 |
| Learning for Optimization | 41 |
| Large Scale, Parallel and Distributed | 34 |
| Non-Convex | 27 |
| Sampling and Optimization | 18 |
| Zero-order and Black-box Optimization | 16 |
| Optimization and Learning under Uncertainty | 12 |
| Global Optimization | 4 |

## Notable findings

- **ICML's LLM concentration versus ICLR's dispersal.** ICML doubled down on the label: Deep
  Learning → Large Language Models went 439 to 1101, becoming 18.3% of labeled 2026 papers and
  2.4x the next-largest subtopic. ICLR did the opposite and deleted the label. Same underlying
  research wave, opposite bookkeeping, which is a warning against reading either taxonomy as a
  measure of what the field works on.
- **Theory is shrinking at ICML but not ICLR.** ICML Theory grew only 1.14x (324 to 368) against
  a conference that grew 1.99x, losing 3.7 points of share. ICLR Theory gained share (5.7% to
  7.0%).
- **Safety and evaluation are the fastest risers at ICML**: Alignment 14 to 76 (5.4x), Safety 43
  to 148 (3.4x), Evaluation 39 to 119 (3.1x), Security 28 to 75 (2.7x), while Privacy managed
  only 1.3x and Fairness 1.6x. ICLR's equivalents grew too, but its trust-related subtopics are
  bundled differently (Trustworthy Machine Learning 84 to 139, Fairness/Equity/Justice/Safety 76
  to 133), so the split between alignment and fairness is not visible in ICLR's labels.
- **Robotics is the standout application at both venues**: ICML 36 to 149 (4.1x), ICLR 65 to 146
  (2.2x).
- **Generative modelling is where the two venues look most alike**: Deep Learning → Generative
  Models and Autoencoders is 321 at ICML 2026 and 288 at ICLR 2026, similar absolute counts at
  similar shares.

## Where the harness papers land

For cross-reference with `icml-iclr-harness-papers-2025-2026.md`. Harness papers do not cluster
in any one category at either venue, which is itself evidence the topic has no home area yet.

- **ICML 2026** (10 papers): four in Deep Learning → Large Language Models, two in Applications →
  Everything Else, one each in General Machine Learning → Evaluation, Social Aspects → Safety and
  Applications → Health / Medicine, and one among the unlabeled.
- **ICLR 2026** (6 papers): Social Aspects → Accountability/Transparency/Interpretability (HAL),
  Computer Vision → Vision Models & Multimodal (lmgame-Bench), Applications → Everything Else
  (Agnostics, From Reproduction to Replication), Social Aspects → Fairness/Equity/Justice/Safety
  (MCP Security Bench), and one unlabeled (ShinkaEvolve).
- **ICML 2025 and ICLR 2025**: none, so no distribution to report.

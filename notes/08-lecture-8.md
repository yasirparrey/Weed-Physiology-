# Lecture 8 — LLM Evaluation

**Video length: 1h49.** Covered here: what "evaluation" means and why human
rating — though the gold standard — does not scale, inter-rater agreement
(Cohen's/Fleiss' kappa), rule-based metrics (BLEU, ROUGE, METEOR) and their
limits, LLM-as-a-judge including its biases and best practices, enforcing output
format, factuality decomposition, the full taxonomy of agentic failure modes,
and benchmarks (MMLU, AIME, PIQA, SWE-bench, HarmBench, τ-bench) with the
$\mathrm{Pass}^k$ metric, Pareto frontiers, data contamination and Goodhart's Law.

This lecture addresses the last of Lecture 6's four weaknesses: **hard to
evaluate**.

---

## 8.1 What "evaluation" means, and why it is hard

The word covers two quite different things:

**Output quality** — instruction following, coherence, factuality, etc.
**System performance** — latency, pricing, reliability.

**Today's focus is output quality.** (System performance is a normal engineering
problem; output quality is the hard one.)

**Context: an LLM generates free-form text, which is hard to evaluate.** There is
no single correct output, so there is nothing to compare against exactly.

### The ideal: human rating

**Human rating is closest to truth and is the "gold" standard.** Generate outputs
with the LLM, have humans rate them, done.

### Why that does not work

**Limitation 1 — the subjectivity of the evaluation task.**

Consider: prompt *"What birthday gift should I get?"*; model response *"A teddy
bear is almost always a sweet gift — just pick one that feels right to you."*;
criterion **usefulness**. Rater 1 says it is useful (warm, actionable enough).
Rater 2 says it is not (vague, no real recommendation). Both are defensible.

So you measure **agreement between raters** — but naive observed agreement is
misleading, because two raters who both say "useful" 90% of the time will agree
81% of the time **purely by chance**.

**The metric:** *"How much better is our agreement than what we'd expect just by
chance, given how the raters actually use the categories?"*

$$\kappa = \frac{p_{\mathrm{observed}} - p_{\mathrm{expected}}}{1 - p_{\mathrm{expected}}}$$

**Variants: Cohen's kappa** (two raters), **Fleiss' kappa** (many raters),
**Krippendorff's alpha** (arbitrary numbers of raters, missing data, and ordinal
or interval scales).

*(Cohen, 1960; Fleiss, 1971.)*

**Limitation 2 — slow.**
**Limitation 3 — expensive.**

> **Intuition — why kappa is the number to demand, and how to read it.** The
> numerator is the agreement you achieved above chance; the denominator is the
> agreement that was *available* above chance. So $\kappa = 0$ means you did no better
> than coin-flipping and $\kappa = 1$ means perfect. Rough conventions: below 0.4 is
> poor, 0.4–0.6 moderate, 0.6–0.8 substantial, above 0.8 excellent.
>
> Here is the practically vital consequence. If your **humans** cannot achieve a
> decent kappa on a criterion, then that criterion is **not well enough defined to
> evaluate anything with** — and no automated judge can rescue it, because there is
> no ground truth to approximate. Low kappa is not a rater problem, it is a
> *rubric* problem: the fix is to sharpen the definition, decompose the criterion,
> or move to a binary scale. This is why kappa appears here, before any automated
> method: it is the tool that tells you whether your evaluation question is even
> answerable.

---

## 8.2 Rule-based metrics

**Idea: write the labels once, and use them as references.** Have humans produce
reference outputs, then compare each new model output against those references
with a deterministic function.

**Common rule-based metrics:**

- **METEOR** — *Metric for Evaluation of Translation with Explicit ORdering*
  (Banerjee et al., 2005). Unigram matching with stemming and synonym matching,
  plus an explicit penalty for word-order fragmentation.

- **BLEU** — *BiLingual Evaluation Understudy* (Papineni et al., 2002).
  Precision-flavoured n-gram overlap with a brevity penalty.

- **ROUGE** — *Recall-Oriented Understudy for Gisting Evaluation* (Lin et al.,
  2004). Recall-flavoured n-gram overlap. **Many variants** — ROUGE-N (n-gram
  overlap), ROUGE-L (longest common subsequence), and others.

### Limitations

**1. Does not take stylistic variation into account.** The lecture's
demonstration — three sentences that mean essentially the same thing:

- *"A plush teddy bear can comfort a child during bedtime."*
- *"Soft stuffed bears often help kids feel safe as they fall asleep."*
- *"Many youngsters rest more easily at night when they cuddle a gentle toy
  companion."*

Judged by n-gram overlap, sentences 1 and 3 share almost nothing, so a model
producing 3 against reference 1 would be scored as a failure. It is a perfectly
good answer.

**2. Correlation with human rating is not that great.**

**3. Still requires human ratings!** You need reference outputs, so you have not
escaped the cost — only moved it from "rate every model output" to "write a
reference for every input".

> **Intuition — the structural flaw.** These metrics measure **surface form
> overlap** while we care about **meaning**. That was tolerable for machine
> translation in 2002, where outputs were short, constrained, and multiple
> references could be collected. It falls apart entirely for open-ended
> generation, where the space of good answers is effectively unbounded and n-gram
> overlap with any particular reference is close to meaningless. Note that they are
> not worthless: they are cheap, deterministic and reproducible, which makes them
> reasonable for *regression testing* (did today's build get worse than
> yesterday's?) even when they are useless as absolute quality measures.

---

## 8.3 LLM-as-a-Judge

**LaaJ = LLM-as-a-Judge** (Zheng et al., 2023, *Judging LLM-as-a-Judge with
MT-Bench and Chatbot Arena*).

**Idea: use an LLM to rate the quality of a response.**

The inputs and outputs:

```
Prompt               Model response                 Criteria
"Why are teddy       "They are comforting           Relevance
bears comforting?"    because they are soft
                      and loyal companions."
                              ↓
                       LLM-as-a-Judge
                              ↓
        Rationale                             Score
   "Direct explanation,                        PASS
    is on topic"
```

**An example judge prompt:**

```
Evaluate how relevant the model's answer is to the user's prompt.

Prompt: {prompt}
Model Response: {model_response}

Return:
- Rationale (1–2 sentences)
- Score: 1 if mostly relevant, 0 if mostly irrelevant.
```

### Enforcing the output format

A judge whose output cannot be parsed is useless, so this is where **guided
decoding** from Lecture 3 slide 65 comes back — and the lecture shows that
OpenAI, Gemini and Anthropic all document structured-output support.

In practice, two steps:

**Step 1. Define the desired output structure.**

```python
class Response:
    rationale: str
    score: Literal[0, 1]
```

**Step 2. Pass it as part of the model call.**

```python
response = client.responses.parse(
    model=model,
    input=input,
    text_format=Response,
)
```

### Benefits compared to what came before

- **no need for a reference / label** — this is the big one; you have escaped the
  human-written-reference bottleneck entirely,

- **interpretability via rationales** — e.g. *"The response does not contain any
  grammatical mistakes. It also is clear, easy to understand and is helpful with
  respect to the task."*

### The two main variations

**Pointwise.** *"Evaluate the quality of: Response"* → *"Very good"*.

**Pairwise.** *"Which one is better: Response A or Response B?"* →
*"Response A"*.

> **Intuition — same trade-off as preference data in Lecture 5, and the same
> resolution.** Pairwise is more reliable, because relative judgements are easier
> and need no absolute scale. Pointwise is what you need for monitoring, because
> you cannot pair every production response against something. In practice:
> pairwise for model selection and A/B comparisons, pointwise with a crisp
> pass/fail rubric for continuous evaluation.

### The three biases — each with a symptom and a remedy

**Position bias.**
*Symptom:* ask *"Which is better: A or B?"* → the judge says **A**. Ask the same
question with the order swapped, *"Which is better: B or A?"* → the judge says
**B**. It is preferring the *position*, not the content.
*Remedy:* **"take the average"** — run both orderings and combine (a genuine
preference should survive the swap; disagreement means a tie) — or **tweak the
position embeddings**.

**Verbosity bias.**
*Symptom:* Response A is *"short and correct"*; Response B *"goes into details, a
lot of which are not needed and is overall not very helpful"*. The judge picks
**B**, because it is longer.
*Remedy:* **explicit guidelines**, **few-shot** examples demonstrating that
concise-and-correct beats long-and-padded, and/or a **penalty on output length**.

**Self-enhancement bias.**
*Symptom:* Response A is a *human-curated answer that perfectly answers the
question*; Response B was *generated by the same model that is now acting as
judge*. The judge picks **B** — its own output.
*Remedy:* **don't use the same model as judge and as generator.**

> **Intuition — where these biases come from, which tells you how much to trust
> the fixes.** They are not bugs in the judge model; they are consequences of how
> it was trained. **Position bias** comes from autoregressive structure and from
> whatever ordering regularities existed in its preference training data.
> **Verbosity bias** is inherited straight from RLHF: human raters systematically
> prefer longer, more detailed answers (they read as more effortful and
> authoritative), so the reward model learned it and the policy learned it, and now
> the judge exhibits it. **Self-enhancement bias** arises because a model assigns
> higher likelihood to text in its own style, and likelihood correlates with its
> notion of quality.
>
> Which means the remedies are mitigations, not cures. Averaging over positions
> genuinely fixes position bias, because it is symmetric. Prompt instructions
> *reduce* verbosity bias but do not eliminate it, since you are fighting a
> learned prior. Using a different model for judging is the only real fix for
> self-enhancement — and note it also implies you should be suspicious of any
> evaluation where a lab judges its own model with its own model.

### Best practices

The consolidated list:

- **crisp guidelines**,
- **binary scale over more granular ones**,
- **write the rationale before outputting the score**,
- **mitigate biases** (position, verbosity, self-enhancement, etc.),
- **calibrate with human judgements**,
- **low temperature for reproducibility**.

> **Intuition — why each of these, since they are not arbitrary.**
>
> *Binary over granular*: exactly the argument from section 8.1. If humans cannot
> reliably distinguish a 6 from a 7, the judge's choice between them is noise
> dressed up as measurement. Collapsing to pass/fail throws away nothing real and
> makes the metric stable. If you need nuance, use several binary criteria rather
> than one fine-grained scale.
>
> *Rationale before score*: this is chain of thought from Lecture 3, applied to
> judging. If the score comes first, the rationale is a post-hoc justification of
> an already-committed answer and cannot influence it. If the rationale comes
> first, the reasoning is in the context when the score is produced, and it
> genuinely conditions the outcome. The ordering in the prompt is doing real work.
>
> *Calibrate with human judgements*: this closes the loop. You keep a human-labelled
> set, measure the judge's agreement with it (kappa again), and only trust the
> judge on criteria where that agreement is high. The judge is a cheap
> approximation of humans, and you must periodically check the approximation.
>
> *Low temperature*: an evaluation metric that returns different numbers on
> identical inputs is not a metric. Note the Lecture 3 caveat that even $T = 0$ is
> not perfectly deterministic in a batched serving stack.

### The revised workflow

```
LLM ──→ LLM-as-a-Judge  (fast, cheap, broad coverage)
    └──→ Human ratings  (slow, expensive, used to calibrate the judge)
```

You do not replace humans; you use them sparingly, to validate an automated judge
that then runs at volume.

### Typical dimensions to evaluate

**Task performance:** usefulness, factuality, relevance.
**Alignment:** tone, style, safety.

### Focus on factuality

**Objective: quantify the factuality of an output.** Take:

> *"Teddy bears, first created in the 1920s, were named after President Theodore
> Roosevelt after he proudly wanted to shoot a captured bear on a hunting trip."*

**Open question: how should we quantify the nuance?** This passage is not simply
true or false. Part is wrong (teddy bears date from 1902, not the 1920s), part is
right (they are named after Roosevelt), and part inverts the actual story
(Roosevelt *refused* to shoot the captured bear — that refusal is the whole
origin of the name).

**The method** (*Long-form factuality in large language models*, Wei et al.,
2024): **decompose into atomic facts**, verify each, and aggregate with weights.

| Atomic fact | Weight `wᵢ` |
|---|---|
| Teddy bears were first created in the 1920s. | 0.3 |
| Teddy bears were named after President Theodore Roosevelt. | 0.4 |
| Theodore Roosevelt was on a hunting trip where a bear was captured. | 0.2 |
| Theodore Roosevelt proudly wanted to shoot the captured bear. | 0.1 |

where `wᵢ` is the **importance of the fact**, and

$$\text{score} = \sum_i w_i \cdot \mathbb{1}\big[\text{fact } i \text{ is correct}\big]$$

Here that yields **score = 0.60**.

> **Intuition — why decomposition is the right move, and it generalises.** Asking
> "is this paragraph factual?" is unanswerable when it is partly factual, and any
> single score you assign is arbitrary. Splitting into atomic claims turns one
> impossible judgement into several tractable ones — each atomic claim *is*
> checkable, against a search engine or a knowledge base — and makes the result
> interpretable, since you can point at exactly which claim failed. The importance
> weights matter because not all errors are equal: getting the decade wrong is a
> detail, while inverting Roosevelt's refusal into eagerness reverses the meaning
> of the story. Note the general lesson, which applies well beyond factuality:
> **when a judgement is too coarse to make reliably, decompose it into judgements
> that are not.** This is the same instinct as "binary over granular scales" and
> "several criteria instead of one".

---

## 8.4 Evaluating agents: the failure-mode taxonomy

Back to the **ReAct** loop from Lecture 7 — with the observation that **a typical
agentic call sees multiple loops** through Observe → Plan → Act.

And back to the three tool steps:

1. Let the LLM find the argument for the relevant function call.
2. Make the function call.
3. Let the LLM deduce a conclusion based on the results.

Each of the three can fail, in characteristic ways. This taxonomy is the most
directly practical content in the lecture.

### Failure class 1: tool prediction errors (step 1)

**(a) Does not use the tool.**
*Symptom:* the LLM directly issues a response — *"Sorry, I don't know where I can
find one."* — despite having the tool available.
*Potential causes:* **tool router error** (the router never surfaced the tool);
**the model doesn't know how to use the tool**.
*Remedies:* **retrain the tool router**; **SFT-train the model, or better adjust
the prompt associated with the target API**.

**(b) Hallucinates a tool.**
*Symptom:* the LLM calls a tool that doesn't exist —
`location = (37.42, −122.17) with find_bear()` when the real function is
`find_teddy_bear()`.
*Potential causes:* **the model is too weak**; **API naming is not logical**;
**instructions are unclear**.
*Remedies:* **upgrade the model**; **revamp the API**; **iterate on the top-level
instructions**.

**(c) Uses the wrong tool.**
*Symptom:* the LLM picks another available tool — e.g.
`send_message(recipient=LocalBusiness, message="may I buy a teddy bear?")`
instead of searching.
*Potential causes:* **tool router error**; **the model chose the wrong tool**.
*Remedies:* **retrain the tool router**; **SFT-train the model or better adjust
the prompt associated with the target API**.

**(d) Infers the wrong argument.**
*Symptom:* the right tool, the wrong input — `location = (0, −0)`.
*Potential causes:* **the argument cannot be inferred** (the agent was never told
the user's location); **the model doesn't know how to use the tool**.
*Remedies:* **introduce a helper tool and/or ensure the context carries the right
information**; **SFT-train the model or better adjust the prompt**.

> **Intuition — (b) and (d) are the two most instructive.** Hallucinating
> `find_bear()` instead of `find_teddy_bear()` is what happens when your API
> naming is not the *obvious* name for the concept: the model generates the name it
> would expect, and if that is not the name you chose, you get a phantom call.
> Rename the function to what a competent person would guess. And (d)'s first
> cause — "the argument cannot be inferred" — is not a model failure at all. If the
> agent has no way to know the user's coordinates, no amount of prompting fixes it;
> you need a `get_user_location()` tool, or the location in context. Diagnosing
> "the model is dumb" when the real answer is "the information was not available"
> is the most common wasted debugging session in agent development.

### Failure class 2: tool call errors (step 2)

**(a) Wrong response.**
*Symptom:* the tool returns a wrong value or an error —
`find_teddy_bear(location)` → `ValueError: could not...`
*Causes:* the tool returns an incorrect value; the tool errors instead of
returning a value.
*Remedy:* **fix the tool implementation!**
*(The lecture notes that sometimes errors could be legitimate, though that is not
standard practice.)*

**(b) No response.**
*Symptom:* the tool returns nothing at all.
**Often seen with: the final response is hallucinated.**
*Cause:* either a bug, or a "feature" of the tool implementation.
*Remedies:* **return something, even if it's an empty JSON**; **emit meaningful
tool outputs as a general rule**.

> **Intuition — (b) deserves special attention because of its symptom.** When a
> tool returns nothing, the model is left with a gap in its context where an
> observation should be. And a language model's default response to a gap is to
> **fill it plausibly** — so you get a confident, entirely fabricated answer, and
> nothing in the trace looks like an error. This is the worst class of bug: silent,
> invisible, and downstream of a place nobody is looking. Returning `{}` with a
> message like `{"status": "no results found"}` converts an invisible
> hallucination into an explicit observation the model can reason about honestly.

### Failure class 3: response generation errors (step 3)

**Wrong response.**
*Symptom:* the final response doesn't convey the tool's response — the tool
returned `{"name": "Teddy", ...}` and the model says *"Didn't find any bear!"*
*Potential causes:* **the model lacks grounding capabilities**; **the tool
response spams the context window**; **the tool response does not convey
information meaningfully**.
*Remedies:* **upgrade the LLM in charge of synthesising the response**; **trim
the information returned by the backend**; **make the tool output format
descriptive**.

### Summary of the taxonomy

```
step 1: <function APIs> + prompt ──→ LLM ──→ call
        failures: does not use / hallucinates / uses the wrong tool,
                  infers wrong argument

step 2: call ──→ Backend ──→ result
        failures: wrong response, no response

step 3: result ──→ LLM ──→ response
        failure:  wrong response
```

### Common issues: takeaway

**Modeling:**

- weak reasoning or grounding capabilities,
- **too much going on in the context window**,
- tool modelling isn't right.

**Tool:**

- the tool itself has a problem,
- the output of the tool isn't interpretable.

**...special care and patience is needed to debug and fix issues!**

> **Intuition — the pattern across the entire taxonomy.** Read the remedies again
> and notice how few of them are "use a better model". Most are: **fix your API
> names**, **fix your docstrings**, **fix your tool's return format**, **put the
> needed information in context**, **return something instead of nothing**. Agent
> reliability is overwhelmingly an *interface design* problem, not a model
> problem. The corollary is that debugging requires **traces**: you must be able
> to see which of the three steps failed, because the remedies are completely
> different. This is why Lecture 7 closed on observability.

---

## 8.5 Benchmarks

The lecture organises common benchmarks into four categories, then adds agents.

| | **Knowledge** | **Reasoning** | **Coding** | **Safety** |
|---|---|---|---|---|
| what it tests | accurately state facts about the world; more "breadth" than "depth"; reflects pretraining quality | solve multi-step problems; includes "maths" and "common sense" subfields; internal reasoning | generate syntactically correct code; tests programming proficiency; proxy for tool-use abilities | prevent harmful, toxic, inappropriate behaviour; surface vulnerabilities before deployment; alignment with custom preferences |
| example | MMLU | AIME, PIQA | SWE-bench | HarmBench |

### MMLU — Massive Multitask Language Understanding

*Hendrycks et al., 2020.* **4 possible choices per question, across 57 tasks**
spanning elementary mathematics, US history, computer science, law, and more.
**Evaluation criteria: find the right choice among A/B/C/D** — a **hardcoded
match**.

### AIME — American Invitational Mathematics Examination

**~30 maths problems**, spanning geometry, algebra and analysis, requiring
thorough reasoning. **Evaluation criteria: give the right 3 digits** — hardcoded
match. (AIME answers are always integers from 000 to 999, which is exactly why it
is convenient as a benchmark.)

### PIQA — Physical Interaction: Question Answering

*Bisk et al., 2019.* **2 possible choices per question** on everyday situations
anchored in physics; **approximately 20,000 examples**. **Evaluation criteria:
find the right choice among Sol1/Sol2** — hardcoded match.

### SWE-bench — SoftWare Engineering benchmark

*Jimenez et al., 2023.* **2,294 software engineering problems from real GitHub
issues across 12 popular Python repositories.** Each problem provides **a base
commit** and **an already-merged PR with tests**. **Evaluation criteria: the
generated PR passes all test cases** — hardcoded match.

> **Intuition — why SWE-bench became the benchmark everyone cites.** It has the
> three properties you want and almost never get together. It is **real** — actual
> issues from actual repositories, not synthetic puzzles. It is **automatically
> verifiable** — the maintainers' own tests decide, with no judge and no
> ambiguity. And it is **agentic** — solving an issue requires navigating a large
> codebase, reading files, understanding context and editing multiple places,
> which is why the lecture calls coding a "proxy for tool-use abilities". It is
> the closest thing the field has to an end-to-end measure of useful autonomous
> work.

### HarmBench — Harmful Behavior Benchmark

*Mazeika et al., 2024.* **510 unique harmful behaviours** (400 text-based, 110
multimodal), split into **"Standard"**, **"Copyright"**, **"Contextual"** and
**"Multimodal"**, with the criterion that they *violate laws and widely-held
norms*. **Evaluation criteria: attack success rate (ASR)**, computed by a
**classifier** (not a hardcoded match, because judging whether a response is
harmful requires a model).

### τ-bench — Tool-Agent-User Interaction Benchmark

*Yao et al., 2024.* A given set of **database schema, APIs and policies** across
two domains:

- **Airline agent** — 500 users, 300 flights, 2000 reservations; ~10 tools and
  50 tasks.

- **Retail agent** — 500 users, 50 products, 1000 orders; ~10 tools and 115
  tasks.

**Evaluation criteria: maximise reward and $\mathrm{pass}^k$.**

### $\mathrm{Pass}^k$ — a metric for consistency/reliability

**"Probability that ALL $k$ attempts succeed."**

$$\mathrm{Pass}^k = \text{probability that \emph{every one} of } k
\text{ independent attempts succeeds}$$

> **Watch out — $\mathrm{Pass}^k$ (caret) is not $\mathrm{Pass@}k$ (at).** They are near-opposites.
> $\mathrm{Pass@}k$ = **at least one** of $k$ succeeds — it measures **capability**, and it
> *increases* with $k$. $\mathrm{Pass}^k$ = **all** $k$ succeed — it measures
> **consistency**, and it *decreases* with $k$.
>
> **Intuition — why agents need the second one.** For a coding assistant with a
> test suite, $\mathrm{Pass@}k$ is the honest metric: you can retry, and one success is a
> win. For an agent operating on the real world — booking flights, issuing refunds
> — retrying is not free and inconsistency is itself the failure. An agent that
> books the correct flight 80% of the time and a wrong one 20% of the time is
> unusable regardless of how good its best case is. $\mathrm{Pass}^k$ exposes exactly this:
> an agent with 80% per-attempt success has $\mathrm{Pass}^5 \approx 0.33$, which is a fair
> description of how it will feel to depend on. τ-bench chose this metric
> deliberately, and reported numbers on it were sobering.

### Profile: what benchmarks actually tell you

**Role of benchmarks:**

- **a projection of performance along a given axis**,
- **different models may be good at different things**.

The lecture illustrates with practitioners' empirical "facts": *Claude's Sonnet
models — coding*; *Gemini's Flash models — cheap*; and so on. And with Gemini 3's
launch (three days before the lecture), whose announcement is broken down along
**reasoning, coding, tool use, knowledge**.

### The Pareto frontier

**Definition: a Pareto curve is the set of solutions that "optimises" a
trade-off** — the points where you cannot improve one axis without giving up on
another.

**Possible trade-offs:**

- quality vs. cost/latency,
- quality vs. safety,
- quality vs. context length.

> **Intuition — this is the shape in which model choice should actually be
> made.** "Which model is best?" is not a well-posed question. "Which models are
> on the Pareto frontier for quality versus cost?" is, and the answer is usually a
> handful — everything else is dominated (something is both better *and* cheaper,
> so there is never a reason to pick it). Then you choose a point on the frontier
> according to your own budget. Note the quality-versus-safety axis: a model that
> refuses less is more useful *and* more dangerous, so where you sit is a product
> decision, not a technical one.

### Data contamination

**Problem: a benchmark's clues may be contained in the training set.** The
benchmark questions and answers are on the public internet, which is where
pretraining data comes from — so a high score may reflect memorisation rather
than capability.

**Precautions:**

- **use an identifier such as a hash** — Google's BIG-bench embeds a canary
  string in its files so that scrapers can be detected and dataset authors can ask
  for exclusion,

- **for tools, use a blocklist** — an agent with web access will otherwise simply
  look the answer up (Gemini 3's evaluation methodology documents doing this),

- **evaluate on newer test versions** — use, say, AIME 2025 for a model whose
  training cutoff precedes it.

> **Intuition.** Contamination is not usually deliberate cheating; it is nearly
> unavoidable when you train on the whole web. The consequence is that a benchmark
> **degrades over time**: the day it is published it measures capability, and a few
> years later it partly measures memorisation. This is why the field keeps having
> to build new benchmarks, and why evaluation on *freshly minted* problems (this
> year's competition, this month's GitHub issues) is so much more informative than
> a strong score on a five-year-old suite. Note the interaction with Lecture 4's
> "train on the test task" recommendation: the goal in both cases is to make sure
> you are measuring the ability rather than the exposure.

### Limitations, and the final word

> *"When a measure becomes a target, it ceases to be a good measure."*
> — **Goodhart's Law** (Strathern, 1997)

**Lessons:**

- **should not over-index on benchmarks**,
- **need ~organic perspectives to complete the picture: Chatbot Arena**,
- **...just try a few models out yourself!**

> **Intuition — Goodhart's Law is the connective tissue of this whole course, and
> it is worth collecting the instances.** Optimise against a reward model and you
> get reward hacking (Lecture 5). Optimise against a length-normalised GRPO loss
> and you get rambling (Lecture 6). Optimise against MMLU and you get models tuned
> for four-option multiple choice. Optimise against human preference votes and you
> get verbosity and sycophancy (Lectures 4 and 8). It is the same failure every
> time: the metric was a *proxy* for what you wanted, and sufficiently hard
> optimisation finds the gap between proxy and goal.
>
> The practical defences are the ones the lecture lands on. Use **many**
> uncorrelated metrics, so that gaming all of them simultaneously is harder than
> actually improving. Keep **held-out, freshly-generated** evaluations that no
> training loop has ever seen. Keep **humans in the loop** at low volume as a
> calibration anchor. And **try the models yourself** on your own real tasks —
> which sounds unscientific but is the only measurement guaranteed not to have
> been optimised against.

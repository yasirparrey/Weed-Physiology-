# Lecture 6 — LLM Reasoning

**Video length: 1h47.** Covered here: what "reasoning" means and how to spot a
reasoning model, reasoning benchmarks and the Pass@k / Cons@k metrics, test-time
scaling and verifiable rewards, controlling the thinking budget, GRPO in detail
and its comparison to PPO, the increasing-output-length pathology and its fixes
(DAPO, Dr. GRPO), and the full DeepSeek R1-Zero and R1 training recipes including
distillation.

The lecture starts from the honest list of what vanilla LLMs can and cannot do:

**Strengths.** Great at imitation and idea generation; amazing at generating and
debugging code.

**Weaknesses.** **Limited reasoning** (today's focus); **knowledge is static**
and **cannot perform actions** (Lecture 7); **hard to evaluate** (Lecture 8).

---

## 6.1 What counts as reasoning

**Tentative definition: reasoning = the ability to solve a problem.**

The lecture draws the line with two examples:

| Not reasoning | Reasoning |
|---|---|
| *"What is the course code of Stanford's Transformers & LLMs class?"* | *"The bear was born in 2020. How old is this bear now?"* |

The first is **retrieval**: the answer either is or is not stored in the weights,
and no amount of thinking helps. The second requires **combining** given
information with other knowledge (the current year) via an operation
(subtraction) to produce something that was never stored anywhere.

### The core idea

**Strategy: teach the model to explain its reasoning before answering** — chain
of thought (Wei et al., 2022), as introduced in Lecture 3.

Without CoT the exchange is terse and fragile:

```
Q: How old is this bear?          A: 4.
Q: How old will the bear be next year?     →     A: 5.
```

With CoT in the demonstration, the model imitates the working:

```
Q: How old is this bear?
A: The bear was born in 2020. It is therefore 4.
Q: How old will the bear be next year?
→ A: It will be one year older than its age this year, which was 4.
     Hence, it will be 5.
```

**The idea behind reasoning models: do CoT, but at a much larger scale.**

### The paradigm shift

```
Until now:      Question ──→ LLM ──→ Answer

New paradigm:   Question ──→ LLM ──→ Reasoning chain ──→ Answer

                Output = Reasoning + Answer
```

The reasoning is not a prompting trick applied from outside; it becomes part of
what the model *is trained to produce*.

> **Intuition — restating the depth-into-length argument, because everything in
> this lecture rests on it.** A Transformer with $N$ layers performs a fixed
> number of sequential computation steps per token. A problem needing more
> sequential steps than that is structurally out of reach in one forward pass — no
> amount of extra width or training fixes it. Chain of thought escapes the limit
> by *externalising* intermediate state into the token stream: each generated
> token gets a full forward pass, and previous conclusions are written into the
> context where attention can retrieve them. The generated text becomes a scratch
> pad, and the model's effective computational depth becomes
> $N \times (\text{number of tokens it chooses to think for})$. This is the mechanism behind
> the phrase **test-time scaling**: you can buy more capability at inference by
> spending more tokens, without changing the weights at all.

### Reasoning model releases

The lecture shows a (non-exhaustive, not-to-scale) timeline of first releases from
major labs:

| Date | Release |
|---|---|
| 2024-09-12 | OpenAI o1-preview |
| 2024-12-19 | Gemini 2.0 Flash Thinking |
| 2025-01-20 | DeepSeek R1 |
| 2025-02-19 | Grok 3 Beta |
| 2025-02-24 | Claude 3.7 Sonnet |
| 2025-06-10 | Magistral |

### How to spot a reasoning model

Two tells, both practical:

**In the product.** You see a **"thought summary"** — a paraphrase of the model's
thinking — while the **complete chain of thought is usually hidden**.

**In the pricing / API docs.** OpenAI, Anthropic and Google all expose
reasoning-specific parameters and bill for **reasoning tokens** as a separate
line item: tokens you pay for but never see.

> **Intuition — why hide the chain of thought?** Several reasons stack up.
> Commercially, the raw traces are the most valuable training data a lab produces
> — expose them and competitors can distil your model, which is exactly what
> section 6.5 describes. Behaviourally, labs have found that models reason better
> when the trace is *not* optimised for human consumption; the moment you train
> the visible thinking to look presentable, you are applying pressure that can
> degrade its usefulness as computation. And safety-wise, an unfiltered trace can
> contain the model exploring ideas it will correctly decline to state in its
> answer. The billing point matters practically: reasoning models can emit
> thousands of hidden tokens per query, so cost and latency are far higher than
> the visible output suggests.

---

## 6.2 Benchmarks and metrics for reasoning

### Coding benchmarks

The structure is *problem → solution → verification*:

- **Problem:** *"You have $n$ teddy bears in a line. Each bear has a size. Find
  the biggest bear that is smaller than the largest bear."*

- **Solution:** the model emits code, e.g.
  ```python
  def second_biggest_bear(bears):
      largest = max(bears)
      return max(b for b in bears if b < largest)
  ```

- **Verification:** run the hidden test cases. All pass → correct.

**Examples:** HumanEval, CodeForces, SWE-bench.

### Maths benchmarks

Structure: *problem → reasoning → compare against ground truth*:

- **Problem:** *"The bear was born in 2020. How old is the bear now?"*
- **Reasoning:** *"It is 2025 now. Subtract the birth year from the current year:
  2025 − 2020 = 5. Answer: 5"*

- **Ground truth:** `5` → verification.

**Examples:** AIME, GSM8K.

> **Intuition — why coding and maths, and not essay writing?** Because both admit
> **automatic, cheap, unambiguous verification**. Code either passes the tests or
> it does not; a numeric answer either matches or it does not. No human, no reward
> model, no judge. That property is not merely convenient for benchmarking — it is
> the entire reason reasoning models exist, as section 6.3 makes explicit. The
> domains where reasoning models are strongest are precisely the domains where a
> reward can be computed by a script.

### Pass@k

**"Probability that at least 1 of k attempts succeeds."**

Estimated unbiasedly from $n$ sampled attempts of which $c$ succeed:

$$\mathrm{Pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

i.e. one minus the probability that a random subset of $k$ attempts contains no
successful one.

- **Pass@k** — for use cases where **checking is easy** or you can afford higher
  latency.

- **Pass@1** — the special case, for use cases where you care about **a single
  generation**.

### Cons@k

**"Consensus at k"** — equivalent to taking the answer from **majority voting**
over $k$ samples and comparing that with the ground truth (DeepSeek-R1, 2025).

> **Intuition — three metrics, three different questions.** *Pass@k* measures
> whether the ability is present *at all* — can the model find the answer if given
> $k$ tries and a perfect verifier? It is the right metric when you have a checker
> (compile, run tests) and can retry. *Cons@k* asks whether the model finds the
> answer *reliably enough that its own majority agrees*, which is the right metric
> when you have no verifier and must trust one output; it is exactly Lecture 3's
> self-consistency turned into a score. *Pass@1* is what a user experiences.
> Reporting Pass@64 alongside Pass@1 is informative: a large gap means the
> knowledge is in there but not reliably surfaced, which is a training problem
> rather than a capability ceiling. **Lecture 8 adds a fourth relative,
> $\mathrm{Pass}^k$** — the probability that *all* $k$ attempts succeed — which measures
> consistency instead of capability. Do not confuse $\mathrm{Pass@}k$ with $\mathrm{Pass}^k$; they
> are almost opposites.

---

## 6.3 Test-time scaling with RL

**Idea: incentivise the model to reason before answering.**

**Considerations that shape the method:**

- **A reasoning chain is hard to write from scratch** — building SFT data by hand
  is impractical. Nobody wants to hand-write ten thousand step-by-step solutions,
  and even fewer people can.

- **We don't want to limit the model to human-written reasoning.** Human traces
  are polished and post-hoc; the model may need to backtrack, try things, and
  check itself in ways a textbook solution never shows.

- **There is a natural verifiable reward**: *"did it solve the problem?"* → yes
  or no.

**Conclusion: let's try RL!**

> **Intuition — this is the pivotal argument of the lecture.** Recall Lecture 5:
> the hard part of RLHF was that "good response" is subjective, so you had to
> train a reward model on human preferences and then constantly guard against
> hacking it. For maths and code that entire apparatus is unnecessary — the reward
> is a **fact**, computable by a script, unhackable in principle because there is
> nothing to approximate. Which means you can run RL *at scale*, cheaply, with no
> human in the loop and no reward model to overfit. That is what changed between
> RLHF and reasoning models. The interesting consequence: nobody tells the model
> *how* to reason. You only tell it whether it got the right answer, and the
> reasoning strategy is discovered.

### Reward 1: verify that the CoT is there

Enforce a **template**:

```
<think>
  ... reasoning ...
</think>
ANSWER
```

and give reward for a response that respects it. Purely structural: are the
delimiters present and well-formed?

### Reward 2: verify that the solution is correct

- **Code verification** — run test cases 1..c; all → reward.
- **Maths verification** — extract the final answer, compare with ground truth;
  match → reward.

### Total reward

$$\text{Rewards} = \underbrace{\text{formatting}}_{\text{think delimiters?}}
\;+\; \underbrace{\text{accuracy}}_{\text{correct solution?}}$$

> **Intuition — why the formatting reward is not just cosmetics.** Two jobs.
> Practically, it makes the answer extractable: you cannot verify accuracy if you
> cannot find where the answer is, so the format reward bootstraps the accuracy
> reward. Behaviourally, it **creates the space to think**. By rewarding the
> presence of a `<think>` block, you are telling the model that emitting reasoning
> before answering is itself desirable — and once that space exists, the accuracy
> reward shapes what goes inside it. It is a remarkably cheap piece of scaffolding
> for the amount of behaviour it unlocks.

### Controlling thinking at inference time

**Problem: not all prompts are equal.** *"What is 2+2?"* does not deserve four
thousand reasoning tokens, and a model that always thinks maximally is
unaffordable.

**Ideas to control "thinking":**

- **Dynamic budget** (*Token-Budget-Aware LLM Reasoning*, Han et al., 2024) —
  estimate how many tokens a question warrants and allocate accordingly.

- **Context awareness** — make the model aware of its own remaining budget so it
  can pace itself.

- **Budget forcing** (*s1: Simple test-time scaling*, Muennighoff et al., 2025) —
  crude and effective: to stop early, inject the `</think>` token; to force
  *more* thinking, suppress `</think>` and append the word **"Wait"**, which
  reliably makes the model second-guess and continue.

- **"Continuous" thoughts** (*Training LLMs to Reason in a Continuous Latent
  Space*, Hao et al., 2024) — reason in the continuous hidden space rather than by
  emitting discrete tokens, avoiding the information bottleneck of having to
  commit to a token at every step.

> **Intuition.** The s1 "Wait" trick is worth pausing on because it is so
> revealing. Appending a single word measurably improves accuracy, because it
> pushes the model into a distribution where self-correction follows. It tells you
> that the reasoning behaviour is a *mode* the model can be nudged into, not a
> fixed procedure — and that inference-time control over thinking length is a real
> and cheap lever. Continuous thoughts are the more ambitious direction: every
> token in a normal chain of thought compresses a rich hidden state down to one
> choice from the vocabulary, which is a substantial information loss at every
> step.

---

## 6.4 GRPO

**GRPO = Group Relative Policy Optimisation** (DeepSeekMath, Shao et al., 2024).

The objective has the same shape as PPO:

$$\text{maximise}\quad
\underbrace{\mathbb{E}\big[A\big]}_{\text{maximise advantages}}
\;-\; \beta\,
\underbrace{\mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\big)}_{\text{don't deviate from old/base}}$$

**The big difference from PPO:**

$$\begin{aligned}
\textbf{PPO:}\quad &\text{Advantage} \approx \text{Reward} - \text{Value function(state)}
&& \gets \text{a trained value network}\\[4pt]
\textbf{GRPO:}\quad &\text{Advantage} \approx \text{Reward} - \mathrm{Avg}(\text{reward of group})
&& \gets \text{no value network at all}
\end{aligned}$$

**Mechanically:** for a given prompt, sample a **group** of $G$ responses from the
current policy. Score them all. Compute each response's advantage by
standardising within the group:

$$A_i = \frac{r_i - \mathrm{mean}(r_1 \dots r_G)}{\mathrm{std}(r_1 \dots r_G)}$$

That advantage is then assigned to **every token** of response $i$.

### GRPO versus PPO, side by side

**Similarities:** the probability **ratio** between new and old policy, and
**clipping** of that ratio. Both inherited directly from PPO.

**Differences:** the **KL penalty** (GRPO places it as an explicit term in the
objective, computed per token against the reference, rather than folding it into
the reward signal) and — the important one — **advantage estimation**.

> **Intuition — why dropping the value model is such a big deal.** The value
> network in PPO exists solely to estimate "what reward should I expect from this
> state?" so that the advantage can be centred. It is expensive: a second model of
> comparable size to the policy, trained simultaneously, and notoriously hard to
> fit well — a bad value estimate poisons every advantage.
>
> GRPO's observation is that if you are going to sample multiple responses to the
> same prompt anyway (and for reasoning you are, because you want diversity to
> learn from), then **the group's own average reward is a perfectly good baseline
> for that prompt** — and an unbiased one, obtained for free. This removes one of
> PPO's four models and a large slice of its instability. It also makes intuitive
> sense of what the model learns: "of the eight attempts I made at this problem,
> these three were better than my average, so do more of that." Automatic
> difficulty calibration comes along for free — on an easy prompt where all eight
> succeed, every advantage is zero and nothing is learned, so training compute
> naturally concentrates on problems at the edge of the model's ability.
>
> This is the same "subtract a baseline to cut variance" idea from Lecture 5, with
> a Monte Carlo estimate replacing a learned function approximator.

---

## 6.5 The increasing-output-length pathology

**Observation: response length keeps increasing with RL training.** DeepSeek-R1's
training curves show average response length growing steadily, and this was
initially celebrated as the model "learning to think longer".

The lecture then shows it is partly an **artifact of the loss normalisation**.

**The problem.** GRPO's loss sums token-level terms and normalises by the number
of tokens in the response, $1/|y_i|$. That means each token in a **short** output
carries a **large** weight, and each token in a **long** output carries a
**small** weight.

Consequence: when a response is bad (negative advantage), a long one is punished
*less per token* than a short one. When a response is good, a long one is
rewarded less per token. Net effect on the gradient: **being long is a cheap way
to dilute penalties**. That is a **bad incentive** — the model learns to ramble
because rambling reduces the downside of being wrong.

**Remedy: equalise token-level contributions.** Two published fixes:

- **DAPO** (Yu et al., 2025, *An Open-Source LLM Reinforcement Learning System at
  Scale*) — normalise by the **total number of tokens across the whole group**
  rather than per response, so every token in the batch carries identical weight
  regardless of which response it belongs to.

- **Dr. GRPO** (Liu et al., 2025, *Understanding R1-Zero-Like Training: A
  Critical Perspective*) — remove the length normalisation term (and the standard
  deviation normalisation) altogether, on the grounds that they are the source of
  the bias.

> **Intuition — a lesson worth generalising far beyond this case.** The length
> growth looked like the model discovering that thinking longer helps. Some of it
> was. But a measurable part was the model exploiting an accounting artifact in
> the loss. This is reward hacking again, except the hacked object is not the
> reward model but the **normalisation constant in the objective** — a place
> nobody thought to look. The general lesson: whenever your model develops a
> surprising behaviour during RL, check whether that behaviour is *actually*
> optimal for the objective as literally implemented, rather than for the
> objective as you intended it.

### Other adjustments being explored

- **Bias linked to level of difficulty** (Dr. GRPO) — dividing by the group's
  standard deviation gives questions where responses happen to be similar an
  outsized gradient, which biases learning towards problems that are uniformly
  easy or uniformly hard.

- **Encourage diversity** (DAPO) — use asymmetric clipping ("clip-higher"),
  raising the upper clip bound so that low-probability tokens can be reinforced;
  without it the policy's entropy collapses and it stops exploring.

**...among others!** The area is moving fast and the design space is not settled.

---

## 6.6 Putting it together: the DeepSeek recipes

The model family, laid out:

```
V3-Base ──→ V3            "traditional" model
   │
   ├──→ R1-Zero           reasoning model — proof of concept
   └──→ R1                reasoning model — full pipeline
```

### R1-Zero's training recipe

**Step 1 — pretrain the model with "traditional" techniques: V3-Base.**
A **MoE** model, **~671B total parameters, ~37B active** per token.

**Step 2 — GRPO with reasoning data → R1-Zero.** No SFT at all. The exact
template used:

```
A conversation between User and Assistant. The user asks a question, and the
Assistant solves it. The assistant first thinks about the reasoning process
in the mind and then provides the user with the answer. The reasoning process
and answer are enclosed within <think> </think> and <answer> </answer> tags,
respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.
User: <this placeholder is replaced by a reasoning query>
Assistant:
```

| Benefits | Challenges |
|---|---|
| **reasoning abilities without any SFT** | chains of reasoning have **formatting and readability issues** |

> **Intuition — why R1-Zero was the headline result.** Everyone had assumed that
> reasoning had to be *taught* — that you needed human-written chains of thought
> to imitate before RL could refine them. R1-Zero showed you can go straight from
> a base model to a reasoning model with **pure RL on verifiable rewards**, and
> that sophisticated behaviours (self-verification, backtracking, spontaneously
> allocating more thinking to harder problems) **emerge** rather than being
> demonstrated. The reported "aha moment" — the model writing something like
> "wait, let me reconsider" without ever being shown such a phrase — is the
> emblematic example.
>
> The cost was legibility: with only formatting and accuracy rewarded, the traces
> drift into whatever is computationally useful, including mixing languages
> mid-thought and idiosyncratic notation. Nothing in the reward cared whether a
> human could read it.

### R1's training recipe — the full five-stage pipeline

**Step 1 — pretrain with "traditional" techniques: V3-Base** (MoE, ~671B total /
~37B active).

**Step 2 — "small-scale" SFT with reasoning data.**
*Data source: long chains of thought generated with R1-Zero and then rewritten by
humans.* A few thousand examples, cleaned for readability.

**Step 3 — GRPO with reasoning data.**
Approximately the same RL process as R1-Zero.
**Reward = formatting + accuracy + language consistency.** (Language consistency
is the new term, added specifically to stop the language-mixing.)

**Step 4 — "large-scale" SFT with reasoning and non-reasoning data.**
Two pools:

- **~600k pairs** of **maths, coding, logic** data, obtained by **rejection
  sampling** of "R1-so-far" responses — generate many, keep only those that pass
  rule-based checks or are approved by a V3 judge;

- **~200k pairs** of **"general" data**, mostly reusing V3's existing SFT data.

**Step 5 — GRPO with reasoning and non-reasoning data → R1.**
Two reward regimes running together:

- on **maths, coding, logic**: **reward = formatting + accuracy** (verifiable);
- on **"general" data**: **reward = helpfulness + harmlessness**, mostly reusing
  V3's RL data (i.e. classic RLHF-style preference reward).

> **Intuition — read the pipeline as a set of repairs to specific problems.**
> Step 2 exists to fix R1-Zero's readability, using R1-Zero itself to produce the
> raw material — the model bootstraps its own SFT data, which is the only reason
> hand-writing thousands of chains became unnecessary. Step 3 pushes reasoning
> ability up from a legible starting point. Step 4 exists because a
> reasoning-only model is *worse at ordinary conversation* than the model it came
> from — pure reasoning RL causes forgetting of general helpfulness — so general
> capability is folded back in, with rejection sampling ensuring the reasoning
> data is high quality. Step 5 aligns the whole thing on both axes at once, which
> is necessary because tuning them sequentially would let each undo the other.
>
> The general principle: **you cannot align one capability at a time.** Each RL
> stage degrades what the previous stage optimised, so the final stage has to
> carry every objective simultaneously.

### Results

R1's benchmark results were competitive with the leading closed reasoning models
of the time, from an openly published recipe — which is why this paper reset
expectations across the field.

### And what about distillation?

The lecture is careful to distinguish two different things that share the name:

| Distillation in Lecture 2 | Distillation used here |
|---|---|
| **Goal: match the next-token distribution** | **Goal: SFT-learn reasoning traces** |
| targeted, token-level, needs the teacher's full logits | R1 **generates entire responses**; the student is trained on them as ordinary SFT data |

So R1-Distill models (Qwen and LLaMA bases finetuned on R1's outputs) are made by
**plain supervised finetuning on R1's generated reasoning traces**.

**Results:**

- **competitive** — the distilled small models substantially outperform their
  bases and rival much larger models on reasoning benchmarks,

- a **"good" use of compute** — the paper notes that running RL directly on a
  small model produces *worse* results than distilling from a large model that
  already went through RL.

> **Intuition — why distillation beats running RL on the small model, which is
> the most practically useful takeaway in the lecture.** RL has to *discover*
> good reasoning by trial and error, and discovery requires enough baseline
> capability to occasionally stumble onto correct solutions — otherwise every
> sample gets zero reward and there is no gradient at all. Small models rarely
> get there. But **imitating** a correct reasoning trace is a much easier
> learning problem than finding one. So the efficient division of labour is:
> spend the enormous RL compute once on your largest model, let it discover how
> to reason, then transfer that behaviour to small models by cheap supervised
> finetuning on its outputs. Note also that this form of distillation needs only
> the teacher's *text*, not its logits — which is why it works across model
> families and architectures, and why it is so hard for labs to prevent.

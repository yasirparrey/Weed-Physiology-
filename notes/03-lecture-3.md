# Lecture 3 — Large Language Models

**Video length: 1h49.** Covered here: what "LLM" actually means, mixture of
experts, how a token is chosen at inference (greedy, beam search, sampling,
top-k, top-p, temperature, guided decoding), context length and context rot,
prompt structure, in-context learning, chain of thought, self-consistency, and
then the full catalogue of inference optimisations (KV caching, GQA,
PagedAttention, multi-head latent attention, speculative decoding, multi-token
prediction).

---

## 3.1 What an LLM is

**LLM = Large Language Model.** Taking the words in order:

**"Language model"** — *a statistical or machine learning model that assigns
probabilities to sequences of tokens.* Nothing more. Given a sequence, it tells
you how likely that sequence is; equivalently, given a prefix, it gives you a
distribution over what comes next.

**"Large"** — quantified in the lecture as:

- **model size:** billions of parameters or more,
- **training data:** hundreds of billions of tokens or more,
- **compute:** a lot of GPUs.

**Architecture:** a **decoder-only Transformer-based model**. Examples: the GPT
series, LLaMA, Gemma, DeepSeek, Mistral, Qwen.

> **Intuition — take the definition seriously, because it explains the failure
> modes.** An LLM is a probability distribution over token sequences, fitted to
> a corpus. It is not a database, not a reasoner, not a fact-checker. Everything
> it does well and everything it does badly follows from this. It hallucinates
> because a fluent-but-false continuation can be more probable than an accurate
> one, given what the corpus rewarded. It has a knowledge cutoff because the
> distribution was fitted at one moment in time. It is sycophantic when
> agreement is the more probable continuation. Lectures 5 through 8 are, in a
> real sense, nine hours of engineering around the consequences of this one
> definition.

---

## 3.2 Mixture of Experts (MoE)

### The motivation

A very large dense model applies **all** of its weights to **every** input
token. The observation that starts MoE: **not all weights are useful in the
forward pass** for a given token. The word "the" does not need the same
computation as a snippet of Rust.

So: replace one huge block with `n` **experts** `E₁, ..., E_n` plus a small
**gating network** `G`. For input `x`, the gate produces weights over experts,
and the output is a combination of expert outputs.

- **Dense MoE** — the output is the weighted average of **all** expert outputs,
  with weights like `[0.1, 0.8, 0.05, ...]` from the gate.

- **Sparse MoE** — the output is the weighted average of only the **selected**
  expert outputs, chosen by **top-k selection** (Shazeer et al., 2017,
  *Outrageously Large Neural Networks*).

Sparse is the interesting one, because only `k` experts actually run.

### Where MoE goes in a Transformer

**The FFNN sub-layer is replaced by the MoE block.** So instead of one
feed-forward network per layer you have `FFNN₁ ... FFNN_n` plus a gate `G`. And
critically: **routing is done for each token**, independently — not per sequence,
not per batch. Token 5 may go to experts 2 and 17 while token 6 goes to experts
1 and 9.

> **Intuition — why replace the FFNN rather than attention?** Two reasons. The
> FFNN is where most of the parameters live (with `d_FF = 4·d_model`, the two
> FFNN matrices hold roughly twice the parameters of all four attention
> projections combined), so that is where sparsity buys the most. And the FFNN is
> already position-independent — it acts on each token separately — so routing
> per token requires no change to the computational structure. Attention, by
> contrast, is inherently about interactions *between* positions; you cannot
> route it per token without breaking what it does.
>
> **The economics of MoE, stated plainly.** Model quality scales with total
> parameter count. Inference cost scales with *active* parameters per token. A
> dense model forces these to be equal. MoE decouples them. DeepSeek-V3 (seen in
> Lecture 6) has **~671B total parameters but only ~37B active** per token — so
> it holds the knowledge of a 671B model at roughly the serving cost of a 37B
> one. You still pay the full 671B in GPU *memory*, since any expert might be
> needed, which is why MoE is a compute-and-latency win rather than a
> memory-and-cost win across the board.

### Training challenge: routing collapse

**Symptom:** the same expert gets selected most of the time. The routing
distribution collapses onto a few experts and the rest never train.

**Remedy:** force the other experts to be "part of the game" via an
**auxiliary loss** (Switch Transformers, Fedus et al., 2021):

```
L_aux = α · n · Σ_i  f_i · P_i
```

where `f_i` is the **fraction of tokens routed to expert `i`** and `P_i` is the
**average routing probability for expert `i`**.

> **Intuition — why that particular product.** The sum `Σ f_i·P_i` is minimised,
> subject to both vectors summing to 1, when the load is spread uniformly; it
> blows up when one expert takes both a large share of tokens and a high average
> probability. Multiplying the two is the trick: `f_i` is a hard count and has no
> gradient, while `P_i` is differentiable. So the gradient flows into the gate
> through `P_i`, scaled by how overloaded that expert currently is — the more
> tokens expert `i` is hogging, the harder the loss pushes the gate's probability
> for it down.
>
> Why does collapse happen at all? A classic rich-get-richer loop. An expert that
> is picked slightly more often trains slightly faster, becomes slightly better,
> and is therefore picked more often still. Without an explicit counterweight
> you end up with a very expensive model that behaves like a small dense one.
> There is also a hard systems reason for balance: experts are distributed across
> GPUs, and if one GPU receives most of the tokens, every other GPU sits idle
> waiting for it.

### Do experts specialise interpretably?

The lecture shows the analysis from **Mixtral of Experts** (Jiang et al., 2024),
colouring tokens by which expert handled them. The honest answer from that paper
is that specialisation is **not** cleanly semantic — experts do not divide into
"the maths expert" and "the French expert". The observed structure is more
syntactic and positional, with some tendency for consecutive tokens to go to the
same expert.

> **Intuition.** This is a useful expectation to hold. MoE is a *capacity* trick
> — a way to hold more parameters without paying for all of them — not an
> interpretability tool. Do not expect to open up a MoE and find human-legible
> modules.

---

## 3.3 Generating a response

### Next-token prediction, as a loop

The model is fed a prefix and produces one token; that token is appended and the
whole thing is fed back:

```
[BOS]                      → A
[BOS] A                    → teddy
[BOS] A teddy              → bear
[BOS] A teddy bear         → is
[BOS] A teddy bear is      → ?
```

Mechanically, at each step: run the decoder stack, take the final-layer vector
at the **last** position, project it to vocabulary size `V` with the output
matrix to get **logits**, and turn logits into probabilities.

### Where the probabilities come from — softmax with temperature

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

where `z` are the logits and `T` is the **temperature**.

**Impact of temperature:**

- **Small `T`** — dividing by a small number magnifies the differences between
  logits, so the distribution becomes **sharper**; in the limit `T → 0` it
  becomes a point mass on the argmax (equivalent to greedy decoding).

- **High `T`** — differences are flattened, the distribution becomes more
  **uniform**; in the limit `T → ∞` you sample uniformly at random from the
  vocabulary.

> **Intuition.** Temperature does not change the *ranking* of tokens, only how
> much probability mass the leaders keep. `T < 1` for tasks with one right answer
> (extraction, classification, code, LLM-as-a-judge in Lecture 8); `T` around 1
> for creative writing. `T` above about 1.2 usually produces incoherence, because
> the tail of a `V`-sized vocabulary contains an enormous amount of nonsense and
> flattening the distribution hands it real probability. The suggested reading
> "Defeating Nondeterminism in LLM Inference" (He et al., 2025) makes the further
> point that even `T = 0` is not truly deterministic in practice — batching and
> floating-point reduction order change results run to run.

### Decoding strategy 1: greedy decoding

Take the token with the highest predicted probability, every time.

**Limitations:** the output is not optimal, not natural, and not diverse. Two
distinct failures. *Not optimal*, because the highest-probability first token
does not necessarily begin the highest-probability *sequence* — a locally greedy
choice can walk you into a dead end. *Not natural*, because real human text is
not the most probable text; greedy decoding produces flat, repetitive prose and
often gets stuck in loops.

### Decoding strategy 2: beam search

**Keep the `k` paths that are the most likely.** At each step, expand every one
of the `k` surviving beams by every possible next token, score the resulting
sequences by cumulative probability, and keep the best `k`.

The lecture's worked example with `k = 3`: from `[BOS]` the candidates are
`a`, `cute`, `the`; expanding `cute` gives `cute teddy` (0.7), `cute bear`,
`cute fluffy` (0.2), and so on; the search continues until beams terminate at
`[EOS]`.

**Limitations:** needs more computation, and **lacks diversity/creativity** —
the `k` beams tend to be near-identical variations of one another, differing by
a word.

> **Intuition.** Beam search dominates machine translation, where there really is
> a single best answer and you want the highest-probability sequence. It is
> almost never used for open-ended generation, because "most probable" and
> "interesting" are close to opposites for creative text. There is also a
> well-known pathology: longer sequences have lower cumulative probability
> (you keep multiplying by numbers below 1), so beam search is biased towards
> short outputs unless you apply a length normalisation.

### Decoding strategy 3: sampling

**Sample the next token from the probability distribution** rather than taking
the maximum. This restores diversity — but naive sampling from the full
distribution occasionally draws a genuinely terrible token from the long tail,
and one bad token derails everything that follows. Hence two truncation
strategies:

- **Top-k** — sample among the `k` most probable tokens only (e.g. `k = 4`),
  renormalising over them.

- **Top-p (nucleus)** — sample from the **smallest set of tokens whose
  cumulative probability is ≥ `p`** (e.g. `p = 90%`).

> **Intuition — why top-p is generally preferred over top-k.** The right number
> of plausible next tokens varies enormously with context. After "The capital of
> France is" there is essentially one; after "She opened the door and saw" there
> are thousands. Top-k with a fixed `k` is either too permissive in the first
> case (admitting three wrong countries) or too restrictive in the second
> (arbitrarily cutting off good options). Top-p adapts automatically: it takes
> few tokens when the model is confident and many when it is not, because it
> is defined on the probability mass rather than the count. In practice the two
> are often combined, with top-k as a hard safety ceiling.

### Guided (constrained) decoding

**Motivation:** you need output in a specific format. Given the prompt *"Generate
a description of my 33-year-old teddy bear who likes reading. Do this in JSON
format."*, you want exactly:

```json
{
  "first_name": "teddy",
  "last_name": "bear",
  "age": 33,
  "hobby": "reading"
}
```

**Idea: only allow "valid" next tokens.** At each step you know, from a grammar
or schema, which tokens could legally come next; you mask the logits of all
others to `−∞` before the softmax, so they cannot be sampled at all.

The lecture's step-by-step: from `[BOS]` the candidates might be `{`, `the`, `}`
— only `{` is legal for JSON, so it is forced. Having emitted `{`, the candidates
are `road`, `sun`, `"first_name"` — only a quoted key is legal. After the key,
candidates `,`, `:`, `to` — only `:` is legal. After the colon, `10`, `"Jane"`,
`"teddy"` — now several are legal (any string), so the model genuinely chooses.
And so on.

> **Intuition — this is a hard guarantee, not a suggestion.** The difference
> between "please reply in JSON" in the prompt and guided decoding is the
> difference between a request and an enforcement. Asking nicely works most of the
> time and fails unpredictably — an apologetic preamble, a trailing comma, a
> markdown fence — which is exactly the kind of failure that breaks a production
> pipeline at 3am. Constrained decoding makes malformed output *impossible*,
> because invalid tokens are never candidates. Note that the model's *content*
> is still its own; you are only restricting the shape. Lecture 8 uses this
> directly to make LLM-as-a-judge outputs parseable, and it is what powers the
> "structured output" APIs from OpenAI, Anthropic and Google.

---

## 3.4 Prompting

### Context length

**Context length** (also *context size*, *window size*) is the maximum number of
tokens the model can attend over — prompt plus generated output together.

The lecture notes that **orders of magnitude vary by input type and by model**,
and then delivers the important caveat: **beware of "context rot"** (Hong et al.,
2025, *Context Rot: How Increasing Input Tokens Impacts LLM Performance*).

> **Intuition — the advertised window is not the usable window.** A model
> advertising a 1M-token context can *accept* a million tokens; that is a
> statement about the position encoding and the memory budget, not about
> comprehension. Measured performance degrades well before the limit, and it
> degrades non-uniformly: information at the very start and very end of a long
> context is retrieved much more reliably than information in the middle (the
> "lost in the middle" effect, which Lecture 7 revisits under
> *needle-in-a-haystack* testing). The practical rule is that a short, curated
> context beats a long, padded one — which is the entire justification for
> retrieval in Lecture 7. Stuffing everything you have into a long window is
> both more expensive and *less accurate*.

### The structure of a good prompt

The lecture decomposes a prompt into four parts:

- **Context** — *"My teddy bear had a long day and needs a bedtime story."*
- **Instructions** — *"Generate a bedtime story that takes place in a specific
  location."*

- **Input** — *"Location: Country of teddy bears"*
- **Constraints** — *"The story needs to be suitable for teddy bears that are
  tired."*

### In-context learning (ICL)

**Brown et al., 2020, *Language Models are Few-Shot Learners*.**

| Zero-shot | Few-shot |
|---|---|
| the question is asked **without examples** | the prompt **contains examples** of input/output |
| performance depends heavily on the quality of the initial model | typically **better performance** |

**Discussion — showing examples in the prompt is generally better, but:**

- it requires effort (you have to write and curate good examples),
- it increases computational complexity and cost (you pay per input token),
- it increases latency.

> **Intuition — why this was startling.** No weights change. The model is
> performing something that looks like learning *purely inside the forward pass*,
> from information in its context. The mechanistic story is that attention can
> implement a form of pattern-matching over the examples: the demonstrations
> establish a mapping, and attention lets the query token retrieve the analogous
> case. Two practical consequences worth knowing. First, the *format* of the
> examples matters more than their correctness — several papers have found that
> examples with deliberately wrong labels still improve performance, because what
> they mostly convey is "here is the shape of the task and the shape of the
> answer". Second, few-shot prompting is the cheapest thing to try before
> finetuning, and it very often removes the need for it entirely.

### Chain of Thought (CoT)

**Wei et al., 2022.** **Idea: explaining the reasoning helps performance.**

Instead of a few-shot example that goes straight to the answer:

```
Q: How old is this bear?
A: 4.
```

give one that shows the working:

```
Q: How old is this bear?
A: The bear was born in 2020. It is therefore 4.
```

The model then imitates the *style* of reasoning on the new question:

```
Q: How old will the bear be next year?
A: It will be one year older than its age this year, which was 4.
   Hence, it will be 5.
```

**Discussion:** you gain interpretability and an explanation; you pay in **more
tokens: higher cost and latency**.

> **Intuition — why writing more helps a model think.** A Transformer performs a
> fixed amount of computation per generated token: `N` layers, once. Some problems
> genuinely need more sequential steps than that — a multi-step arithmetic
> problem, a logical deduction with intermediate conclusions. Forcing the answer
> out in one token asks the model to do all that work in one forward pass, which
> it structurally cannot. Chain of thought converts *depth* into *length*: each
> intermediate token gets its own full forward pass, and the intermediate results
> are written into the context where later steps can attend to them. The
> generated text becomes an external working memory. This single observation is
> the seed of the entire Lecture 6 — reasoning models are chain of thought scaled
> up and trained in, rather than prompted.

### Self-consistency

**Wang et al., 2022.** **Idea: aggregating over reasoning paths improves
performance.**

Sample **several** chains of thought for the same question (which requires
`T > 0`), extract the final answer from each, and take a **majority vote**:

- *"It will be one year older than its age this year, which was 4. Hence, it
  will be 5."* → 5

- *"The bear was born in 2020. It will therefore be 5."* → 5
- *"Next year is 2024. The bear will then be 4."* → 4

Majority answer: **5**.

**Discussion:** a trade-off between performance and added cost.

> **Intuition.** Reasoning chains fail in idiosyncratic ways — a slipped
> arithmetic step here, a misread premise there — but they tend to fail
> *differently* each time, while the correct path is a single attractor that many
> chains converge on. So errors scatter and correctness concentrates, and a vote
> recovers the signal. It is bagging, applied to reasoning. Note that this only
> works when the answer is easily extractable and comparable (a number, a
> multiple-choice letter); you cannot majority-vote over essays. Lecture 6 makes
> this a formal metric, **Cons@k**, and Lecture 5's Best-of-N is the same idea
> with a reward model doing the selecting instead of a vote.

---

## 3.5 Inference optimisations

### Framing the problem

**Motivation: computations are expensive — is there any way to reduce
complexity?** The lecture organises the answer into two categories, which is a
genuinely useful mental filing system:

**"Exact" efficiency** — same output, less work:

- avoid redundancies,
- memory management,
- reformulate the maths.

**Approximations** — accept a (hopefully tiny) change in output:

- architectural changes,
- embedding representations,
- token prediction.

And the summary mapping given at the end of the lecture:

| Category | Technique |
|---|---|
| Avoid redundancies | KV cache |
| Memory management | PagedAttention |
| Reformulate the maths | *(FlashAttention — Lecture 4)* |
| Architectural changes | Grouped-query attention |
| Embedding representations | Latent attention |
| Token prediction | Speculative decoding, multi-token prediction |

### KV caching (avoid redundancies)

**Motivation:** a new token needs to interact with all previous tokens. Naively,
generating token 100 means running attention over the whole prefix from
scratch — recomputing the keys and values for tokens 1–99 that you already
computed when generating token 99.

**Idea: keep keys and values in a cache.** At each step, compute `k` and `v` for
the *new* token only, append them to the cache, and compute attention using the
new query against the entire cached `K` and `V`.

> **Intuition — why K and V but not Q.** Each generation step has exactly one
> query: the one belonging to the token you are currently predicting from. That
> query is needed once and then never again. Keys and values, by contrast, are
> needed at *every* future step, and — crucially — they never change, because
> causal masking means token 5's key does not depend on token 6. So they are
> perfectly cacheable. The saving is large: it turns per-step cost from `O(n²)`
> to `O(n)`, making total generation `O(n²)` instead of `O(n³)`.
>
> **The cost is memory, and that cost drives the next three techniques.** The
> cache size is `2 · n_tokens · N_layers · n_kv_heads · d_head · bytes`. For a
> large model with a long context this reaches tens of gigabytes per sequence —
> often exceeding the model weights. Every optimisation that follows is an attack
> on this number: GQA reduces `n_kv_heads`, latent attention reduces `d_head`,
> PagedAttention reduces the waste in how it is stored.

This also explains the two distinct phases of LLM inference, worth naming even
though the lecture does not: **prefill**, where the whole prompt is processed in
one parallel pass and the cache is populated (compute-bound), and **decode**,
where tokens are produced one at a time (memory-bandwidth-bound, because you
must stream the entire cache and all the weights through the compute units for
each single token).

### Grouped-query attention (architectural change)

Repeated from Lecture 2, now with its true motivation. In vanilla MHA,
`#query heads = #key heads = #value heads = h`. In **GQA**, key/value heads are
**shared within groups of queries**: `#query = h`, `#key = #value = G < h`.
MQA is the `G = 1` extreme. The cache shrinks by a factor of `h/G`.

### PagedAttention (memory management)

**Kwon et al., 2023** — the paper behind **vLLM**.

**Observation: lots of memory is wasted when storing the KV cache.** The
standard approach reserves one contiguous block per sequence, sized for the
maximum possible length. If a request needs 2000 tokens but you reserved 32,000,
94% of that allocation is dead space — and you cannot use it for another request
because it must stay contiguous.

**Idea: store K and V in non-contiguous space to minimise wasted memory.** Break
the cache into fixed-size **blocks** ("pages"), allocate them on demand, and keep
a per-sequence block table mapping logical positions to physical blocks.

> **Intuition — this is literally operating-system virtual memory, applied to
> attention.** The analogy is exact: block table = page table, KV block = memory
> page, and the attention kernel is modified to follow the indirection. Three
> wins follow. Waste drops to at most one partially-filled block per sequence
> (a few percent instead of ~90%). Throughput rises, because you can fit many
> more concurrent sequences in the same GPU. And blocks become **shareable**:
> if ten requests share a long system prompt, they can point at the *same*
> physical blocks for that prefix instead of holding ten copies — which is the
> mechanism behind "prompt caching" pricing, and which Lecture 7 relies on for
> contextual retrieval.

### Multi-head latent attention (embedding representations)

**DeepSeek-V2.** **Goal: reduce the dimension of K and V stored in memory** —
the per-head `d_head` is *too big*.

**Solution: store compressed representations instead.**

- **Before:** for each token you store `h` full-size key vectors and `h`
  full-size value vectors.

- **After:** you store one **shared low-dimensional latent vector** per token.
  When attention needs them, the full keys and values for all heads are
  **reconstructed** from that latent by up-projection matrices.

So the down-projection to the latent is a compression you pay for once per token;
the up-projections are absorbed into the attention computation.

> **Intuition.** This is a learned autoencoder on the KV cache. Instead of the
> crude sharing of GQA — "these four heads must use literally the same keys" —
> MLA says "all heads' keys and values live in a shared low-dimensional subspace,
> and each head has its own learned readout from it". That is strictly more
> expressive than GQA at comparable cache size, which is why DeepSeek reported
> both a smaller cache *and* better quality than GQA. The clever part is that the
> up-projection matrices can be algebraically folded into `W_Q` and `W_O`, so you
> never actually materialise the full-size keys and values — the compression is
> free at inference. RoPE needs special handling here, since rotation does not
> commute with the folding, which is why the paper carries a small
> "decoupled RoPE" component alongside the latent.

### Speculative decoding (token prediction)

**Chen et al., 2023.** **Idea: use a draft (small) model to generate tokens that
are validated by a target (big) model.**

**The procedure:**

1. The **draft LLM** autoregressively generates `k` candidate tokens cheaply —
   e.g. from `[BOS] my teddy bear` it proposes `is cute and smart` — recording
   its probabilities `P₁, ..., P_k`.

2. The **target LLM** processes the whole proposed sequence
   `[BOS] my teddy bear is cute and smart` in **one parallel forward pass**,
   producing its own probabilities `Q₁, ..., Q_k, Q_{k+1}`.

3. Accept or reject each proposed token in order:
   - if `Q_i(token) ≥ P_i(token)` → **accept**;
   - otherwise → accept with probability `Q_i(token) / P_i(token)`, and
     **reject** with probability `1 − Q_i(token)/P_i(token)`.
   - **If a rejection happens, re-sample the next token from the residual
     distribution `[Q_i − P_i]₊` (normalised) and exit** the loop for this round.

Then start again from the new accepted prefix.

> **Intuition — the crucial guarantee, and where the speed comes from.** This
> accept/reject rule is *exact*: the resulting token sequence is distributed
> **identically** to sampling from the target model directly. Speculative
> decoding is not an approximation; it changes the speed and nothing else. The
> reason it wins is the asymmetry noted under KV caching: decoding is
> memory-bandwidth-bound, so verifying `k` tokens in one parallel pass costs
> barely more wall-clock time than generating one token, since either way you
> must stream the model's weights through the compute units once. If the draft
> model agrees with the target on, say, 70% of tokens — and it will, because most
> tokens in text are easy, being punctuation, common words, and forced
> continuations — you get most of those `k` tokens for the price of one big-model
> pass. Typical speedups are 2–3×.
>
> The `[Q_i − P_i]₊` residual is what makes the maths exact. When you reject, you
> must not simply sample from `Q` — the rejection itself carries information, and
> sampling from `Q` again would bias the result. You sample from the part of `Q`
> that the draft *under*-weighted, which precisely repairs the discrepancy.

### Multi-token prediction (MTP)

**Gloeckle et al., 2024.** **Idea: train `k` prediction heads**, so that from
position `t` the model directly predicts positions `t+1, t+2, ..., t+k`.

The selling point stated in the lecture: **the same model is both draft and
target**. You no longer need a separate small model — the extra heads produce
the speculative continuation, and the main head verifies it.

> **Intuition.** Beyond the inference speedup, there is a training benefit that
> the paper emphasises: predicting several tokens ahead forces the representation
> at position `t` to encode information about the *near future*, not merely the
> immediate next token. That is a denser learning signal and it measurably
> improves quality on tasks like code generation, where you must plan a few
> tokens ahead. DeepSeek-V3 uses MTP both as a training objective and as a
> built-in speculative decoder.

<!-- -->

> **Watch out — a summary of what is exact and what is not.** KV caching,
> PagedAttention and speculative decoding are **exact**: bit-for-bit (or
> distributionally) identical output, purely faster. GQA and latent attention are
> **architectural choices** that change the model, so they must be trained in and
> carry a (small) quality cost. Multi-token prediction changes the training
> objective. When someone tells you an inference optimisation is "free", this is
> the distinction to check.

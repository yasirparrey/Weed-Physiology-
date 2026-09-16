# Lecture 4 — LLM Training

**Video length: 1h47.** Covered here: the transfer-learning paradigm shift,
pretraining (objective, data mixtures, scale), FLOPs and scaling laws including
Chinchilla, the challenges and costs of pretraining, training optimisations
(data parallelism, ZeRO, model parallelism, FlashAttention, mixed precision),
supervised finetuning and instruction tuning, benchmarks and Chatbot Arena, and
then parameter-efficient finetuning with LoRA and QLoRA.

---

## 4.1 The paradigm shift

**Traditional ML:** given a task, train a model from scratch. Want spam
detection? Train a spam model. Want sentiment extraction? Train a sentiment
model. Want translation? Train a translation model. Three tasks, three models,
three labelled datasets, three training runs.

**Transfer learning:** reuse a trained model to some capacity.

**The LLM paradigm** applies this in two stages: *train an LLM to understand
language, then tune it for the end task.*

```
              Pretraining              "Tuning"
initialised  ─────────────→  pretrained  ─────────→  spam detection model
   model                        model                sentiment extraction model
                                                     translation model
```

> **Intuition — why this reordering was so consequential.** Language
> understanding is almost entirely shared across tasks: you cannot detect spam
> without knowing what words mean, and you cannot summarise without knowing
> syntax. Under the old paradigm, every project relearned all of that from its
> own small labelled dataset, which is both wasteful and hopeless — no sentiment
> dataset is large enough to teach English. The pretraining/tuning split pays the
> enormous cost of learning language *once*, from unlabelled text that is
> essentially free and effectively unlimited, and then amortises it over every
> downstream task. The economics inverted completely: pretraining costs millions
> and is done by a handful of labs; tuning costs very little and is done by
> everyone.

---

## 4.2 Pretraining

**Goal: learn the patterns of language and code.**

The lecture illustrates the data with a deliberately varied set of samples:
English prose about a teddy bear reading by the window; the same in French;
the same in Persian; a Python class `TeddyBear` with a `hug()` method; the
equivalent Go program. Natural language in many languages, and code in many
languages, all in the same mixture.

**Objective function: predict the next token.** That is all. Feed
`[BOS] A teddy bear is` and train the model to put high probability on the true
continuation, with cross-entropy loss, at every position simultaneously.

**Data mixtures:**

- web-scraped text (Common Crawl, Wikipedia),
- code (GitHub, StackOverflow).

**Size — approximately trillions of tokens:**

| Model | Pretraining size (tokens) |
|---|---|
| GPT-3 | 300 billion |
| LLaMA 3 | 15 trillion |

> **Intuition — why does next-token prediction produce *knowledge*?** It looks
> like a shallow objective, and the fact that it works is genuinely surprising. The
> argument is this: to predict the next token *as well as possible* over a corpus
> containing all of Wikipedia, you must implicitly learn the facts in Wikipedia,
> because guessing the token after "The capital of France is" correctly requires
> knowing the answer. To predict the closing brace of a Python function you must
> track scope. To predict the next line of a proof you must follow the argument.
> Compression and understanding converge: the *only* way to compress text well is
> to model the process that generated it. And the 50× jump from GPT-3's 300B
> tokens to LLaMA 3's 15T is not incidental — the Chinchilla result below explains
> exactly why that happened.

### Notation: FLOPs versus FLOPS

The lecture is careful here because the two are constantly confused:

- **FLOPs** — **FL**oating-point **OP**eration**s**. A *count*. "This training
  run took 3×10²³ FLOPs." A measure of work.

- **FLOPS** or **FLOP/s** — floating-point operations **per second**. A *rate*.
  "This GPU delivers 1000 TFLOPS." A measure of speed.

Work divided by rate gives time. Mixing them up makes every cost estimate
nonsense.

### Scaling laws

**Kaplan et al., 2020, *Scaling Laws for Neural Language Models*.** The finding:
loss follows a smooth **power law** in each of model size $N$, dataset size $D$,
and compute $C$, over many orders of magnitude:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}
\qquad
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}
\qquad
L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}$$

Plotted on log–log axes these are straight lines. The practical consequence is
extraordinary: you can train a series of small models, fit the line, and
**predict the loss of a model 100× larger before building it**.

**Sample efficiency** — the second finding: larger models reach any given loss
level in *fewer* tokens than smaller models. Bigger models learn faster per
example, not just better in the limit.

### The Chinchilla law

**Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models*.**

The question: given a fixed compute budget $C$, how should you split it between
making the model bigger and training on more tokens? Kaplan's original answer
leaned heavily towards bigger models. Chinchilla redid the experiment more
carefully and found something different:

$$C \approx 6ND \qquad \text{(compute} \approx 6 \times \text{parameters} \times \text{tokens)}$$

$$\text{optimal:}\quad D \approx 20N \qquad \text{— roughly 20 tokens per parameter}$$

Both $N$ and $D$ should scale as roughly $C^{0.5}$ — that is, **in equal
proportion**.

The headline demonstration: **Chinchilla (70B parameters, 1.4T tokens)
outperformed Gopher (280B parameters, 300B tokens)** using the *same* compute
budget. A model four times smaller, trained on five times more data, won.

> **Intuition — and why every model since is "overtrained".** The implication was
> that essentially every large model of the GPT-3 era was badly
> **undertrained**: far too large for the number of tokens it had seen. Money had
> been spent on parameters that never got enough data to be worth having. The
> field pivoted immediately, which is exactly why LLaMA 3 trained a modest model
> on 15 trillion tokens.
>
> There is an important refinement, though. Chinchilla optimises *training*
> compute. But a deployed model is trained once and served billions of times, and
> serving cost scales with parameter count. So if you weight inference cost at
> all, the optimum shifts even further towards *smaller models trained on even
> more tokens* than Chinchilla prescribes. LLaMA 3's 15T tokens for an 8B model is
> around 1,875 tokens per parameter — roughly 90× past the Chinchilla point. That
> is not a mistake; it is deliberately trading extra training compute for
> permanently cheaper inference. Recognising this distinction is one of the more
> useful things to take from the lecture.

### Challenges of pretraining

**Cost:**

- **at least millions of dollars**,
- it takes a long time (weeks to months on thousands of GPUs),
- environmental and electricity impact.

**Learned knowledge:**

- **"knowledge cutoff"** — the model knows nothing after its training data ends.
  The lecture shows this literally, as the cutoff date printed on OpenAI's GPT-5
  model page.

- **hard to edit knowledge** — there is no `UPDATE` statement for a fact stored
  distributed across billions of weights. Retraining is the blunt option;
  Lecture 7's RAG is the practical one.

- **"plagiarism"** — models can reproduce memorised training text verbatim,
  with the legal and ethical questions that entails.

---

## 4.3 Training optimisations

### What must be held in memory

Walking through one training step makes the memory budget explicit:

1. **Initialisation.** Model parameters: $O(\text{billions})$ to
   $O(\text{hundreds of billions})$.

2. **Forward pass** — compute the loss. This requires storing **activations**,
   needed later to compute gradients. Activation memory is a function of model
   size, **batch size**, and **context length**.

3. **Backward pass** — compute **gradients**, needed for the weights update. One
   gradient per parameter.

4. **Weights update** — apply the optimiser. This requires the **optimiser
   state**.

With the **Adam** optimiser:

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t && \text{first moment (momentum)}\\[1pt]
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 && \text{second moment (variance)}\\[1pt]
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^{\,t}} && \text{bias correction}\\[3pt]
\theta_t &= \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} && \text{parameter update}
\end{aligned}$$

*(Suggested readings: Kingma et al., 2014 for Adam; Loshchilov et al., 2017 for
AdamW's decoupled weight decay.)*

**Bottleneck: memory.** The lecture puts an NVIDIA H100 on screen and notes its
**limited memory, $O(\text{10s of GB})$** — 80 GB for the flagship.

> **Intuition — do the arithmetic, because it is shocking.** Take a 7B-parameter
> model in mixed-precision training. Parameters in FP32: 28 GB. Gradients: 28 GB.
> Adam's two moments: 56 GB. That is **112 GB before a single activation**, on a
> GPU with 80 GB. A 70B model needs over a terabyte. This is why distributed
> training is not an optimisation but a *requirement*, and why the optimiser state
> — 2 bytes-per-parameter × 2 moments — is the first thing anyone attacks.
> Note also that Adam costs 8 bytes per parameter of state versus SGD's 0; the
> reason nobody uses SGD anyway is that Adam's per-parameter adaptive step sizes
> are what make Transformer training stable at all.

### Data parallelism

**Idea:** divide the batch of data across devices; **replicate the model on each
device**. Each GPU computes gradients on its own shard of the batch; gradients
are averaged across GPUs (an all-reduce); every GPU applies the same update.

Each GPU therefore holds a full copy of: parameters, gradients, optimiser state.

### ZeRO — Zero Redundancy Optimisation

**Rajbhandari et al., 2019.** **Idea: that redundant information across devices
is pure waste.** Every GPU is storing an identical copy of things it only needs
part of at any moment. So shard them:

- **ZeRO-1** — shard the **optimiser state** across GPUs.
- **ZeRO-2** — shard the **optimiser state + gradients**.
- **ZeRO-3** — shard the **optimiser state + gradients + parameters**.

> **Intuition — and the trade-off.** ZeRO-1 is nearly free: the optimiser state
> is only needed at update time, so each GPU can update its own slice and share
> the result. Since Adam's state is the single largest consumer, this alone cuts
> memory by roughly half with negligible communication overhead. ZeRO-3 goes
> furthest — no GPU holds the whole model — but it means that during the forward
> pass each GPU must *gather* the parameters of each layer from its peers just
> before using it, then discard them. Memory scales down almost linearly with GPU
> count, at the price of substantially more communication. Hence the ladder: pick
> the lowest stage that fits.

### Model parallelism

**Idea: split the model computations across several devices.** Variations named
in the lecture:

- **Tensor Parallelism (TP)** — split individual matrices across GPUs, so a
  single matmul is performed cooperatively. Needs very fast interconnect;
  normally used *within* a node.

- **Pipeline Parallelism (PP)** — assign different *layers* to different GPUs;
  micro-batches flow through the pipeline. Cheap on communication, but suffers
  from pipeline "bubbles" where GPUs idle.

- **Sequence Parallelism (SP)** — split along the sequence dimension for the
  operations where that is valid.

- **Context Parallelism (CP)** — split the context/attention computation across
  devices, for very long sequences.

- **Expert Parallelism (EP)** — place different MoE experts on different GPUs.

*(Suggested reading: "The Ultra-Scale Playbook: Training LLMs on GPU Clusters",
HuggingFace, 2025.)*

> **Intuition.** Real frontier training runs combine several of these into a
> "4D parallelism" mesh — data × tensor × pipeline × expert — chosen to match the
> hardware topology, with the fastest-communicating dimension (tensor) inside a
> node and the slowest (pipeline, data) across nodes. The engineering here is a
> serious specialism in its own right.

### FlashAttention (Dao et al., 2022)

**Strategy: leverage the components of the GPU to speed up attention
computations.** The relevant hardware facts:

- **HBM** (high-bandwidth memory) — **big and slow**, tens of GB.
- **SRAM** (on-chip) — **small and fast**, tens of MB, orders of magnitude
  faster.

- **CU** — the compute units.

**What standard self-attention does**, step by step, with every data movement
made explicit:

```
LOAD  Q, K from HBM by blocks
COMPUTE S = QKᵀ
WRITE S to HBM              ← n × n matrix!
READ  S from HBM
COMPUTE P = softmax(S)
WRITE P to HBM              ← n × n matrix again!
LOAD  P, V from HBM by blocks
COMPUTE O = PV
WRITE O to HBM
```

The $n \times n$ score matrix is written to slow memory and read back — **twice**.

**Idea 1: minimise reads/writes to HBM with "tiling" via SRAM.** Load blocks of
$Q$, $K$, $V$ into SRAM, compute a block of the output $O$ entirely on-chip, and
write only $O$ back to HBM. The $n \times n$ intermediate is **never materialised in
HBM at all**.

**The trick that makes tiling possible:** *there is no need to compute the full
$QK^{\top}$ before applying softmax.* Softmax appears to need a global normaliser
(the sum over the whole row), but it can be computed incrementally: maintain a
running maximum and a running sum per row, and rescale the partial output as new
blocks arrive. This is the **online softmax**, and it is the mathematical heart
of FlashAttention.

**Idea 2 (backward pass): sometimes it is better to recompute than to store.**
Rather than keeping the attention matrix in HBM for the backward pass, throw it
away and recompute it with tiling via SRAM when needed. **More FLOPs, but less
runtime!**

**Result: FlashAttention gives a significant speedup with exact computation.**

> **Intuition — this is the single most important idea in the lecture for
> understanding modern ML performance.** Attention is not compute-bound, it is
> **memory-bandwidth-bound**. The arithmetic is cheap; moving the $n \times n$ matrix
> in and out of HBM is what takes the time. So an algorithm that does *more*
> arithmetic while doing *less* memory movement runs faster — which sounds
> paradoxical until you internalise that a modern GPU can perform hundreds of
> floating-point operations in the time it takes to fetch one number from HBM.
> "More FLOPs, but less runtime" should be read as the general lesson, not a
> curiosity. There is a second, enormous benefit: because the $n \times n$ matrix is
> never stored, FlashAttention's memory use is **linear** in sequence length
> rather than quadratic. Long-context models are practical largely because of
> this. Note that unlike Longformer or sliding-window attention, FlashAttention is
> **exact** — the output is identical to standard attention, it is purely a better
> implementation. It belongs to the "reformulate the maths" bucket from
> Lecture 3.

### Number precision and mixed-precision training

A float is stored as **sign + exponent + mantissa**:

| Format | Sign | Exponent | Mantissa |
|---|---|---|---|
| FP16 (Floating-Point 16) | 1 | 5 | 10 |
| FP32 (Floating-Point 32) | 1 | 8 | 23 |
| FP64 (Floating-Point 64) | 1 | 11 | 52 |
| BFLOAT16 (Brain Float 16) | 1 | **8** | **7** |

The **exponent** sets the *range* of representable magnitudes; the **mantissa**
sets the *precision* within that range.

On an H100: **lower precision → faster processing.** Fewer bits means less
memory moved and higher arithmetic throughput, typically doubling with each
halving of width.

**Mixed precision training (Micikevicius et al., 2017).**
**Objective: speed up training and decrease memory requirements.**

- **Forward pass** — activations in **low** precision.
- **Backward pass** — gradient computations in **low** precision.
- **Weights update** — keep a **master copy of the weights in high precision**.

> **Intuition — why BF16 beat FP16 for deep learning, and why the master copy is
> non-negotiable.** Compare the formats: FP16 and BF16 both use 16 bits, but FP16
> spends them on 5 exponent + 10 mantissa while BF16 spends 8 + 7. BF16 therefore
> has *exactly the same dynamic range as FP32* and only loses precision. That
> matters because the thing that destroys low-precision training is not
> imprecision, it is **overflow and underflow** — gradients that are tiny become
> exactly zero in FP16 and the signal is lost entirely. With BF16 they stay
> representable, merely coarse. Coarse gradients still point roughly the right
> way; zero gradients teach nothing. This is why BF16 needs no loss-scaling
> tricks while FP16 does, and why BF16 is the default today.
>
> The high-precision master weights are needed for a subtler reason. A single
> update is often smaller than the smallest increment representable at the
> weight's magnitude — you would compute $w + \text{tiny}$ and get back exactly $w$, so
> the model would silently stop learning. Accumulating updates into an FP32
> master copy lets thousands of tiny updates add up into a change large enough to
> matter, with the low-precision copy re-derived each step for the forward pass.

---

## 4.4 Supervised finetuning (SFT)

### The problem with a pretrained model

A purely pretrained model is a **text completer**, not an assistant. Ask it:

> *"Can I put my teddy bear in the washer?"*

and it may well reply:

> *"Teddy bears are often made of materials like polyester and cotton, with
> plastic eyes and sometimes small accessories."*

That is not a wrong continuation — it is a *perfectly reasonable* continuation of
a document that begins with that sentence. It is simply not an answer, because
nothing ever told the model that a question should be followed by an answer
rather than by more text on the topic.

**Remedy:** add a finetuning stage.

```
              Pretraining                    Finetuning
initialised ────────────→ model with "basic ───────────→ model tuned for
   model                   knowledge" about              specific tasks
                           language, code, ...
```

### Supervised finetuning

**Idea: change the model's behaviour by tuning its weights.**

**Strategy:**

- collect pairs of inputs/outputs exhibiting the desired behaviour (**SFT
  data**),

- train using the **next-word-prediction objective given the input**.

**Special case:** SFT on instruction-following data is called **instruction
tuning**.

> **Watch out.** The objective function is *exactly the same* as pretraining —
> next-token prediction with cross-entropy. Nothing about the mathematics
> changes. What changes is the **data distribution**: instead of arbitrary web
> text, you train on curated (instruction, good response) pairs. And in practice
> the loss is masked so that it is only computed over the *response* tokens, not
> the prompt — you want the model to learn to produce answers, not to produce
> questions.

### Instruction tuning

**Wei et al., 2022, *Finetuned Language Models are Zero-Shot Learners* (FLAN).**
**Goal: "graduate" the model to being a helpful assistant.**

The training data covers many task types, all phrased as instructions with
demonstrated responses. The lecture's examples:

- **Story writing** — *"Write a short story about a teddy bear who likes to read
  poetry."* → a short story.

- **Poem creation** — *"Create a poem about my cute teddy bear."* → *"Soft and
  cuddly, full of charm, / Always keeps me safe from harm, / With button eyes
  and fur so neat, / My teddy bear is oh so sweet."*

- **List generation** — *"List three fun activities a teddy bear might do on a
  rainy day."* → *"Sure! 1. Read poetry with friends. 2. Be cute. 3. Hug its
  owner tightly."*

- **Explanation** — *"Explain why a teddy bear is a great friend."* → a
  paragraph of explanation.

**Objective function:** predict the next token given the input. Feed
`[BOS] Do X . Sure ...` and train on the response portion.

**Data mixtures** — both human-written and synthetic:

- assistant dialogues,
- synthetic instructions,
- maths, reasoning, code,
- safety alignment,
- and more.

**Size — thousands to millions of examples:**

| Model | SFT size (examples) |
|---|---|
| GPT-3 | 13 thousand |
| LLaMA 3 | 10 million |

**Result.** The same question now gets an answer:

> *"Can I put my teddy bear in the washer?"* →
> *"No, it might get damaged. Try hand washing instead."*

<!-- -->

> **Intuition — why so few examples suffice.** 13,000 examples is nothing next to
> 300 billion pretraining tokens, yet it transforms the model's behaviour. The
> explanation is that instruction tuning does not *teach* anything new — the
> knowledge and the ability to write helpfully are already in there, learned from
> the many helpful documents in the pretraining corpus. What SFT does is **select
> a mode**: it tells the model which of the many text-generating personas latent
> in its distribution to adopt. This is sometimes called the "superficial
> alignment hypothesis", and it explains both why SFT is cheap and why the
> *quality* of SFT data matters far more than the quantity. A thousand excellent
> examples beat a hundred thousand mediocre ones, because you are demonstrating a
> style rather than transferring information.

### Challenges of SFT

- **very high-quality data needed**,
- **sensitive to prompt distribution** — the model becomes good at the kinds of
  request it saw, and can regress on others,

- **generalisation** — will it behave well on requests unlike the training set?
- **difficult to evaluate**,
- **computationally expensive**.

### Benchmarks

**Dimensions measured:**

- **general knowledge:** MMLU,
- **basic reasoning:** ARC-Challenge,
- **math reasoning:** GSM8K,
- **code generation:** HumanEval.

**Validity.** The lecture flags a subtle and important methodological point,
from *Training on the Test Task Confounds Evaluation and Emergence*
(Dominguez-Olmedo et al., 2024): it is **recommended to train on the test task**
when comparing across models. Meaning: give every model a small equal amount of
adaptation to the *format* of the benchmark before scoring, because otherwise you
are measuring "which model happened to have seen this answer format during
pretraining" rather than capability — and much of the apparent "emergence" of
abilities at scale is an artifact of larger models being better at guessing the
expected output format.

### "Real-life" feeling: Chatbot Arena

**Idea:** websites like **Chatbot Arena / LMArena** run **A/B tests on real user
prompts**. A user types a prompt, receives two anonymous responses, and votes;
the votes are aggregated into an Elo-style ranking.

**Benefit: it puts a number on "vibes."**

**Outstanding challenges** (drawing on *Exploring and Mitigating Adversarial
Manipulation of Voting-Based Leaderboards*, Huang et al., 2025):

- **unequal exposure of models / "cold start"** — a new model has few votes and a
  noisy rating,

- **easy to "rig"** — coordinated voting can move a model's position,
- **user inability to accurately assess important aspects**, notably
  **factuality** — a confident wrong answer often reads better than a hedged
  right one,

- **personal preference bias** — the voter population is not a representative
  distribution of real usage,

- **safety penalisation** — a model that correctly refuses a harmful request
  loses the vote to one that complies.

**...evaluation is a hard problem in itself!** — which is precisely why
Lecture 8 exists.

### The full lifecycle

```
   Pretraining          Finetuning          Preference tuning
initialised ──→ "basic knowledge" ──→ tuned for  ──→ does not misbehave
   model         about language,       specific        as much
                 code, etc.            tasks
                              └──────── "alignment" of the model ────────┘
```

Preference tuning is Lecture 5.

---

## 4.5 Parameter-efficient finetuning: LoRA

**Context: SFT is resource-intensive and not everyone has big GPUs.** Full
finetuning of a 70B model requires the same terabyte-scale memory as pretraining
it.

### The idea

**LoRA = Low-Rank Adaptation** (Hu et al., 2021). **Approximate the weight
update with the product of two low-rank matrices:**

$$W = W_0 + BA$$

where $W_0$ is the frozen pretrained weight matrix ($d \times d$), and $B$ is $d \times r$,
$A$ is $r \times d$, with the **rank $r \ll d$** (typical values 8, 16, 64).

- **Before:** regular finetuning optimises the **full** matrix $W$ — $d^2$
  parameters.

- **After:** LoRA freezes $W_0$ and optimises only $A$ and $B$ — $2dr$
  parameters.

With $d = 4096$ and $r = 8$, that is 16.7M parameters down to 65k: a **256×**
reduction for that matrix.

**Discussion:**

- a **fraction of the parameters** need to be trained, with **similar
  performance**,

- other methods in the same family include **prefix tuning** and **adapters**.

> **Intuition — why a low-rank update is enough.** The hypothesis, which turns
> out to hold empirically, is that adapting a pretrained model to a task requires
> only a *low-rank* change to its weights. This connects directly to the point
> made about SFT above: you are not teaching new capability, you are selecting a
> behaviour that already exists. Selecting a mode is a low-dimensional operation;
> it does not need the full $d^2$ degrees of freedom.
>
> Two implementation details worth knowing, since they explain the training
> dynamics below. $A$ is initialised randomly and $B$ is initialised to **zero**,
> so that $BA = 0$ at the start and the model begins exactly as the pretrained
> one — no discontinuity. And the update is usually scaled by $\alpha/r$ so that
> changing the rank does not require retuning the learning rate.

### The big practical benefit: swap matrices = swap tasks

Because $W_0$ never changes, one copy of the base model serves every task; you
just attach a different $(A, B)$ pair:

$$\begin{aligned}
W_0 + B_{\mathrm{spam}} A_{\mathrm{spam}} &\;\to\; \text{spam detection task}\\[1pt]
W_0 + B_{\mathrm{sentiment}} A_{\mathrm{sentiment}} &\;\to\; \text{sentiment extraction task}\\[1pt]
W_0 + B_{\mathrm{translate}} A_{\mathrm{translate}} &\;\to\; \text{translation task}
\end{aligned}$$

> **Intuition — this is why LoRA took over industry, and it is a serving argument
> more than a training one.** Full finetuning for $k$ tasks means $k$ complete
> copies of a multi-hundred-gigabyte model, each needing its own GPUs. With LoRA
> you hold **one** base model in memory and swap adapters of a few megabytes.
> You can serve hundreds of task-specific or customer-specific variants from a
> single deployment, and switching costs a memory copy rather than a model load.
> Note also that at inference you can *merge* $BA$ into $W_0$ once, so a LoRA
> model has **exactly zero added latency** compared to the base — unlike adapter
> layers, which insert extra computation into the forward pass. Zero-overhead
> inference plus tiny checkpoints is the whole story.

### Where to apply LoRA

- **Originally:** as experimented in the LoRA paper — the attention projections,
  typically $W_Q$ and $W_V$.

- **Updated guidance** (Schulman et al., 2025, *LoRA Without Regret*): the
  **MLP / feed-forward layers are the most important location**, and current
  recommendation is to apply LoRA to **all** linear layers.

### Training dynamics — "fun facts", i.e. traps

From *LoRA Without Regret*, two empirical differences from full finetuning:

- **LoRA needs a higher learning rate** than full finetuning (roughly an order of
  magnitude higher).

- **LoRA does poorly at large batch size** compared with full finetuning.

> **Intuition.** Both follow from the same cause. The effective update is
> $\Delta W = BA$, a product of two trained factors, so gradients with respect to each
> factor are attenuated — a given step in $A$ and $B$ produces a smaller change in
> $W$ than the same-sized step in $W$ would. You compensate with a larger learning
> rate. The batch-size finding is the more surprising and more practically
> important one: the usual "bigger batch = more stable" intuition does not carry
> over, so if you scale up your batch and your LoRA run degrades, the batch size
> is the first thing to suspect.

### QLoRA (Dettmers et al., 2023)

**Idea: quantise all frozen weights to relieve the memory bottleneck.**

The arrangement:

- $W_0$ is **stored quantised** (4-bit),
- $A$ and $B$ are **stored in full precision**,
- **computations are performed in full precision** — $W_0$ is dequantised
  block-by-block on the fly as it is used.

**Trick: use 4-bit NormalFloat (NF4) to best split the space.** Standard INT8
quantisation uses **uniform** cut-offs across the range. NF4 instead places its
16 levels at the **quantiles of a normal distribution**.

> **Intuition.** Neural network weights are empirically close to
> zero-centred and normally distributed. Uniformly-spaced quantisation levels
> therefore waste most of their resolution on the extreme tails, where almost no
> weights live, while being too coarse near zero, where nearly all of them are.
> NF4 puts the levels where the mass is: fine resolution near zero, coarse in the
> tails. It is *information-theoretically optimal* for normally distributed data,
> and it is why NF4 loses so little quality compared to naive 4-bit rounding.

**Double quantisation.** Quantisation is done in blocks, and each block needs a
**quantisation constant** (its scale) stored in full precision. With small blocks
there are many constants, and they add up to real memory. So: quantise the
quantisation constants too.

- *No quantisation* — weights in full precision.
- *Single quantisation* — weights quantised, constants in full precision.
- *Double quantisation* — weights quantised, **and constants quantised as well**.

**Benefits:**

- VRAM savings enable finetuning on smaller GPUs, and faster,
- a better trade-off between memory resources and quality.

**Orders of magnitude, reported for LLaMA 65B:**

- **~16× VRAM savings** during finetuning,
- the double-quantisation trick saves an **extra ~6%**.

> **Intuition — what QLoRA actually unlocked.** 16× is the difference between
> "you need a cluster" and "you need one gaming GPU". QLoRA is the single reason
> the open finetuning ecosystem exists: a 65B model became finetunable on a single
> 48 GB card, and 7B models on consumer hardware. Note the careful separation of
> concerns in the design — **storage** is 4-bit (that is where the memory saving
> lives) while **computation** is full precision (that is where the quality
> lives). You are not doing 4-bit arithmetic; you are keeping a compressed copy
> of frozen weights and expanding each piece just in time. And because the base
> weights are frozen, quantisation error is a fixed, non-accumulating offset that
> the trainable LoRA matrices can partly compensate for.

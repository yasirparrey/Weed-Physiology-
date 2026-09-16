# Lecture 1 — The Transformer

**Video length: 1h42.** Covered here: what NLP tasks look like and how they are
scored, tokenisation, word representations and Word2vec, RNNs and LSTMs and why
they fail, the origin of attention, the self-attention mechanism, the full
encoder–decoder Transformer architecture, and one complete end-to-end worked
example.

---

## 1.1 The landscape of NLP tasks

Before any architecture, the lecture sets up *what we are trying to do*. Every
classical NLP task falls into one of three shapes, and the shape determines
what the output head of your model looks like.

**Shape 1 — Classification.** Text in, one label out. Sentiment extraction
("this teddy bear is SO CUTE!" → positive), intent detection, language
detection, topic modelling. The model reduces a whole sequence to a single
decision.

**Shape 2 — "Multi"-classification (token-level labelling).** Text in, one
label *per token* out. Part-of-speech tagging, named entity recognition
(marking "teddy bear" as an entity), dependency parsing, constituency parsing.
The input and output have the same length; you are labelling positions, not the
whole sequence.

**Shape 3 — Generation.** Text in, *new* text out, of arbitrary length. Machine
translation, question answering, summarisation, open-ended text generation.

> **Intuition.** This taxonomy quietly predicts the whole rest of the course.
> Shapes 1 and 2 only need a good *representation* of the input, which is
> exactly what an encoder-only model like BERT provides (Lecture 2). Shape 3
> needs the model to produce tokens one at a time conditioned on what it has
> already produced, which is exactly what a decoder-only LLM does (Lecture 3).
> The reason decoder-only models took over is that shape 3 can *simulate*
> shapes 1 and 2 — you can always ask a generative model to emit the word
> "positive" — while the reverse is not true.

### How each shape is evaluated

**Classification and token labelling** use the standard confusion-matrix family:

- **Accuracy** — fraction of observations predicted correctly.
- **Precision** — of the items predicted positive, what fraction really were?
- **Recall** — of the items that really are positive, what fraction did we find?
- **F1** — the harmonic mean of precision and recall,
  `F1 = 2·P·R / (P + R)`, which punishes a model that is excellent at one and
  terrible at the other.

For NER specifically, these are computed **at token level, per entity type** —
you get a precision/recall/F1 for `PERSON`, another for `ORG`, and so on, which
matters because entity types are wildly imbalanced.

Typical datasets: Amazon reviews, IMDB critiques and Twitter for sentiment;
annotated Reuters newswire (CoNLL-2003, CoNLL++) for NER.

**Generation** is much harder to score, because there are many correct answers.
The lecture introduces three metrics that reappear in Lecture 8:

- **BLEU** — measures how much of the *generated* text appears in the
  reference. It is precision-flavoured: it asks "of the n-grams I produced, how
  many were legitimate?" Designed for machine translation.
- **ROUGE** — the mirror image. It is recall-flavoured: "of the n-grams in the
  reference, how many did I manage to produce?" Designed for summarisation,
  where coverage matters more than terseness.
- **Perplexity (PPL)** — not a comparison against a reference at all, but a
  measure of how *surprised* the model is by real text. Formally it is the
  exponentiated average negative log-likelihood; intuitively, a perplexity of
  20 means the model is, on average, as uncertain as if it were choosing
  uniformly among 20 tokens at each step. Lower is better.

> **Intuition on why BLEU/ROUGE are a pair.** A translation system that outputs
> a single very safe word would score well on precision-style metrics if that
> word is always right, and terribly on recall. A system that dumps every
> plausible word would do the opposite. Reporting both is a crude way of pinning
> a system between the two failure modes. Lecture 8 will demolish both metrics
> as poor proxies for human judgement, but they remain the historical baseline.

### The historical timeline

The lecture frames the field in four eras, and the framing explains *why*
progress happened when it did:

| Era | What appeared | What enabled it |
|---|---|---|
| 1980s | Recurrent neural networks | theoretical foundations |
| 1997 | LSTM | theoretical foundations |
| 2013 | Word2vec | lots of data, growing compute |
| 2017 | Transformers | lots of data, growing compute |
| 2020s | Large Language Models | fast iteration on ideas |

The point being made is that the theory largely predates the results by decades.
What changed was data volume and hardware, and then — in the 2020s — the sheer
speed at which the community could try ideas.

> **Watch out.** The lecture opens with a deliberately intimidating wall of
> abbreviations (BERT, RLHF, ROUGE, MLM, BLEU, LSTM, GRU, RAG, FLAN, T5, GLUE,
> DPO, NER, PEFT, PoS, NLG, METEOR, SQuAD, C4, BPE, PPL, GPT, F1, MRPC, LLaMA,
> MT, QA, LaaJ, WP, SFT, SP, PPO...) and then immediately organises them into
> six buckets: architectures/techniques, Transformer-based models, datasets,
> metrics, training strategies, and tasks. If you keep those six buckets in mind
> as you read, every new acronym in the course slots into one of them.

---

## 1.2 Tokenisation

A neural network consumes numbers, so the first job is chopping text into
discrete units and assigning each an index. The lecture walks the same sentence
— `A cute teddy bear is reading.` — down four levels of granularity.

**Arbitrary / naive split.** `A | cute | teddy | bear | is | reading | .`
Note that punctuation becomes its own token; splitting on whitespace alone would
glue `reading.` together and make it a different token from `reading`.

**Word-level.** `A | cute | teddy bear | is | reading | .` Multi-word
expressions can be treated as single units. Simple and interpretable.

**Sub-word level.** `A | cute | ted | ##dy | bear | is | read | ##ing | .`
Words are broken into stems and affixes. The `##` prefix marks "this piece
continues the previous word rather than starting a new one" — that is the
WordPiece convention, and it is what lets you reconstruct the original string
unambiguously.

**Character level.** `A | c | u | t | e | ...` Every character is a token.

### The trade-off table

| Method | Pros | Cons |
|---|---|---|
| Word-level | simple, interpretable | risk of out-of-vocabulary tokens; doesn't exploit shared roots |
| Sub-word (WordPiece, BPE) | exploits common prefixes/suffixes; vocabulary is *learned from data* | small residual OOV risk, but much less than word-level |
| Character-level | almost no OOV; robust to casing and misspellings | sequences become very long → slow; individual embeddings are not interpretable |

> **Intuition — why sub-word won.** Consider the words *read*, *reading*,
> *reader*, *reads*, *reread*. Word-level tokenisation gives you five unrelated
> indices, and if *reread* never appeared in training you cannot represent it at
> all. Character-level lets you represent anything but forces the model to
> relearn that `r-e-a-d` means something, at every position, from scratch — and
> multiplies your sequence length by about four, which for a Transformer means
> roughly sixteen times the attention cost. Sub-word tokenisation is the
> compromise: frequent words stay whole (cheap, and their embedding is
> well-trained), rare words decompose into pieces the model has seen before
> (`ted` + `##dy`), and nothing is unrepresentable. This is why essentially
> every modern model uses BPE or a close relative.

> **Watch out.** "Risk of OOV, though less than word-level" for sub-word is a
> deliberately careful phrasing. Sub-word tokenisers usually include a
> byte-level fallback, so in practice they are OOV-free — but the vocabulary
> itself is learned on a specific corpus, so text from a very different
> distribution (another language, heavy code, emoji) tokenises inefficiently
> even when it tokenises successfully. This inefficiency is a real cost: it
> consumes context window and money, since you are billed per token.

---

## 1.3 Representing tokens as vectors

### Why one-hot encoding is not enough

The naive representation of a vocabulary of size `V` is a one-hot vector: a
`V`-dimensional vector of zeros with a single 1 at the token's index. Two
problems, both fatal:

1. **Size.** `V` is typically 30,000–100,000+. Every token is a vector that
   large, almost entirely zeros.
2. **No notion of similarity.** Every pair of distinct one-hot vectors is
   exactly equally distant. The representation of *cat* is precisely as far
   from *kitten* as it is from *bulldozer*. All the structure of language is
   thrown away.

The fix is a **learned embedding**: a dense vector of dimension `d` (with
`d ≪ V`) per token, whose values are trained. Mechanically this is a lookup
table — a `V × d` matrix — and looking up a token is exactly the same operation
as multiplying its one-hot vector by that matrix. Now similarity is meaningful:
tokens that behave alike end up nearby in the `d`-dimensional space.

### Word2vec (Mikolov et al., 2013)

The canonical way to *obtain* such embeddings without labelled data. The recipe:

- Train a neural network on a **proxy task** over billions of words of text.
- The task itself is disposable; what you keep is the **embedding layer** it
  learned along the way.

Two standard proxy tasks:

- **CBOW (Continuous Bag Of Words)** — given the surrounding context words,
  predict the missing centre word.
- **Skip-gram** — the reverse: given the centre word, predict the surrounding
  context words.

**The architecture** is deliberately shallow: an input layer of size `V`, one
hidden layer of size `d`, an output layer of size `V`. No non-linearity of
consequence, no depth.

**The worked example in the lecture** uses next-word prediction over
`A cute teddy bear is reading`, with `V = 6` and `d = 2`:

- Input `A` as one-hot `[1,0,0,0,0,0]`.
- The hidden layer produces its embedding, e.g. `[0.2, 0.9]`.
- The output layer produces a distribution over the vocabulary, e.g.
  `[0.2, 0.4, 0.1, 0.1, 0.1, 0.1]`, and we compare that against the true next
  word `cute`.
- Then slide along: input `cute` as `[0,1,0,0,0,0]`, hidden `[0.8, 0.4]`, output
  `[0.2, 0.2, 0.2, 0.1, 0.2, 0.1]`, target `teddy`. And so on across the corpus.

After training on enough text, the hidden-layer weights *are* your embedding
table, and they exhibit the famous geometric regularities (vector arithmetic
like `king − man + woman ≈ queen`).

> **Intuition — why a throwaway task produces useful vectors.** The only way for
> a network this simple to predict context words well is to place words that
> appear in similar contexts at similar points in the hidden space. This is the
> *distributional hypothesis* — "you shall know a word by the company it keeps"
> — implemented as gradient descent. The task is a pretext; the geometry is the
> product. This exact logic reappears twice more in the course: BERT's masked
> language modelling (Lecture 2) and LLM pretraining (Lecture 4) are both
> "disposable proxy task, keep the representations".

---

## 1.4 Recurrent networks, and why they were abandoned

### RNNs

A class of network where connections form a **temporal sequence**: the network
processes one token at a time and carries a hidden state forward. In general
form, at step `t`:

```
a_t = g₁(W_aa·a_{t−1} + W_ax·x_t + b_a)
y_t = g₂(W_ya·a_t + b_y)
```

The crucial property is **weight sharing across time**: the same `W` matrices
are applied at every step, so the network can handle sequences of any length.

The lecture animates this on the running example: feed `A`, predict `cute`;
carry the state; feed `cute`, predict `teddy bear`; carry the state; and so
forth. Because the hidden state is a function of everything seen so far, **word
order matters** — a genuine improvement over Word2vec, where a sentence is just
a bag of vectors.

RNNs cover all three task shapes, using the same body with different heads:
one output at the end for classification (sentiment, opinion, text source), one
output per step for token labelling (tags), and a generation loop for
sequence-to-sequence work (translation).

### LSTM (Hochreiter & Schmidhuber, 1997)

Same recurrent skeleton, but with a **more structured hidden state**. Instead of
one vector overwritten at each step, an LSTM cell maintains a *cell state* and
uses learned **gates** to decide what to forget, what to write, and what to
expose:

```
Γ_f = σ(W_f·[a_{t−1}, x_t] + b_f)        forget gate
Γ_u = σ(W_u·[a_{t−1}, x_t] + b_u)        update / input gate
Γ_o = σ(W_o·[a_{t−1}, x_t] + b_o)        output gate
c̃_t = tanh(W_c·[a_{t−1}, x_t] + b_c)     candidate cell content
c_t  = Γ_f ⊙ c_{t−1} + Γ_u ⊙ c̃_t        new cell state
a_t  = Γ_o ⊙ tanh(c_t)                   new hidden state
```

> **Intuition.** The reason gating helps is the term `Γ_f ⊙ c_{t−1}`. In a plain
> RNN, information from step 1 reaching step 50 must survive being multiplied by
> a weight matrix and squashed through a non-linearity 49 times; gradients
> flowing back shrink (or blow up) geometrically. That is the **vanishing
> gradient problem**. In an LSTM there is a nearly-linear path along the cell
> state, and if the forget gate stays near 1 the information passes through
> almost unchanged. The gate is *learned*, so the network decides for itself
> what deserves long-term storage. GRU is a simplification with two gates
> instead of three and no separate cell state.

### The scorecard that motivates the rest of the course

| Method | Pros | Cons |
|---|---|---|
| Word2vec (CBOW, Skip-gram) | very simple yet powerful; intuitive embeddings | word order ignored; embeddings are **not context-aware** |
| RNNs (vanilla, LSTM) | word order matters; state-of-the-art results at the time | vanishing gradients; **slow computation** |

> **Intuition — the two words that killed RNNs.** "Slow computations" is not a
> minor engineering gripe, it is the whole reason Transformers exist. An RNN's
> step `t` needs the output of step `t−1`, so training on a sequence of length
> `n` takes `n` sequential steps *no matter how many GPUs you own*. You cannot
> parallelise along the sequence. When the 2010s handed the field enormous
> corpora and enormous GPU clusters, the bottleneck stopped being ideas and
> started being "how much text can you push through per second" — and there, an
> architecture that processes all positions simultaneously wins by an enormous
> margin. Note carefully that this argument is about *training*: at inference
> time, generation is still sequential for a Transformer too, which becomes the
> central problem of Lecture 3 and reappears in Lecture 9.
>
> "Embeddings not context-aware" is the other half. Word2vec assigns *bank* one
> vector, so the river bank and the savings bank share a representation. Whatever
> replaces it must produce a *different* vector for the same token depending on
> its neighbours.

---

## 1.5 Where attention came from

Attention was **not** invented for Transformers. It was introduced by Bahdanau
et al. (2014) to fix a specific failure in neural machine translation:
**long-term dependencies**. A seq2seq model compressed the entire source
sentence into one fixed-size vector, and by the time the decoder was emitting
its fifth French word it had effectively forgotten what the English sentence
said.

The lecture illustrates this with `A cute teddy bear is reading` →
`Un ours en peluche mignon ...`, where the decoder gets stuck. Attention's fix:
instead of one summary vector, let the decoder **look back at all source
positions at every output step**, and learn *which* ones to weight. To produce
`ours` (bear) it should be looking at `teddy bear`; to produce `mignon` (cute)
it should be looking at `cute`; to produce `lit` (reads) it should be looking at
`reading`.

> **Intuition.** Note that this also solves the reordering problem for free.
> French puts the adjective after the noun ("ours en peluche mignon" vs "cute
> teddy bear"), so a strictly monotonic aligner would fail. Attention weights
> are just a learned soft alignment, with no ordering constraint — which is why
> attention maps are often readable as translation alignments.

The 2017 leap was to notice that if attention is the part doing the real work,
you can **delete the recurrence entirely**. Hence the title: *Attention Is All
You Need*.

---

## 1.6 The self-attention mechanism

### Query, key, value

The mechanism is best understood by analogy to a soft dictionary lookup. Each
token produces three vectors, all linear projections of its embedding `x`:

```
q = x·W_Q      the query:  "what am I looking for?"
k = x·W_K      the key:    "what do I offer to others?"
v = x·W_V      the value:  "what do I actually contribute if attended to?"
```

For one query, attention proceeds in four steps:

1. **Score** the query against every key by dot product: `score_j = q · k_j`.
   A large dot product means "this key matches what I was looking for".
2. **Scale** by `√d_k`.
3. **Normalise** the scores into weights with a softmax, so they are positive
   and sum to 1.
4. **Combine**: output the weighted average of the *value* vectors, using those
   weights.

The compact matrix form, applied to all queries at once:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

where `Q` is `n × d_k`, `K` is `n × d_k`, `V` is `n × d_v`, so `Q·Kᵀ` is the
`n × n` matrix of all pairwise scores, and the output is `n × d_v`.

> **Intuition — what each piece is for.**
>
> *Why three separate projections?* Because "what I need" and "what I offer" are
> different questions. The token `bear` might be *looking for* adjectives while
> *offering* itself as a noun. One shared vector could not encode both roles.
> Splitting off the value is a further separation: the key decides *whether* a
> token is relevant; the value decides *what gets copied* if it is. A token can
> be easy to find but contribute something quite different from its identity.
>
> *Why divide by `√d_k`?* If the components of `q` and `k` are roughly
> independent with unit variance, their dot product over `d_k` dimensions has
> variance about `d_k`, so it grows like `√d_k` in magnitude. Feed large numbers
> into a softmax and it saturates: one weight goes to ~1, the rest to ~0, and
> the gradient through the softmax approaches zero. Dividing by `√d_k` holds the
> scores in the regime where softmax is smooth and trainable. It is a
> variance-normalisation trick, nothing deeper.
>
> *Why does this beat recurrence?* Every output position depends on every input
> position through a *single* matrix multiplication — path length 1, not `n`. So
> gradients do not have to survive `n` sequential steps, and the whole thing is
> one big matmul, which is exactly what GPUs are built for.

> **Watch out.** `V` here is the value matrix, not the vocabulary size. The
> course reuses the letter. Context disambiguates: inside an attention formula
> it is values; in "`V`: vocabulary size" it is the vocabulary.

### The cost

`Q·Kᵀ` is an `n × n` matrix, so self-attention is **O(n²)** in both time and
memory with respect to sequence length. This single fact drives an enormous
amount of later material: sparse attention and sliding windows (Lecture 2), KV
caching and PagedAttention (Lecture 3), FlashAttention (Lecture 4), and even the
hardware discussion in Lecture 9.

### Multi-head attention

Rather than one attention operation of width `d_model`, run `h` of them in
parallel, each with its own `W_Q`, `W_K`, `W_V` projecting into a smaller `d_k`,
then concatenate the `h` outputs and pass them through a final projection `W_O`:

```
head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)
MHA(X) = Concat(head_1, ..., head_h) · W_O
```

The benefit stated in the lecture: it lets the model **capture different
attention features in parallel**, and the explicit comparison offered is to
**multiple filters in a convolutional layer** in computer vision.

> **Intuition.** That CNN analogy is the right one. A single conv filter detects
> one kind of pattern; a bank of filters detects many. Likewise one attention
> head can only implement one relational pattern per layer — softmax forces it
> to commit its probability mass. But a sentence has several simultaneous
> relations to track: syntactic subject-of, adjective-modifies, coreference,
> local phrase structure. With `h` heads, different heads specialise. The
> Transformer paper itself shows heads in layer 5 of 6 doing **anaphora
> resolution** — tracking what the pronoun "its" refers to — and Lecture 2 opens
> by displaying exactly that attention map. Note that heads are usually sized so
> that `h · d_k = d_model`, meaning multi-head attention costs about the same as
> single-head attention of full width: you get diversity for free.

---

## 1.7 The full architecture

The original Transformer is an **encoder–decoder** model with three kinds of
component.

**Attention layers (MHA), in three roles:**
- *Encoder self-attention* (encoder–encoder): source tokens attend to source
  tokens.
- *Decoder self-attention* (decoder–decoder): output tokens attend to previously
  generated output tokens. This one must be **masked** so that position `t`
  cannot see positions `> t`, otherwise the model would cheat at training time
  by reading the answer.
- *Encoder–decoder cross-attention*: queries come from the decoder, keys and
  values from the encoder output. This is the direct descendant of Bahdanau
  attention, and it is the only place where source information enters the
  decoder.

**A position-wise feed-forward network (FFNN)** after each attention layer: two
linear layers with a non-linearity between them, applied identically and
independently at every position, expanding to `d_FF` and back to `d_model`.

**Positional encoding (PE)**, added to the input embeddings.

Around every sub-layer there are **residual connections** and a
**normalisation layer** — both essential for training depth, and both examined
in detail at the start of Lecture 2.

### Walking the stack

**Input.** Text is tokenised, then each token is mapped to a learned embedding.
Parameters: `V` (vocabulary size), `d_model` (embedding dimension).

**Positional encoding — "input, with a trick".** Attention as defined is
*permutation-equivariant*: reorder the input tokens and you reorder the outputs
identically, but nothing in the computation knows which token came first. The
fix is to add position information directly to the input vectors. It can be
**learned** or **hardcoded**; the goal is to let the model understand relative
input position. The lecture shows the classic visualisation: the sinusoidal
encoding as a heat map over (position × dimension), and a second plot showing
that the dot product between two position encodings decays smoothly as the
distance between the positions grows — the property that makes the encoding
useful.

**Encoder.** `N` stacked identical layers. Each has self-attention, an FFNN, and
normalisation. Parameters: `N` layers, `h` heads, `d_FF`/`d_key`/`d_value` for
sub-layer widths, `d_model` overall.

**Output, "shifted right".** The decoder's input is the target sequence offset
by one position, so that when predicting token `t` the decoder has been given
tokens `1..t−1`. During translation you begin with a `[BOS]` (beginning of
sequence) token.

**Decoder.** `N` stacked layers, each with masked self-attention, then
encoder–decoder cross-attention, then an FFNN, with normalisation throughout.
Same parameter list as the encoder.

**Output head.** A linear projection from `d_model` up to `V`, followed by a
softmax. As the lecture puts it, this is *a classification problem where the
classes are words*.

> **Intuition — why "shifted right" and masking are the same idea.** Both exist
> to enforce causality during *parallel* training. You want to train on the whole
> target sentence in one forward pass, computing the loss at every position
> simultaneously — but position `t`'s prediction must not depend on the true
> token at position `t` or later. Shifting the input right by one handles the
> "not itself" part; the causal mask on decoder self-attention handles the "not
> the future" part. At inference time neither is needed as a trick, because you
> genuinely do not have future tokens: you feed back what you generated.

### Two computational tricks from the paper

**Multi-head attention** (above): run several attention layers in parallel to
capture different features simultaneously; analogous to multiple CNN filters.

**Label smoothing.** Borrowed from a 2015 vision paper whose message was
*overconfidence is bad*. Instead of a one-hot target (probability 1 on the
correct token, 0 elsewhere), use a slightly softened target — e.g. `1 − ε` on
the correct token and `ε` spread over the rest. It prevents overfitting and, in
the Transformer paper, improved both accuracy and BLEU.

> **Intuition.** A one-hot target instructs the model to drive the correct
> logit to `+∞` relative to all others — a target it can never reach, so it just
> keeps sharpening, producing enormous logits and brittle overconfidence. Since
> language is genuinely ambiguous (several next words are often fine), a target
> that says "mostly this, but leave a little mass elsewhere" is closer to the
> truth. Interesting wrinkle noted in the paper: label smoothing *hurts*
> perplexity — the model is deliberately less certain than it could be — while
> *helping* BLEU. A useful early lesson that the loss you optimise and the
> metric you care about are not the same thing.

---

## 1.8 End-to-end worked example

The lecture closes by pushing `A cute teddy bear is reading.` through the whole
machine to produce `Un ours en peluche mignon lit.` It is worth following
step by step, because it is the only place where every component is seen in
sequence.

**Encoder side.**

1. **Tokenise:** `A | cute | teddy | bear | is | reading | .`
2. **Add special tokens:** `[BOS] A cute teddy bear is reading . [EOS]`
3. **Embed** each token → one vector per token.
4. **Add the position embedding** to each → *position-aware embeddings*.
5. **Stack** them into a matrix (one row per token, `d_model` columns).
6. Feed that matrix into the encoder. Multiply it by `W_Q`, `W_K`, `W_V` to get
   `Q`, `K`, `V`.
7. Compute `Q·Kᵀ`: an `n × n` grid where entry `(i, j)` is how much token `i`
   attends to token `j`. Softmax each row.
8. Multiply by `V`: each output row is a **weighted average of value vectors,
   with weights determined by the query–key match**. That sentence is the whole
   mechanism in one line.
9. This happens `h` times in parallel; concatenate the heads and project through
   `W_O`.
10. Pass through the feed-forward network. Output: **context-aware encoded
    embeddings** — one vector per input token, each now informed by the whole
    sentence.
11. Repeat for all `N` encoder layers.

**Decoder side** — now generating, one token at a time.

12. Start the decoder with `[BOS]`.
13. Decoder self-attention over what has been generated so far.
14. Encoder–decoder attention: the decoder queries the encoder's output.
15. Feed-forward network.
16. Linear projection to vocabulary size, then softmax →
    `[0.001, 0.0003, ..., 0.4, ..., 0.002]`.
17. Pick the highest-probability token: `Un`.
18. Append it and repeat. Decoder input `[BOS] Un` → predicts `ours`. Then
    `[BOS] Un ours` → `en`, then `peluche`, `mignon`, `lit`, and finally
    `[EOS]`, at which point generation stops.

Final output: `Un ours en peluche mignon lit.`

> **Intuition — the one thing to take away.** Step 8 is the heart of it. Every
> Transformer layer does the same thing: *each position replaces its own vector
> with a weighted mixture of information gathered from other positions, where
> the weights are computed from content, not from position*. Stack that
> operation `N` times, with a feed-forward network between rounds to reshape the
> mixed information, and you have the architecture that everything in the next
> eight lectures is built on.

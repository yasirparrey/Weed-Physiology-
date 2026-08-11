# `DLLiteA_TBox.v` explained line-by-line (Coq for absolute beginners)

> This is the file you actually asked about: the **TBox reasoning** file
> (helpers + inductive inclusion closure). It builds on top of `DLLiteA.v`.
> This doc explains **every line** and teaches the Coq syntax as it appears.
> Throwaway teaching file — delete it whenever you're done.

---

## Part 0 — What this file is (30-second version)

`DLLiteA.v` (the other file) only **defined** the language and what a "model"
is. It did **no reasoning**.

This file, `DLLiteA_TBox.v`, adds the first bit of *reasoning*: given a TBox (a
set of schema rules like "Student ⊑ Person"), it defines what it means to
**derive** new inclusions by combining the rules. For example, from
`Student ⊑ Person` and `Person ⊑ Animal` you can derive `Student ⊑ Animal`
(transitivity). That "what can be derived" relation is the core deliverable.

The header comment even lays out the roadmap:
- **Weeks 3–4 (this file):** helpers + the inductive *closure* relations.
- **Weeks 5–6:** soundness (proving the closure actually matches the semantics
  from `DLLiteA.v`) and subsumption.
- **Stretch (not here):** negative inclusions and functionality.

So concretely, this file contains:
1. Two small **helper definitions** (`role_inverse`, and two "is this axiom in the
   TBox?" predicates) plus one tiny helper lemma.
2. Two **inductive relations** (`closes_concept`, `closes_role`) — the derivation
   engine.
3. Two **example lemmas** proving specific derivations hold for the example TBox.

---

## Part 1 — Just-enough Coq syntax for this file

If you've read the `DLLiteA.v` doc, this repeats a little, but focuses on what
*this* file uses. Refer back as needed.

### 1.1 Every command ends with a period `.`
Like sentences. `Require Import Foo.` is one command. Line breaks don't matter;
the `.` ends it.

### 1.2 `Require Import`
`Require Import X.` loads compiled module `X` and brings its names into scope so
you can use them unqualified. Here it pulls in project libraries and, crucially,
`DLLiteA` — so all the types from the other file (`Role`, `Basic_Concept`,
`TBox`, `ConceptInclusion`, `example_tbox`, ...) are available.

### 1.3 The three `Set`/`Open` config lines
- `Open Scope type_scope.` — makes symbols like `*` mean *type pairing*, not
  multiplication (not really exercised in this file, but harmless/consistent).
- `Set Implicit Arguments.` — let Coq infer arguments it can, so you type less.
- `Set Maximal Implicit Insertion.` — do that inference aggressively.
These are knobs about *verbosity*, not *meaning*.

### 1.4 `Section ... End` and `Context`
```coq
Section DLLiteA_TBox.
Context {lp : Logic_Primitives}.
  ...
End DLLiteA_TBox.
```
- `Section NAME. ... End NAME.` groups definitions that share parameters.
- `Context {lp : Logic_Primitives}.` declares an **implicit** shared parameter
  `lp`. `Logic_Primitives` is a *typeclass* (a bundle of assumptions) from the
  project. The curly braces `{ }` mean "implicit" — Coq fills it in for you; you
  never type it. Practically, `lp` silently supplies "what a role name is", "what
  a concept name is", etc., to everything in the section. When `End` is reached,
  every definition that used `lp` automatically gets it added as an argument.

### 1.5 `Definition NAME (args) : type := body.`
Names a function or value. Args in `(x : T)` are inputs; the part after `:` is the
result type; after `:=` is the body.

### 1.6 `match ... with | pattern => result ... end`
Pattern matching = a `switch` on which constructor built a value. `=>` separates
the pattern from its result; `end` closes it. You can name the constructor's
payload in the pattern (e.g. `Direct Rn` binds the role name to `Rn`).

### 1.7 `Prop` — the type of logical statements
A `Prop` is a proposition: something you can (try to) prove. `True` is trivially
true. `A -> B` reads "A implies B". `forall x, P x` is "for all x". These are how
math statements are written.

### 1.8 `Inductive` used as a *relation* (this is the key idea!)
You already saw `Inductive` used to make *data types*. Here it's used to define a
**relation** — a family of `Prop`s — by listing the ways it can be *true*. Example:

```coq
Inductive even : nat -> Prop :=
| even_0 : even 0
| even_SS : forall n, even n -> even (S (S n)).
```
Read this as a set of **inference rules**:
- `even_0`: "0 is even" (a base fact).
- `even_SS`: "if `n` is even, then `n+2` is even" (a rule with a premise).

The `-> Prop` in the header means "this defines a proposition (about the indices
before the `->`)". `even` takes a `nat` and yields a `Prop`. The **only** way
`even k` can be proven is by applying these constructors. That's how you encode
"derivable by these rules and nothing else." The `closes_concept` /`closes_role`
relations in this file work exactly like this — each constructor is a derivation
rule.

Key sub-points you'll see:
- A constructor with premises above the conclusion, like
  `forall n, even n -> even (S (S n))`, means "**to** conclude `even (S (S n))`,
  you must **first** supply a proof of `even n`." The `->` separates premises from
  the conclusion.
- The name before the first `->` (e.g. `closes_concept (T : TBox)`) is a fixed
  **parameter** that stays the same in every rule.

### 1.9 `Lemma NAME : statement. Proof. <tactics> Qed.`
- `Lemma NAME : goal.` states what you want to prove.
- `Proof.` starts it.
- The indented words are **tactics**: commands that transform the goal until it's
  solved. (Each specific tactic used here is explained where it appears in Part 2.)
- `Qed.` closes and re-checks the proof; Coq accepts only if the goal is fully
  discharged.

### 1.10 Project-specific notations you'll meet
Not built into Coq — they come from the imported libraries and from `DLLiteA.v`:
- `X.(field)` — projection (field access). E.g. `Role_Name.(T)` isn't in this file,
  but `T!!` is: the postfix **`!!`** turns a `Finite_Set`/`TBox` into its
  underlying **list**, so you can talk about list-membership. (`T!!` = "the list of
  axioms of T".)
- `LIn xs x` — "**L**ist membership": `x` is an element of the list `xs`. (From the
  project's list utilities. It behaves like the standard `In`.)
So `LIn T!! (ConceptInclusion B (Pos B'))` reads: "the axiom `B ⊑ B'` is literally
one of the axioms in the list of `T`."

That's the toolkit. Now the file.

---

## Part 2 — The file, block by block

### 2.1 Header comment
```coq
(**
    DL-Lite_A TBox reasoning — Weeks 3–4 (positive core).
    ...
**)
```
- `(* ... *)` is a comment; `(** ... **)` is a *documentation* comment. It has **no
  effect** on behavior. It just states the file's scope and roadmap (see Part 0):
  this file does helpers + inductive inclusion closure for the "positive core";
  soundness/subsumption and negative/functionality features are for later.

### 2.2 Imports + config
```coq
Require Import Mem TotalMem ListUtils Sets Lia List.
Require Import DLLiteA.
```
- Loads project libraries `Mem`, `TotalMem`, `ListUtils`, `Sets`, the arithmetic
  tactic library `Lia`, the standard `List` library, and — importantly — the
  **previous file** `DLLiteA`. That last one is why we can use `Role`, `TBox`,
  `Basic_Concept`, `ConceptInclusion`, `RoleInclusion`, `Pos`, `Direct`, `Inverse`,
  `example_tbox`, `example_student`, etc., without redefining them.
```coq
Open Scope type_scope.
Set Implicit Arguments.
Set Maximal Implicit Insertion.
```
- Config knobs, see 1.3.

### 2.3 Section + context
```coq
Section DLLiteA_TBox.
Context {lp : Logic_Primitives}.
```
- Opens the section (closed at the end) and declares the implicit `lp` parameter
  that everything inside shares. See 1.4. Because `DLLiteA.v` was also written
  inside a section with the *same* `Context`, the imported names line up with this
  file's `lp`.

### 2.4 Helper: `role_inverse`
```coq
(** Inverse of a role expression: R⁻⁻ = R. *)
Definition role_inverse (R : Role) : Role :=
  match R with
  | Direct Rn => Inverse Rn
  | Inverse Rn => Direct Rn
  end.
```
- A function taking a `Role` and returning a `Role`.
- Recall from `DLLiteA.v`: a `Role` is either `Direct Rn` (role name used
  forwards, e.g. `teaches`) or `Inverse Rn` (used backwards, e.g. `teaches⁻`).
- `match R with ... end` inspects which one it is:
  - `Direct Rn => Inverse Rn` — the inverse of a forward role is the backward role.
  - `Inverse Rn => Direct Rn` — the inverse of a backward role is the forward role.
- `Rn` is bound to the underlying role name in each case and reused on the right.
- So this just *flips the direction flag*. Applying it twice gets you back where
  you started — which is exactly the next lemma.

### 2.5 Helper lemma: `role_inverse_involutive`
```coq
Lemma role_inverse_involutive : forall R,
  role_inverse (role_inverse R) = R.
Proof.
  intros [Rn | Rn]; reflexivity.
Qed.
```
- **Statement:** for every role `R`, inverting twice gives `R` back. ("Involutive"
  = "applying it twice is the identity.") `=` is ordinary equality.
- **Proof, tactic by tactic:**
  - `intros [Rn | Rn]` — `intros` moves the `forall R` into your context (i.e.
    "take an arbitrary `R`"). The **square brackets** `[ ... | ... ]` do an
    immediate **case split** on `R` by its two constructors: the first `Rn` is the
    `Direct Rn` case, the second `Rn` is the `Inverse Rn` case. So one tactic both
    introduces `R` and splits it into the two possible shapes. (This "intro
    pattern" is a common Coq shorthand.)
  - `;` — the semicolon means "apply the next tactic to **every** subgoal produced
    so far." We now have two subgoals (Direct case, Inverse case).
  - `reflexivity` — proves goals of the form `x = x`. In the `Direct Rn` case,
    `role_inverse (role_inverse (Direct Rn))` **computes** to `Direct Rn` (Coq
    unfolds the two matches), so the goal becomes `Direct Rn = Direct Rn`, closed
    by `reflexivity`. Same for the `Inverse` case. The `;` runs `reflexivity` on
    both.
  - `Qed.` checks and seals it.
- Takeaway: `intros [..|..]; reflexivity` is the idiomatic way to prove "this
  function definition satisfies an equation, by cases, each holding by computation."

### 2.6 Helper: `tbox_has_pos_ci` (is a positive concept inclusion in the TBox?)
```coq
(** Positive concept inclusion present in the TBox: B ⊑ B'. *)
Definition tbox_has_pos_ci (T : TBox) (B B' : Basic_Concept) : Prop :=
  LIn T!! (ConceptInclusion B (Pos B')).
```
- Takes a TBox `T` and two basic concepts `B`, `B'` (`(B B' : Basic_Concept)` =
  two args of the same type). Returns a `Prop`.
- Body: `LIn T!! (ConceptInclusion B (Pos B'))`.
  - `T!!` — the underlying **list** of axioms of the TBox `T` (postfix `!!`, 1.10).
  - `ConceptInclusion B (Pos B')` — builds the axiom "`B ⊑ B'`" (using the
    `ConceptInclusion` constructor from `DLLiteA.v`; `Pos B'` wraps `B'` as a
    positive general concept, since the right-hand side of an inclusion is a
    `General_Concept`).
  - `LIn list item` — "item is a member of list".
- So the whole thing is the proposition: **"the axiom `B ⊑ B'` literally appears in
  the TBox."** It's a *membership check*, not a derivation. This is the raw
  ingredient the `cc_axiom` rule will use.

### 2.7 Helper: `tbox_has_ri` (is a role inclusion in the TBox?)
```coq
(** Role inclusion present in the TBox: R ⊑ R'. *)
Definition tbox_has_ri (T : TBox) (R R' : Role) : Prop :=
  LIn T!! (RoleInclusion R R').
```
- Same idea for roles: the proposition "the axiom `R ⊑ R'` is one of the axioms in
  `T`." Uses the `RoleInclusion` constructor. Feeds the `cr_axiom` rule below.

### 2.8 The main event: `closes_concept` (derivable concept inclusions)
```coq
Inductive closes_concept (T : TBox) : Basic_Concept -> Basic_Concept -> Prop :=
| cc_axiom : forall B B',
    tbox_has_pos_ci T B B' ->
    closes_concept T B B'
| cc_refl : forall B,
    closes_concept T B B
| cc_trans : forall B1 B2 B3,
    closes_concept T B1 B2 ->
    closes_concept T B2 B3 ->
    closes_concept T B1 B3.
```
This is an **inductive relation** (see 1.8). Read the header first:
- `Inductive closes_concept (T : TBox) : Basic_Concept -> Basic_Concept -> Prop`
  - `(T : TBox)` is a fixed **parameter**: the TBox is the same throughout.
  - `Basic_Concept -> Basic_Concept -> Prop` means `closes_concept T` is a
    two-argument relation on basic concepts producing a `Prop`. So
    `closes_concept T B1 B2` is the proposition **"from TBox `T`, we can derive
    `B1 ⊑ B2`."**

The three constructors are the **derivation rules** — the only ways this can be
true:
- **`cc_axiom`**: `forall B B', tbox_has_pos_ci T B B' -> closes_concept T B B'`.
  "If `B ⊑ B'` is literally an axiom in `T` (the premise, `tbox_has_pos_ci ...`),
  then it's derivable." The `->` marks the premise; the conclusion is after it.
  (Base rule: axioms are derivable.)
- **`cc_refl`**: `forall B, closes_concept T B B`.
  "Every concept includes itself: `B ⊑ B` is always derivable." No premise
  (reflexivity). (Base rule.)
- **`cc_trans`**: `forall B1 B2 B3, closes_concept T B1 B2 -> closes_concept T B2 B3 -> closes_concept T B1 B3`.
  "If you can derive `B1 ⊑ B2` **and** `B2 ⊑ B3`, then you can derive `B1 ⊑ B3`."
  Two premises chained with `->`. (Transitivity — this is what lets you compose
  axioms into longer chains.)

So `closes_concept T B1 B2` holds **iff** you can build it from these rules: it's
in the TBox, or it's reflexive, or it's a transitive chain of such steps. This is
the "positive core" closure the header promised. Notice there's no rule involving
`Exists`/roles here yet (that interaction is deferred).

### 2.9 The other relation: `closes_role` (derivable role inclusions)
```coq
Inductive closes_role (T : TBox) : Role -> Role -> Prop :=
| cr_axiom : forall R R',
    tbox_has_ri T R R' ->
    closes_role T R R'
| cr_refl : forall R,
    closes_role T R R
| cr_trans : forall R1 R2 R3,
    closes_role T R1 R2 ->
    closes_role T R2 R3 ->
    closes_role T R1 R3
| cr_inverse : forall R R',
    closes_role T R R' ->
    closes_role T (role_inverse R) (role_inverse R').
```
Same shape as `closes_concept`, but over `Role`s, and with **one extra rule**:
- `cr_axiom` — role inclusions that are literally in `T` are derivable.
- `cr_refl` — `R ⊑ R` always.
- `cr_trans` — chain `R1 ⊑ R2` and `R2 ⊑ R3` into `R1 ⊑ R3`.
- **`cr_inverse`** (the new one): `forall R R', closes_role T R R' -> closes_role T (role_inverse R) (role_inverse R')`.
  "If `R ⊑ R'` is derivable, then their inverses are related the same way:
  `R⁻ ⊑ R'⁻`." This is a genuine feature of roles (unlike concepts): flipping both
  sides of a role inclusion preserves it. It uses the `role_inverse` helper from
  2.4. This is why roles get four rules and concepts only three.

### 2.10 Example 1: deriving `Student ⊑ Person`
```coq
Lemma example_closes_student_person :
  closes_concept example_tbox
    (Atomic example_student) (Atomic example_person).
Proof.
  apply cc_axiom.
  unfold tbox_has_pos_ci, example_tbox, example_tbox_axioms; simpl.
  left; reflexivity.
Qed.
```
- **Statement:** for the `example_tbox` defined in `DLLiteA.v`, we can derive
  `Student ⊑ Person`. (`Atomic example_student` / `Atomic example_person` are the
  basic concepts built from those example names.)
- **Proof, step by step:**
  - `apply cc_axiom.` — `apply` says "prove the goal by using this rule/lemma."
    The goal is `closes_concept example_tbox (Atomic student) (Atomic person)`. Rule
    `cc_axiom` concludes exactly `closes_concept T B B'` *provided* you prove its
    premise `tbox_has_pos_ci T B B'`. So after `apply cc_axiom`, the **new goal**
    becomes that premise: `tbox_has_pos_ci example_tbox (Atomic student) (Atomic
    person)` — i.e. "this inclusion is literally in the example TBox." (We chose the
    axiom rule because `Student ⊑ Person` is a stated axiom, not something needing
    reflexivity/transitivity.)
  - `unfold tbox_has_pos_ci, example_tbox, example_tbox_axioms; simpl.` — `unfold`
    replaces those names with their definitions (so `tbox_has_pos_ci ...` becomes
    the `LIn T!! (...)` expression, and `example_tbox` becomes the concrete finite
    set built from `example_tbox_axioms`, which is the concrete 3-element list).
    Then `simpl` computes/simplifies — in particular it turns `LIn` over a concrete
    list into a plain **disjunction** of equalities, roughly:
    `(ConceptInclusion (Atomic student) (Pos (Atomic person)) = <axiom 1>) \/
     (... = <axiom 2>) \/ (... = <axiom 3>) \/ False`.
    Recall the list order from `DLLiteA.v`:
    `Student⊑Person :: Professor⊑Person :: teaches⊑works_at :: nil`. So our target
    matches the **first** disjunct.
  - `left; reflexivity.` — the goal is now `A \/ B` (an "or"). `left` chooses to
    prove the **left** side (the first disjunct), because our axiom is the first in
    the list. That leaves an equality goal `<our axiom> = <first axiom>`, which is
    literally the same term, so `reflexivity` (`x = x`) closes it. The `;` runs
    `reflexivity` right after `left`.
  - `Qed.` checks it.
- Big picture: this proof says "the derivation is trivial — the inclusion is
  already an axiom sitting at position 1 of the TBox list."

### 2.11 Example 2: deriving `teaches ⊑ works_at`
```coq
Lemma example_closes_teaches_works_at :
  closes_role example_tbox
    (Direct example_teaches) (Direct example_works_at).
Proof.
  apply cr_axiom.
  unfold tbox_has_ri, example_tbox, example_tbox_axioms; simpl.
  (* ConceptInclusion :: ConceptInclusion :: RoleInclusion :: nil *)
  do 2 right; left; reflexivity.
Qed.
```
- **Statement:** for `example_tbox`, we can derive the role inclusion
  `teaches ⊑ works_at` (both wrapped as `Direct`, i.e. forward roles).
- **Proof:**
  - `apply cr_axiom.` — same idea as before but for roles: `cr_axiom` reduces the
    goal to its premise `tbox_has_ri example_tbox (Direct teaches) (Direct
    works_at)` — "this role inclusion is literally in the TBox."
  - `unfold ...; simpl.` — unfold the definitions and compute, turning the
    membership into a disjunction of equalities over the same 3-element list.
  - The inline comment `(* ConceptInclusion :: ConceptInclusion :: RoleInclusion ::
    nil *)` is a reminder of the list order: the role inclusion is the **3rd** item.
  - `do 2 right; left; reflexivity.` — now navigate the disjunction to the 3rd
    slot:
    - `do 2 right` — `right` chooses the **right** side of an `A \/ B` (i.e. "skip
      the current first option"). `do 2 right` does that **twice**, skipping the two
      `ConceptInclusion` disjuncts. (`do N tac` = "run `tac` exactly N times.")
    - `left` — now at the third position, pick it (the left of the remaining
      `RoleInclusion \/ False`).
    - `reflexivity` — the chosen disjunct is `<our inclusion> = <third axiom>`,
      identical terms, so it closes.
  - `Qed.`
- Big picture: identical strategy to Example 1, but because the target sits at
  position 3 in the list, you skip past two options (`do 2 right`) before selecting
  (`left`). This is the direct payoff of the list ordering defined back in
  `DLLiteA.v`.

### 2.12 Closing the section
```coq
End DLLiteA_TBox.
```
- Ends `Section DLLiteA_TBox`. Every definition/lemma that used the implicit `lp`
  now gets it threaded through as a parameter automatically. The names
  (`role_inverse`, `closes_concept`, `closes_role`, the lemmas) become available to
  any later file that imports this one.

---

## Part 3 — The mental model, all together

Think of it in three layers:

1. **"Is it stated?" (helpers):** `tbox_has_pos_ci` / `tbox_has_ri` just check
   whether an inclusion is *literally written* in the TBox list. `role_inverse`
   (+ its involutive lemma) is plumbing for the role-inverse rule.

2. **"What can we derive?" (the engine):** `closes_concept` and `closes_role` are
   inductive relations whose constructors are inference rules:
   - start from stated axioms (`*_axiom`),
   - everything includes itself (`*_refl`),
   - chain inclusions (`*_trans`),
   - and for roles only, mirror across inverses (`cr_inverse`).
   `closes_concept T B1 B2` is provable **exactly when** you can construct it from
   these rules — nothing more. That's the whole meaning of "closure".

3. **"Does it work on a concrete case?" (examples):** the two lemmas show the
   engine deriving two inclusions from the example TBox. Both happen to be
   single-step derivations via the `axiom` rule, and the proofs reduce to
   *navigating the TBox list* to the right position (`left` vs `do 2 right; left`)
   and closing with `reflexivity`.

What's intentionally **not** here yet (per the header): soundness (proving these
syntactic derivations agree with the model-theoretic `is_a_model` semantics from
`DLLiteA.v`), and the harder features (negative inclusions, functionality). Those
are the Weeks 5–6 / stretch items.

### Cheat-sheet: tactics used in this file
- `intros [a | b]` — take the universally-quantified variable and immediately split
  it into its constructor cases.
- `apply RULE` — prove the goal by a rule/lemma, leaving its premises as new goals.
- `unfold NAME` — replace a defined name by its definition.
- `simpl` — compute/simplify (e.g. expand `LIn` over a concrete list into an `\/`
  chain).
- `left` / `right` — pick the left / right side of an `A \/ B` goal.
- `do N tac` — run `tac` exactly `N` times.
- `reflexivity` — close a goal of the form `x = x`.
- `;` — apply the following tactic to all subgoals produced by the previous one.
- `Qed.` — finish and re-check the proof; `Admitted.` (not used here) would leave
  it as an unproven assumption.

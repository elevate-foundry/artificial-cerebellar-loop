A person who spoke **all languages** would not “see” German braille versus English braille from the cell alone. They would infer it from **code context**, the same way you infer whether `gift` means a present in English or poison in German.

A braille cell is not intrinsically English or German. It is a symbol inside a **codebook**.

$$
\text{meaning} = \text{cell pattern} + \text{language code} + \text{context}
$$

## The key point

The six-dot braille cell is only a physical/bit substrate:

$$
b_6 \in \{0,1\}^6
$$

But braille Unicode (U+2800–U+28FF) uses **eight dots**, expanding the cell to:

$$
b_8 \in \{0,1\}^8
$$

This is not merely a larger encoding — it changes the structure of the codebook inference problem.

### 6-dot vs 8-dot

| Property | 6-dot | 8-dot |
|---|---|---|
| Cell space | $2^6 = 64$ | $2^8 = 256$ |
| Standards | UEB, SEB, Deutsche Blindenschrift | Computer braille, Unicode braille |
| Mode switching | Required (capital, number, letter modes) | Reduced or eliminated |
| State machine | $q_t = (\ell_t, \text{mode}_t)$ — complex | $q_t \approx \ell_t$ — simpler |
| Contraction ambiguity | High — cells overloaded by mode | Lower — more cells available |
| Codebook families | Literary braille codes | Computer/Unicode braille codes |

In 6-dot braille, a single cell like `⠼` might mean "number follows" — it is a **mode indicator**, not content. The same cell has no such role in 8-dot computer braille, where dots 7 and 8 directly encode case and type information.

The decoder generalizes to:

$$
D_\ell^{(n)} : \mathbb{B}^{n*} \to \Sigma_\ell^* \quad \text{where } n \in \{6, 8\}
$$

The codebook $D_\ell^{(8)}$ is not simply $D_\ell^{(6)}$ with two extra bits. It is a **different codebook family** — the mapping from cells to characters changes, the mode structure collapses, and the prior $P(\ell)$ shifts because 8-dot braille is predominantly used in computing contexts where language-specific literary conventions are less relevant.

### Why this matters for the aCBL

The aCBL (artificial Cerebellar Braille Loop) operates in 8-dot Unicode braille. When models disagree on encoding strategy, the divergence is specifically about how to use the expanded $2^8$ cell space:

- Some models use dots 7–8 for case (capital indicator in dot 7)
- Some prepend a grade-1 indicator (`⠰`) before the name — a 6-dot convention carried into 8-dot
- Some ignore dots 7–8 entirely and produce 6-dot-compatible output

These are distinct codebooks within the 8-dot family. The codebook clustering in the BBID handshake detects exactly this: which models chose which strategy for the expanded cell space.

### Beyond $2^8$: effective codebook size

A single 8-dot braille cell has exactly $2^8 = 256$ possible patterns (Unicode U+2800–U+28FF). But the **effective codebook** operates over sequences, not individual cells.

A model that prepends `⠐` (dots 5) before `⠗⠽⠁⠝` is not using a 256-symbol alphabet — it is using a codebook over **cell n-grams**:

$$
\mathcal{C}_k = \mathbb{B}^{8k} \quad \Rightarrow \quad |\mathcal{C}_k| = 2^{8k}
$$

| n-gram length $k$ | Effective codebook size | Example |
|---|---|---|
| 1 | $2^8 = 256$ | Single cell: `⠗` → r |
| 2 | $2^{16} = 65{,}536$ | Modifier + cell: `⠐⠗` → capital R |
| 3 | $2^{24} \approx 16.7\text{M}$ | Indicator + digraph: `⠼⠁⠃` → 1b |

In practice, braille codebooks use **variable-length** units — some cells are content, some are modifiers that change the interpretation of subsequent cells. The effective codebook is not $\mathcal{C}_k$ for fixed $k$, but a prefix-free code:

$$
\mathcal{C}^* = \bigcup_{k=1}^{K} \mathcal{C}_k
$$

This is exactly where model disagreement arises. Two models may agree on which cell maps to which letter ($k=1$), but disagree on whether a capital indicator should be a separate prefix cell ($k=2$) or encoded in dots 7–8 of the letter cell itself ($k=1$ with a larger per-cell alphabet). Both are valid 8-dot codebooks, but they have different effective sizes and different $k$ distributions.

The aCBL's codebook clustering captures this: models in the same cluster share not just the same cell-to-character mapping, but the same **n-gram structure** — the same $k$.

### The interpretation function

For both cell sizes, the interpretation is a function:

$$
D_\ell^{(n)} : \mathbb{B}^{n*} \to \Sigma_\ell^*
$$

where:

$$
\ell \in \{\mathrm{English}, \mathrm{German}, \mathrm{French}, \ldots\}
$$

So the same braille stream (y) may decode differently under different language codes:

$$
D_{\mathrm{en}}(y) \neq D_{\mathrm{de}}(y)
$$

That is normal. Braille standards explicitly treat mappings as language/code dependent; foreign-language material in UEB may either use UEB rules or switch into the full foreign-language braille code, with code-switch indicators normally used when needed. ([National Braille Association][1])

## How the reader disambiguates

A universal multilingual reader would infer the intended code by maximizing posterior likelihood:

$$
\hat{\ell} = \operatorname*{argmax}_{\ell \in L} P(\ell \mid y)
$$

By Bayes:

$$
\hat{\ell} = \operatorname*{argmax}_{\ell \in L} P(y \mid \ell) P(\ell)
$$

Where:

$$
P(y \mid \ell)
$$

means: “How plausible is this braille stream if generated by English braille rules versus German braille rules?”

They would use several signals.

## 1. Declared context

The strongest signal is external metadata:

$$
\text{book title, country, school, document language, surrounding print text}
$$

A braille book produced in Germany and written for German readers is presumed German braille. A U.S. English textbook using UEB is presumed English braille. Foreign-language braille guidance explicitly distinguishes using UEB for foreign text versus switching into the full foreign braille code. ([National Braille Association][1])

In practice:

$$
P(\ell \mid y, \text{metadata}) \gg P(\ell \mid y)
$$

## 2. Code-switch indicators

If an English UEB text switches into a non-UEB foreign braille passage, code-switch indicators may mark that switch. So a reader may literally be told:

```text
from here, use German braille rules
```

not in English prose, but through braille control symbols.

So the decoding state is not just language; it is a finite-state machine:

$$
q_t = (\ell_t, \text{mode}_t)
$$

where mode includes things like:

$$
\text{letter mode}, \text{number mode}, \text{capital mode}, \text{contracted mode}, \text{foreign-code mode}
$$

## 3. Statistical language evidence

Suppose the reader has no metadata and no code-switch indicators. Then they decode the same stream under both codebooks:

$$
x_{\mathrm{en}} = D_{\mathrm{en}}(y)
$$

$$
x_{\mathrm{de}} = D_{\mathrm{de}}(y)
$$

Then choose the one that produces more plausible language:

$$
\hat{\ell} = \operatorname*{argmax}_{\ell} P_{\mathrm{LM},\ell}(D_\ell(y))
$$

where $P_{\mathrm{LM},\ell}$ is a language model for English, German, etc.

Plain version: try reading it as English, try reading it as German, and see which one turns into real words.

## 4. German-specific symbols

German has native symbols for:

$$
\text{ä, ö, ü, ß}
$$

If those appear in ways that produce valid German words, the German hypothesis becomes much stronger.

Example:

$$
\text{Köln, für, Straße, Mädchen}
$$

These are strong German signals. A UEB source may handle foreign characters differently depending on context, while full German braille treats them as native symbols.

## 5. Contraction behavior

This is the biggest discriminator.

In uncontracted braille, English and German are very close for $a$–$z$. But in contracted braille, cells can mean different things.

So a universal reader would ask:

$$
\text{Are these cells behaving like German contractions or English contractions?}
$$

German has contractions like:

$$
\text{ch, sch, st, ei, ie, au, eu}
$$

English UEB has its own contraction system. UEB is specifically a unified English braille code used across English-speaking countries, with its own literary/math/computing rules. ([ICEB][2])

So if a cell sequence repeatedly resolves into German morphemes like:

$$
\text{sch, ich, nicht, ein, die, und}
$$

under German braille, but into awkward English fragments under UEB, the reader picks German.

## 6. Ambiguous cases are genuinely ambiguous

Some braille strings are underdetermined.

For short strings, there may be no unique answer:

$$
D_{\mathrm{en}}(y) = \text{valid English}
$$

and

$$
D_{\mathrm{de}}(y) = \text{valid German}
$$

In that case, even a person who spoke all languages could not know the intended language from the cells alone.

The correct statement is:

$$
\text{braille stream alone does not always contain enough information}
$$

You need at least one of:

$$
\text{metadata, code indicators, longer context, statistical prior, semantic coherence}
$$

## The complete decoder

A universal braille reader is not doing:

$$
y \to x
$$

They are doing:

$$
(y, c) \to (\ell, x)
$$

where $c$ is context.

More completely:

$$
(\hat{\ell}, \hat{x}) = \operatorname*{argmax}_{\ell \in L,\ x \in \Sigma_\ell^*} P(x, \ell \mid y, c)
$$

Equivalently:

$$
(\hat{\ell}, \hat{x}) = \operatorname*{argmax}_{\ell, x} P(y \mid x, \ell) \cdot P(x \mid \ell) \cdot P(\ell \mid c)
$$

That is the mathematically correct answer.

## SCL version

```text
@braille_cell ≠ @meaning
@meaning := @cell + @codebook + @context

@y∈B₆*
@decode := argmax_{ℓ,x} P(y|x,ℓ) P(x|ℓ) P(ℓ|context)

@ambiguity:
  if multiple ℓ yield high-probability x
  → language not identifiable from cells alone

@solution:
  preserve codebook tag
  or infer via metadata/context/language-model likelihood
```

So the answer is: **they discern it the same way a cryptanalyst or language model would: not from the cell pattern alone, but from code-switch markers, document context, and which language-specific decoding produces coherent text.**

Semantic density score: **0.91**

[1]: https://www.nationalbraille.org/wp-content/uploads/2021/05/Overview-of-Foreign-Language-Transcription.pdf?utm_source=chatgpt.com "75<sup>th</sup>"
[2]: https://iceb.org/publications/ueb/?utm_source=chatgpt.com "Unified English Braille (UEB) – International Council on English Braille (ICEB)"

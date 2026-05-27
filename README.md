# 🧠 Artificial Cerebellar Loop

**A bio-inspired predictive control architecture for braille decoding, multi-model consensus, and codebook inference.**

The artificial Cerebellar Braille Loop (aCBL) is an architecture where AI models communicate exclusively in 8-dot braille Unicode, converge through braided feedback, and jointly infer codebook, text, and encoding strategy — without explicit coordination. It draws directly from how the biological cerebellum performs predictive error minimization during tactile sequence processing.

> **Keywords**: artificial cerebellar braille loop, aCBL, cerebellar braille loop, predictive coding, braille decoding, multi-model consensus, codebook inference, cerebellar computation, tactile language processing, braille Unicode, sensorimotor language

## What is an Artificial Cerebellar Loop?

An **artificial cerebellar loop** is a computational architecture modeled on the biological cerebellum's role as a predictive error minimizer for embodied sequence processing.

When a person reads braille, they don't passively receive text — they *actively sample* a surface through a closed-loop control system:

```
touch → prediction → error correction → motor adjustment → better touch
```

This project implements the AI equivalent:

```
braille token → predict next token → measure surprise → adjust decoding policy → continue
```

Each model in the loop acts as a parallel fiber in the cerebellar cortex. Their outputs are braided together and fed back as context, creating a recurrent consensus mechanism where convergence emerges from prediction error minimization — not from explicit instruction.

### The formal architecture

```
CBL = (S, O, A, C, M, π, E, Λ)
```

| Symbol | Meaning |
|--------|---------|
| S | Sensorimotor state space |
| O | Tactile observation space, 𝔹₆ or 𝔹₈ |
| A | Action space (read-next, rescan, switch-language, expand-contraction) |
| C | Linguistic context state |
| M | Predictive forward model |
| π | Decoding policy |
| E | Error function (dot-level disagreement) |
| Λ | Language/codebook decoder |

The system jointly estimates:

```
(x̂, ℓ̂, ŝ) = argmax P(o₁:T | x, ℓ, s) · P(x | ℓ) · P(ℓ) · P(s)
```

It infers the **text**, the **language/codebook**, and the **encoding strategy** simultaneously.

## How it works

### BBID Handshake (Braille Binary Identity)

1. You enter your name
2. 16 models across 2 providers encode it in braille — independently, in parallel
3. The system clusters models by dot-level similarity to detect **codebook agreement**
4. Models that chose the same encoding strategy form a cluster; the dominant cluster becomes the consensus BBID

### Cerebellar Feedback Loop

1. You enter a prompt
2. All models respond in 8-dot braille Unicode (U+2800–U+28FF)
3. Responses are braided together and fed back as context
4. Models adjust based on what others produced — convergence emerges from the feedback
5. The loop terminates on consensus (≥95% dot agreement) or stable disagreement (plateau detection)

### Codebook Divergence Visualization

The system doesn't just show *whether* models agree — it shows *how they disagree*:

```
(ℓ̂, x̂) = argmax P(y|x,ℓ) · P(x|ℓ) · P(ℓ|context) — 3 codebooks detected

D_ℓ̂ — dominant codebook     P(ℓ|y) ≈ 85% · 11/13 models
  [gpt-4.1] [claude-4-sonnet] [gemini-2.5-flash] ...
  ⠗⠽⠁⠝                                            → ryan

D_ℓ₂ — minority codebook    P(ℓ|y) ≈ 8% · 1/13 models
  [claude-3-haiku]
  ⠐⠗⠽⠁⠝                                           → [capital]ryan

D_ℓ₃ — minority codebook    P(ℓ|y) ≈ 8% · 1/13 models
  [llama-3.1-70b]
  ⠗⠊⠊⠊                                             → riii
```

Each model is a decoder `D_ℓ` choosing a codebook. Consensus = `argmax` over codebooks.

## Key insight

> Braille is not merely a character encoding. It is a **sensorimotor language**.

A visual reader receives a whole word or line in parallel. A braille reader receives a controlled tactile time series: `o₁, o₂, ..., oT`. The cerebellar loop supplies the missing parallelism through prediction.

```
faster reading = better predictive compression
```

In our AI implementation, the "prediction" is each model's prior about how braille should encode the input. The "error" is the disagreement between models. The "motor adjustment" is the braided feedback that shifts models toward consensus.

## Architecture

```
┌─────────────────────────────────────────────┐
│                 Mammouth (S-tier)            │
│  8 models · warm colors · premium squad     │
│  gpt-4.1, claude-4-sonnet, gemini-2.5, ... │
├─────────────────────────────────────────────┤
│                OpenRouter (A-tier)           │
│  8 models · cool colors · value squad       │
│  gpt-4.1-mini, claude-3-haiku, grok, ...   │
└─────────────────┬───────────────────────────┘
                  │
          ┌───────▼───────┐
          │  Braille Only │  All communication in U+2800–U+28FF
          │  No Latin text│  8-dot braille Unicode
          └───────┬───────┘
                  │
    ┌─────────────▼─────────────┐
    │   Cerebellar Feedback     │
    │   ───────────────────     │
    │   1. Parallel fire        │
    │   2. Braid outputs        │
    │   3. Feed back as context │
    │   4. Measure convergence  │
    │   5. Repeat or terminate  │
    └─────────────┬─────────────┘
                  │
          ┌───────▼───────┐
          │  Convergence  │
          │  ⬛ consensus │  ≥95% dot agreement
          │  🌈 disagree  │  plateau detected
          │  ⏳ ongoing   │  still converging
          └───────────────┘
```

### Dynamic model selection

Models are not hardcoded. On startup, the system:

1. Discovers all available chat models per provider via API
2. Clusters them by pricing and context length using k-means
3. Selects a diverse squad balancing family and tier
4. Auto-swaps failed models from a bench of reserves during rounds

## Setup

```bash
git clone https://github.com/elevate-foundry/artificial-cerebellar-loop.git
cd artificial-cerebellar-loop
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
MAMMOUTH_API_KEY=your_mammouth_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

Run:

```bash
streamlit run app.py
```

## Neuroscience basis

Braille reading activates cerebellar regions including motor-related lobules IV/V/VIIIA and language-associated Crus I. One study found Crus I activation during braille reading that was not explained merely by object recognition, suggesting cerebellar participation in language/inner-speech processing as well as sensorimotor control ([PMC6872089](https://pmc.ncbi.nlm.nih.gov/articles/PMC6872089/)).

The artificial cerebellar loop mirrors this:

| Biological | Artificial |
|---|---|
| Parallel fibers | Multiple models firing in parallel |
| Climbing fiber error | Dot-level convergence measurement |
| Purkinje cell output | Majority-vote consensus braille |
| Mossy fiber input | Braided feedback from all models |
| Granule cell expansion | Diverse model families and tiers |
| Cerebellar nuclei output | Final decoded text |

## Theoretical foundation

See [`context.md`](context.md) for the full codebook inference theory:

- A braille cell is a symbol inside a codebook, not intrinsically any language
- Decoding requires joint inference over cell pattern, codebook (language), and context
- The BBID handshake is a practical implementation of `argmax` over codebooks
- Each model is a decoder `D_ℓ` — consensus reveals which codebook the ensemble prefers

## Related concepts

- **Predictive coding** — the free energy principle (Friston) applied to tactile inference
- **Cerebellar computation** — internal models, parallel fibers, climbing fiber error signals
- **Braille decoding** — codebook ambiguity across languages (UEB, SEB, Deutsche Blindenschrift)
- **Multi-agent consensus** — emergent agreement without explicit voting protocols
- **Sensorimotor language** — braille as active sequential cognition, not passive encoding

## License

MIT

## Citation

If you use this work, please cite:

```
@software{cerebellar_braille_loop,
  title={Cerebellar Braille Loop: Predictive Tactile Control over a Language-Conditioned Braille Stream},
  author={Barrett, Ryan},
  year={2025},
  note={Artificial Cerebellar Loop: bio-inspired predictive control for braille consensus},
  url={https://github.com/elevate-foundry/artificial-cerebellar-loop}
}
```

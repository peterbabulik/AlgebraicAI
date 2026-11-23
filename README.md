# AlgorithmicAI

Exploratory numerical studies and falsification-first benchmarks across two connected threads:
1. **Algebraic Structure for Quantum Advantage (`NoGo` & `ToGo`):** Probing where exponential/computational quantum speedups do *not* hide (Cayley–Dickson algebras) and where structured algebraic dualities survive cryptanalytic and classical baselines ($\mathbb{F}_{2^n}$ trace-inverse hidden shift, non-MM bent functions, Temperley–Lieb path algebras, and DQI).
2. **Neuro-Symbolic Algebraic/Geometry AI:** Generating verified synthetic geometry theorems from exact rational coordinates + symbolic deduction (`UltraDD_AR`), training a raw-DSL Transformer (with zero handcrafted features) to propose auxiliary constructions, and evaluating out-of-distribution (OOD) generalization via strict symbolic verification and ablations.

**Repository:** [github.com/peterbabulik/AlgorithmicAI](https://github.com/peterbabulik/AlgorithmicAI)

---

## Part I — Algebraic Structure for Quantum Advantage (`NoGo` & `ToGo`)

### 1. `NoGo`: Where Quantum Advantage Is *Not* Hiding in Cayley–Dickson Algebras
Cayley–Dickson algebras of dimension $2^n$ ($\mathbb{R}, \mathbb{C}, \mathbb{H}, \mathbb{O}, \mathbb{S}, \dots$) are sign-twisted group algebras of $\mathbb{Z}_2^n$ with an explicit $O(n)$ recursion for the sign cocycle $s(a, b)$. Three candidate formulations all fail to produce classical hardness:

- **Attempt A — Twisted Simon:** Sampling output is **identically uniform** ($P(y) = 2^{-n}$ for all $n$, all $p \neq q$, and $\sigma = \pm 1$; verified numerically for $n \le 5$ and proved inductively from the recursion).
- **Attempt B — Zero Divisors / $\ker L_u$ for $k$-sparse $u$:** $L_u$ decomposes into coset blocks of size $\le 2^{k-1}$, and singular blocks are dense ($\text{median } \rho \ge 0.5$). For $k=2$, the singular-coset indicator is $s(p,b)\oplus s(p,b\oplus s)\oplus s(q,b)\oplus s(q,b\oplus s) = 0$ (independent of $\sigma$, empirical $\mathbb{F}_2$-degree $n-3$).
- **Attempt C — Fourier Structure & IQP:** Walsh spectra of $f_a(b) = s(a,b)$ stay concentrated with entropy $H(P) \approx 2.2$ bits independent of $n$ (proved via recursion), have algebraic immunity $\text{AI} = 2$, $81.25\%$ quadratic correlation ($n=6$), and fail anti-concentration even for sums of $8$ signs ($Z = 105$ at $n=10$ vs. $3.6$ for a random cubic).

### 2. `ToGo`: Candidates That Survive Screening

#### Candidate A — Hidden Shift of $\mathbb{F}_{2^n}$ Trace-Inverse
Given known $a \in \mathbb{F}_{2^n}^*$, $f(x) = \text{Tr}(a/x)$ ($0^{-1} := 0$), and oracle $g(x) = f(x \oplus s)$, recover $s$:
- **Quantum sign oracle:** Via $\#E_u(\mathbb{F}_{2^n}) = 2^n + K(\sqrt{u})$ on $E_u: y^2 + xy = x^3 + u$ (0 mismatches vs. brute force for $n \le 10$ and all $4{,}095$ nonzero $y$ at $n=12$ with trace-dual conversion), PARI `ellcard` computes Kloosterman signs in $0.62\text{ s}$ at $n=1024$ ($\sim n^{2.5}$).
- **1-query quantum success (sign-corrected Rötteler circuit):** **$P(s) = 0.7208$** at $n=12$ (Sato–Tate limit $64/(9\pi^2) \approx 0.7205$); for $m=2$ shifted inversions, $W_f(y)$ reduces to a genus-2 Artin–Schreier point count (0 mismatches for $n \le 10$; ideal success $0.6216$ at $n=12$ vs. Gaussian $2/\pi \approx 0.6366$).
- **Cryptanalytic floor & classical attacks:** Optimal algebraic immunity ($\text{AI} = \lceil n/2 \rceil$ for $n=6..8$, full rank), random-like 2nd-order nonlinearity ($\text{NL}_2 = 14/32$ at $n=6$), and differential bias $O(2^{-n/2})$.
  - Generic birthday fingerprinting: $O(n \cdot 2^{n/2})$ queries/time (`16–20/20` recovery up to $n=18$).
  - CryptoMiniSat SAT/XOR on $(x_i \oplus s)z_i = a$: times out ($\ge 40\text{–}60\text{ s}$) by $n=14$ (far slower than brute force).
  - Unknown-pole learning (`triple.c`, $m=3$): scales as $\approx (2^n)^m / m! \cdot \text{poly}(n)$ ($203\text{ s}$ at $n=10$).
- **Open:** Coherent reversible Toffoli/T-gate count for characteristic-2 AGM point counting at $n \gtrsim 80\text{–}128$, and Gröbner/XL/lattice resistance.

#### Additional Screened Directions
- **Non-MM Bent Hidden Shift:** Dillon $\text{Tr}(a x^{2^{n/2}-1})$ ($Z=1$, $\text{AI}=n/2$, same-exponent analytic dual) gives $P(s)=1.000$ in 1 quantum query using one $\mathbb{F}_{2^n}$ inversion, but admits a memory-less $O(n \cdot 2^{n/2})$ coset-line classical attack.
- **Temperley–Lieb Path Algebras $\text{TL}_{12}(\delta)$:** Distinguishes Ising ($r=4$, Clifford, `distinct P>0 = 1`) and $\text{SU}(2)_4$ ($r=6$, finite image, `14`) from Fibonacci ($r=5$, BQP-complete, `89`, $Z=2.413$).
- **Decoded Quantum Interferometry (DQI, $\ell=2$):** Verified exact statevector expectation `0.711948` on $\text{ExtBCH}(16,7,6)$ ($d=6 \ge 2\ell+2$) and explained $v$-dependent spread (`0.700–0.737`) on $\text{BCH}(15,7,5)$ ($d=5 = 2\ell+1$) via weight-2/3 syndrome collisions.

---

## Part II — Neuro-Symbolic Synthetic Geometry AI (4-Cell Colab Pipeline)

A modular 4-cell pipeline that generates verified geometric theorems, trains an Encoder–Decoder Transformer directly on **raw tokenized DSL** (zero handcrafted features), answers English or DSL geometry questions via 1/2-step beam search + exact symbolic verification, and runs controlled OOD ablations.

### Pipeline Overview
- **Cell 1 — Data Generation & Verifier:** Exact `fractions.Fraction` coordinate engine + sound Deductive Database & integer Angle-Chasing Algebraic Reasoner (`UltraDD_AR`). Generates `10,000` training problems on `6–7` point diagrams (`0/100` verifier round-trip errors) and `500` held-out OOD test problems on `8–9` point diagrams (`avg_cands = 44.6`, strictly unsolvable by `UltraDD_AR` alone) with randomized point labels `A..Z` and shuffled premises.
- **Cell 2 — AI Training:** PyTorch Encoder–Decoder Transformer trained with on-the-fly alpha-renaming augmentation across `A..Z` so embeddings generalize to unseen OOD point letters (`H, I`).
- **Cell 3 — Question Asking (`ask_geometry`):** Lightweight English-to-DSL parser (`nl_to_dsl`) + 1-step & 2-step Transformer beam search verified by `UltraDD_AR`.
  - Solves **Trapezoid Diagonal Midpoints** from plain English in **1 step at AI rank #1** (`add E = mid(B,C)`), proves **Varignon's Theorem**, **Triangle Midsegment**, and **Altitude Cyclic Quad**, and solves `4/5` sampled OOD `8–9` pt diagrams (discovering valid alternative `para4` constructions not in the ground-truth label set).
- **Cell 4 — Ablation Study (`500` OOD `8–9` pt diagrams):**

### Key Ablation Results (500 Held-Out OOD `8–9` pt Diagrams)

| Method / Condition | `B=1` | `B=3` | `B=5` | `B=10` | `MedRank` |
|---|---|---|---|---|---|
| Symbolic `DD+AR` alone (0 aux) | 0.0% | 0.0% | 0.0% | 0.0% | — |
| Random Aux + `DD+AR` | 3.6% | 10.2% | 16.6% | 31.8% | 42.0 |
| Goal-Touching Heuristic + `DD+AR` | 2.4% | 13.6% | 21.0% | 30.8% | 41.0 |
| `4L-192d` Free Beam (Unconstrained) | 16.8% | 35.4% | 48.0% | 66.4% | 6.0 |
| `4L-192d` Free Beam (Masked valid pts) | 16.8% | 35.4% | 48.2% | 66.4% | 6.0 |
| `4L-192d` Scored Pool (`140 ep`) | 24.4% | 56.4% | 68.0% | 83.0% | 3.0 |
| No Alpha-Renaming (`augment=False`, `9k`) | 17.8% | — | 56.2% | 71.4% | 4.0 |
| Data Scaling (`N=1,000`, `aug=True`) | 21.0% | — | 65.4% | 85.2% | 3.0 |
| Data Scaling (`N=3,000`, `aug=True`) | 25.0% | — | 69.6% | 86.2% | 3.0 |
| **`2L-128d` Scored Pool (`N=9k`, `50 ep`)** | **28.8%** | — | **71.2%** | **86.6%** | **3.0** |

*(On `2L-128d`, `MaskedBeam` also improves to `21.8% / 55.6% / 70.6%`.)*

**Main ablation takeaways:**
1. **Alpha-renaming is critical (`+11%` at `B=1`, `+15%` at `B=10`):** Without it (`val_loss = 0.6613`), the model overfits to `A..G` and degrades on OOD `8–9` pt letters `H, I`.
2. **Compact `2L-128d` outperforms `4L-192d`:** Smaller capacity + `best_val_loss` checkpointing generalizes better to larger diagrams while training ~3.5× faster.
3. **Logit masking barely changes Free Beam (`+0.2%`):** The Transformer already learned `100%` valid syntax and diagram-local letters; the gap to `Scored Pool` comes from beam competition between `mid(...)` and `para4(...)` templates.
4. **Complementary failure modes:** On `para` (`n=400`) and `mid` (`n=402`), the Goal-Touching Heuristic is worse than Random (`1.8% / 15.0%`) while the Transformer hits `25.8% / 69.0%` (`74.1%` on `mid` at `B=5`). On rare `cyclic` (`n=34`, only `2%` of training data), Goal-Touching reaches `73.5%` at `B=5` vs. `61.8%` for the Transformer.

---

## Requirements & Running

- **Part I (Quantum / Algebra scripts):** Python 3.10+, `numpy`, plus `cypari` (PARI/GP) for elliptic-curve point counting, `pycryptosat` for SAT/XOR checks, and `gcc` for `triple.c`.
- **Part II (Geometry AI Colab pipeline):** Python 3.10+, `numpy`, `torch` (recommended: Colab T4 GPU). Run **Cells 1 → 2 → 3 → 4** sequentially in a single notebook.

---

## Status & Caveats

- Exploratory empirical studies (`n ≤ 12` for full quantum/geometry simulations, `n ≤ 1024` for PARI `ellcard`); negative results in `NoGo` include inductive proofs for twisted-Simon uniformity and $O(1)$ Walsh entropy, while classical hardness of $\mathbb{F}_{2^n}$ trace-inverse hidden shift remains a conjecture (bounded above by $O(n \cdot 2^{n/2})$ birthday fingerprinting).

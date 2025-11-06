
# Solving ECDLP

# Goal
Given an elliptic curve $E$ over a prime field $F_p$, a publicly known base point $G ∈ E(F_p)$ of order n, and a public point $Q ∈ ⟨G⟩$, find the integer $d (0 < d < n)$ such that $Q = d \cdot G$.

### Assumptions (inputs)

* Prime modulus p (p is prime).
* Curve coefficients a, b ∈ F_p satisfying $4a^3 + 27b^2 ≠ 0 (mod p)$.
* Elliptic curve $E: y^2 = x^3 + a x + b (mod p)$.
* Base point $G = (x_G, y_G) ∈ E(F_p)$.
* Order n of G (smallest positive integer with n·G = O). n is usually prime.
* Public point $Q = (x_Q, y_Q) ∈ ⟨G⟩$.
* Goal: find integer d with $1 ≤ d < n$ so that d·G = Q.


# a. Brute Force

## High-level idea
Compute successive multiples of G:
```
R ← G, k ← 1
While R ≠ Q and k < n:
    R ← R + G
    k ← k + 1
If R = Q 
    return k else report “not found in subgroup”.
```
This is the naive/baseline search. It requires O(n) group operations; on realistic curves n is astronomically large so brute force is infeasible.

### Pseudocode (precise)

```
Input: p, a, b, G, Q, n
Output: d such that d·G = Q, or FAIL if none found within [1..n-1]

FUNCTION BruteForceECDLP(p, a, b, G, Q, n):
if Q == O:                   # optional check for point at infinity
    if 0 < n and (0 * G) == Q:
        return 0
else:
    return FAIL

R ← G
k ← 1
WHILE k < n:
    if R == Q:
        return k
    R ← EC_Point_Add(R, G, a, p)   # elliptic curve point addition mod p
    k ← k + 1    
    
// last check if k == n (optional): R should have become O

if R == Q:
    return k
return FAIL
```

Notes on EC_Point_Add:

* Use modular arithmetic mod p.
* If R == G you will be doing doubling formula automatically when implementing addition.
* Handle special cases: point at infinity, $P == -Q (sum = O)$, denominator inverse via modular inverse.

### Time & space complexity

* Time: O(n) elliptic-curve group operations (point additions/doublings). If n ≈ 2^b, brute force cost ≈ 2^b operations.
* Space: O(1) extra memory (just current R and counter k) — minimal memory baseline.
* Average-case: expected ~n/2 group ops to find d (if uniformly random).

Practicality

* Feasible only for tiny toy curves (n small).
* For secure curves n ≈ 2^256 (or ≥2^160 in old curves) → brute force impossible.

Toy example (concrete numbers, manually traceable)
Use this in demonstrations:

* p = 17
* curve: y^2 = x^3 + 2x + 2 (mod 17)
* G = (5, 1), assume order n = 19 (toy assumption)
* secret d = 7 (private)
* compute Q = d·G = (10, 6)  (by repeated addition; you can show 1G, 2G, 3G,...)

```
Run brute-force:
k=1 → R=1G=(5,1)   R≠Q
k=2 → R=2G=(6,3)   R≠Q
k=3 → R=3G=(10,6)  R == Q → return k=3
```

(If in your toy numbers the observed first match is k=3 while private d chosen was 7, point order considerations mean d ≡ 3 (mod order). For consistent toy demonstration pick d < n or show modular equivalence.)


Security takeaway
Brute-force illustrates the problem and serves as a baseline, but it is not an attack on real ECC: standard curves are chosen so that n is huge (e.g., ~2^256) making brute force infeasible.

Below is a clean, formal explanation of how **partial key leakage** (known bits, bounded interval, etc.) alters the brute-force attack on ECDLP. It includes adapted search strategies, complexity estimates, small toy examples you can present, and short mitigation advice.

---

## 1 — The attack model (partial leakage)

Assume the attacker knows:

* Curve parameters and generator (G) (public).
* Public key (Q=dG) (public).
* **Partial leakage** about the private key (d). Examples:

  * **Known bits**: some subset of bits of (d) (e.g., lowest `b` bits, or several scattered bit positions).
  * **Bounded interval**: $(d \in [L, U])$.
  * **Masked value or truncated value**: attacker sees `d` modulo a small value or sees only high/low portion.
    The attacker’s goal: exploit that leakage to reduce the brute-force search space.

---

## 2 — High-level effect of leakage

* If `b` low (or high) consecutive bits are known exactly, the search space reduces by factor ≈ $(2^{b})$.
  Example: if `b=40`, a 256-bit private key’s brute force drops from $(2^{256})$ to $(2^{216})$ — still infeasible, but dramatically smaller.
* If `d` is confined to an interval of length `m` (i.e., (U-L+1=m)), complexity becomes (O(m)).
* Combined leaks intersect constraints: e.g., known bits *and* interval → candidate set = {numbers in interval matching bit pattern}.

---

## 3 — Adaptations of brute force

### (A) Brute force with known consecutive bits (LSBs or MSBs)

Idea: enumerate only integers whose known-bit positions match leaked values.

Pseudocode (conceptual):

* Input: public (G, Q), order (n), leaked bits specification `bits` (positions and values).
* For `candidate` in `[0 .. n-1]`:

  * If `candidate` matches `bits` then test `candidate·G == Q`.
* Return candidate if match found.

Time complexity: $(O(n / 2^{b}))$ EC ops when `b` bits are known (assuming those bits uniformly constrain the space).
Space: O(1).

### (B) Brute force over a bounded interval

Idea: search only `k ∈ [L..U]`.

Pseudocode (conceptual):

* Input: `L, U`.
* For `k` from `L` to `U`: if `k·G == Q` return `k`.

Time complexity: (O(U-L+1)).

### (C) Combined strategy (bits + interval)

* Intersect constraints; iterate only values inside `[L,U]` that also match leaked bit pattern.
* Complexity roughly `O(#candidates)` where `#candidates` ≈ `(U-L+1)/2^b` (if bits independent and uniform).

### (D) Incremental / prioritized search

* If attacker believes leakage is noisy or partial (e.g., some bits may be wrong), rank candidates by probability (e.g., Hamming distance to leaked pattern) and test in that order to maximize early success probability.

---

## 4 — Toy examples (concrete, small numbers you can show in class)

Use the toy ECC from earlier or a simple integer example to illustrate the reduction clearly.

### Toy ECC parameters (small)

* Field (p=17), curve (y^2 = x^3 + 2x + 2 \pmod{17}).
* Base point (G=(5,1)), order (n=19) (toy).
* Private (d = 7) → public (Q = 7G = (10,6)).

#### Example A — Known low 2 bits (b = 2)

* `d = 7` in binary (within [0..18]) → 7 ≡ `11` (mod 4).
* Candidate set = {k ∈ [0..18] | k mod 4 = 3} = {3, 7, 11, 15} (4 candidates).
* Brute force tests only 4 values → far cheaper than testing 19 values. (In the toy this cuts work by factor ~4.)

#### Example B — Bounded interval

* Suppose leak gives `d ∈ [5, 10]`.
* Candidates = {5,6,7,8,9,10} (6 candidates).
* Brute force tests 6 values.

#### Example C — Combined: interval [0,18] & LSBs = `11`

* Intersection as in A: still {3,7,11,15}.

Use these steps live on the board or slide: list all kG values and show which candidate hits Q first.

---

## 5 — Quantitative impacts & complexity estimates

* **Known `b` consecutive bits** reduces expected operations by factor ≈ (2^{b}). Formal time: (T \approx c\cdot (n/2^{b})) where `c` is cost per EC op.
* **Interval length `m`** gives time $(T \approx c\cdot m)$.
* **Combined**: $(T \approx c\cdot \frac{m}{2^{b}})$ (if bits and interval are independent).
* **Practical meaning**: For a 256-bit key, knowing 32 bits reduces search from $(2^{256})$ to $(2^{224})$ — still infeasible, but useful if the attacker can combine many leaks or target weak curves.

---

## 6 — Practical notes (attacks vs realistic curves)

* Real ECC uses (n ≈ 2^{256}). Even dramatic leaks (e.g., 64 known bits → (2^{192}) remaining) remain infeasible for a single attacker.
* **Partial leakage becomes critical** when:

  * The leak is large (e.g., many consecutive bits).
  * The leak reduces entropy below practical brute-force thresholds (e.g., remaining entropy ≤ 60–80 bits).
  * Many keys with similar leakage are available (parallelism/meet-in-the-middle style).
  * Implementation errors produce biased keys (not uniformly random).
* For moderate leakage, the attacker will prefer sub-exponential or √n algorithms (Pollard’s Rho, baby-step giant-step) combined with the constrained search — e.g., run Pollard’s Rho but initialize or restrict to found cosets if that’s applicable.

---

## 7 — Defenses / mitigations (short list for your write-up)

* **Avoid leaking bits**: constant-time implementations, mask and blinding techniques (randomize ephemeral nonces, scalar blinding).
* **Use full-entropy keys**: ensure private keys are uniformly random in ([1, n-1]).
* **Key derivation and key-stretching**: if keys originate from passwords or low-entropy sources, apply KDFs and salts to raise entropy.
* **Side-channel resistant code**: protect against timing/power/EM leaks that expose bits.
* **Ephemeral keys** (where applicable): limit value lifetime so any partial leak is less useful.

---

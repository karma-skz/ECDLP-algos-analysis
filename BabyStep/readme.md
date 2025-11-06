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


# Baby-step Giant-step

**Baby-Step Giant-Step (BSGS): Meet-in-the-Middle for ECDLP**

---

### 🔹 Problem

Given base point **G** and public key **Q = d·G**, find **d** (the private key).

Elliptic curve group order = *n*.

Brute force ⇒ O(n) steps → infeasible.

---

### 🔹 Idea

Split secret **d** into two parts:

$d = i·m + j, \quad m = ⌈√n⌉$

Then
$$
Q = d·G = i(mG) + jG \Rightarrow Q - i(mG) = jG
$$

Compute all possible **jG** once (baby steps),
then loop over **i** to find matching **$Q − i(mG)$** (giant steps).

---

### 🔹 Algorithm

1. **Baby Steps:**
   Precompute and store ( jG ) for ( j = 0, 1, …, m-1 )
   → store in a hash map `{point → j}`

2. **Giant Steps:**
   Compute ( S = mG ).
   For each ( i = 0, 1, …, m-1 ):

   * Compute ( cur = Q - iS )
   * If ( cur ) in table → ( d = i·m + j )

---

### 🔹 Complexity

| Metric | Brute Force | BSGS  |
| ------ | ----------- | ----- |
| Time   | O(n)        | O(√n) |
| Space  | O(1)        | O(√n) |

---

### 🔹 Toy Example

Curve: y² = x³ + 2x + 3 mod 97
G = (3,6), order n = 5
Q = (80,87), find d.

1. √n ≈ 3 ⇒ m = 3
2. Baby steps:
   0G=O, 1G=(3,6), 2G=(80,10)
3. Giant-step factor: S = 3G = (80,87)

Check:

* i=0 → Q-(0·S)=(80,87) ❌
* i=1 → Q-(1·S)=O ✅ baby[O]=0
  → **d = 1·3 + 0 = 3**

✅ Found private key **d = 3**

---

### 🔹 Key Takeaways

* “Meet-in-the-middle” halves the exponent search.
* Time ≈ 2^(b/2) for b-bit keys → much faster than brute force.


**Baby-Step Giant-Step (BSGS) — Partial Key Leakage Adaptation**

---

### 🔹 Core Idea

If some bits of the secret key *d* are leaked or *d* lies in a bounded interval,
reduce the search domain before running BSGS.

Standard ECDLP:
 Find *d* such that **Q = d·G**, with *G* generator of order *n*.
 Complexity ≈ **O(√n)** time & space.

---

### 🔹 Known Bits (Low *b* Bits Leak)

If *d ≡ r (mod 2ᵇ)* ⇒ *d = s·u + r*, where *s = 2ᵇ*.
Transform:
 Q' = Q − rG = u·(sG)

Now solve reduced ECDLP:
 **Q' = u·G′**, with *G′ = sG*, order ≈ *n/s*.

**Cost:** O(√(n/s))  → Speedup ≈ 2^(b/2)

---

### 🔹 Known Interval Leak

If *d ∈ [L, U]* ⇒ *Q' = Q − L·G = t·G*, with *t = d − L*.
Run BSGS only for *t ∈ [0, U−L]*.

**Cost:** O(√(U−L))

---

### 🔹 Combined Leak

1. Express *d = s·u + r* using leaked bits.
2. Intersect with interval constraint [L, U].
3. Run BSGS for *u* only over that smaller interval.

---

### 🔹 Toy Example

Curve: *p = 97, G = (3,6), n = 5*, public *Q = (80,87)*
Known 1 LSB → *b = 1, r = 1, s = 2*
⇒ *G′ = 2G = (80,10)*, *Q′ = Q − G*
Domain shrinks from 5 → 3 ⇒ cost ~√3 ≈ 2 steps (vs √5 ≈ 3)

---

### 🔹 Summary

| Leak Type         | Domain Size | Time/Space | Example Speedup |
| ----------------- | ----------- | ---------- | --------------- |
| None              | *n*         | O(√n)      | —               |
| b known bits      | n / 2ᵇ      | O(√(n/2ᵇ)) | 2^(b/2)         |
| Interval length m | m           | O(√m)      | depends on m    |

---

**Key takeaway:**
➡ Partial information shrinks the effective ECDLP domain,
making BSGS faster by roughly √(reduction factor).


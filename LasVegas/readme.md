# Las Vegas Algorithm for ECDLP: Theory & Implementation

[cite_start]This document explains the theory and implementation of the Las Vegas algorithm for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP), as described in the paper "A Las Vegas algorithm to solve the elliptic curve discrete logarithm problem"[cite: 1].

The code in `main.py` is a direct implementation of this paper's methodology.

## 🎯 The ECDLP Problem

Given an elliptic curve $E$, a public base point $G$ of order $n$, and a public point $Q$, the ECDLP is to find the secret integer $d$ (where $1 \le d < n$) such that:

**$Q = d \cdot G$**

## 🏛️ The Core Theory: The "Magic" Link

[cite_start]The entire algorithm is built on a key theorem [cite: 50] that connects the elliptic curve group operation (point addition) to algebraic geometry (curves).

> **Theorem 1 (Simplified):**
> [cite_start]A set of $k$ points $P_1, P_2, \dots, P_k$ on an elliptic curve sums to the identity element $\mathcal{O}$ (i.e., $\sum_{i=1}^{k} P_i = \mathcal{O}$) **if and only if** there exists another algebraic curve $\mathcal{C}$ of degree $n'$ (where $k = 3n'$) that passes through all $k$ of those points[cite: 51].

### How This Is Used to Solve ECDLP

Our goal is to find $d$. The algorithm's approach is to find a set of random integers $n_i$ and $n_j'$ that satisfy the following equation:

[cite_start]$\sum (n_i G) + \sum (-n_j' Q) = \mathcal{O}$ [cite: 69]

If we can find such a set, we can solve for $d$:
1.  Let $A = \sum n_i$ and $B = \sum n_j'$.
2.  The equation becomes: $A \cdot G + B \cdot (-Q) = \mathcal{O}$
3.  $A \cdot G = B \cdot Q$
4.  Substitute $Q = d \cdot G$: $A \cdot G = B \cdot (d \cdot G) = (B \cdot d) \cdot G$
5.  This implies (working in $\pmod n$): $A \equiv B \cdot d \pmod n$
6.  [cite_start]Therefore, the secret is: **$d = A \cdot B^{-1} \pmod n$** [cite: 164]

The entire problem is now reduced to: **How do we find such a set of $n_i$ and $n_j'$?**

---

## ⚙️ The Algorithm: A Two-Part Strategy

[cite_start]The algorithm is split into two main parts[cite: 23, 122]:

1.  [cite_start]**Algorithm 1:** A Las Vegas algorithm that efficiently reduces the ECDLP into a specific linear algebra problem, which the paper calls **"Problem L"** [cite: 123-128].
2.  [cite_start]**Algorithm 2:** A heuristic (and non-guaranteed) algorithm to solve Problem L[cite: 237]. [cite_start]This is the "bottleneck" of the attack[cite: 26].

### Algorithm 1: Reducing ECDLP to "Problem L"

This algorithm is a clever way to test many subsets of points at once.

[cite_start]Instead of picking $k=3n'$ points and *hoping* they sum to $\mathcal{O}$, we pick **more** points, $k+l$ (where $l=3n'$)[cite: 102, 130].

[cite_start]The paper's **Corollary 1** [cite: 114] [cite_start]states that finding a $3n'$ subset of these points that sums to $\mathcal{O}$ is **equivalent** to finding a special vector $v$ in the **left-kernel ($\mathcal{K}$)** of a matrix $\mathcal{M}$[cite: 115]. [cite_start]This vector $v$ must have exactly **$l$ zeros**[cite: 114].

Here is the step-by-step process:

1.  **Setup (Choose $n'$, set $l=3n'$):**
    * We choose a small integer $n'$ (e.g., $n'=2$). This means $l=6$.
    * We will generate a total of $3n' + l = 6n' = 12$ points.
    * We will build a matrix $\mathcal{M}$ with $3n' + l = 12$ rows.
    * [cite_start]The number of columns will be $\binom{n'+2}{2}$[cite: 130]. For $n'=2$, this is $\binom{4}{2} = 6$ columns.

2.  [cite_start]**Build Matrix $\mathcal{M}$ [cite: 133-151]:**
    * Generate $3n'-1 = 5$ random points $P_i = r_i G$. Store the $r_i$ values in a list $\mathcal{I}$.
    * Generate $l+1 = 7$ random points $Q_j = -r_j' Q$. Store the $r_j'$ values in a list $\mathcal{J}$.
    * [cite_start]Each row in $\mathcal{M}$ is a **monomial expansion** of its corresponding point's coordinates $(x, y, z)$[cite: 75, 79]. For $n'=2$, the monomials are $(x^2, y^2, z^2, xy, xz, yz)$. Using affine coordinates $(x, y, 1)$, a row for point $(x_i, y_i)$ is:
        `[x_i^2, y_i^2, 1, x_i*y_i, x_i, y_i]` (all $\pmod p$)

3.  [cite_start]**Compute Left-Kernel $\mathcal{K}$[cite: 152]:**
    * [cite_start]The left-kernel $\mathcal{K}$ is the standard (right) kernel of the transpose of $\mathcal{M}$, i.e., $\text{Kernel}(\mathcal{M}^T)$[cite: 82].
    * This is a standard linear algebra operation. The result is a basis for $\mathcal{K}$.

4.  [cite_start]**Define "Problem L"[cite: 153]:**
    * [cite_start]The ECDLP is now reduced to **Problem L**: Find a vector $v$ (which is a linear combination of the basis vectors of $\mathcal{K}$) that contains exactly $l=6$ zeros [cite: 224-225].

### Problem L & Algorithm 2: The Bottleneck 🧩

[cite_start]This is the hard part and the "bottleneck" of the attack[cite: 26].

* We have a basis matrix for $\mathcal{K}$ of size $l \times (3n'+l)$.
* [cite_start]Since we set $l=3n'$, our kernel basis matrix $\mathcal{K}$ has $l$ rows and $2l$ columns ($l \times 2l$)[cite: 228].
* [cite_start]**Algorithm 2** is a heuristic to solve this[cite: 237]:
    1.  Treat $\mathcal{K}$ as two $l \times l$ blocks, $[\mathcal{K}_1 | \mathcal{K}_2]$.
    2.  [cite_start]Perform Gaussian elimination to diagonalize the *first* block $\mathcal{K}_1$[cite: 232].
    3.  After this, every row has at least $l-1$ zeros (from the diagonalization).
    4.  [cite_start]Check if any row *also* has a zero in the $\mathcal{K}_2$ block[cite: 234]. If yes, we found our vector $v$!
    5.  [cite_start]If not, repeat the process but this time diagonalize the *second* block $\mathcal{K}_2$ [cite: 236] and look for a zero in $\mathcal{K}_1$.
    6.  If this *still* fails, Algorithm 2 has failed. [cite_start]We must **go back to Algorithm 1** and generate a new matrix $\mathcal{M}$[cite: 132].

---

## 🏁 Final Calculation (If Successful)

If Algorithm 2 finds a solution vector $v$ (with $l$ zeros), we can calculate the final answer:

1.  Initialize $A=0$ and $B=0$.
2.  [cite_start]Iterate through the $3n'-1$ $G$-points: if $v[i] \ne 0$, add its corresponding random number $A = A + \mathcal{I}[i]$ [cite: 154-158].
3.  [cite_start]Iterate through the $l+1$ $Q$-points: if $v[j] \ne 0$, add its corresponding random number $B = B + \mathcal{J}[k]$ [cite: 159-163].
4.  Calculate the final secret (using the group order $n$ as the modulus):

    [cite_start]**$d = (A \cdot B^{-1}) \pmod n$** [cite: 164]

## 🐍 How to Use the Code

The `main.py` script implements this entire flow. It can be run as follows, using the test cases provided:

```python
# --- Example: Test Case 1 ---
p = 17
a = 2
b = 2
Gx = 5
Gy = 1
n = 19
d_secret = 13

# 1. Setup the curve and points
my_curve = EllipticCurve(a, b, p)
G = Point(Gx, Gy)
Q = my_curve.multiply(G, d_secret)

# 2. Initialize the solver
solver = LasVegasECDLP(curve=my_curve, G=G, Q=Q, n=n)

# 3. Run the solver
# We use n_prime=2 and give it 100 tries
found_d, attempts = solver.solve(n_prime=2, max_tries=100)

if found_d is not None:
    print(f"\nFinal Result: d = {found_d} (found in {attempts} attempts)")
else:
    print("\nFinal Result: d not found.")
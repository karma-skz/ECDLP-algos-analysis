# 🔒 Pohlig-Hellman Algorithm for ECDLP

The Pohlig-Hellman algorithm is a **special-purpose** method for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP): finding the integer $d$ such that $Q = d \cdot G$, where $n$ is the order of the base point $G$.

Its efficiency relies entirely on the order $n$ being a **smooth integer** (i.e., its largest prime factor is small).

---

## 💡 Algorithm Steps

Pohlig-Hellman breaks the main ECDLP into smaller, independent subproblems and uses the Chinese Remainder Theorem (CRT) to combine the solutions.

### 1. Factor the Order $n$

First, the order $n$ of the base point $G$ is factored into its prime power decomposition:

$$n = q_1^{e_1} \cdot q_2^{e_2} \cdots q_k^{e_k}$$

### 2. Reduce to Subgroup Problems (Solve $d \pmod{q_i^{e_i}}$)

For each prime power factor $n_i = q_i^{e_i}$, the goal is to find a solution $d_i$ such that $d \equiv d_i \pmod{n_i}$.

The process involves projecting the original problem onto a smaller subgroup of order $n_i$:

* **Project Points:** Calculate the cofactor $h = n / n_i$. Use this cofactor to define the new generator $G_1 = h \cdot G$ and the new target $Q_1 = h \cdot Q$. Both $G_1$ and $Q_1$ belong to a subgroup of order $n_i$.
* **Subgroup ECDLP:** Solve the reduced problem: Find $d_i$ such that $Q_1 = d_i \cdot G_1$.
    * This is typically solved using the **Baby-step Giant-step (BSGS)** algorithm, which is fast because the subgroup order $n_i$ (and its prime factor $q_i$) is assumed to be small. The result $d_i$ is the logarithm modulo $n_i$.

### 3. Combine Results using CRT

Once a solution $d_i$ is found for every modulus $n_i$, the final discrete logarithm $d$ is recovered using the Chinese Remainder Theorem (CRT):

$$d \equiv d_1 \pmod{n_1}$$
$$d \equiv d_2 \pmod{n_2}$$
$$\vdots$$
$$d \equiv d_k \pmod{n_k}$$

The CRT provides the unique solution $d$ modulo $n$.

---

## ⏱️ Complexity and Limitations

### Time Complexity

The total runtime is dominated by the time required to solve the subproblem corresponding to the **largest prime factor**, $q_{\max}$.

$$O\left(\sum_{i=1}^k e_i \cdot (\log n + \sqrt{q_i})\right)$$

### Limitations (Cryptographic Defense)

* **Reliance on Smoothness:** The algorithm is efficient *only* if the largest prime factor ($q_{\max}$) of the order $n$ is small.
* **Security Requirement:** To ensure cryptographic security, the order $n$ of an elliptic curve base point $G$ **must** contain a sufficiently large prime factor ($q_{\max} > 2^{160}$), which renders the $\mathcal{O}(\sqrt{q_{\max}})$ complexity of the BSGS step infeasible and effectively neutralizes the Pohlig-Hellman attack.
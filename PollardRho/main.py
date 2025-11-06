"""
Pollard-rho for ECDLP (Floyd cycle detection)
Reads 5-line positional input (p | a b | Gx Gy | n | Qx Qy) from a file path given as CLI arg,
or defaults to ./input/filename.txt relative to this file.
Outputs metadata, stats, and verification, aligned with other modules.
"""

from typing import Optional, Tuple
import random
import math
import time
from pathlib import Path
import sys
import math

Point = Optional[Tuple[int, int]]

# ---------- EC helpers (same contracts as other scripts) ---------- #

def egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended gcd: returns (g, x, y) such that a*x + b*y = g = gcd(a,b)."""
    if b == 0:
        return (a, 1, 0)
    else:
        g, x1, y1 = egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)

def mod_inv_euclid(x: int, m: int) -> int:
    """Modular inverse using extended Euclid. Raises ZeroDivisionError if inverse doesn't exist."""
    x %= m
    g, u, v = egcd(x, m)
    if g != 1:
        raise ZeroDivisionError(f"no inverse: gcd({x},{m}) = {g}")
    return u % m

def ec_point_neg(P: Point, p: int) -> Point:
    if P is None:
        return None
    x, y = P
    return (x % p, (-y) % p)

def ec_point_add(P: Point, Q: Point, a: int, p: int) -> Point:
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if P != Q:
        num = (y2 - y1) % p
        den = (x2 - x1) % p
        lam = (num * mod_inv_euclid(den, p)) % p
    else:
        if y1 % p == 0:
            return None
        num = (3 * (x1 * x1 % p) + a) % p
        den = (2 * y1) % p
        lam = (num * mod_inv_euclid(den, p)) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)

def ec_scalar_mul(k: int, P: Point, a: int, p: int) -> Point:
    if P is None:
        return None
    if k < 0:
        return ec_point_neg(ec_scalar_mul(-k, P, a, p), p)
    R: Point = None
    Q: Point = P
    while k:
        if k & 1:
            R = ec_point_add(R, Q, a, p)
        Q = ec_point_add(Q, Q, a, p)
        k >>= 1
    return R

# ---------- Sanity checks ---------- #

def is_curve_valid(p: int, a: int, b: int) -> bool:
    return ((4 * (a % p) ** 3) + (27 * (b % p) ** 2)) % p != 0

def is_point_on_curve(P: Point, p: int, a: int, b: int) -> bool:
    if P is None:
        return True
    x, y = P
    if not (0 <= x < p and 0 <= y < p):
        return False
    return (y * y - (x * x * x + a * x + b)) % p == 0

# ---------- Pollard-rho implementation ---------- #

def _partition_index(P: Point, p: int, m: int) -> int:
    """Partition using low bits of x coordinate: returns 0..m-1."""
    if P is None:
        return 0
    # use several low bits of x to get better mixing; ensure non-negative
    return P[0] % m

def pollard_rho_ecdlp(p: int, a: int, b: int,
                      G: Point, Q: Point, n: int,
                      max_steps: int = 2_000_000,
                      partition_m: int = 16) -> Tuple[Optional[int], int]:
    """
    Pollard-rho for ECDLP using Floyd's cycle detection.
    Returns (d, steps) where d satisfies Q = d*G (mod n), else (None, steps).
    """
    if G is None or Q is None:
        return None, 0

    def random_state():
        A = random.randrange(n)
        B = random.randrange(n)
        X1 = ec_scalar_mul(A, G, a, p)
        X2 = ec_scalar_mul(B, Q, a, p)
        X = ec_point_add(X1, X2, a, p)
        return (X, A % n, B % n)

    def f(state):
        X, A, B = state
        idx = _partition_index(X, p, partition_m)
        # partition into three classes but selection depends on idx
        # we'll use 3 actions but partition size m controls variety of idx values.
        r = idx % 3
        if r == 0:
            # X <- X + G
            X_new = ec_point_add(X, G, a, p)
            return (X_new, (A + 1) % n, B)
        elif r == 1:
            # X <- 2*X
            X_new = ec_point_add(X, X, a, p)
            return (X_new, (2 * A) % n, (2 * B) % n)
        else:
            # X <- X + Q
            X_new = ec_point_add(X, Q, a, p)
            return (X_new, A, (B + 1) % n)

    # initialize tortoise & hare
    state_t = random_state()
    state_h = (state_t[0], state_t[1], state_t[2])
    steps = 0

    while steps < max_steps:
        state_t = f(state_t)
        state_h = f(state_h)
        state_h = f(state_h)

        Xt, At, Bt = state_t
        Xh, Ah, Bh = state_h

        steps += 1

        if Xt == Xh:
            num = (At - Ah) % n
            den = (Bh - Bt) % n
            g, _, _ = egcd(den, n)
            if g == 0:
                return None, steps
            if g != 1:
                # gcd != 1: no unique inverse modulo n; caller should restart/retry
                # but sometimes we can still have a solution if num % g == 0 (multiple solutions).
                # For simplicity: treat as a restart.
                return None, steps
            # den and n coprime -> invertible
            try:
                den_inv = mod_inv_euclid(den, n)
            except ZeroDivisionError:
                return None, steps
            d = (num * den_inv) % n
            # verify
            if ec_scalar_mul(d, G, a, p) == Q:
                return d, steps
            else:
                return None, steps

    return None, steps

# ---------- Input loader ---------- #

def load_positional_input(file_path: Path):
    with file_path.open('r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 5:
        raise ValueError("Input file must contain 5 non-empty lines: p | a b | Gx Gy | n | Qx Qy")
    def ints(s: str):
        return list(map(int, s.split()))
    p = ints(lines[0])[0]
    a, b = ints(lines[1])
    Gx, Gy = ints(lines[2])
    n = ints(lines[3])[0]
    Qx, Qy = ints(lines[4])
    return p, a, b, (Gx, Gy), n, (Qx, Qy)

# ------------------------- Runner ------------------------- #
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    default_path = script_dir / 'input' / 'filename.txt'
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    p, a, b, G, n, Q = load_positional_input(input_path)

    # Sanity checks
    if not is_curve_valid(p, a, b):
        raise ValueError("Invalid curve: 4a^3 + 27b^2 ≡ 0 (mod p)")
    if G is None or Q is None:
        raise ValueError("G/Q cannot be point at infinity")
    if not is_point_on_curve(G, p, a, b):
        raise ValueError("G is not on the curve")
    if not is_point_on_curve(Q, p, a, b):
        raise ValueError("Q is not on the curve")
    nG = ec_scalar_mul(n, G, a, p)
    if nG is not None:
        print("Warning: n*G != O; provided n may not be the exact order")

    method = "Pollard-rho"
    partition_m = 16      # stronger partitioning (uses x % 16 then mod 3 for action)
    max_steps = 5_000_000 # increase budget
    attempts = 50         # number of independent restarts
    steps_total = 0
    start = time.perf_counter()
    d_found = None
    for attempt in range(attempts):
        d_try, steps = pollard_rho_ecdlp(p, a, b, G, Q, n, max_steps=max_steps, partition_m=partition_m)
        steps_total += steps
        if d_try is not None:
            d_found = d_try
            break
    elapsed = time.perf_counter() - start

    # Output metadata & stats
    print(f"Curve: p={p}, a={a}, b={b}")
    print(f"G=({G[0]},{G[1]}), Q=({Q[0]},{Q[1]}), n={n}")
    print(f"Method: {method}")
    print(f"Partition m: {partition_m}")
    print(f"Steps taken: {steps_total}")
    print(f"Attempts: {attempts}")
    print(f"Time elapsed: {elapsed:.6f} s")

    if d_found is not None:
        print(f"Found d: {d_found}")
        Q_check = ec_scalar_mul(d_found % n, G, a, p)
        match = (Q_check == Q)
        print(f"Verified: {match}")
        print(f"Match: {match}")
    else:
        print("No solution found (None)")

# ECDLP Baby-Step Giant-Step (BSGS) Solver
# Point representation: (x, y) tuples; point at infinity = None
# Curve: y^2 = x^3 + a x + b (mod p), with prime p

from typing import Optional, Tuple, Dict
import math
import sys
from pathlib import Path
import time
import sys as _sys

Point = Optional[Tuple[int, int]]  # None denotes the point at infinity

# ------------------------- Finite field / EC primitives ------------------------- #

def mod_inv(x: int, p: int) -> int:
    """Modular inverse of x modulo prime p using Fermat's little theorem."""
    x %= p
    if x == 0:
        raise ZeroDivisionError("no inverse for 0 modulo p")
    return pow(x, p - 2, p)

def ec_point_neg(P: Point, p: int) -> Point:
    """Additive inverse of a point P on E(F_p)."""
    if P is None:
        return None
    x, y = P
    return (x % p, (-y) % p)

def ec_point_add(P: Point, Q: Point, a: int, p: int) -> Point:
    """Add two points P and Q on the curve y^2 = x^3 + a x + b (mod p).
    Returns the resulting point (or None for point at infinity).
    """
    # Handle identity cases
    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    # P + (-P) = O
    if x1 == x2 and (y1 + y2) % p == 0:
        return None

    if P != Q:
        # slope = (y2 - y1) / (x2 - x1)
        num = (y2 - y1) % p
        den = (x2 - x1) % p
        lam = (num * mod_inv(den, p)) % p
    else:
        # Point doubling: slope = (3*x1^2 + a) / (2*y1)
        if y1 % p == 0:
            return None  # tangent is vertical -> infinity
        num = (3 * (x1 * x1 % p) + a) % p
        den = (2 * y1) % p
        lam = (num * mod_inv(den, p)) % p

    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)

def ec_scalar_mul(k: int, P: Point, a: int, p: int) -> Point:
    """Double-and-add scalar multiplication: k * P."""
    if P is None:
        return None
    if k < 0:
        # handle negative scalars: (-k)P = -(kP)
        return ec_point_neg(ec_scalar_mul(-k, P, a, p), p)
    R: Point = None
    Q: Point = P
    while k > 0:
        if k & 1:
            R = ec_point_add(R, Q, a, p)
        Q = ec_point_add(Q, Q, a, p)
        k >>= 1
    return R

def find_order(p: int, a: int, b: int, G: Point, max_iter: int = 100000) -> Optional[int]:
    """Compute the order n of point G on E(F_p) by brute-force addition up to max_iter."""
    if G is None:
        return 1
    R = G
    for k in range(1, max_iter + 1):
        if R is None:
            return k
        R = ec_point_add(R, G, a, p)
    return None

# ------------------------- BSGS for ECDLP ------------------------- #

def bsgs_ecdlp(p: int, a: int, b: int, G: Point, Q: Point, n: int) -> Optional[int]:
    """Solve for d in Q = d * G using Baby-Step Giant-Step in subgroup of order n.
    Returns d in [0, n-1] if found, else None.
    """
    if G is None:
        return None
    if Q is None:
        return 0  # 0 * G = O
    if n <= 0:
        return None

    m = int(math.isqrt(n)) + 1  # ceil(sqrt(n))

    # Baby steps: store j*G for j = 0..m-1
    baby: Dict[Point, int] = {}
    R: Point = None  # R = 0*G = O
    for j in range(m):
        # Store the first j for which j*G equals this point (keeps smallest j)
        if R not in baby:
            baby[R] = j
        # advance to (j+1)G
        R = ec_point_add(R, G, a, p)

    # Compute -mG
    mG = ec_scalar_mul(m, G, a, p)
    neg_mG = ec_point_neg(mG, p)

    # Giant steps: Gamma = Q + i*(-mG), i = 0..m
    Gamma: Point = Q
    for i in range(m + 1):
        if Gamma in baby:
            j = baby[Gamma]
            d = (i * m + j) % n
            return d
        Gamma = ec_point_add(Gamma, neg_mG, a, p)

    return None

def bsgs_ecdlp_with_stats(p: int, a: int, b: int, G: Point, Q: Point, n: int) -> Tuple[Optional[int], int, int, int]:
    """Same as bsgs_ecdlp, but also returns (d, m, baby_count, giant_count)."""
    if G is None:
        return None, 0, 0, 0
    if Q is None:
        return 0, 0, 0, 0
    if n <= 0:
        return None, 0, 0, 0

    m = int(math.isqrt(n)) + 1
    baby: Dict[Point, int] = {}
    R: Point = None
    for j in range(m):
        if R not in baby:
            baby[R] = j
        R = ec_point_add(R, G, a, p)

    baby_count = len(baby)

    mG = ec_scalar_mul(m, G, a, p)
    neg_mG = ec_point_neg(mG, p)

    Gamma: Point = Q
    giant_count = 0
    for i in range(m + 1):
        giant_count += 1
        if Gamma in baby:
            j = baby[Gamma]
            d = (i * m + j) % n
            return d, m, baby_count, giant_count
        Gamma = ec_point_add(Gamma, neg_mG, a, p)

    return None, m, baby_count, giant_count

def is_curve_valid(p: int, a: int, b: int) -> bool:
    return ((4 * (a % p) ** 3) + (27 * (b % p) ** 2)) % p != 0

def is_point_on_curve(P: Point, p: int, a: int, b: int) -> bool:
    if P is None:
        return True
    x, y = P
    if not (0 <= x < p and 0 <= y < p):
        return False
    return (y * y - (x * x * x + a * x + b)) % p == 0

def is_infinity(P: Point) -> bool:
    return P is None

def load_positional_input(file_path: Path) -> Tuple[int, int, int, Point, int, Point]:
    """Read 5-line integer input format:
    1) p
    2) a b
    3) Gx Gy
    4) n
    5) Qx Qy
    Returns (p, a, b, G, n, Q).
    """
    with file_path.open('r') as f:
        # keep non-empty lines only
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 5:
        raise ValueError("Input file must contain at least 5 non-empty lines: p | a b | Gx Gy | n | Qx Qy")

    # Parse integers from each line (space-separated)
    def ints(line: str):
        return list(map(int, line.split()))

    p_line = ints(lines[0])
    ab_line = ints(lines[1])
    G_line = ints(lines[2])
    n_line = ints(lines[3])
    Q_line = ints(lines[4])

    if len(p_line) != 1:
        raise ValueError("Line 1 must have exactly 1 integer: p")
    if len(ab_line) != 2:
        raise ValueError("Line 2 must have exactly 2 integers: a b")
    if len(G_line) != 2:
        raise ValueError("Line 3 must have exactly 2 integers: Gx Gy")
    if len(n_line) != 1:
        raise ValueError("Line 4 must have exactly 1 integer: n")
    if len(Q_line) != 2:
        raise ValueError("Line 5 must have exactly 2 integers: Qx Qy")

    p = p_line[0]
    a, b = ab_line
    G: Point = (G_line[0], G_line[1])
    n = n_line[0]
    Q: Point = (Q_line[0], Q_line[1])

    return p, a, b, G, n, Q


# ------------------------- File-driven runner ------------------------- #
if __name__ == "__main__":
    # Determine input path: argv[1] if provided, else default to ./input/filename.txt
    script_dir = Path(__file__).parent
    default_path = script_dir / 'input' / 'filename.txt'
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    p, a, b, G, n, Q = load_positional_input(input_path)

    # Pre-run sanity checks
    if not is_curve_valid(p, a, b):
        raise ValueError("Invalid curve: 4a^3 + 27b^2 ≡ 0 (mod p)")
    if G is None:
        raise ValueError("G is point at infinity")
    if Q is None:
        raise ValueError("Q is point at infinity")
    if not is_point_on_curve(G, p, a, b):
        raise ValueError("G is not on the curve")
    if not is_point_on_curve(Q, p, a, b):
        raise ValueError("Q is not on the curve")
    # Optional: sanity that n*G == O
    nG = ec_scalar_mul(n, G, a, p)
    if nG is not None:
        print("Warning: n*G != O; provided n may not be the exact order")

    method = "BSGS"
    start = time.perf_counter()
    d_found, m, baby_count, giant_count = bsgs_ecdlp_with_stats(p, a, b, G, Q, n)
    elapsed = time.perf_counter() - start

    # Output metadata and results
    print(f"Curve: p={p}, a={a}, b={b}")
    print(f"G=({G[0]},{G[1]}), Q=({Q[0]},{Q[1]}), n={n}")
    print(f"Method: {method}")
    print(f"m (ceil(sqrt(n))): {m}")
    print(f"Baby steps stored: {baby_count}")
    print(f"Giant steps tried: {giant_count}")
    print(f"Time elapsed: {elapsed:.6f} s")

    if d_found is not None:
        print(f"Found d: {d_found}")
        Q_check = ec_scalar_mul(d_found % n, G, a, p)
        match = (Q_check == Q)
        print(f"Verified: {match}")
        print(f"Match: {match}")
    else:
        print("No solution found (None)")

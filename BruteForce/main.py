# Brute-force ECDLP solver
# Point representation: (x, y) tuples; point at infinity = None

from typing import Optional, Tuple
from pathlib import Path
import sys
import time

Point = Optional[Tuple[int, int]]  # None for point at infinity

def mod_inv(x: int, p: int) -> int:
    """Modular inverse of x modulo p (p is prime)."""
    # Using Fermat's little theorem: x^(p-2) mod p
    return pow(x % p, p - 2, p)

def ec_point_add(P: Point, Q: Point, a: int, p: int) -> Point:
    """
    Add two points P and Q on the curve y^2 = x^3 + a x + b (mod p).
    Returns the resulting point (or None for point at infinity).
    """
    # Handle identity cases
    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    if x1 == x2 and (y1 + y2) % p == 0:
        # P + (-P) = O
        return None

    if P != Q:
        # slope = (y2 - y1) / (x2 - x1)
        num = (y2 - y1) % p
        den = (x2 - x1) % p
        if den == 0:
            return None  # should not happen because handled above, but safe
        lam = (num * mod_inv(den, p)) % p
    else:
        # Point doubling: slope = (3*x1^2 + a) / (2*y1)
        if y1 % p == 0:
            return None  # tangent is vertical -> point at infinity
        num = (3 * (x1 * x1 % p) + a) % p
        den = (2 * y1) % p
        lam = (num * mod_inv(den, p)) % p

    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)

def find_order(p: int, a: int, b: int, G: Point, max_iter: int = 10000) -> Optional[int]:
    """Compute the order n of point G on E(F_p) by brute-force addition.
    Stops at max_iter for safety (to avoid infinite loops).
    """
    R = G
    for k in range(1, max_iter + 1):
        if R is None:
            return k  # found n
        R = ec_point_add(R, G, a, p)
    return None  # did not loop back to infinity within max_iter


def brute_force_ecdlp(p: int, a: int, b: int, G: Point, Q: Point, n: int) -> Optional[int]:
    """
    Brute-force search for d in [1..n-1] such that d*G = Q on the curve over F_p.
    Returns d if found, otherwise None.
    """
    # Optional quick checks
    if Q is None:
        # If Q is the point at infinity, d must be 0 mod n; but we search for 1..n-1
        # return None here unless you want to allow d==0 as valid.
        return None

    R = G
    k = 1
    while k < n:
        if R == Q:
            return k
        R = ec_point_add(R, G, a, p)
        k += 1

    if R == Q:
        return k
    return None

def ec_scalar_mul(k: int, P: Point, a: int, p: int) -> Point:
    if P is None:
        return None
    R: Point = None
    Q: Point = P
    while k > 0:
        if k & 1:
            R = ec_point_add(R, Q, a, p)
        Q = ec_point_add(Q, Q, a, p)
        k >>= 1
    return R

def is_curve_valid(p: int, a: int, b: int) -> bool:
    return ((4 * (a % p) ** 3) + (27 * (b % p) ** 2)) % p != 0

def is_point_on_curve(P: Point, p: int, a: int, b: int) -> bool:
    if P is None:
        return True
    x, y = P
    if not (0 <= x < p and 0 <= y < p):
        return False
    return (y * y - (x * x * x + a * x + b)) % p == 0

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

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    default_path = script_dir / 'input' / 'filename.txt'
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    p, a, b, G, n, Q = load_positional_input(input_path)

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

    method = "Brute-force"
    start = time.perf_counter()
    d_found = brute_force_ecdlp(p, a, b, G, Q, n)
    elapsed = time.perf_counter() - start

    print(f"Curve: p={p}, a={a}, b={b}")
    print(f"G=({G[0]},{G[1]}), Q=({Q[0]},{Q[1]}), n={n}")
    print(f"Method: {method}")
    print(f"Time elapsed: {elapsed:.6f} s")

    if d_found is not None:
        print(f"Found d: {d_found}")
        Q_check = ec_scalar_mul(d_found % n, G, a, p)
        match = (Q_check == Q)
        print(f"Verified: {match}")
        print(f"Match: {match}")
    else:
        print("No solution found (None)")

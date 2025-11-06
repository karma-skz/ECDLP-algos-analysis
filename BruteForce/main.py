# Brute-force ECDLP solver
# Point representation: (x, y) tuples; point at infinity = None

from typing import Optional, Tuple

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
        # R <- R + G
        R = ec_point_add(R, G, a, p)
        k += 1

    # final check (k == n)
    if R == Q:
        return k
    return None

# Example usage (small toy curve)
if __name__ == "__main__":
    # Toy example: choose small prime p, curve y^2 = x^3 + ax + b (mod p),
    # base point G of small order n, and compute Q = d*G for testing.
    p = 9739
    a = 497
    b = 1768
    G = (1804, 5368)
    n =  (  # in real use you'd compute or be given n; here we assume a small known order for illustration
        0  # placeholder; if you know n put it here, otherwise brute_force_ecdlp will iterate up to n-1
    )

    # If you want a quick self-test, construct Q by repeated addition and set n:
    # (Below is an example that computes Q and uses the brute-force to recover d.)
    # WARNING: In real cryptographic curves n is huge; this toy example is only for demonstration.

    # build a small subgroup by computing small multiples of G
    # let's compute first 50 multiples to form a toy subgroup
    d_true = 13  # secret scalar we will look for
    R = None
    for _ in range(d_true):
        R = ec_point_add(R, G, a, p) if R is not None else G
    Q = R
    # set n to an upper bound (e.g., 100)
    n = find_order(p, a, b, G, max_iter=100)
    
    if n is None:
        n = 100  # fallback if order not found
    

    found = brute_force_ecdlp(p, a, b, G, Q, n)
    print("True d:", d_true)
    print("Found d:", found)

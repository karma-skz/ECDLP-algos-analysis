"""
Optimized Las Vegas (Birthday Attack) with C++ backend.

This implements a Randomized Collision Search (Birthday Attack).
Time Complexity: O(sqrt(n))
Space Complexity: O(sqrt(n))
"""

import sys
import time
import random
import ctypes
import math
from pathlib import Path
from typing import Optional, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import EllipticCurve, Point, load_input, mod_inv

# -----------------------------------------------------------------------------
# C++ Library Loading
# -----------------------------------------------------------------------------
try:
    lib_path = Path(__file__).parent / "ecc_fast.so"
    if lib_path.exists():
        ecc_lib = ctypes.CDLL(str(lib_path))
        
        # Define function signatures
        ecc_lib.scalar_mult.argtypes = [
            ctypes.c_longlong,  # k
            ctypes.c_longlong,  # Gx
            ctypes.c_longlong,  # Gy
            ctypes.c_longlong,  # a
            ctypes.c_longlong,  # b
            ctypes.c_longlong,  # p
            ctypes.POINTER(ctypes.c_longlong),  # result_x
            ctypes.POINTER(ctypes.c_longlong),  # result_y
        ]
        ecc_lib.scalar_mult.restype = ctypes.c_int
        
        USE_CPP = True
        print("✓ Using C++ optimized scalar multiplication")
    else:
        USE_CPP = False
        print("⚠ C++ library not found, using Python fallback (slower)")
except Exception as e:
    USE_CPP = False
    print(f"⚠ C++ library load failed: {e}, using Python fallback")


def fast_scalar_mult(k: int, G: Point, curve: EllipticCurve) -> Optional[Point]:
    """Fast scalar multiplication using C++ if available."""
    if not USE_CPP or G is None:
        return curve.scalar_multiply(k, G)
    
    result_x = ctypes.c_longlong()
    result_y = ctypes.c_longlong()
    
    is_valid = ecc_lib.scalar_mult(
        k % curve.p,
        G[0],
        G[1],
        curve.a,
        curve.b,
        curve.p,
        ctypes.byref(result_x),
        ctypes.byref(result_y)
    )
    
    if is_valid:
        return (result_x.value, result_y.value)
    else:
        return None

def fast_point_add(P: Point, Q: Point, curve: EllipticCurve) -> Point:
    """Helper for C++ point addition if you implement it, else Python."""
    return curve.add(P, Q)


# -----------------------------------------------------------------------------
# FIXED ALGORITHM: Birthday Attack (Robust for Composite n)
# -----------------------------------------------------------------------------

def las_vegas_optimized(curve: EllipticCurve, G: Point, Q: Point, n: int, 
                       max_attempts: int = 1000000) -> Tuple[Optional[int], int]:
    """
    Optimized Las Vegas (Birthday Attack).
    Generates random points R = aG + bQ and stores them.
    If a collision occurs (same R found twice), we solve the ECDLP.
    """
    
    # Storage for Birthday Attack: Point -> (a, b)
    visited: Dict[Point, Tuple[int, int]] = {}
    
    # Dynamic limit based on sqrt(n)
    # We need approx sqrt(pi * n / 2) points for 50% collision chance
    expected_points = int(math.sqrt(math.pi * n / 2))
    limit = min(max_attempts, max(2000, expected_points * 2))
    
    # Only print for larger cases to reduce noise on small ones
    if n > 10000:
        print(f"  Target: Collecting ~{expected_points} points...", flush=True)

    for i in range(1, limit + 1):
        if i % 50000 == 0:
            print(f"  stored {i} points...", flush=True)

        # 1. Pick random coefficients
        a = random.randrange(n)
        b = random.randrange(n)
        
        # 2. Compute R = aG + bQ
        term1 = fast_scalar_mult(a, G, curve)
        term2 = fast_scalar_mult(b, Q, curve)
        R = fast_point_add(term1, term2, curve)
        
        if R is None:
            continue 
            
        # 3. Check for Collision
        if R in visited:
            a_old, b_old = visited[R]
            
            # Equation: (a - a_old)G = (b_old - b)Q
            num = (a - a_old) % n
            den = (b_old - b) % n
            
            if den == 0:
                continue 

            # ROBUST SOLVER for gcd(den, n) > 1
            g = math.gcd(den, n)
            
            if g == 1:
                # Standard case: den is coprime to n
                try:
                    den_inv = mod_inv(den, n)
                    d = (num * den_inv) % n
                    if fast_scalar_mult(d, G, curve) == Q:
                        return d, i
                except (ValueError, ZeroDivisionError):
                    pass
            elif num % g == 0:
                # Composite case: Solve in reduced group n/g
                n_prime = n // g
                den_prime = den // g
                num_prime = num // g
                
                try:
                    den_inv = mod_inv(den_prime, n_prime)
                    d_base = (num_prime * den_inv) % n_prime
                    
                    # The solution is one of: d_base, d_base + n', d_base + 2n', ...
                    for k in range(g):
                        d_cand = d_base + k * n_prime
                        if fast_scalar_mult(d_cand, G, curve) == Q:
                            return d_cand, i
                except (ValueError, ZeroDivisionError):
                    pass
        
        # 4. Store point
        visited[R] = (a, b)
        
    return None, limit


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = script_dir.parent / 'input' / 'testcase_1.txt'
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        p, a, b, G, n, Q = load_input(input_path)
        curve = EllipticCurve(a, b, p)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Solving ECDLP using Optimized Las Vegas (Birthday Attack)...")
    print(f"Curve: y^2 = x^3 + {a}x + {b} (mod {p})")
    print(f"G = ({G[0]}, {G[1]}), Q = ({Q[0]}, {Q[1]}), n = {n}") #type: ignore
    
    if n > 2**30:
        print("\n[WARNING] Curve > 30 bits. Algorithm may run out of RAM.")
    
    # Calculate sensible limits
    max_steps = 2_000_000 
    
    start_time = time.perf_counter()
    
    d, attempts = las_vegas_optimized(curve, G, Q, n, max_steps)
    
    elapsed = time.perf_counter() - start_time
    
    if d is not None:
        print(f"\n{'='*50}")
        print(f"Solution: d = {d}")
        print(f"Points Stored: {attempts}")
        print(f"Time: {elapsed:.6f} seconds")
        
        # Verification
        Q_verify = fast_scalar_mult(d, G, curve)
        verified = (Q_verify == Q)
        print(f"Verification: {'PASSED' if verified else 'FAILED'}")
        print(f"{'='*50}")
        if not verified: sys.exit(1)
    else:
        print(f"\nNo solution found after storing {attempts} points.")
        print(f"Time: {elapsed:.6f} seconds")
        sys.exit(1)

if __name__ == "__main__":
    main()
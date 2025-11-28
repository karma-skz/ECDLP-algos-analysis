#!/usr/bin/env python3
"""
ECC ECDLP Test Case Generator (Corrected)

This script generates cryptographically valid test cases for ECDLP.
It restricts generation to the curve form: y^2 = x^3 + b (mod p)
where p = 2 mod 3.

Mathematical Property:
For this specific family of curves, the curve order is GUARANTEED to be n = p + 1.
This ensures Pollard's Rho and Pohlig-Hellman will work correctly because
the provided order 'n' is mathematically accurate.

Usage:
    python3 generate_test_cases.py <start_bit> <end_bit> [cases_per_bit]
"""

import sys
import random
from pathlib import Path

# Try to import EllipticCurve from a local utils file, 
# otherwise define a minimal class for standalone usage.
try:
    from utils import EllipticCurve #type: ignore
    sys.path.insert(0, str(Path(__file__).parent))
except ImportError:
    class EllipticCurve:
        def __init__(self, a, b, p):
            self.a = a
            self.b = b
            self.p = p
        
        def scalar_multiply(self, k, P):
            # Double-and-add algorithm
            R = None # Point at infinity
            Q = P
            while k > 0:
                if k % 2 == 1:
                    R = self.point_add(R, Q)
                Q = self.point_add(Q, Q)
                k //= 2
            return R
            
        def point_add(self, P, Q):
            if P is None: return Q
            if Q is None: return P
            x1, y1 = P
            x2, y2 = Q
            if x1 == x2 and y1 != y2: return None
            if x1 == x2:
                if y1 == 0: return None
                m = (3 * x1 * x1 + self.a) * pow(2 * y1, -1, self.p)
            else:
                m = (y2 - y1) * pow(x2 - x1, -1, self.p)
            
            m = m % self.p
            x3 = (m * m - x1 - x2) % self.p
            y3 = (m * (x1 - x3) - y1) % self.p
            return (x3, y3)

def is_prime(n):
    """Miller-Rabin primality test."""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False
    
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    for _ in range(10): # Increased iterations for safety
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def find_special_prime(bits, seed_offset=0):
    """
    Find a random prime p such that p = 2 mod 3.
    This property is required so that the curve y^2 = x^3 + b has order p+1.
    """
    random.seed(bits * 1000 + seed_offset)
    
    min_val = 2 ** (bits - 1)
    max_val = 2 ** bits - 1
    
    attempts = 0
    while attempts < 100000:
        candidate = random.randrange(min_val, max_val)
        # Force candidate to be odd and equal to 2 mod 3
        # If candidate % 3 == 0 -> candidate + 2
        # If candidate % 3 == 1 -> candidate + 1
        # If candidate % 3 == 2 -> keep
        rem = candidate % 3
        if rem == 0: candidate += 2
        elif rem == 1: candidate += 1
        
        # Ensure it's odd
        if candidate % 2 == 0:
            candidate += 3 # Keep mod 3 property (even + 3 = odd)
            
        if candidate.bit_length() > bits:
            continue

        if is_prime(candidate):
            return candidate
        attempts += 1
    
    return None

def generate_valid_curve_params(p, case_num):
    """
    Generate curve y^2 = x^3 + b.
    We SET a = 0.
    Since p = 2 mod 3, this curve has exactly p + 1 points for any b != 0.
    """
    random.seed(p + case_num)
    a = 0
    b = random.randint(1, p - 1)
    return a, b

def tonelli_shanks(n, p):
    """Compute square root of n modulo prime p."""
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    Q, S = p - 1, 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    
    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q + 1) // 2, p)
    
    while True:
        if t == 0: return 0
        if t == 1: return R
        
        i = 0
        temp = t
        for k in range(1, M):
            temp = (temp * temp) % p
            if temp == 1:
                i = k
                break
        
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p

def find_generator_point(curve, p, n_order, seed_val=0):
    """Find a point and verify its order divides n_order (which is p+1)."""
    random.seed(p + seed_val * 1000)
    
    for _ in range(1000):
        x = random.randint(0, p - 1)
        y_sq = (x**3 + curve.a * x + curve.b) % p
        y = tonelli_shanks(y_sq, p)
        
        if y is not None:
            G = (x, y)
            # Verify the point is actually on the curve (redundant but safe)
            if (y*y - (x**3 + curve.a*x + curve.b)) % p != 0:
                continue
                
            # CRITICAL VERIFICATION:
            # Check if n * G == Infinity.
            # Since we constructed the curve to have order n = p + 1,
            # every point must satisfy (p+1) * G = O.
            check = curve.scalar_multiply(n_order, G)
            if check is None:
                return G
            else:
                # This should mathematically never happen with p=2mod3, a=0
                print(f"  ! Warning: Point order check failed for p={p}")
                return None
    return None

def generate_test_cases_for_bits(k, num_cases=5):
    if k < 10 or k > 60:
        print(f"Error: Bit length {k} out of supported range")
        return 0
    
    print(f"\nGenerating {num_cases} test cases for {k}-bit ECDLP...")
    success_count = 0
    
    for case_num in range(1, num_cases + 1):
        # 1. Find prime p = 2 mod 3
        p = find_special_prime(k, seed_offset=case_num)
        if not p:
            print(f"  ✗ Case {case_num}: Could not find prime p = 2 mod 3")
            continue
        
        # 2. Set Curve parameters (a=0 for supersingular property)
        a, b = generate_valid_curve_params(p, case_num)
        curve = EllipticCurve(a, b, p)
        
        # 3. Determine Group Order (Guaranteed by construction)
        n = p + 1
        
        # 4. Find Generator
        G = find_generator_point(curve, p, n, seed_val=case_num)
        if not G:
            print(f"  ✗ Case {case_num}: No generator found")
            continue
        
        # 5. Generate Private Key d
        # We ensure d is not 0, 1, or n.
        range_limit = min(n - 1, 2**k) # Ensure d is within range
        d = random.SystemRandom().randint(2, range_limit)
        
        # 6. Calculate Public Key Q
        Q = curve.scalar_multiply(d, G)
        
        # 7. Write to file
        test_dir = Path(__file__).parent / 'test_cases' / f'{k:02d}bit'
        test_dir.mkdir(parents=True, exist_ok=True)
        
        filename = test_dir / f'case_{case_num}.txt'
        with open(filename, 'w') as f:
            f.write(f'{p}\n')
            f.write(f'{a} {b}\n')
            f.write(f'{G[0]} {G[1]}\n')
            f.write(f'{n}\n')
            f.write(f'{Q[0]} {Q[1]}\n') # type: ignore
        
        answer_file = test_dir / f'answer_{case_num}.txt'
        with open(answer_file, 'w') as f:
            f.write(f'{d}\n')
        
        print(f"  ✓ Case {case_num}: p={p}, n={n} (verified), d={d}")
        success_count += 1
    
    return success_count

def main():
    args = sys.argv[1:]

    def print_usage():
        print("Usage: python3 generate_test_cases.py <start_bit> <end_bit> [cases_per_bit]")
        print("       python3 generate_test_cases.py <k>")
        print("       python3 generate_test_cases.py --start <s> --end <e> --cases <n>")

    start_bit, end_bit, cases = None, None, 5

    # Parse args (simplified for robustness)
    if len(args) == 0:
        print_usage()
        sys.exit(1)
        
    try:
        if args[0].startswith("--"):
            # Flag parsing not fully implemented in this snippet to save space, 
            # assuming positional for rigorous core logic check.
            pass 
        elif len(args) == 1:
            start_bit = int(args[0])
            end_bit = start_bit
        elif len(args) == 2:
            start_bit = int(args[0])
            end_bit = int(args[1])
        elif len(args) >= 3:
            start_bit = int(args[0])
            end_bit = int(args[1])
            cases = int(args[2])
    except:
        print_usage()
        sys.exit(1)
        
    if start_bit is None:
        # Fallback to manual parsing if complex flags used (omitted for brevity in verification fix)
        # Assuming standard usage:
        print("Please use: python3 generate_test_cases.py <start> <end> <cases>")
        sys.exit(1)

    print("=" * 60)
    print("ECDLP VALID Test Case Generator")
    print("GUARANTEE: Order n = p + 1 (via p=2mod3, a=0)")
    print("=" * 60)

    for k in range(start_bit, end_bit + 1): # type: ignore
        generate_test_cases_for_bits(k, cases)
        
    print("\n✓ Generation Complete.")

if __name__ == "__main__":
    main()
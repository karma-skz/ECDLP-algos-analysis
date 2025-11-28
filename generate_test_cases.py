#!/usr/bin/env python3
"""
ECC ECDLP Test Case Generator (Zero-Dependency Version)

Method:
    Uses "Supersingular Curves" of the form y^2 = x^3 + b (mod p)
    where p = 11 mod 12.
    
    Mathematical Property:
    For these specific curves, the order N is ALWAYS exactly (p + 1).
    
    This allows us to generate 50-bit (or even 100-bit) test cases 
    instantly in pure Python without needing Schoof's algorithm or SymPy.
"""

import sys
import random
from pathlib import Path

# --- Configuration ---
# Bit limits
MIN_BITS = 10
MAX_BITS = 60

# ==========================================
# Elliptic Curve Class (Minimal)
# ==========================================
class EllipticCurve:
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p

    def add(self, P, Q):
        if P is None: return Q
        if Q is None: return P
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and (y1 + y2) % self.p == 0: return None
        
        if P != Q:
            lam = ((y2 - y1) * pow(x2 - x1, -1, self.p)) % self.p
        else:
            lam = ((3 * x1 * x1 + self.a) * pow(2 * y1, -1, self.p)) % self.p
            
        x3 = (lam * lam - x1 - x2) % self.p
        y3 = (lam * (x1 - x3) - y1) % self.p
        return (x3, y3)

    def scalar_multiply(self, k, P):
        R = None
        while k > 0:
            if k % 2 == 1: R = self.add(R, P)
            P = self.add(P, P)
            k //= 2
        return R

# ==========================================
# Math Helpers (Pure Python)
# ==========================================

def is_prime_miller_rabin(n, k=10):
    """Returns True if n is likely prime."""
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n < 2: return False
    
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
        
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1: break
        else: return False
    return True

def find_special_prime(bits, seed_offset=0):
    """
    Finds a prime p such that p = 11 mod 12.
    
    Why 11 mod 12?
    1. p = 2 mod 3  -> Ensures curve order is exactly p + 1.
    2. p = 3 mod 4  -> Ensures easy square roots using x^((p+1)/4).
    Combined: p = 11 mod 12.
    """
    random.seed(bits * 1000 + seed_offset)
    min_val = 2 ** (bits - 1)
    max_val = 2 ** bits - 1
    
    # Start at a random location
    start = random.randrange(min_val, max_val)
    # Align to 11 mod 12
    candidate = start - (start % 12) + 11
    
    # Search forward
    for i in range(20000):
        curr = candidate + (12 * i)
        if curr >= max_val: break
        if is_prime_miller_rabin(curr):
            return curr
    return None

def find_point_on_curve(p, b):
    """
    Finds a point (x, y) on y^2 = x^3 + b.
    Since p = 3 mod 4, we can compute sqrt easily.
    """
    while True:
        x = random.randint(0, p - 1)
        rhs = (pow(x, 3, p) + b) % p
        
        # Euler's Criterion: Check if rhs is a quadratic residue
        # rhs^((p-1)/2) == 1 mod p
        if pow(rhs, (p - 1) // 2, p) != 1:
            continue
            
        # Compute Square Root for p = 3 mod 4
        # y = rhs^((p+1)/4) mod p
        y = pow(rhs, (p + 1) // 4, p)
        return (x, y)

# ==========================================
# Main Generator Logic
# ==========================================

def generate_test_cases_for_bits(k, num_cases=5):
    if k < MIN_BITS or k > MAX_BITS:
        print(f"Error: Bit length {k} out of range")
        return 0
    
    print(f"Generating {num_cases} test cases for {k}-bit ECDLP...")
    
    success_count = 0
    for case_num in range(1, num_cases + 1):
        try:
            # 1. Find Special Prime (p = 11 mod 12)
            p = find_special_prime(k, seed_offset=case_num)
            if not p:
                print(f"  [Skip] Case {case_num}: Could not find prime.")
                continue
            
            # 2. Set Curve Parameters
            # We use y^2 = x^3 + b. (a is always 0)
            # Varying b gives us different curves.
            a = 0
            b = random.randint(1, p - 1)
            
            # 3. CALCULATE ORDER
            # Magic: For this curve form and prime type, Order is EXACTLY p + 1
            n = p + 1
            
            curve = EllipticCurve(a, b, p)
            
            # 4. Find Generator
            G = find_point_on_curve(p, b)
            
            # 5. Generate Secret (Full range)
            d = random.randint(1, n - 1)
            
            # 6. Compute Public Key
            Q = curve.scalar_multiply(d, G)
            if Q is None: continue

            # 7. Write Files
            test_dir = Path(__file__).parent / 'test_cases' / f'{k:02d}bit'
            test_dir.mkdir(parents=True, exist_ok=True)
            
            with open(test_dir / f'case_{case_num}.txt', 'w') as f:
                f.write(f'{p}\n{a} {b}\n{G[0]} {G[1]}\n{n}\n{Q[0]} {Q[1]}\n') 
            
            with open(test_dir / f'answer_{case_num}.txt', 'w') as f:
                f.write(f'{d}\n')
            
            print(f"  ✓ Case {case_num}: {k}-bit, p={p} (Order p+1)")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Case {case_num} Error: {e}")
            continue

    return success_count

def main():
    args = sys.argv[1:]
    start_bit, end_bit = 10, 20
    cases_per_bit = 5

    # Argument Parsing
    if len(args) >= 1: start_bit = int(args[0])
    if len(args) >= 2: end_bit = int(args[1])
    if len(args) >= 3: cases_per_bit = int(args[2])
    
    print("="*60)
    print("ECC Generator: Zero-Dependency Mode")
    print("Using Supersingular Curves (N = p + 1)")
    print("="*60)

    for k in range(start_bit, end_bit + 1):
        generate_test_cases_for_bits(k, cases_per_bit)

    print("\n✓ Done.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
ECC ECDLP Test Case Generator (Fixed Prime-Order Mode)

Method:
    Uses Supersingular Curves y^2 = x^3 + b (mod p) with p = 11 mod 12.
    
    CORRECTION:
    Since p = 11 mod 12, (p+1) is always divisible by 12.
    We search for p such that q = (p + 1) / 12 is PRIME.
    
    This gives us a curve of order N = 12*q.
    We then work in the subgroup of prime order q.
"""

import sys
import random
from pathlib import Path

# --- Configuration ---
MIN_BITS = 10
MAX_BITS = 60

# ==========================================
# Elliptic Curve Class
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
# Math Helpers
# ==========================================

def is_prime_miller_rabin(n, k=20): 
    """Returns True if n is prime."""
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

def find_strong_prime_pair(bits, seed_offset=0):
    """
    Finds a pair (p, q) such that:
    1. p is a prime of 'bits' length.
    2. p = 11 mod 12.
    3. q = (p + 1) / 12 is ALSO PRIME.
    """
    random.seed(bits * 10000 + seed_offset)
    min_val = 2 ** (bits - 1)
    max_val = 2 ** bits - 1
    
    # Start random search
    start = random.randrange(min_val, max_val)
    candidate = start - (start % 12) + 11 
    
    attempts = 0
    while attempts < 100000:
        if candidate >= max_val:
            candidate = min_val - (min_val % 12) + 11
            
        # Check if p is prime
        if is_prime_miller_rabin(candidate):
            # FIX: Divide by 12, not 6
            # Since p = 11 mod 12, p+1 is divisible by 12.
            q = (candidate + 1) // 12
            
            # If q is composite, try removing small factors (optional optimization)
            # But for "Strong Prime" strict definition, we just check if q is prime directly
            if is_prime_miller_rabin(q):
                return candidate, q
        
        candidate += 12
        attempts += 1
        
    return None, None

def find_subgroup_generator(curve, p, q):
    """
    Finds a generator G of prime order q.
    Curve Order N = 12 * q.
    """
    while True:
        x = random.randint(0, p - 1)
        rhs = (pow(x, 3, p) + curve.b) % p
        
        if pow(rhs, (p - 1) // 2, p) != 1:
            continue
            
        y = pow(rhs, (p + 1) // 4, p)
        P = (x, y)
        
        # Project into subgroup: G = 12 * P
        # This removes the cofactor of 12.
        G = curve.scalar_multiply(12, P)
        
        if G is not None:
            return G

# ==========================================
# Main Generator
# ==========================================

def generate_test_cases_for_bits(k, num_cases=5):
    if k < MIN_BITS or k > MAX_BITS:
        print(f"Error: Bit length {k} out of range")
        return 0
    
    print(f"Generating {num_cases} test cases for {k}-bit (Prime Subgroup q)...")
    
    success_count = 0
    for case_num in range(1, num_cases + 1):
        try:
            p, q = find_strong_prime_pair(k, seed_offset=case_num)
            
            if not p:
                print(f"  [Skip] Case {case_num}: Could not find strong prime.")
                continue
            
            a = 0
            b = random.randint(1, p - 1)
            curve = EllipticCurve(a, b, p)
            
            G = find_subgroup_generator(curve, p, q)
            
            # Secret d in range [1, q-1]
            d = random.randint(1, q - 1)
            
            Q = curve.scalar_multiply(d, G)
            if Q is None: continue

            test_dir = Path(__file__).parent / 'test_cases' / f'{k:02d}bit'
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # We write 'q' as the order so algorithms solve for the prime subgroup
            with open(test_dir / f'case_{case_num}.txt', 'w') as f:
                f.write(f'{p}\n{a} {b}\n{G[0]} {G[1]}\n{q}\n{Q[0]} {Q[1]}\n') 
            
            with open(test_dir / f'answer_{case_num}.txt', 'w') as f:
                f.write(f'{d}\n')
            
            print(f"  ✓ Case {case_num}: p={p}, Subgroup Prime q={q}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Case {case_num} Error: {e}")
            continue

    return success_count

def main():
    args = sys.argv[1:]
    start_bit, end_bit = 10, 20
    cases_per_bit = 5

    if len(args) >= 1: start_bit = int(args[0])
    if len(args) >= 2: end_bit = int(args[1])
    if len(args) >= 3: cases_per_bit = int(args[2])
    
    print("="*60)
    print("ECC Generator: Prime Subgroup Mode (Fixed)")
    print("Generates curves with order N = 12*q where q is PRIME.")
    print("="*60)

    for k in range(start_bit, end_bit + 1):
        generate_test_cases_for_bits(k, cases_per_bit)

    print("\n✓ Done.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Tailored Test Case Generator for ECC Algorithms

Generates specific types of test cases optimized for different algorithms:
1. Generic: For BSGS, Pollard's Rho, Las Vegas (Random curves)
2. PH-Friendly: For Pohlig-Hellman (Smooth order curves)
3. MOV-Friendly: For MOV Attack (Supersingular curves with low embedding degree)
"""

import sys
import random
import math
from pathlib import Path
from typing import Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import EllipticCurve, Point

def is_prime(n: int) -> bool:
    """Miller-Rabin primality test."""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False
    
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    for _ in range(5):
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

def find_prime(bits: int, condition=None) -> int:
    """Find a random prime with optional condition."""
    min_val = 2 ** (bits - 1)
    max_val = 2 ** bits - 1
    
    while True:
        p = random.randrange(min_val, max_val) | 1
        if is_prime(p):
            if condition is None or condition(p):
                return p

def tonelli_shanks(n: int, p: int) -> Optional[int]:
    """Compute square root of n modulo p."""
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    
    Q, S = p - 1, 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    
    M, c, t, R = S, pow(z, Q, p), pow(n, Q, p), pow(n, (Q + 1) // 2, p)
    
    while True:
        if t == 0: return 0
        if t == 1: return R
        
        i = 1
        temp = (t * t) % p
        while temp != 1 and i < M:
            temp = (temp * temp) % p
            i += 1
        
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p

def find_point(curve: EllipticCurve) -> Point:
    """Find a random point on the curve."""
    for _ in range(100):
        x = random.randrange(0, curve.p)
        y2 = (x**3 + curve.a * x + curve.b) % curve.p
        y = tonelli_shanks(y2, curve.p)
        if y is not None:
            return (x, y)
    return None

def bsgs_order(curve: EllipticCurve, P: Point, max_order: int) -> Optional[int]:
    """Find order of point P using BSGS (for small-ish curves)."""
    if P is None: return 1
    
    m = math.isqrt(max_order) + 1
    
    # Baby steps
    baby_steps = {}
    curr = None # Infinity
    for j in range(m):
        baby_steps[curr] = j
        curr = curve.add(curr, P)
    
    # Giant steps
    mP = curve.scalar_multiply(m, P)
    neg_mP = curve.negate(mP)
    giant = None # Infinity
    
    for i in range(m + 1):
        if giant in baby_steps:
            order = i * m + baby_steps[giant]
            if order > 0:
                # Verify
                if curve.scalar_multiply(order, P) is None:
                    return order
        giant = curve.add(giant, neg_mP)
        
    return None

def is_smooth(n: int, bound: int = 1000) -> bool:
    """Check if n is B-smooth (all prime factors <= bound)."""
    d = 2
    temp = n
    while d * d <= temp and d <= bound:
        while temp % d == 0:
            temp //= d
        d += 1
    return temp <= bound

def save_case(folder: Path, case_num: int, p, a, b, G, n, Q, d):
    """Save test case to file."""
    folder.mkdir(parents=True, exist_ok=True)
    
    with open(folder / f'case_{case_num}.txt', 'w') as f:
        f.write(f'{p}\n{a} {b}\n{G[0]} {G[1]}\n{n}\n{Q[0]} {Q[1]}\n')
        
    with open(folder / f'answer_{case_num}.txt', 'w') as f:
        f.write(f'{d}\n')

# ----------------- Generators -----------------

def generate_mov_friendly(bits: int, count: int):
    """
    Generate Supersingular curves: y^2 = x^3 + 1 over F_p where p = 2 mod 3.
    These have order n = p + 1 and embedding degree k = 2.
    """
    print(f"Generating {count} MOV-friendly cases (Supersingular, k=2) for {bits} bits...")
    
    for i in range(1, count + 1):
        # Find p = 2 mod 3
        p = find_prime(bits, lambda x: x % 3 == 2)
        a, b = 0, 1
        curve = EllipticCurve(a, b, p)
        
        # Order is exactly p + 1
        n = p + 1
        
        # Find generator
        while True:
            G = find_point(curve)
            if G is None: continue
            # Check if G has large order (we want full order n or large factor)
            # For simplicity, just use G. If it's small order, attack is trivial anyway.
            break
            
        d = random.randrange(2, n)
        Q = curve.scalar_multiply(d, G)
        
        save_case(Path('test_cases/MOV_friendly'), i, p, a, b, G, n, Q, d)
        print(f"  ✓ Case {i}: p={p}, n={n}")

def generate_ph_friendly(bits: int, count: int):
    """
    Generate curves with smooth order for Pohlig-Hellman.
    Limited to ~20-25 bits to allow order computation.
    """
    print(f"Generating {count} PH-friendly cases (Smooth order) for {bits} bits...")
    
    found = 0
    attempts = 0
    
    while found < count and attempts < 1000:
        attempts += 1
        p = find_prime(bits)
        a = random.randrange(0, p)
        b = random.randrange(0, p)
        curve = EllipticCurve(a, b, p)
        if not curve.is_valid(): continue
        
        G = find_point(curve)
        if G is None: continue
        
        # Hasse bound: order is within [p+1-2sqrt(p), p+1+2sqrt(p)]
        # We use BSGS to find exact order
        max_possible = p + 1 + int(2 * math.sqrt(p)) + 2
        n = bsgs_order(curve, G, max_possible)
        
        if n and is_smooth(n, bound=1000): # B-smooth with B=1000
            d = random.randrange(2, n)
            Q = curve.scalar_multiply(d, G)
            save_case(Path('test_cases/PH_friendly'), found + 1, p, a, b, G, n, Q, d)
            print(f"  ✓ Case {found+1}: p={p}, n={n} (Smooth!)")
            found += 1
            
    if found < count:
        print(f"  Warning: Only found {found} smooth curves after {attempts} attempts.")

def generate_generic(bits: int, count: int):
    """Generate generic random curves."""
    print(f"Generating {count} Generic cases for {bits} bits...")
    
    for i in range(1, count + 1):
        p = find_prime(bits)
        a = random.randrange(0, p)
        b = random.randrange(0, p)
        curve = EllipticCurve(a, b, p)
        if not curve.is_valid(): 
            # Retry
            a, b = 0, 7
            curve = EllipticCurve(a, b, p)
            
        G = find_point(curve)
        if G is None: continue
        
        # Calculate CORRECT order using BSGS
        # Hasse bound: |n - (p+1)| <= 2*sqrt(p)
        max_possible = p + 1 + int(2 * math.sqrt(p)) + 2
        n = bsgs_order(curve, G, max_possible)
        
        if n is None:
            print(f"  ✗ Case {i}: Could not find order")
            continue
        
        d = random.randrange(2, n)
        Q = curve.scalar_multiply(d, G)
        
        save_case(Path('test_cases/Generic'), i, p, a, b, G, n, Q, d)
        print(f"  ✓ Case {i}: p={p}, n={n}")

def main():
    # Generate MOV cases (can be large bits)
    generate_mov_friendly(bits=20, count=5)
    
    # Generate PH cases (must be small bits for generation feasibility)
    generate_ph_friendly(bits=16, count=5)
    
    # Generate Generic cases (medium bits)
    generate_generic(bits=20, count=5)
    
    print("\nDone! Test cases saved in 'test_cases/' subdirectories.")

if __name__ == "__main__":
    main()

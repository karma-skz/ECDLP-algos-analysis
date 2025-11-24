#!/usr/bin/env python3

"""
Verify ECDLP test cases by checking if d * G = Q.
Scans 'input/' and 'test_cases/' directories for test case files and corresponding answer files.
"""

import sys
import os
from pathlib import Path

# ----------------- Finite field utilities -----------------

def inv_mod(x, p):
    """Modular inverse of x modulo p (p must be prime, x != 0)."""
    # Using extended Euclidean algorithm to avoid relying on pow(..., -1, p)
    if x % p == 0:
        raise ZeroDivisionError("No inverse for 0 modulo p")
    a, b = x % p, p
    u0, u1 = 1, 0
    while b:
        q = a // b
        a, b = b, a - q * b
        u0, u1 = u1, u0 - q * u1
    return u0 % p

# ----------------- Elliptic curve operations -----------------

def is_on_curve(P, a, b, p):
    """Check if point P = (x, y) lies on y^2 = x^3 + a x + b over F_p."""
    if P is None:
        return True  # Point at infinity is on the curve by definition
    x, y = P
    return (y * y - (x * x * x + a * x + b)) % p == 0

def point_add(P, Q, a, p):
    """Add two points P and Q on the curve over F_p."""
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
        # λ = (y2 - y1) / (x2 - x1)
        num = (y2 - y1) % p
        den = (x2 - x1) % p
        lam = (num * inv_mod(den, p)) % p
    else:
        # Point doubling:
        # λ = (3 x1^2 + a) / (2 y1)
        if y1 % p == 0:
            return None
        num = (3 * x1 * x1 + a) % p
        den = (2 * y1) % p
        lam = (num * inv_mod(den, p)) % p

    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)

def scalar_mul(k, P, a, p):
    """Compute k * P using double-and-add."""
    if k == 0 or P is None:
        return None
    if k < 0:
        # k * P = -k * (-P)
        x, y = P
        return scalar_mul(-k, (x, (-y) % p), a, p)

    result = None
    addend = P

    while k > 0:
        if k & 1:
            result = point_add(result, addend, a, p)
        addend = point_add(addend, addend, a, p)
        k >>= 1

    return result

# ----------------- Main verification logic -----------------

def parse_case_file(filepath):
    """Parse a test case file containing curve parameters and points."""
    try:
        with open(filepath, 'r') as f:
            content = f.read().split()
        
        # Expected format:
        # p
        # a b
        # Gx Gy
        # n
        # Qx Qy
        
        if len(content) < 8:
            return None
            
        p = int(content[0])
        a = int(content[1])
        b = int(content[2])
        Gx = int(content[3])
        Gy = int(content[4])
        n = int(content[5])
        Qx = int(content[6])
        Qy = int(content[7])
        
        return (p, a, b, Gx, Gy, n, Qx, Qy)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def parse_answer_file(filepath):
    """Parse an answer file containing the private key d."""
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
        if not content:
            return None
        return int(content)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def verify_case(case_path, answer_path):
    """Verify a single test case."""
    params = parse_case_file(case_path)
    if not params:
        return False
    
    p, a, b, Gx, Gy, n, Qx, Qy = params
    
    d = parse_answer_file(answer_path)
    if d is None:
        print(f"FAIL: {case_path} - Could not read answer d")
        return False
        
    G = (Gx, Gy)
    Q = (Qx, Qy)
    
    # Basic validity checks
    if not is_on_curve(G, a, b, p):
        print(f"FAIL: {case_path} - Generator G is not on the curve")
        return False
        
    if not is_on_curve(Q, a, b, p):
        print(f"FAIL: {case_path} - Target Q is not on the curve")
        return False
        
    # Verify d * G == Q
    try:
        Q_computed = scalar_mul(d, G, a, p)
    except Exception as e:
        print(f"FAIL: {case_path} - Error during scalar multiplication: {e}")
        return False
    
    if Q_computed == Q:
        print(f"PASS: {case_path}")
        return True
    else:
        print(f"FAIL: {case_path} - Verification failed (d*G != Q)")
        print(f"  d = {d}")
        print(f"  G = {G}")
        print(f"  Q (expected) = {Q}")
        print(f"  Q (computed) = {Q_computed}")
        return False

def main():
    base_dir = Path(__file__).parent.absolute()
    
    # Directories to search for test cases
    dirs_to_check = [
        base_dir / 'input',
        base_dir / 'test_cases'
    ]
    
    total_checked = 0
    total_passed = 0
    
    print(f"Starting verification in: {base_dir}")
    
    for d in dirs_to_check:
        if not d.exists():
            print(f"Directory not found: {d}")
            continue
            
        print(f"\nScanning {d}...")
        
        for root, _, files in os.walk(d):
            # Group files by case number
            case_files = {}
            answer_files = {}
            
            for file in files:
                if file.endswith('.txt'):
                    path = Path(root) / file
                    if file.startswith('testcase_'):
                        # input/testcase_N.txt
                        num = file[9:-4]
                        case_files[num] = path
                    elif file.startswith('case_'):
                        # test_cases/XXbit/case_N.txt
                        num = file[5:-4]
                        case_files[num] = path
                    elif file.startswith('answer_'):
                        # answer_N.txt
                        num = file[7:-4]
                        answer_files[num] = path
            
            # Sort by number for consistent output
            sorted_nums = sorted(case_files.keys(), key=lambda x: int(x) if x.isdigit() else x)
            
            for num in sorted_nums:
                case_path = case_files[num]
                if num in answer_files:
                    answer_path = answer_files[num]
                    total_checked += 1
                    if verify_case(case_path, answer_path):
                        total_passed += 1
                else:
                    print(f"WARNING: Missing answer file for {case_path}")

    print(f"\n{'='*40}")
    print(f"Summary: {total_passed}/{total_checked} tests passed.")
    print(f"{'='*40}")

    if total_checked > 0 and total_passed == total_checked:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

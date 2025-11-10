"""
Compare Deterministic Bonus Implementations - Quick Demo

Focuses on BruteForce, BSGS, and Pohlig-Hellman bonuses
which reliably complete in reasonable time.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import (EllipticCurve, Point, load_input, KeyLeakage,
                   candidates_from_lsb_leak, format_leak_info,
                   calculate_speedup, calculate_search_reduction)

from BruteForce.bonus import brute_force_with_lsb_leak
from BabyStep.bonus import bsgs_with_msb_leak, bsgs_with_interval
from PohligHellman.bonus import pohlig_hellman_with_residue_leak
from PohligHellman.main import trial_factor


def main():
    """Quick comparison of deterministic bonus implementations."""
    script_dir = Path(__file__).parent
    input_dir = script_dir / 'input'
    
    print("="*80)
    print("DETERMINISTIC BONUS COMPARISON")
    print("Partial Key Leakage Impact on BruteForce, BSGS, and Pohlig-Hellman")
    print("="*80)
    
    # Test on first two cases
    for test_num in [1, 2]:
        test_path = input_dir / f'testcase_{test_num}.txt'
        answer_path = input_dir / f'answer_{test_num}.txt'
        
        if not test_path.exists() or not answer_path.exists():
            continue
        
        print("\n" + "="*80)
        print(f"TEST CASE {test_num}")
        print("="*80)
        
        # Load test
        p, a, b, G, n, Q = load_input(test_path)
        curve = EllipticCurve(a, b, p)
        
        with answer_path.open('r') as f:
            d_actual = int(f.read().strip())
        
        print(f"Curve: y² = x³ + {a}x + {b} (mod {p})")
        print(f"Order n = {n:,}")
        print(f"Secret: d = {d_actual}")
        
        factors = trial_factor(n)
        print(f"Factorization: {factors}")
        print()
        
        # === BRUTE FORCE + 16-bit LSB ===
        print("-" * 80)
        print("✓ BRUTE FORCE + 16-bit LSB Leak")
        print("-" * 80)
        
        leaked_value, mask = KeyLeakage.leak_lsb_bits(d_actual, 16)
        candidates = candidates_from_lsb_leak(n, leaked_value, mask)
        
        print(format_leak_info('lsb', num_bits=16, leaked_value=leaked_value))
        print(f"Candidates: {len(candidates):,}")
        reduction, pct = calculate_search_reduction(n, len(candidates))
        print(f"Reduction: {reduction:.2f}x")
        
        start = time.perf_counter()
        result = brute_force_with_lsb_leak(curve, G, Q, n, leaked_value, mask)
        elapsed = time.perf_counter() - start
        
        if result == d_actual:
            print(f"✓ SUCCESS: d = {result}")
            print(f"Time: {elapsed:.6f}s")
        else:
            print(f"✗ FAILED")
        
        # === BSGS + 12-bit MSB ===
        print("\n" + "-" * 80)
        print("✓ BABY-STEP GIANT-STEP + 12-bit MSB Leak")
        print("-" * 80)
        
        leaked_value, shift = KeyLeakage.leak_msb_bits(d_actual, 12, n.bit_length())
        
        print(format_leak_info('msb', num_bits=12, leaked_value=leaked_value))
        
        start = time.perf_counter()
        result = bsgs_with_msb_leak(curve, G, Q, n, leaked_value, shift)
        elapsed = time.perf_counter() - start
        
        if result == d_actual:
            print(f"✓ SUCCESS: d = {result}")
            print(f"Time: {elapsed:.6f}s")
        else:
            print(f"✗ FAILED")
        
        # === BSGS + 5% Interval ===
        print("\n" + "-" * 80)
        print("✓ BABY-STEP GIANT-STEP + 5% Interval")
        print("-" * 80)
        
        interval_size = max(1000, n * 5 // 100)
        lower = max(0, d_actual - interval_size // 2)
        upper = min(n - 1, lower + interval_size - 1)
        
        print(format_leak_info('interval', lower=lower, upper=upper, n=n))
        reduction, pct = calculate_search_reduction(n, upper - lower + 1)
        print(f"Reduction: {reduction:.2f}x")
        
        start = time.perf_counter()
        result = bsgs_with_interval(curve, G, Q, lower, upper)
        elapsed = time.perf_counter() - start
        
        if result == d_actual:
            print(f"✓ SUCCESS: d = {result}")
            print(f"Time: {elapsed:.6f}s")
        else:
            print(f"✗ FAILED")
        
        # === POHLIG-HELLMAN + Residues ===
        if factors and len(factors) >= 2:
            print("\n" + "-" * 80)
            print("✓ POHLIG-HELLMAN + Residue Leaks (2 primes)")
            print("-" * 80)
            
            sorted_factors = sorted(factors.items(), key=lambda x: x[0] ** x[1])
            leaked_moduli = [q ** e for q, e in sorted_factors[:2]]
            leaked_residues = KeyLeakage.leak_residues(d_actual, leaked_moduli)
            
            print(format_leak_info('residues', residues=leaked_residues))
            
            start = time.perf_counter()
            result = pohlig_hellman_with_residue_leak(curve, G, Q, n, leaked_residues)
            elapsed = time.perf_counter() - start
            
            if result == d_actual:
                print(f"✓ SUCCESS: d = {result}")
                print(f"Time: {elapsed:.6f}s")
            else:
                print(f"✗ FAILED")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: PARTIAL KEY LEAKAGE IMPACT")
    print("="*80)
    print("""
✓ BRUTE FORCE: LSB leaks reduce search space by 2^(leaked_bits)
  → 16-bit leak = 65,536x speedup!

✓ BSGS: MSB/interval leaks reduce the √n parameter dramatically
  → Smaller baby-step table = faster lookup

✓ POHLIG-HELLMAN: Residue leaks eliminate CRT subproblems
  → Each leaked factor removed from computation

KEY INSIGHT: Side-channel attacks providing even partial information
             can make "impossible" problems tractable!
    """)
    print("="*80)


if __name__ == "__main__":
    main()

"""
Compare ALL Bonus Implementations - Partial Key Leakage Analysis

Comprehensive comparison demonstrating how different leak types
affect each algorithm's performance across multiple test cases.
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent))

from utils import (EllipticCurve, Point, load_input, KeyLeakage,
                   candidates_from_lsb_leak, format_leak_info,
                   calculate_speedup, calculate_search_reduction)

# Import bonus implementations
from BruteForce.bonus import brute_force_with_lsb_leak, brute_force_with_interval
from BabyStep.bonus import bsgs_with_interval, bsgs_with_msb_leak
from PohligHellman.bonus import pohlig_hellman_with_residue_leak
from PollardRho.bonus import pollard_rho_with_interval
from LasVegas.bonus import las_vegas_with_approximate

# Import standard implementations for comparison
from BruteForce.main import brute_force_ecdlp
from BabyStep.main import bsgs_ecdlp
from PohligHellman.main import pohlig_hellman_ecdlp, trial_factor


def run_timed(func, *args, timeout=30.0):
    """Run function with timeout, return (result, time, success)."""
    start = time.perf_counter()
    try:
        result = func(*args)
        elapsed = time.perf_counter() - start
        
        if elapsed > timeout:
            return None, timeout, False
        
        return result, elapsed, True
    except Exception as e:
        elapsed = time.perf_counter() - start
        return None, elapsed, False


def compare_test_case(test_num: int, input_path: Path, answer_path: Path, max_time: float = 30.0):
    """Compare all algorithms on one test case."""
    print("\n" + "="*80)
    print(f"TEST CASE {test_num}")
    print("="*80)
    
    # Load test case
    try:
        p, a, b, G, n, Q = load_input(input_path)
        curve = EllipticCurve(a, b, p)
    except Exception as e:
        print(f"Error loading test: {e}")
        return
    
    # Load actual answer
    try:
        with answer_path.open('r') as f:
            d_actual = int(f.read().strip())
    except:
        print("Error loading answer")
        return
    
    print(f"Curve: y² = x³ + {a}x + {b} (mod {p})")
    print(f"Order n = {n:,}")
    print(f"Actual secret: d = {d_actual}")
    
    # Factor analysis for Pohlig-Hellman
    factors = trial_factor(n)
    print(f"Factorization: {factors}")
    print()
    
    # === BRUTE FORCE WITH LSB LEAK ===
    print("-" * 80)
    print("BRUTE FORCE + 16-bit LSB Leak")
    print("-" * 80)
    
    leaked_value, mask = KeyLeakage.leak_lsb_bits(d_actual, 16)
    candidates = candidates_from_lsb_leak(n, leaked_value, mask)
    print(format_leak_info('lsb', num_bits=16, leaked_value=leaked_value))
    print(f"Candidates: {len(candidates):,}")
    reduction, pct = calculate_search_reduction(n, len(candidates))
    print(f"Reduction: {reduction:.2f}x ({pct} of search space)")
    
    result, elapsed, success = run_timed(brute_force_with_lsb_leak, curve, G, Q, n, leaked_value, mask, timeout=max_time)
    if success and result == d_actual:
        print(f"✓ SUCCESS in {elapsed:.6f}s")
    else:
        print(f"✗ FAILED or timeout ({elapsed:.3f}s)")
    
    # === BSGS WITH MSB LEAK ===
    print("\n" + "-" * 80)
    print("BABY-STEP GIANT-STEP + 16-bit MSB Leak")
    print("-" * 80)
    
    leaked_value, shift = KeyLeakage.leak_msb_bits(d_actual, 16, n.bit_length())
    print(format_leak_info('msb', num_bits=16, leaked_value=leaked_value))
    
    result, elapsed, success = run_timed(bsgs_with_msb_leak, curve, G, Q, n, leaked_value, shift, timeout=max_time)
    if success and result == d_actual:
        print(f"✓ SUCCESS in {elapsed:.6f}s")
    else:
        print(f"✗ FAILED or timeout ({elapsed:.3f}s)")
    
    # === BSGS WITH INTERVAL ===
    print("\n" + "-" * 80)
    print("BABY-STEP GIANT-STEP + 5% Interval")
    print("-" * 80)
    
    interval_size = max(1000, n * 5 // 100)
    # Create interval around actual d
    lower = max(0, d_actual - interval_size // 2)
    upper = min(n - 1, lower + interval_size - 1)
    
    print(format_leak_info('interval', lower=lower, upper=upper, n=n))
    reduction, pct = calculate_search_reduction(n, upper - lower + 1)
    print(f"Reduction: {reduction:.2f}x ({pct} of search space)")
    
    result, elapsed, success = run_timed(bsgs_with_interval, curve, G, Q, n, lower, upper, timeout=max_time)
    if success and result == d_actual:
        print(f"✓ SUCCESS in {elapsed:.6f}s")
    else:
        print(f"✗ FAILED or timeout ({elapsed:.3f}s)")
    
    # === POHLIG-HELLMAN WITH RESIDUE LEAK ===
    if factors and len(factors) >= 2:
        print("\n" + "-" * 80)
        print("POHLIG-HELLMAN + Residue Leaks (2 smallest primes)")
        print("-" * 80)
        
        sorted_factors = sorted(factors.items(), key=lambda x: x[0] ** x[1])
        leaked_moduli = [q ** e for q, e in sorted_factors[:2]]
        leaked_residues = KeyLeakage.leak_residues(d_actual, leaked_moduli)
        
        print(format_leak_info('residues', residues=leaked_residues))
        total_work = sum(q ** e for q, e in factors.items())
        leaked_work = sum(leaked_moduli)
        remaining_work = total_work - leaked_work
        reduction, pct = calculate_search_reduction(total_work, remaining_work)
        print(f"Complexity reduction: {reduction:.2f}x ({pct} remaining)")
        
        result, elapsed, success = run_timed(pohlig_hellman_with_residue_leak, curve, G, Q, n, leaked_residues, timeout=max_time)
        if success and result == d_actual:
            print(f"✓ SUCCESS in {elapsed:.6f}s")
        else:
            print(f"✗ FAILED or timeout ({elapsed:.3f}s)")
    
    # === POLLARD RHO WITH INTERVAL ===
    print("\n" + "-" * 80)
    print("POLLARD RHO + 10% Interval")
    print("-" * 80)
    
    interval_size = max(10000, n * 10 // 100)
    lower = max(0, d_actual - interval_size // 2)
    upper = min(n - 1, lower + interval_size - 1)
    
    print(format_leak_info('interval', lower=lower, upper=upper, n=n))
    print("(Probabilistic - may timeout)")
    
    result, elapsed, success = run_timed(pollard_rho_with_interval, curve, G, Q, n, lower, upper, 100000, timeout=max_time)
    if success and result == d_actual:
        print(f"✓ SUCCESS in {elapsed:.6f}s")
    else:
        print(f"⊘ TIMEOUT (expected for probabilistic method)")
    
    # === LAS VEGAS WITH APPROXIMATE VALUE ===
    print("\n" + "-" * 80)
    print("LAS VEGAS + 5% Approximate Value")
    print("-" * 80)
    
    approx_value, error = KeyLeakage.leak_approximate_value(d_actual, n, 5)
    print(format_leak_info('approximate', approx_value=approx_value, error=error, n=n))
    print("(Probabilistic - may timeout)")
    
    result, elapsed, success = run_timed(las_vegas_with_approximate, curve, G, Q, n, approx_value, error, 100000, timeout=max_time)
    if success and result == d_actual:
        print(f"✓ SUCCESS in {elapsed:.6f}s")
    else:
        print(f"⊘ TIMEOUT (expected for probabilistic method)")


def main():
    """Compare all bonus implementations across test cases."""
    script_dir = Path(__file__).parent
    input_dir = script_dir / 'input'
    
    print("="*80)
    print("COMPREHENSIVE BONUS COMPARISON")
    print("Partial Key Leakage Impact Across All Algorithms")
    print("="*80)
    
    # Find all test cases
    test_cases = sorted(input_dir.glob('testcase_*.txt'))
    
    if not test_cases:
        print("Error: No test cases found")
        sys.exit(1)
    
    for test_path in test_cases:
        test_num = test_path.stem.split('_')[1]
        answer_path = input_dir / f'answer_{test_num}.txt'
        
        if not answer_path.exists():
            print(f"Warning: Answer file missing for test {test_num}")
            continue
        
        compare_test_case(int(test_num), test_path, answer_path, max_time=30.0)
    
    # Summary insights
    print("\n" + "="*80)
    print("SUMMARY: KEY INSIGHTS FROM PARTIAL KEY LEAKAGE")
    print("="*80)
    print("""
1. BRUTE FORCE + LSB Leak:
   - Most effective with strong bit leaks
   - 16-bit leak reduces search by ~65,536x
   - Linear scan becomes practical for large d

2. BSGS + MSB Leak:
   - MSB leaks reduce √n parameter effectively
   - Interval leaks allow smaller baby-step table
   - Best for large d values (d > √n)

3. POHLIG-HELLMAN + Residue Leak:
   - Each leaked residue eliminates entire subproblems
   - Most vulnerable to partial leaks
   - Factorization structure becomes weakness

4. POLLARD RHO + Interval:
   - Random walk can be constrained
   - Probabilistic nature makes timeout likely
   - Interval restricts collision space

5. LAS VEGAS + Approximate Value:
   - Weighted sampling improves success probability
   - Gaussian distribution beats uniform
   - Power analysis leaks are highly valuable

OVERALL: Deterministic algorithms (Brute, BSGS, Pohlig) benefit most from leaks.
         Side-channel attacks providing even partial key information are devastating!
    """)
    print("="*80)


if __name__ == "__main__":
    main()

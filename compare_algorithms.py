#!/usr/bin/env python3
"""
Compare ECDLP algorithms across multiple test cases.
Shows which algorithm performs best under different conditions.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from utils import EllipticCurve, load_input

# Import algorithms
sys.path.insert(0, str(Path(__file__).parent / 'BruteForce'))
from BruteForce.main import brute_force_ecdlp

sys.path.insert(0, str(Path(__file__).parent / 'BabyStep'))
from BabyStep.main import bsgs_ecdlp

sys.path.insert(0, str(Path(__file__).parent / 'PohligHellman'))
from PohligHellman.main import pohlig_hellman_ecdlp


def main():
    """Compare algorithms across all test cases."""
    input_dir = Path(__file__).parent / 'input'
    test_files = sorted(input_dir.glob('testcase_*.txt'))
    
    if not test_files:
        print("Error: No test cases found")
        sys.exit(1)
    
    print("="*80)
    print("ECDLP ALGORITHM COMPARISON")
    print("="*80)
    print(f"Testing {len(test_files)} test cases with diverse curve parameters\n")
    
    results = {
        'Brute Force': {'times': [], 'wins': 0},
        'BSGS': {'times': [], 'wins': 0},
        'Pohlig-Hellman': {'times': [], 'wins': 0},
    }
    
    # Test each case
    for test_file in test_files:
        case_num = test_file.stem.split('_')[1]
        print(f"\n{'='*80}")
        print(f"Test Case {case_num}")
        print(f"{'='*80}")
        
        # Load test
        try:
            p, a, b, G, n, Q = load_input(test_file)
            curve = EllipticCurve(a, b, p)
            sqrt_n = int(n ** 0.5)
            
            print(f"Curve: y² = x³ + {a}x + {b} (mod {p})")
            print(f"Base point G = ({G[0]}, {G[1]})")
            print(f"Target point Q = ({Q[0]}, {Q[1]})")
            print(f"Order n = {n}, √n ≈ {sqrt_n}")
            print(f"Task: Find d such that Q = d*G")
            print()
            
        except Exception as e:
            print(f"Error loading: {e}\n")
            continue
        
        # Run algorithms
        times = {}
        
        # Brute Force
        print(f"{'Brute Force':<20}", end="", flush=True)
        start = time.perf_counter()
        d_bf = None
        try:
            d = brute_force_ecdlp(curve, G, Q, n)
            elapsed = time.perf_counter() - start
            verified = curve.scalar_multiply(d, G) == Q
            if verified:
                times['Brute Force'] = elapsed
                results['Brute Force']['times'].append(elapsed)
                d_bf = d
                print(f"✓  {elapsed:8.4f}s  (d = {d})")
            else:
                print(f"✗  Failed verification")
        except Exception as e:
            print(f"✗  Error: {e}")
        
        # BSGS
        print(f"{'BSGS':<20}", end="", flush=True)
        start = time.perf_counter()
        d_bsgs = None
        try:
            d = bsgs_ecdlp(curve, G, Q, n)
            elapsed = time.perf_counter() - start
            verified = curve.scalar_multiply(d, G) == Q
            if verified:
                times['BSGS'] = elapsed
                results['BSGS']['times'].append(elapsed)
                d_bsgs = d
                print(f"✓  {elapsed:8.4f}s  (d = {d})")
            else:
                print(f"✗  Failed verification")
        except Exception as e:
            print(f"✗  Error: {e}")
        
        # Pohlig-Hellman
        print(f"{'Pohlig-Hellman':<20}", end="", flush=True)
        start = time.perf_counter()
        d_ph = None
        try:
            d = pohlig_hellman_ecdlp(curve, G, Q, n)
            elapsed = time.perf_counter() - start
            if d is not None:
                verified = curve.scalar_multiply(d, G) == Q
                if verified:
                    times['Pohlig-Hellman'] = elapsed
                    results['Pohlig-Hellman']['times'].append(elapsed)
                    d_ph = d
                    print(f"✓  {elapsed:8.4f}s  (d = {d})")
                else:
                    print(f"✗  Failed verification")
            else:
                print(f"✗  No solution found")
        except Exception as e:
            print(f"✗  Error: {e}")
        
        # Show actual answer for verification
        answer_file = input_dir / f"answer_{case_num}.txt"
        if answer_file.exists():
            with answer_file.open('r') as f:
                d_actual = int(f.read().strip())
            d_pos = "d < √n" if d_actual < sqrt_n else "d > √n"
            print(f"\nActual answer: d = {d_actual} ({d_pos})")
            
            # Check if all algorithms agree
            found_vals = [v for v in [d_bf, d_bsgs, d_ph] if v is not None]
            if found_vals and all(v == d_actual for v in found_vals):
                print("✓ All algorithms found correct answer")
            elif found_vals:
                print("⚠ Warning: Algorithm results differ!")
        
        # Determine winner
        if times:
            winner = min(times, key=times.get)
            results[winner]['wins'] += 1
            print(f"\n→ Fastest: {winner} ({times[winner]:.4f}s)")
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Algorithm':<20} {'Avg Time':<12} {'Wins':<8} {'Success Rate'}")
    print("-"*80)
    
    for name in ['Brute Force', 'BSGS', 'Pohlig-Hellman']:
        data = results[name]
        if data['times']:
            avg_time = sum(data['times']) / len(data['times'])
            success_rate = f"{len(data['times'])}/{len(test_files)} ({100*len(data['times'])/len(test_files):.0f}%)"
            print(f"{name:<20} {avg_time:8.4f}s    {data['wins']:<8} {success_rate}")
        else:
            print(f"{name:<20} {'N/A':<12} {data['wins']:<8} 0/{len(test_files)} (0%)")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS:")
    print("="*80)
    print("• Brute Force: Best when d < √n/2 (small secrets)")
    print("• BSGS: Best when d > √n (large secrets) and n is large prime")
    print("• Pohlig-Hellman: Best when n has small prime factors (usually fastest!)")
    print("\nNote: Pollard Rho and Las Vegas are probabilistic algorithms that may")
    print("take very long or fail. They are implemented but not included in comparison.")
    print("Run them individually on specific test cases if needed.")
    print("="*80)


if __name__ == "__main__":
    main()

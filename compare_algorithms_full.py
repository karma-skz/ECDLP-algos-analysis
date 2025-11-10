#!/usr/bin/env python3
"""
Compare all ECDLP algorithms across multiple test cases.
Shows success rates, timing statistics, and attempts for probabilistic algorithms.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import EllipticCurve, load_input

# Import algorithm functions
sys.path.insert(0, str(Path(__file__).parent / 'BruteForce'))
from BruteForce.main import brute_force_ecdlp

sys.path.insert(0, str(Path(__file__).parent / 'BabyStep'))
from BabyStep.main import bsgs_ecdlp

sys.path.insert(0, str(Path(__file__).parent / 'PohligHellman'))
from PohligHellman.main import pohlig_hellman_ecdlp

sys.path.insert(0, str(Path(__file__).parent / 'PollardRho'))
from PollardRho.main import pollard_rho_ecdlp

sys.path.insert(0, str(Path(__file__).parent / 'LasVegas'))
from LasVegas.main import las_vegas_ecdlp


class AlgorithmResult:
    """Store results for one algorithm across all test cases."""
    def __init__(self, name: str):
        self.name = name
        self.successes = 0
        self.failures = 0
        self.times: List[float] = []
        self.attempts: List[int] = []  # For probabilistic algorithms
    
    def add_success(self, time_taken: float, attempts: int = 1):
        self.successes += 1
        self.times.append(time_taken)
        self.attempts.append(attempts)
    
    def add_failure(self, time_taken: float):
        self.failures += 1
        self.times.append(time_taken)
    
    @property
    def total_tests(self) -> int:
        return self.successes + self.failures
    
    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return 100.0 * self.successes / self.total_tests
    
    @property
    def avg_time(self) -> float:
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)
    
    @property
    def avg_attempts(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(self.attempts) / len(self.attempts)


def run_algorithm(name: str, algo_func, curve: EllipticCurve, G, Q, n: int, 
                  timeout: float = 30.0, is_probabilistic: bool = False) -> Tuple[bool, float, int]:
    """
    Run an algorithm on one test case.
    
    Returns:
        (success, time_taken, attempts)
    """
    start = time.perf_counter()
    
    try:
        if is_probabilistic:
            # For probabilistic algorithms, return might include attempts info
            result = algo_func(curve, G, Q, n)
            
            if isinstance(result, tuple) and len(result) >= 2:
                d, extra = result[0], result[1]
                # extra might be steps/attempts depending on algorithm
                attempts = extra if isinstance(extra, int) else 1
            else:
                d = result
                attempts = 1
        else:
            d = algo_func(curve, G, Q, n)
            attempts = 1
        
        elapsed = time.perf_counter() - start
        
        # Check for timeout
        if elapsed > timeout:
            return False, elapsed, attempts
        
        if d is not None:
            # Verify
            Q_verify = curve.scalar_multiply(d, G)
            if Q_verify == Q:
                return True, elapsed, attempts
        
        return False, elapsed, attempts
    
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"    Error in {name}: {e}")
        return False, elapsed, 1


def main():
    """Compare all algorithms across all test cases."""
    input_dir = Path(__file__).parent / 'input'
    
    # Find all test cases
    test_files = sorted(input_dir.glob('testcase_*.txt'))
    
    if not test_files:
        print("Error: No test cases found in input/")
        sys.exit(1)
    
    print("="*70)
    print("ECDLP ALGORITHM COMPARISON")
    print("="*70)
    print(f"Found {len(test_files)} test cases\n")
    
    # Initialize results
    results = {
        'Brute Force': AlgorithmResult('Brute Force'),
        'BSGS': AlgorithmResult('BSGS'),
        'Pohlig-Hellman': AlgorithmResult('Pohlig-Hellman'),
        'Pollard Rho': AlgorithmResult('Pollard Rho'),
        'Las Vegas': AlgorithmResult('Las Vegas'),
    }
    
    # Run algorithms on each test case
    for test_file in test_files:
        case_num = test_file.stem.split('_')[1]
        print(f"\n{'='*70}")
        print(f"Test Case {case_num}: {test_file.name}")
        print(f"{'='*70}")
        
        # Load test case
        try:
            p, a, b, G, n, Q = load_input(test_file)
            curve = EllipticCurve(a, b, p)
            sqrt_n = int(n ** 0.5)
            
            # Load answer to show d position
            answer_file = input_dir / f"answer_{case_num}.txt"
            if answer_file.exists():
                with answer_file.open('r') as f:
                    d_actual = int(f.read().strip())
                d_pos = "small" if d_actual < sqrt_n else "large"
                print(f"Curve: y² = x³ + {a}x + {b} (mod {p})")
                print(f"n = {n}, √n ≈ {sqrt_n}")
                print(f"d = {d_actual} ({d_pos})")
            else:
                print(f"Curve: y² = x³ + {a}x + {b} (mod {p})")
                print(f"n = {n}")
            
            print()
        except Exception as e:
            print(f"Error loading test case: {e}")
            continue
        
        # Run each algorithm
        # Brute Force
        print(f"  Brute Force...", end=" ", flush=True)
        success, t, att = run_algorithm('Brute Force', brute_force_ecdlp, curve, G, Q, n, timeout=10.0)
        if success:
            results['Brute Force'].add_success(t, att)
            print(f"✓ {t:.4f}s")
        else:
            results['Brute Force'].add_failure(t)
            print(f"✗ {t:.4f}s")
        
        # BSGS
        print(f"  BSGS...", end=" ", flush=True)
        success, t, att = run_algorithm('BSGS', bsgs_ecdlp, curve, G, Q, n, timeout=10.0)
        if success:
            results['BSGS'].add_success(t, att)
            print(f"✓ {t:.4f}s")
        else:
            results['BSGS'].add_failure(t)
            print(f"✗ {t:.4f}s")
        
        # Pohlig-Hellman
        print(f"  Pohlig-Hellman...", end=" ", flush=True)
        success, t, att = run_algorithm('Pohlig-Hellman', pohlig_hellman_ecdlp, curve, G, Q, n, timeout=10.0)
        if success:
            results['Pohlig-Hellman'].add_success(t, att)
            print(f"✓ {t:.4f}s")
        else:
            results['Pohlig-Hellman'].add_failure(t)
            print(f"✗ {t:.4f}s")
        
        # Pollard Rho (probabilistic, give it more time)
        print(f"  Pollard Rho...", end=" ", flush=True)
        success, t, att = run_algorithm('Pollard Rho', pollard_rho_ecdlp, curve, G, Q, n, 
                                       timeout=20.0, is_probabilistic=True)
        if success:
            results['Pollard Rho'].add_success(t, att)
            print(f"✓ {t:.4f}s")
        else:
            results['Pollard Rho'].add_failure(t)
            print(f"✗ {t:.4f}s (timeout/failed)")
        
        # Las Vegas (probabilistic, runs multiple attempts internally)
        print(f"  Las Vegas...", end=" ", flush=True)
        success, t, att = run_algorithm('Las Vegas', las_vegas_ecdlp, curve, G, Q, n, 
                                       timeout=15.0, is_probabilistic=True)
        if success:
            results['Las Vegas'].add_success(t, att)
            print(f"✓ {t:.4f}s")
        else:
            results['Las Vegas'].add_failure(t)
            print(f"✗ {t:.4f}s (failed)")
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Algorithm':<20} {'Success Rate':<15} {'Avg Time':<12} {'Avg Attempts':<12}")
    print("-"*70)
    
    for name in ['Brute Force', 'BSGS', 'Pohlig-Hellman', 'Pollard Rho', 'Las Vegas']:
        result = results[name]
        success_str = f"{result.successes}/{result.total_tests} ({result.success_rate:.1f}%)"
        time_str = f"{result.avg_time:.4f}s" if result.times else "N/A"
        attempts_str = f"{result.avg_attempts:.1f}" if result.attempts else "N/A"
        
        print(f"{name:<20} {success_str:<15} {time_str:<12} {attempts_str:<12}")
    
    print("\n" + "="*70)
    print("NOTES:")
    print("- Brute Force: Fast when d < √n (test case 1)")
    print("- BSGS: Fast when d > √n or d position unknown")
    print("- Pohlig-Hellman: Fastest when n has small prime factors")
    print("- Pollard Rho: Probabilistic, may need multiple attempts")
    print("- Las Vegas: Probabilistic, complex linear algebra approach")
    print("="*70)


if __name__ == "__main__":
    main()

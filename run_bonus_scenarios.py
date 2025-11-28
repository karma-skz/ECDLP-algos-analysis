#!/usr/bin/env python3
"""
MASTER BONUS RUNNER
Runs all bonus scenarios on a specific test case and generates a summary.
Usage: python3 run_bonus_scenarios.py <path_to_test_case>
"""
import subprocess
import sys
import shutil
from pathlib import Path

def print_separator(char="=", length=80):
    print(char * length)

def run_scenario(algo, test_file):
    script = Path(algo) / 'bonus.py'
    if not script.exists(): 
        print(f"⚠ Script not found: {script}")
        return
    
    print(f"\n>>> Running {algo} Bonus...")
    # Use the current python executable to ensure we stay in venv
    try:
        subprocess.run([sys.executable, str(script), str(test_file)], check=False)
    except KeyboardInterrupt:
        print("\n[Aborted by user]")
        sys.exit(1)

def main():
    # 1. Parse Arguments
    if len(sys.argv) > 1:
        case_path = Path(sys.argv[1])
    else:
        # Default fallback
        print("No test case specified. Looking for a default...")
        case_path = Path("test_cases/20bit/case_1.txt")
        if not case_path.exists():
            # Find first available txt file
            cases = list(Path("test_cases").glob("*/*.txt"))
            if cases:
                case_path = cases[0]
            else:
                print("Error: No test cases found in test_cases/ folder.")
                sys.exit(1)

    if not case_path.exists():
        print(f"Error: File not found: {case_path}")
        sys.exit(1)

    # 2. Print Header
    print_separator("=")
    print(f"  ECC SIDE-CHANNEL / BONUS SCENARIOS TEST SUITE")
    print_separator("=")
    print(f"Target Case: {case_path}")
    print(f"File Size:   {case_path.stat().st_size} bytes")
    
    # Check bit size from path name (heuristic)
    try:
        bits = int(case_path.parent.name.replace('bit', ''))
        print(f"Bit Length:  {bits}-bit curve")
        if bits > 40:
            print("⚠ WARNING: You are running scenarios on a LARGE curve (>40 bits).")
            print("  - Brute Force & BSGS baselines will be skipped (mocked) to prevent hanging.")
            print("  - Pollard's Rho & Las Vegas will rely on the hints to finish.")
    except:
        pass
    
    print_separator("=")

    # 3. Run All Scenarios
    algos = ['BruteForce', 'BabyStep', 'PollardRho', 'PohligHellman', 'LasVegas']
    
    for algo in algos:
        run_scenario(algo, case_path)

    print("\n")
    print_separator("=")
    print("✓ SCENARIO SUITE COMPLETE")
    print_separator("=")

if __name__ == "__main__":
    main()
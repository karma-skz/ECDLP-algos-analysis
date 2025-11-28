#!/usr/bin/env python3
"""
Interactive ECC Encryption/Decryption Demo (ECIES)

This script demonstrates the full lifecycle of an Elliptic Curve message:
1. Alice generates a key pair (d, Q).
2. Bob encrypts a text message for Alice using Q (ECIES method).
3. The user acts as an attacker/recipient to recover the private key 'd'.
4. The script uses 'd' to decrypt the message step-by-step.

Uses AES-128-GCM for the symmetric part and a simple SHA-256 KDF.
"""

import sys
import os
import hashlib
import time
import random
import argparse
from pathlib import Path

# Try to import cryptography for AES (standard in most python envs)
# If not available, we can mock the symmetric part or use simple XOR for demo
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    print("Warning: 'cryptography' library not found. Installing it is recommended.")
    print("         Falling back to simple XOR encryption for demonstration.")

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent))

try:
    from utils import EllipticCurve, Point, load_input
    
    # Import Solvers
    # NOTE: We import from 'main.py' for most algorithms because 'main_optimized.py' 
    # often contains script-only logic without reusable functions.
    
    from BruteForce.main import brute_force_ecdlp as solve_bf
    from BabyStep.main import bsgs_ecdlp as solve_bsgs
    from PohligHellman.main import pohlig_hellman_ecdlp as solve_ph
    
    # For Pollard's Rho, we prefer the optimized version if available
    try:
        from PollardRho.main_optimized import pollard_rho_optimized as solve_rho
    except ImportError:
        from PollardRho.main import pollard_rho_ecdlp as solve_rho

except ImportError as e:
    print(f"\n[CRITICAL ERROR] Could not import project modules.")
    print(f"Details: {e}")
    print("Ensure you are running this script from the PROJECT ROOT directory.")
    print("Example: python3 demo_ecies.py")
    sys.exit(1)

# ==============================================================================
# ECIES HELPER FUNCTIONS
# ==============================================================================

def kdf(point_x: int, length: int = 16) -> bytes:
    """
    Key Derivation Function (ANSI X9.63 compliant simplified).
     Hashes the x-coordinate of the shared secret to generate a symmetric key.
    """
    # We use SHA-256 on the bytes of x
    x_bytes = str(point_x).encode()
    full_hash = hashlib.sha256(x_bytes).digest()
    return full_hash[:length] # Truncate to desired key length (16 bytes for AES-128)

def symmetric_encrypt(key: bytes, plaintext: str) -> dict:
    """Encrypts text using AES-GCM (or XOR fallback)."""
    data = plaintext.encode('utf-8')
    
    if HAS_CRYPTO:
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, data, None)
        return {
            'mode': 'AES-GCM',
            'nonce': nonce.hex(),
            'ciphertext': ciphertext.hex()
        }
    else:
        # XOR Fallback (Insecure, just for demo logic)
        # XOR data with repeated key
        cipher = bytearray()
        for i, b in enumerate(data):
            cipher.append(b ^ key[i % len(key)])
        return {
            'mode': 'XOR-Simple',
            'nonce': '00'*12,
            'ciphertext': cipher.hex()
        }

def symmetric_decrypt(key: bytes, packet: dict) -> str:
    """Decrypts text."""
    try:
        if packet['mode'] == 'AES-GCM':
            if not HAS_CRYPTO: return "[Error: cryptography lib missing]"
            aesgcm = AESGCM(key)
            nonce = bytes.fromhex(packet['nonce'])
            ct = bytes.fromhex(packet['ciphertext'])
            pt = aesgcm.decrypt(nonce, ct, None)
            return pt.decode('utf-8')
        else:
            # XOR
            ct = bytes.fromhex(packet['ciphertext'])
            pt = bytearray()
            for i, b in enumerate(ct):
                pt.append(b ^ key[i % len(key)])
            return pt.decode('utf-8')
    except Exception as e:
        return f"[Decryption Failed: {e}]"

# ==============================================================================
# MAIN DEMO
# ==============================================================================

def main():
    parser = argparse.figure_description = argparse.ArgumentParser(description="Interactive ECIES Demo")
    parser.add_argument("testcase", nargs="?", default="test_cases/20bit/case_1.txt", help="Path to the test case file (default: test_cases/20bit/case_1.txt)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  ELLIPTIC CURVE INTEGRATED ENCRYPTION SCHEME (ECIES) DEMO")
    print("="*70)
    
    # 1. SETUP: Load a curve
    # We use a small 20-bit curve so algorithms run fast for the demo
    # But large enough to look cool.
    print("\n[STEP 1] System Setup")
    print("Loading elliptic curve parameters...")
    
    case_path = Path(args.testcase)
    if not case_path.exists():
        # Fallback to creating one or finding another
        try:
            # Try to find any case file
            candidates = list(Path("test_cases").glob("*/*.txt"))
            if not candidates:
                raise FileNotFoundError("No case files found")
            # Sort to pick a smaller one preferably
            candidates.sort(key=lambda p: p.stat().st_size)
            case_path = candidates[0]
            print(f"Warning: Specified file not found. Falling back to {case_path}")
        except:
            print("Error: No test cases found. Run 'generate_test_cases.py' first.")
            return

    try:
        p, a, b, G, n, _ = load_input(case_path)
    except:
        # Handle cases where the file might have 4 lines instead of 5 (older gen)
        # Or just retry with a simpler parsing if needed, but Utils should handle it.
        print(f"Error loading {case_path}. Please regenerate test cases.")
        return

    curve = EllipticCurve(a, b, p)
    print(f"  Curve: y^2 = x^3 + {a}x + {b} (mod {p})")
    print(f"  Generator G: {G}")
    print(f"  Order n: {n}")
    
    # 2. KEY GENERATION (Alice)
    print("\n[STEP 2] Key Generation (Alice)")
    # Alice picks private key d_A
    d_A = random.randint(1, n-1)
    # Alice computes public key Q_A = d_A * G
    Q_A = curve.scalar_multiply(d_A, G)
    
    print(f"  Alice's Private Key (d_A):  [HIDDEN] (You need to find this!)")
    print(f"  Alice's Public Key  (Q_A):  {Q_A}")
    print("  Alice publishes Q_A to the world.")
    
    # 3. ENCRYPTION (Bob sends message to Alice)
    print("\n[STEP 3] Message Encryption (Bob -> Alice)")
    # Flush stdin just in case
    try:
        import termios, sys
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except:
        pass

    msg = input("  Enter a short message to encrypt (default: 'Attack at Dawn'): ").strip()
    if not msg: msg = "Attack at Dawn"
    
    print(f"\n  Encrypting: '{msg}'")
    
    # ECIES Encryption Logic
    # A. Bob generates temporary ephemeral key k
    k = random.randint(1, n-1)
    print(f"  A. Bob picks random ephemeral k: [HIDDEN]")
    
    # B. Bob computes Ephemeral Public Key R = k*G
    R = curve.scalar_multiply(k, G)
    print(f"  B. Bob computes Ephemeral Point R = k*G: {R}")
    
    # C. Bob computes Shared Secret Point S = k*Q_A
    S = curve.scalar_multiply(k, Q_A)
    print(f"  C. Bob computes Shared Secret S = k*Q_A: ({S[0]}, ...)") # type: ignore
    
    # D. Derive Symmetric Key
    sym_key = kdf(S[0]) # type: ignore
    print(f"  D. Bob derives AES Key from S.x: {sym_key.hex()}...")
    
    # E. Encrypt Data
    encrypted_packet = symmetric_encrypt(sym_key, msg)
    print(f"  E. Bob encrypts message with AES.")
    print(f"     Ciphertext: {encrypted_packet['ciphertext'][:30]}...")
    
    print("\n  --> Bob sends [R, Ciphertext] to Alice (and you intercept it!)")
    
    # 4. ATTACK PHASE
    print("\n" + "="*70)
    print("[STEP 4] The Attack")
    print("="*70)
    print("  You have intercepted:")
    print(f"  1. Alice's Public Key Q_A: {Q_A}")
    print(f"  2. Ephemeral Point R:      {R}")
    print(f"  3. Encrypted Ciphertext")
    print("\n  To decrypt, you need the Shared Secret S.")
    print("  S = d_A * R  (You need Alice's private key d_A)")
    print("  OR")
    print("  S = k * Q_A  (You need Bob's ephemeral k)")
    print("\n  Let's solve the ECDLP on Alice's key: Q_A = d_A * G")
    
    print("\n  Choose your weapon:")
    print("  1. Brute Force (Slow)")
    print("  2. Baby-Step Giant-Step (Memory Heavy)")
    print("  3. Pollard's Rho (Fast & Low Memory)")
    print("  4. Pohlig-Hellman (Cheats if n is smooth)")
    
    choice = input("\n  Select algorithm [1-4]: ").strip()
    
    print(f"\n  Cracking d_A where {Q_A} = d_A * {G}...")
    start_crack = time.time()
    
    found_d = None
    
    if choice == '1':
        found_d = solve_bf(curve, G, Q_A, n)
    elif choice == '2':
        found_d = solve_bsgs(curve, G, Q_A, n)
    elif choice == '3':
        # Need to handle the return tuple from rho
        res = solve_rho(curve, G, Q_A, n, max_steps=1000000)
        # Check if result is a tuple (d, steps) or just d
        if isinstance(res, tuple):
            found_d = res[0]
        else:
            found_d = res
    elif choice == '4':
        found_d = solve_ph(curve, G, Q_A, n)
    else:
        print("  Invalid choice. Defaulting to Pollard's Rho.")
        res = solve_rho(curve, G, Q_A, n, max_steps=1000000)
        # Check if result is a tuple (d, steps) or just d
        if isinstance(res, tuple):
            found_d = res[0]
        else:
            found_d = res
        
    crack_time = time.time() - start_crack
    
    if found_d == d_A:
        print(f"\n  ✓ SUCCESS! Private Key Found: {found_d}")
        print(f"  Time taken: {crack_time:.4f}s")
    else:
        print(f"\n  ✗ Failed to find key (Found {found_d}, Real {d_A}).")
        print("  Using real key to continue demo...")
        found_d = d_A

    # 5. DECRYPTION
    print("\n" + "="*70)
    print("[STEP 5] Decryption")
    print("="*70)
    
    # A. Recover Shared Secret
    # S = d_A * R
    S_recovered = curve.scalar_multiply(found_d, R)
    print(f"  A. Compute Shared Secret S = d_A * R")
    print(f"     {found_d} * {R} = {S_recovered}")
    
    # Check match (S must match Bob's S)
    if S_recovered != S:
        # This might happen if we found a 'd' that satisfies dG=Q but isn't the original 'd'
        # in a composite order curve (subgroup attack).
        # In that case, dR might NOT equal kQ if R is outside the subgroup.
        # But for ECIES, valid 'd' should work if we are in the same subgroup.
        print("  Warning: Recovered secret might be in a different subgroup!")
        print(f"  Real S: {S}")
        print(f"  Calc S: {S_recovered}")
        # Proceed anyway to see garbage output, which is educational

    # B. Derive Key
    recovered_key = kdf(S_recovered[0]) # type: ignore
    print(f"  B. Derive AES Key: {recovered_key.hex()}...")
    
    # C. Decrypt
    print("  C. Decrypt Ciphertext...")
    plaintext = symmetric_decrypt(recovered_key, encrypted_packet)
    
    print(f"\n  🔓 DECRYPTED MESSAGE: \"{plaintext}\"")
    print("="*70)

if __name__ == "__main__":
    main()
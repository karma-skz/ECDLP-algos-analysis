"""
BSGS with Interval Leakage - BONUS
Adaptation: Reduces RAM/Time by searching only the effective range.
"""
import sys, time, math, ctypes
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import EllipticCurve, Point, load_input

USE_CPP = False
ecc_lib = None
lib_paths = [Path(__file__).parent.parent / "utils" / "cpp" / "ecc_fast.so"]
for p in lib_paths:
    if p.exists():
        ecc_lib = ctypes.CDLL(str(p))
        ecc_lib.scalar_mult.argtypes = [ctypes.c_longlong]*6 + [ctypes.POINTER(ctypes.c_longlong)]*2
        ecc_lib.scalar_mult.restype = ctypes.c_int
        USE_CPP = True
        break

def fast_mult(k, G, curve):
    if not USE_CPP: return curve.scalar_multiply(k, G)
    rx, ry = ctypes.c_longlong(), ctypes.c_longlong()
    # FIX: No % p
    valid = ecc_lib.scalar_mult(k, G[0], G[1], curve.a, curve.b, curve.p, ctypes.byref(rx), ctypes.byref(ry))
    return (rx.value, ry.value) if valid else None

def bsgs_range(curve, G, Q, lower, upper):
    Q_prime = curve.add(Q, curve.negate(fast_mult(lower, G, curve)))
    limit = upper - lower
    m = int(math.ceil(math.sqrt(limit)))
    
    table = {}
    P = None 
    for j in range(m):
        table[P] = j
        P = curve.add(P, G) if P else G
        
    mG = fast_mult(m, G, curve)
    neg_mG = curve.negate(mG)
    curr = Q_prime
    
    for i in range(m + 1):
        if curr in table:
            return lower + (i * m + table[curr])
        curr = curve.add(curr, neg_mG)
    return None

def main():
    if len(sys.argv) < 2: return
    p, a, b, G, n, Q = load_input(Path(sys.argv[1]))
    curve = EllipticCurve(a, b, p)
    try:
        num = Path(sys.argv[1]).stem.split('_')[1]
        with open(Path(sys.argv[1]).parent / f"answer_{num}.txt") as f: d_real = int(f.read())
    except: d_real = n // 2

    print(f"\n{'='*70}")
    print(f"BSGS ADAPTATIONS (d={d_real})")
    print(f"{'='*70}")
    print(f"{'SCENARIO':<25} | {'TABLE SIZE':<12} | {'TIME':<10}")
    print(f"{'-'*70}")

    m_std = int(math.sqrt(n))
    print(f"{'Standard':<25} | {m_std:<12,} | {'(Baseline)':<10}")

    width = n // 100
    low = max(1, d_real - width//2)
    t0 = time.time()
    d = bsgs_range(curve, G, Q, low, low + width)
    t = time.time() - t0
    m_opt = int(math.sqrt(width))
    print(f"{'1% Interval Leak':<25} | {m_opt:<12,} | {t:.4f}s")
    print(f"{'-'*70}")
    print(f"RAM Savings: {m_std/max(1, m_opt):.1f}x less memory needed")

if __name__ == "__main__":
    main()
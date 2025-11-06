import random
import sys
from typing import List, Tuple, Optional

# Set higher recursion depth for deep computations if needed
sys.setrecursionlimit(2000)

# ==============================================================================
# 
#  Part 1: Modular Math Helpers
#
# ==============================================================================

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Returns (gcd, x, y) s.t. a*x + b*y = gcd"""
    if a == 0:
        return (b, 0, 1)
    else:
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return (gcd, x, y)

def mod_inv(a: int, m: int) -> int:
    """Calculates the modular inverse of a (mod m)"""
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f'Modular inverse does not exist for {a} mod {m}')
    else:
        return x % m

# ==============================================================================
#
#  Part 2: Elliptic Curve and Point Classes (Affine Coords)
#
# ==============================================================================

class Point:
    """A point on an elliptic curve [x, y] or the identity O (None)"""
    
    def __init__(self, x: Optional[int], y: Optional[int]):
        self.x = x
        self.y = y

    def is_identity(self) -> bool:
        """Check if the point is the identity"""
        return self.x is None or self.y is None

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __str__(self) -> str:
        if self.is_identity():
            return "Point(O)"
        return f"Point({self.x}, {self.y})"
        
    def __hash__(self):
        return hash((self.x, self.y))

class EllipticCurve:
    """
    Represents an elliptic curve y^2 = x^3 + ax + b (mod p)
    
    Args:
        a, b: Curve coefficients
        p: The prime modulus of the finite field
    """
    
    def __init__(self, a: int, b: int, p: int):
        self.a = a
        self.b = b
        self.p = p
        # Validate curve
        if (4 * a**3 + 27 * b**2) % p == 0:
            raise ValueError("Curve is singular")

    def identity(self) -> Point:
        """Returns the identity point O"""
        return Point(None, None)

    def negate(self, p1: Point) -> Point:
        """Returns -P"""
        if p1.is_identity():
            return self.identity()
        return Point(p1.x, -p1.y % self.p)

    def add(self, p1: Point, p2: Point) -> Point:
        """Adds two points P1 + P2 on the curve"""
        if p1.is_identity():
            return p2
        if p2.is_identity():
            return p1

        # Handle P1 == P2 (doubling)
        if p1 == p2:
            return self.double(p1)

        # Handle P1 == -P2
        if p1.x == p2.x and p1.y != p2.y:
            return self.identity()

        # Standard addition
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y

        # Calculate slope 's'
        try:
            # s = (y2 - y1) * mod_inv(x2 - x1, p)
            s_num = (y2 - y1) % self.p
            s_den = mod_inv((x2 - x1) % self.p, self.p)
            s = (s_num * s_den) % self.p
        except ValueError:
            # This case (x1=x2, y1!=y2) is already handled,
            # but a duplicate x1=x2,y1=y2 should go to double()
            return self.identity()

        # Calculate new point
        x3 = (s**2 - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p

        return Point(x3, y3)

    def double(self, p1: Point) -> Point:
        """Doubles a point P (P + P)"""
        if p1.is_identity() or p1.y == 0:
            return self.identity()
            
        x, y = p1.x, p1.y
        
        # Calculate slope 's'
        # s = (3*x^2 + a) * mod_inv(2*y, p)
        s_num = (3 * x**2 + self.a) % self.p
        s_den = mod_inv((2 * y) % self.p, self.p)
        s = (s_num * s_den) % self.p

        # Calculate new point
        x3 = (s**2 - 2 * x) % self.p
        y3 = (s * (x - x3) - y) % self.p
        
        return Point(x3, y3)

    def multiply(self, p_in: Point, n: int) -> Point:
        """Scalar multiplication n*P using double-and-add"""
        result = self.identity()
        current = p_in
        
        if n < 0:
            n = abs(n)
            current = self.negate(current)

        while n > 0:
            if n % 2 == 1:
                result = self.add(result, current)
            current = self.double(current)
            n //= 2
            
        return result

# ==============================================================================
#
#  Part 3: Linear Algebra over a Finite Field (F_p)
#
# ==============================================================================

class LinearAlgebraFp:
    """Contains static methods for linear algebra over F_p"""

    @staticmethod
    def _create_matrix_copy(M: List[List[int]]) -> List[List[int]]:
        return [[cell for cell in row] for row in M]

    @staticmethod
    def find_left_kernel_basis(M: List[List[int]], p: int) -> List[List[int]]:
        """
        Computes the basis for the left kernel of M over F_p.
        This is the standard (right) kernel of M_transpose.
        """
        if not M:
            return []
        
        num_rows_M = len(M)
        num_cols_M = len(M[0])
        
        # Transpose M
        Mt = [[M[r][c] for r in range(num_rows_M)] for c in range(num_cols_M)]
        
        # Compute RREF of M_transpose
        rref, pivots = LinearAlgebraFp.compute_rref(Mt, p)
        
        # Extract kernel basis
        kernel_basis = []
        pivot_cols = [p[1] for p in pivots]
        free_var_cols = [c for c in range(num_rows_M) if c not in pivot_cols]
        
        pivot_row_map = {p[1]: p[0] for p in pivots}

        for free_col in free_var_cols:
            vec = [0] * num_rows_M
            vec[free_col] = 1
            for pivot_col in pivot_cols:
                row = pivot_row_map[pivot_col]
                if row < len(rref) and free_col < len(rref[row]):
                    vec[pivot_col] = -rref[row][free_col] % p
            kernel_basis.append(vec)
            
        return kernel_basis

    @staticmethod
    def compute_rref(A: List[List[int]], p: int) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        """Computes the Row-Reduced Echelon Form (RREF) of matrix A over F_p"""
        M = LinearAlgebraFp._create_matrix_copy(A)
        rows = len(M)
        cols = len(M[0]) if rows > 0 else 0
        
        pivots = []
        r = 0
        c = 0
        
        while r < rows and c < cols:
            # Find pivot
            i_pivot = -1
            for i in range(r, rows):
                if M[i][c] != 0:
                    i_pivot = i
                    break
            
            if i_pivot == -1:
                c += 1
                continue

            # Swap rows
            M[r], M[i_pivot] = M[i_pivot], M[r]
            pivots.append((r, c))

            # Normalize pivot row
            pivot_val = M[r][c]
            inv_pivot = mod_inv(pivot_val, p)
            for j in range(c, cols):
                M[r][j] = (M[r][j] * inv_pivot) % p
                
            # Eliminate other rows
            for i in range(rows):
                if i == r:
                    continue
                factor = M[i][c]
                if factor == 0:
                    continue
                for j in range(c, cols):
                    M[i][j] = (M[i][j] - factor * M[r][j]) % p
            
            r += 1
            c += 1
            
        return M, pivots

    @staticmethod
    def solve_problem_l_heuristic(K_basis: List[List[int]], l: int, p: int) -> Optional[List[int]]:
        """
        Implements Algorithm 2 from the paper.
        This is a *heuristic* to find a vector with 'l' zeros.
        """
        if not K_basis:
            return None
        
        # Total columns = 2l (since 3n' = l)
        total_cols = len(K_basis[0])
        
        # --- Attempt 1: Diagonalize first block (cols 0 to l-1) ---
        K_copy1 = LinearAlgebraFp._create_matrix_copy(K_basis)
        
        # Perform Gaussian elimination on the first block
        # This is a partial RREF, not a full one
        r = 0
        for c in range(l):
            if r >= l: break
            
            # Find pivot
            i_pivot = -1
            for i in range(r, l):
                if K_copy1[i][c] != 0:
                    i_pivot = i
                    break
            if i_pivot == -1: continue

            # Swap
            K_copy1[r], K_copy1[i_pivot] = K_copy1[i_pivot], K_copy1[r]
            
            # Normalize
            inv_pivot = mod_inv(K_copy1[r][c], p)
            for j in range(total_cols):
                K_copy1[r][j] = (K_copy1[r][j] * inv_pivot) % p
                
            # Eliminate
            for i in range(l):
                if i == r: continue
                factor = K_copy1[i][c]
                for j in range(total_cols):
                    K_copy1[i][j] = (K_copy1[i][j] - factor * K_copy1[r][j]) % p
            r += 1
        
        # Check for a solution
        for row in K_copy1:
            if row[l:total_cols].count(0) > 0:
                # This row has at least l-1 zeros from the diagonal block
                # and at least 1 zero in the second block.
                # Let's count total zeros to be sure.
                if row.count(0) >= l:
                    return row

        # --- Attempt 2: Diagonalize second block (cols l to 2l-1) ---
        K_copy2 = LinearAlgebraFp._create_matrix_copy(K_basis)
        
        r = 0
        for c in range(l, total_cols):
            if r >= l: break
            
            i_pivot = -1
            for i in range(r, l):
                if K_copy2[i][c] != 0:
                    i_pivot = i
                    break
            if i_pivot == -1: continue

            K_copy2[r], K_copy2[i_pivot] = K_copy2[i_pivot], K_copy2[r]
            
            inv_pivot = mod_inv(K_copy2[r][c], p)
            for j in range(total_cols):
                K_copy2[r][j] = (K_copy2[r][j] * inv_pivot) % p
                
            for i in range(l):
                if i == r: continue
                factor = K_copy2[i][c]
                for j in range(total_cols):
                    K_copy2[i][j] = (K_copy2[i][j] - factor * K_copy2[r][j]) % p
            r += 1

        # Check for a solution
        for row in K_copy2:
            if row[0:l].count(0) > 0:
                if row.count(0) >= l:
                    return row
                    
        # No solution found by this heuristic
        return None


# ==============================================================================
#
#  Part 4: The Las Vegas ECDLP Algorithm
#
# ==============================================================================

class LasVegasECDLP:
    """
    Implements the full algorithm based on the paper.
    
    Args:
        curve: An EllipticCurve object
        G: The base point (generator)
        Q: The public point (Q = d*G)
        n: The order of the subgroup generated by G
    """
    
    def __init__(self, curve: EllipticCurve, G: Point, Q: Point, n: int):
        self.curve = curve
        self.G = G
        self.Q = Q
        self.n = n  # Group order
        self.p = curve.p # Field prime
        self.rng = random.SystemRandom()

    def _build_monomial_row(self, point: Point, n_prime: int) -> List[int]:
        """
        [cite_start]Creates a row for matrix M from a point [cite: 75, 79]
        Assumes projective coords (x, y, 1)
        """
        x, y, z = point.x, point.y, 1
        
        if n_prime == 2:
            # Monomials: x^2, y^2, z^2, xy, xz, yz
            # With z=1: x^2, y^2, 1, xy, x, y
            return [
                (x*x) % self.p,
                (y*y) % self.p,
                1,
                (x*y) % self.p,
                x,
                y
            ]
        elif n_prime == 3:
            # Monomials: x^3, y^3, z^3, x^2y, x^2z, y^2x, y^2z, z^2x, z^2y, xyz
            # With z=1: x^3, y^3, 1, x^2y, x^2, y^2x, y^2, x, y, xy
            x2, y2 = (x*x) % self.p, (y*y) % self.p
            x3, y3 = (x*x2) % self.p, (y*y2) % self.p
            return [
                x3, y3, 1,
                (x2*y) % self.p, x2,
                (y2*x) % self.p, y2,
                x, y,
                (x*y) % self.p
            ]
        else:
            # This can be generalized, but hard-coded for now
            raise ValueError(f"n_prime = {n_prime} not supported. Use 2 or 3.")

    def _build_matrix_M(self, n_prime: int, l: int) -> Tuple[List[List[int]], List[int], List[int]]:
        """
        [cite_start]Generates random points and builds the matrix M [cite: 133-151]
        """
        num_cols = (n_prime + 1) * (n_prime + 2) // 2
        num_rows_G = 3 * n_prime - 1
        num_rows_Q = l + 1
        
        M = []
        I_list = [] # List of r for r*G
        J_list = [] # List of r for -r*Q
        
        used_r_G = set()
        used_r_Q = set()

        # Generate (3n' - 1) rows from r*G
        for _ in range(num_rows_G):
            point_rG = self.curve.identity()
            r = -1 # Init to invalid value
            
            # Loop until we get a non-identity point
            while point_rG.is_identity():
                r = self.rng.randrange(1, self.n)
                while r in used_r_G:
                    r = self.rng.randrange(1, self.n)
                
                point_rG = self.curve.multiply(self.G, r)
            
            # We now have a valid r and a valid (non-identity) point
            used_r_G.add(r)
            I_list.append(r)
            M.append(self._build_monomial_row(point_rG, n_prime))


        # Generate (l + 1) rows from -r*Q
        for _ in range(num_rows_Q):
            point_minus_rQ = self.curve.identity()
            r = -1 # Init to invalid value

            # Loop until we get a non-identity point
            while point_minus_rQ.is_identity():
                r = self.rng.randrange(1, self.n)
                while r in used_r_Q:
                    r = self.rng.randrange(1, self.n)
                
                point_minus_rQ = self.curve.multiply(self.Q, -r)

            # We now have a valid r and a valid (non-identity) point
            used_r_Q.add(r)
            J_list.append(r)
            M.append(self._build_monomial_row(point_minus_rQ, n_prime))
            
        return M, I_list, J_list

    def solve(self, n_prime: int, max_tries: int = 50) -> Tuple[Optional[int], int]:
        """
        Attempts to solve the ECDLP
        
        Args:
            n_prime: The 'n'' parameter from the paper. Small integer (e.g., 2 or 3)
            max_tries: Number of times to repeat Algorithm 1
            
        Returns:
            A tuple (found_d, attempts):
            - found_d: The discovered secret (int) or None if not found.
            - attempts: The number of attempts taken.
        """
        
        l = 3 * n_prime
        print(f"--- Starting ECDLP Solver ---")
        print(f"Field: F_{self.p}, Group Order: {self.n}")
        print(f"Params: n' = {n_prime}, l = {l}")
        
        for attempt in range(1, max_tries + 1):
            print(f"\n[Attempt {attempt}/{max_tries}]")
            
            # 1. [cite_start]Build Matrix M and get random multipliers [cite: 133-151]
            print("  Building matrix M...")
            M, I_list, J_list = self._build_matrix_M(n_prime, l)
            
            # 2. [cite_start]Compute Left-Kernel K of M [cite: 152]
            print("  Computing left-kernel K...")
            K_basis = LinearAlgebraFp.find_left_kernel_basis(M, self.p)
            
            if not K_basis:
                print("  Kernel is empty. Retrying...")
                continue
                
            # 3. [cite_start]Solve Problem L (Heuristic) [cite: 153]
            print("  Solving Problem L (Algorithm 2)...")
            v = LinearAlgebraFp.solve_problem_l_heuristic(K_basis, l, self.p)
            
            if v is None:
                print("  Problem L not solved by heuristic. Retrying...")
                continue

            # 4. Solution Found! [cite_start]Calculate A and B [cite: 154-163]
            print("  SUCCESS: Problem L solved. Found vector v.")
            A = 0
            B = 0
            
            # Sum for G points (indices 0 to 3n'-2)
            for i in range(len(I_list)):
                if v[i] != 0:
                    A = (A + I_list[i]) % self.n
                    
            # Sum for Q points (indices 3n'-1 to 3n'+l)
            for i in range(len(J_list)):
                v_index = len(I_list) + i
                if v[v_index] != 0:
                    B = (B + J_list[i]) % self.n
            
            print(f"  A = {A}, B = {B}")

            # 5. [cite_start]Final Calculation: d = A * B^-1 (mod n) [cite: 164]
            # Note: We use mod 'n' (group order) as per ECDLP
            # [cite_start]The paper uses 'p' [cite: 164] [cite_start]because it *defined* p as the group order [cite: 32]
            try:
                B_inv = mod_inv(B, self.n)
                d = (A * B_inv) % self.n
                
                print(f"  B_inv (mod {self.n}) = {B_inv}")
                print(f"  d = (A * B_inv) % n = {d}")
                
                # Verification
                Q_check = self.curve.multiply(self.G, d)
                if Q_check == self.Q:
                    print(f"\n*** VERIFIED! {d} * G = Q ***")
                    return d, attempt # <-- CHANGED: Return d and attempt count
                else:
                    print(f"  VERIFICATION FAILED. {d}*G != Q. Retrying...")
            
            except ValueError:
                print(f"  B = {B} is not invertible (mod {self.n}). Retrying...")
                
        print("\n--- Solver finished. No solution found in max_tries. ---")
        return None, max_tries # <-- CHANGED: Return None and max attempts

# ==============================================================================
#
#  Part 5: Main Execution (Example)
#
# ==============================================================================


# [p, a, b, Gx, Gy, n, d_secret]
# [p, a, b, Gx, Gy, n, d_secret]

test_cases = [
    # Test Case 1: Original Sanity Check (p=17)
    [17, 2, 2, 5, 1, 19, 13],

    # Test Case 2: Small Secret (p=17)
    [17, 2, 2, 5, 1, 19, 5],

    # Test Case 3: Large Secret (p=17)
    [17, 2, 2, 5, 1, 19, 18],

    # Test Case 4: Medium Prime (p=113)
    [113, 1, 6, 2, 4, 127, 42],

    # Test Case 5: Medium Prime (p=97)
    [97, 1, 0, 17, 75, 104, 27],
]

def display_result(testcase, n_prime=2, max_tries=100):
    p, a, b, Gx, Gy, n, d_secret = testcase
    my_curve = EllipticCurve(a, b, p)
    G = Point(Gx, Gy)
    Q = my_curve.multiply(G, d_secret)
    print(f"Curve: y^2 = x^3 + {a}x + {b} (mod {p})")
    print(f"Base G: {G}")
    print(f"Order n: {n}")
    print(f"Public Q: {Q}") 

    print(f"(Secret d is {d_secret} for verification)\n")

    solver = LasVegasECDLP(curve=my_curve, G=G, Q=Q, n=n)
    
    # --- CHANGED: Capture both return values ---
    found_d, attempts = solver.solve(n_prime=n_prime, max_tries=max_tries)
    
    if found_d is not None:
        print(f"\nFinal Result: d = {found_d}")
    else:
        print("\nFinal Result: d not found.")
        
    # --- CHANGED: Return all info for the summary table ---
    return (p, n, d_secret, found_d, attempts, max_tries)


if __name__ == "__main__":

    # List to store results: (p, n, d_secret, found_d, attempts, max_tries)
    results_data = []
    
    # --- Parameters for all tests ---
    N_PRIME = 2
    MAX_TRIES = 100

    for i in range (len(test_cases)):
        print(f"\n=== Test Case {i+1} ===")
        result = display_result(test_cases[i], n_prime=N_PRIME, max_tries=MAX_TRIES)
        results_data.append(result)
        print("\n=========================\n")
        
    
    # --- Print Summary Table ---
    print("\n\n--- FINAL TEST SUMMARY ---")
    print(f"{'Test Case':<12} | {'Params (p, n)':<18} | {'Secret d':<10} | {'Found d':<10} | {'Status':<7} | {'Attempts':<10}")
    print("-" * 84)

    for i, res in enumerate(results_data):
        p, n, d_secret, found_d, attempts, max_tries = res
        
        test_id = f"Test {i+1}"
        params = f"p={p}, n={n}"
        status = "PASS" if d_secret == found_d else "FAIL"
        found_d_str = str(found_d) if found_d is not None else "None"
        attempts_str = f"{attempts}/{max_tries}"
        
        print(f"{test_id:<12} | {params:<18} | {d_secret:<10} | {found_d_str:<10} | {status:<7} | {attempts_str:<10}")
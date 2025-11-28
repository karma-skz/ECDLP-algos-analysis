#include "EC_lasVegas.tcc"
#include "constants.hpp"
#include "utils.hpp"
#include "containment.tcc"

#include <bits/stdc++.h>

template <class _mat_, class FiniteField>
_mat_ get_Ljk_Minor(const _mat_ &M, const int k, const int j)
{
    _mat_ minor;
    minor.SetDims(k + 1, k + 1);

    ulong row_1 = 0, col_1 = 0;
    for (ulong row = j; row < (j + k + 1); ++row)
    {
        for (ulong col = 0; col < k + 1; ++col)
        {
            minor[row_1][col_1] = M[row][col];
            ++col_1;
        }
        ++row_1;
        col_1 = 0;
    }
    return minor;
}

template <class _mat_, class FiniteField>
bool is_ObliqueEliminationSuccessful(const _mat_ &newKer, const _mat_ &L, const _mat_ &U)
{
    int n = newKer.NumRows();
    int end = n - 1;

    // Test L_{k,j} matrices
    for (int j = 1; j <= end; ++j)
    {
        for (int k = 1; k <= (n - j); ++k)
        {
            _mat_ minor = get_Ljk_Minor<_mat_, FiniteField>(L, (k - 1), (j - 1));
            if (IsZero(determinant(minor)))
            {
                cout << "\n L => Procssing - k :: " << (k - 1) << "\t j :: " << (j - 1);
                cout << "\t det :: " << determinant(minor) << endl;
                int a;
                cin >> a;
                return false;
            }
        }
    }

    _mat_ U_transpose;
    U_transpose = transpose(U);
    // Test for U_{k,j} matrices
    for (int j = 1; j <= end; ++j)
    {
        for (int k = 1; k <= (n - j); ++k)
        {
            _mat_ minor = transpose(get_Ljk_Minor<_mat_, FiniteField>(U_transpose, (k - 1), (j - 1)));
            if (IsZero(determinant(minor)))
            {
                cout << "\n U => Procssing - k :: " << (k - 1) << "\t j :: " << (j - 1);
                cout << "\t det :: " << determinant(minor) << endl;
                int a;
                cin >> a;
                return false;
            }
        }
    }

    return true;
}

template <class _mat_, class FiniteField>
void LUdecomposition(const _mat_ &a, _mat_ &l, _mat_ &u, int n)
{
    int i = 0, j = 0, k = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (j < i)
                l[j][i] = 0;
            else
            {
                l[j][i] = a[j][i];
                for (k = 0; k < i; k++)
                {
                    l[j][i] = l[j][i] - l[j][k] * u[k][i];
                }
            }
        }
        for (j = 0; j < n; j++)
        {
            if (j < i)
                u[i][j] = 0;
            else if (j == i)
                u[i][j] = 1;
            else
            {
                u[i][j] = a[i][j] / l[i][i];
                for (k = 0; k < i; k++)
                {
                    u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
                }
            }
        }
    }
}

/**
 * Date : 28th July 2021
 * Oblique Elimination implementation to solve ECDLP
 * This implementation of O.E. is incomplete.
 */
template <class _mat_, class FiniteField>
bool obliqueElimination_old(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    _mat_ newKer = getKernelFromFile<_mat_>(processorId, totalNumberOfProcessors);
    ulong kerColCnt = conv<ulong>(newKer.NumCols());

    ulong columnReduceConstant = 1;
    ulong columnsToBeReduced = kerColCnt / columnReduceConstant;
    ulong cloumnReduceCount = 0;

    _mat_ ker;
    ker = makeKernelFromMatrix<_mat_, FiniteField>(newKer);

    _mat_ orgMat = ker;

    // Step 1: Compute ker = LU
    _mat_ L, U;
    L.SetDims(newKer.NumRows(), newKer.NumCols());
    U.SetDims(newKer.NumRows(), newKer.NumCols());

    LUdecomposition<_mat_, FiniteField>(newKer, L, U, newKer.NumRows());

    // Step 2: Check if Minimum Variable Oblique Elimination is possible
    //         - i.e. Theorem-25 from Paul Gader

    if (is_ObliqueEliminationSuccessful<_mat_, FiniteField>(newKer, L, U))
    {
        cout << "\n O.E. is successful...\n";
    }
    else
    {
        cout << "\n O.E. is NOT possible...\n";
    }

    return false;
}

/**
 * Date : 1st September 2021
 * This function reduces the matrix mat w.r.t. obliqueCnt.
 * This function is implementation of the algorithm from P. Gader paper.
 * Minimum variable oblique elimination.
 */
template <class _mat_, class FiniteField>
_mat_ reduceOblique(int obliqueCnt, _mat_ M)
{
    ulong n = M.NumRows();

    _mat_ L;
    L.SetDims(n, n);

    if (obliqueCnt < 1 || obliqueCnt >= n)
    {
        cerr << "\n in-valid value of obliqueCnt :: " << obliqueCnt << endl;
        return L;
    }

    // Step #1 : Make identity matrix
    for (ulong i = 0; i < n; ++i)
        L[i][i] = 1;

    // Step #2 : Calcuate the number of terms
    const ulong numberOfTerms = ((obliqueCnt + 1) * obliqueCnt) / 2;

    // Step 3 : Compute X_i
    FiniteField X_i[n - 1];
    ulong X_i_cnt = 0;

    // Making the frist few x_i's zero
    for (; X_i_cnt < (n - obliqueCnt - 1); ++X_i_cnt)
        X_i[X_i_cnt] = 0;

    int start = (n - obliqueCnt);
    int end = (n - 1);

    // For-loop to compute each x_i from the solution set
    for (int i = start, j = 0; i <= end; ++i, ++j)
    {
        // Step 3.1 : Compute numerator
        FiniteField numerator = -(M[i][j]);

        FiniteField a_i = conv<FiniteField>(0);

        // k : number of terms in the denominator w.r.t '+'
        int numberOfTermsInDenominator = j + 1;
        for (int k = 0; k < numberOfTermsInDenominator; ++k)
        {
            FiniteField x_i_sum = conv<FiniteField>(1);

            // l : number of elements in each term
            int l_end = numberOfTermsInDenominator - 1 - k;
            for (int l = 0; l < l_end; ++l)
                x_i_sum *= X_i[(n - 1) - obliqueCnt + k + l];

            ulong mat_row = n - obliqueCnt + k - 1;
            a_i += M[mat_row][j] * x_i_sum;
        }

        // Step 3.2 :
        if (IsZero(a_i))
        {
            cout << " This is going to crash now ... denominator :: " << a_i;
            cout << " \t Calulating X_" << i << "\n\n";
        }
        X_i[X_i_cnt++] = numerator / a_i;
    }

    // Step 4  : Compute L
    // Step 4.1 : Fill up the diagonal of L with x1, x2, x3 ... x(n-1)
    ulong row = n - obliqueCnt;
    ulong col = row - 1;
    for (int i = 0; i < obliqueCnt; ++i)
        L[row++][col++] = X_i[(n - 1) - obliqueCnt + i];

    // Step 4.2 : Fill up other entries of L
    if (numberOfTerms > 1)
    {
        // Step 4.3 : Determinine the start row & col
        ulong startRow = (n + 1) - obliqueCnt;
        ulong startCol = startRow - 2;

        for (ulong row = startRow; row < n; ++row)
        {
            FiniteField mulElement = L[row][row - 1];
            ulong colEnd = row - 1;
            for (ulong col = startCol; col < colEnd; ++col)
                L[row][col] = mulElement * L[row - 1][col];
        }
    }

    return L;
}

/**
 * Given oblique is reduced for the given matrix.
 * The reduction is better than MVOE algorithm.
 * The element above is used to reduce the current entry.
 */
template <class _mat_, class FiniteField>
bool reduceOblique_2(int obliqueCnt, _mat_ &M, resultData_2x2 &rD)
{
    ulong matRow = M.NumRows();
    ulong startRow = matRow - obliqueCnt;
    ulong col = 0;

    for (ulong i = startRow; i < matRow; ++i)
    {
        if (!IsZero(M[i - 1][col]))
        {
            FiniteField ele = -M[i][col] / M[i - 1][col];
            M[i] = M[i] + (ele * M[i - 1]);
            ++col;
        }
        else
        {
            rD.row1 = i - 1;
            rD.col1 = col;
            rD.row2 = i;
            rD.col2 = col;

            return true;
        }
    }
    return false;
}

template <class _mat_>
ulong getNumberOfZerosInRow(_mat_ mat, ulong row)
{
    ulong cnt = 0;
    for (ulong i = 0; i < mat.NumCols(); ++i)
        if (IsZero(mat[row][i]))
            ++cnt;
    return cnt;
}

template <class _mat_, class FiniteField>
bool schurComplement_OE(const _mat_ &mat, ulong obliqueCnt, ZZ ordP, const _mat_ orgMat)
{
    ulong row_ = mat.NumRows();
    // Search for a 2-minor
    if (obliqueCnt > 1)
    {
        const ulong row1_start = mat.NumRows() - obliqueCnt;
        const ulong col1_start = obliqueCnt;

        for (ulong row1 = row1_start; row1 < row_; ++row1)
        {
            for (ulong row2 = row1 + 1; row2 < row_; ++row2)
            {
                for (ulong col1 = col1_start; col1 < row_; ++col1)
                {
                    for (ulong col2 = col1 + 1; col2 < row_; ++col2)
                    {
                        FiniteField det = (mat[row1][col1] * mat[row2][col2]) - (mat[row1][col2] * mat[row2][col1]);

                        if (IsZero(det))
                        {
                            _mat_ twoM, twoMOrg;
                            twoM.SetDims(2, 2);

                            twoM[0][0] = mat[row1][col1];
                            twoM[0][1] = mat[row1][col2];

                            twoM[1][0] = mat[row2][col1];
                            twoM[1][1] = mat[row2][col2];

                            {
                                // Make I & J
                                ulong startRow = mat.NumRows() - obliqueCnt - 1;
                                ulong I_cnt = 0, J_cnt = 0;
                                ulong I_dim = (row1 - startRow) + 2;
                                ulong I[I_dim], J[I_dim];

                                for (ulong i = startRow; i < row1; ++i)
                                    I[I_cnt++] = i;

                                I[I_cnt++] = row1;
                                I[I_cnt++] = row2;

                                for (ulong i = 0; i < (I_dim - 2); ++i)
                                    J[i] = i;

                                J[I_dim - 2] = col1;
                                J[I_dim - 1] = col2;
                                _mat_ minor, minor2;
                                minor.SetDims(I_dim, I_dim);
                                minor2.SetDims(I_dim, I_dim);

                                makeMatrixFromRowColCombination(I, J, orgMat, minor);
                                makeMatrixFromRowColCombination(I, J, mat, minor2);
                                if (IsZero(minor))
                                {
                                    cout << "\n orgMatr => minor :: \n"
                                         << minor << endl;
                                    cout << " orgMat => det :: " << conv<FiniteField>(determinant(minor)) << endl;

                                    cout << "\n hPrime => minor2 :: \n"
                                         << minor2 << endl;

                                    cout << " hPrime => det :: " << conv<FiniteField>(determinant(minor2)) << endl;
                                    printCombination(I, I_dim);
                                    printCombination(J, I_cnt);

                                    int a;
                                    cin >> a;
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return false;
}

/**
 * Date : 1st September 2021
 * Oblique Elimination implementation
 */
template <class _mat_, class FiniteField>
bool obliqueElimination(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    _mat_ newKer = getKernelFromFile<_mat_>(processorId, totalNumberOfProcessors, ordP);
    ulong kerColCnt = conv<ulong>(newKer.NumCols());

    ulong columnReduceConstant = 1;
    ulong columnsToBeReduced = kerColCnt / columnReduceConstant;
    ulong cloumnReduceCount = 0;

    _mat_ ker;
    ker = makeKernelFromMatrix<_mat_, FiniteField>(newKer);

    _mat_ orgMat = ker;

    ulong matDim = newKer.NumRows();
    _mat_ mat1 = ker;

    masterPrint(processorId) << " mat1.r :: " << mat1.NumRows() << "\t mat1.c :: " << mat1.NumCols();
    masterPrint(processorId) << "\t Looking for r :: " << (mat1.NumCols() / 2) << " zeros in there...\n";
    for (int i = 1; i < (matDim / 2); ++i)
    {
        resultData_2x2 rD;
        if (reduceOblique_2<_mat_, FiniteField>(i, mat1, rD))
        {
            ulong r = (mat1.NumCols() / 2);
            long rowIndex = -1;
            if (isKernelHaving_r_Zeros(mat1, r, rowIndex))
            {
                cout << "\n DLP solved r :: " << r << "\t rowIndex :: " << rowIndex << endl;
                cout << "\n mat1[rowIndex] :: " << mat1[rowIndex] << endl;
                ZZ DLP = solveDLP_rowIndex<_mat_, FiniteField>(mat1, ordP, rowIndex);
                cout << "\n DLP :: " << DLP << endl;
                return true;
            }
        }
        schurComplement_OE<_mat_, FiniteField>(mat1, i, ordP, ker);
    }
    return false;
}
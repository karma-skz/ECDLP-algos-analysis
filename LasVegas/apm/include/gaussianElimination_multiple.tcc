#include <cmath>
#include <math.h>

#include "EC_GF2E.hpp"
#include "EC_ZZp.hpp"
#include "EC_ZZp_Point.hpp"

#include <NTL/mat_ZZ_p.h>
#include <mpi.h>

/**
 * Performs Gaussian Elimination on the second half of the matrix
 * @param arr : The matrix to be reduced
 */
template <class T, class U>
void second_GE(T &mat, ulong col)
{
    const ulong n = mat.NumRows();
    const long mat_col = mat.NumCols();
    try
    {
        ulong curRow = 1;
        cout << "\n n :: " << n << "\t n+col :: " << (n + col) << endl;

        // for (ulong k = n; k < ((2 * n) - 1); ++k)
        for (ulong k = n; k < (n + col); ++k)
        {
            U tmp = mat[curRow - 1][k];
            cout << "\n k :: " << k << "\t tmp :: " << tmp << endl;

            for (ulong i = curRow; i < n; ++i)
            {
                U factor = mat[i][k];
                for (ulong j = 0; j < mat_col; ++j)
                {
                    if ((!IsZero(tmp)) && (!IsZero(factor)))
                    {
                        if (!IsZero(mat[curRow - 1][j]))
                        {
                            if (IsOne(tmp))
                            {
                                mat[i][j] = (factor * mat[curRow - 1][j]) + (mat[i][j]);
                            }
                            else
                            {
                                mat[i][j] = ((factor / tmp) * mat[curRow - 1][j]) + (mat[i][j]);
                            }
                        }
                    }
                }
            }
            curRow++;
        }
    }
    catch (...)
    {
        std::cerr << "\n Exception ";
    }
}

/**
 * This is the re-implementation of the algorithm published in the
 * first Indo-Crypt paper.
 */
template <class FiniteField, class _mat_>
bool gaussianElimination_multiple(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    _mat_ newKer = getKernelFromFile<_mat_>(processorId, totalNumberOfProcessors, ordP);
    ulong kerColCnt = conv<ulong>(newKer.NumCols());

    _mat_ ker;
    ker = makeKernelFromMatrix<_mat_, FiniteField>(newKer);
    ulong r = ker.NumCols() / 2;
    bool isSolvedByFirst_GE = false;
    bool isSolvedBySecond_GE = false;

    // 1. First Gaussian elimination
    for (ulong i = 1; i < ker.NumRows(); ++i)
    {
        gauss(ker, i);
        long rowIndex = -1;
        // 2. Check for required number of zeros
        if (isKernelHaving_r_Zeros<_mat_>(ker, r, rowIndex))
        {
            // 2.2 Solve DLP
            ZZ PQ_randomNumbers[ker.NumCols()];
            getRandomNumbersFromFile(processorId, totalNumberOfProcessors, PQ_randomNumbers, ordP);
            ulong k_randomNums = (ker.NumCols() / 2) - 1;
            ulong t_randomNums = (ker.NumCols() / 2) + 1;

            ZZ_p DLP = conv<ZZ_p>(getDlp(ker, rowIndex, k_randomNums, t_randomNums, PQ_randomNumbers, ordP));

            // cout << "\n ker :: \n"
            //      << ker << endl;

            cout << "\nFIRST - G.E. - DLP :: " << DLP << "\t row-reduce cnt :: " << i << "\t rowIndex :: " << rowIndex << "\t pId :: " << processorId << endl;

            // isSolvedByFirst_GE = true;

            MPI_Abort(MPI_COMM_WORLD, 73);
            return true;
        }
    }
    return false;
    // ---------- Second G.E. ----------

    // long rowIndex = -1;
    // 4. Check for required number of zeros
    // if (isKernelHaving_r_Zeros<mat_ZZ_p>(ker, r, rowIndex))
    // {
    //     // 4.1 Solve DLP
    //     ZZ PQ_randomNumbers[ker.NumCols()];
    //     getRandomNumbersFromFile(processorId, totalNumberOfProcessors, PQ_randomNumbers);
    //     ulong k_randomNums = (ker.NumCols() / 2) - 1;
    //     ulong t_randomNums = (ker.NumCols() / 2) + 1;
    //     ZZ_p DLP = conv<ZZ_p>(getDlp(ker, rowIndex, k_randomNums, t_randomNums, PQ_randomNumbers, ordP));
    //     cout << "\n 2-DLP :: " << DLP << "\t rowIndex :: " << rowIndex << endl;
    //     cout << "\n 2-ker :: " << ker[rowIndex] << endl;
    //     cout << "\n SOLVED - SECOND G.E. \n";
    //     isSolvedBySecond_GE = true;
    // }
}
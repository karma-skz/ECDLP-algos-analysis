#include <cmath>
#include <math.h>

#include "EC_GF2E.hpp"
#include "EC_ZZp.hpp"
#include "EC_ZZp_Point.hpp"

#include <NTL/mat_ZZ_p.h>
#include <mpi.h>

void testRow(const mat_ZZ_p &mat, ulong row1, ulong col, ulong row2, ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ZZ_p vec[mat.NumCols()];

    if (IsZero(mat[row1][col]))
    {
        cout << "\n Divisior Zero.... Noooooo...\n";
        return;
    }

    ZZ_p c = (mat[row2][col] * (1 / mat[row1][col]));

    for (ulong i = 0; i < mat.NumCols(); ++i)
    {
        vec[i] = mat[row2][i] - (c * mat[row1][i]);
    }

    ulong r = mat.NumCols() / 2;

    ulong zeroCnt = 0;
    for (ulong i = 0; i < mat.NumCols(); ++i)
    {
        if (IsZero(vec[i]))
            ++zeroCnt;
    }

    if (zeroCnt == r)
    {
        ulong col2;

        for (ulong i = 0; i < r; ++i)
        {
            if (i != col && IsZero(vec[i]))
            {
                col2 = i;
                break;
            }
        }

        cout << "\n zeroCnt :: " << zeroCnt << "\t r :: " << r << "\t row1 :: " << row1 << "\t col1 :: " << col << "\t row2 :: " << row2 << "\t col2 :: " << col2 << endl;

        // Solving the DLP
        mat_ZZ_p newMat;
        newMat = mat;

        for (ulong i = 0; i < mat.NumCols(); ++i)
            newMat[row2][i] = vec[i];

        ZZ PQ_randomNumbers[newMat.NumCols()];
        getRandomNumbersFromFile(processorId, totalNumberOfProcessors, PQ_randomNumbers, ordP);

        ulong rowIndex = row2;
        ulong k_randomNums = (newMat.NumCols() / 2) - 1;
        ulong t_randomNums = (newMat.NumCols() / 2) + 1;

        ZZ_p DLP = conv<ZZ_p>(getDlp(newMat, rowIndex, k_randomNums, t_randomNums, PQ_randomNumbers, ordP));

        cout << "\n DLP :: " << DLP << " @ pID :: " << processorId << endl;

        // mat_ZZ_p minor;
        // minor.SetDims(2, 2);
        // minor[0][0] = mat[row1][col];
        // minor[0][1] = mat[row1][col2];
        // minor[1][0] = mat[row2][col];
        // minor[1][1] = mat[row2][col2];

        // cout << "\n minor :: \n"
        //      << minor << endl;
        // cout << "\n det(minor) :: " << determinant(minor) << endl;
        // cout << "\n row1 :: " << mat[row1] << endl;
        // cout << "\n row2 :: " << mat[row2] << endl;
    }

    delete NodeName;
}
void testElement(const mat_ZZ_p &mat, ulong row, ulong col, ZZ ordP)
{
    for (ulong i = (row + 1); i < mat.NumRows(); ++i)
    {
        if (i != row)
            testRow(mat, row, col, i, ordP);
    }
}

/**
 * Instead of executing G.E. algorithm.
 * Here we test a row with all other rows along with
 * testing all entries of a row as the pivot.
 */
void GE_allRowAllPivot(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    mat_ZZ_p newKer = getKernelFromFile<mat_ZZ_p>(processorId, totalNumberOfProcessors, ordP);
    ulong kerColCnt = conv<ulong>(newKer.NumCols());

    mat_ZZ_p ker;
    ker = makeKernelFromMatrix<mat_ZZ_p, ZZ_p>(newKer);
    ulong r = ker.NumCols() / 2;

    ulong rowCnt = 0;

    while (rowCnt < r)
    {
        gauss(ker, rowCnt);

        for (ulong i = rowCnt; i < ker.NumRows(); ++i)
        {
            for (ulong j = 0; j < ker.NumCols() / 2; ++j)
            {
                testElement(ker, i, j, ordP);
            }
        }
    }
}
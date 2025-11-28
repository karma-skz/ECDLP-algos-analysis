#include "EC_GF2E.hpp"
#include "EC_ZZp.hpp"
#include "EC_GF2E_Point.hpp"
#include "EC_lasVegas.tcc"

#include "MPI_utils.hpp"

#include <iomanip>
#include <NTL/matrix.h>
#include <NTL/mat_GF2E.h>

#include "constants.hpp"

template <class _mat_>
bool IsKernelVaid(const _mat_ &nonIdentityKernel)
{
    for (ulong i = 0; i < nonIdentityKernel.NumRows(); ++i)
        if (IsZero(nonIdentityKernel[i][0]) || IsZero(nonIdentityKernel[i][1]) || IsZero(nonIdentityKernel[i][2]))
            return false;

    for (ulong i = 0; i < nonIdentityKernel.NumRows(); ++i)
        for (ulong j = 0; j < nonIdentityKernel.NumCols(); ++j)
            if (IsZero(nonIdentityKernel[i][j]))
                return false;

    return true;
}

/**
 * @param P : Point P
 * @param Q : Point Q i.e. Q = mP
 * @param ordP : Order of P
 * @return : DLP
 */
template <class _EC_Point_, class V, class _mat_, class FiniteField>
void makeKernelDB(_EC_Point_ &P, _EC_Point_ &Q, ZZ ordP, ulong _p, const int offset, V *EC, ulong numberOfKernelsToGenerate)
{
    int processorId;
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);

    if (processorId != MASTER_NODE)
        return;

    cout << "\n Generating " << numberOfKernelsToGenerate << " kernels on MASTER_NODE => pId :: " << processorId << endl;

    const ulong n = offset * _p;
    const ulong r = 3 * n;

    const ulong k_randomNums = (3 * n) - 1, t_randomNums = r + 1;
    const ulong mat_row = r + r, mat_col = ((n + 1) * (n + 2)) / 2;

    ZZ PQ_randomNumbers[(k_randomNums + t_randomNums)];

    ulong weightedVector_arr[mat_col][3];
    generateWeightedVector(n, weightedVector_arr);

    ulong iterationCnt = 1;
    ulong accidentCnt = 0;

    cout << "\n iterationCnt :: " << iterationCnt << "\t numberOfkernesTOGenerate :: " << numberOfKernelsToGenerate << endl;
    while (iterationCnt <= numberOfKernelsToGenerate)
    {
        cout << " Processing kernel :: " << iterationCnt << endl;

        double s_time = GetTime();
        generateRandomNumbers(mat_row, PQ_randomNumbers, ordP);

        _mat_ M;
        M.SetDims(mat_row, mat_col);
        double time_MStart = GetTime();
        int result = generateMatrix(M, P, Q, k_randomNums, t_randomNums, PQ_randomNumbers, weightedVector_arr, EC);
        double time_MEnd = GetTime();
        if (result == 1)
        {
            accidentCnt++;
            continue;
        }

        _mat_ ker;
        double start_kTime = GetTime();
        kernel(ker, M);
        double end_kTime = GetTime();

        if (ker.NumRows() == 0)
            continue;

        if (ker.NumRows() < r)
        {
            continue;
        }

        _mat_ nonIdentityKernel = getNonIdentityMatrixFromKernel(ker);

        if (!IsKernelVaid(nonIdentityKernel))
            continue;

        char *kernelFileName = new char[200];
        sprintf(kernelFileName, "kernel_DB/new/kernel_%u_%u.txt", _p, iterationCnt);
        cout << " FILE-NAME :: " << kernelFileName << endl;
        ofstream kernelFile(kernelFileName);

        kernelFile << nonIdentityKernel;
        kernelFile.close();

        char *randomNumberfileName = new char[200];
        sprintf(randomNumberfileName, "kernel_DB/new/kernel_%u_%u_RN.txt", _p, iterationCnt);
        saveRandomNumberToFile(randomNumberfileName, PQ_randomNumbers, mat_row);
        iterationCnt++;
    }
}
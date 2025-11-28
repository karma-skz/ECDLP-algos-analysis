#include "EC_lasVegas.tcc"
#include "constants.hpp"
#include "utils.hpp"
#include "containment.tcc"

#include <bits/stdc++.h>

/**
 * This function spoils the fun.
 * Template paramater FiniteField can take only two values
 * GF2E and ZZ_p
 * Add more else if(...) conditions for other finite-fields.
 */
template <class _mat_, class FiniteField>
_mat_ makeKernelFromMatrix(const _mat_ &ker)
{
    _mat_ newKer;
    ulong rowCnt = 0;
    newKer.SetDims(ker.NumRows(), (ker.NumCols() * 2));

    for (int i = 0; i < newKer.NumRows(); ++i)
    {
        for (int j = 0; j < ker.NumCols(); ++j)
        {
            if (i < ker.NumCols())
            {
                newKer[i][j] = ker[i][j];
            }
        }

        if (typeid(FiniteField) == typeid(GF2E))
        {
            newKer[i][(newKer.NumCols() - i - 1)] = conv<FiniteField>("[1]");
        }
        else if (typeid(FiniteField) == typeid(ZZ_p))
        {
            newKer[i][(newKer.NumCols() - i - 1)] = conv<FiniteField>("1");
        }
        else
        {
            cout << "\n FiniteField (" << typeid(FiniteField).name() << ") not supported w.r.t. IDENTITY Element...";
            cout << "\n Exiting program from makeKernelFromMatrix() " << endl;
            exit(0);
        }
    }
    return newKer;
}

/**
 * After the kernels are generated each processor will process seperate
 * kernels instead of all processing the same kernel.
 * This will also have ability for processing minors of any valid dimention
 */
template <class _mat_, class FiniteField>
bool schurComplement_serial(ZZ ordP)
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
    while (cloumnReduceCount < columnsToBeReduced)
    {
        ker = orgMat;
        gauss(ker, cloumnReduceCount);

        // Get H' from the reduced matrix
        _mat_ hPrime;
        hPrime.SetDims(ker.NumRows() - cloumnReduceCount, ker.NumRows() - cloumnReduceCount);
        getSubMatrix_extended(ker, hPrime, cloumnReduceCount);

        // bool flag = processAllSubMatricesOfDimension_FirstContinousRow<_mat_>(7, hPrime);
        // if (flag)
        //     break;

        resultData_2x2 rD;
        partitionData_2x2 pD;
        pD.i_start = 0;
        pD.j_start = 1;
        pD.quota = -1;
        // isMinorPresent<_mat_>(hPrime, 9, 9);
        if (is_2by2_DeterminantZero_2<_mat_, FiniteField>(hPrime, pD, rD, ker, ordP))
        {
            // cout << " @pId :: " << processorId << "\t r1 :: " << rD.row1 << "\t r2 :: " << rD.row2 << "\t c1 :: " << rD.col1 << "\t c2 :: " << rD.col2 << "\t cloumnReduceCount :: " << cloumnReduceCount << endl;
            return true;
        }

        cloumnReduceCount++;
    }
    return false;
}
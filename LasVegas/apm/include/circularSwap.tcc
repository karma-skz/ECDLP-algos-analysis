#include "EC_lasVegas.tcc"
#include "constants.hpp"
#include "utils.hpp"
#include "containment.tcc"

#include <bits/stdc++.h>

template <class _mat_>
_mat_ circulatShiftMatrixRow(const _mat_ &kernel2)
{
    ulong row = kernel2.NumRows();
    _mat_ newMat;
    newMat.SetDims(row, row);
 
    for (ulong i = 0; i < (row - 1); ++i)
        newMat[i] = kernel2[i + 1];

    newMat[row - 1] = kernel2[0];
    return newMat;
}

template <class _mat_, class FiniteField>
void schurComplement_circularSwap(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong fileId = 0;

    // Iterate over all the kernel files.
    for (ulong i = 0; i < totalNumberOfProcessors; ++i)
    {

        // Loop to swap all rows of ker
        _mat_ ker;
        ker = getKernelFromFile<_mat_>(i, totalNumberOfProcessors);
        for (ulong j = 0; j < ker.NumRows(); j++)
        {
            masterPrint(processorId) << " Processing fileID :: " << i << "\t swap-count :: " << j
                                     << " / " << ker.NumRows() << endl;

            if (processorId == MASTER_NODE)
            {
                if (ker.NumRows() != ker.NumCols())
                {
                    cout << "\n ker row != col ==> asd jkl \n";
                    int lll;
                    cin >> lll;
                }

                //Circular Swap the kernel
                ker = circulatShiftMatrixRow<_mat_>(ker);
            }
            // schurComplement_internal<_mat_, FiniteField>(ker, ordP);
        }
    }
}

#include "EC_GF2E.hpp"
#include "EC_ZZp.hpp"
#include "EC_GF2E_Point.hpp"
#include "EC_lasVegas.tcc"

#include "MPI_utils.hpp"

#include <iomanip>
#include <NTL/matrix.h>
#include <NTL/mat_GF2E.h>

#include "constants.hpp"

/**
 * LasVegas algorithm
 * High Level Algorithm
 *   Step 1 :
 *   Step 2 :
 *   Step 3 :
 *   Step 4 :
 *
 * @param P : Point P
 * @param Q : Point Q i.e. Q = mP
 * @param ordP : Order of P
 * @return : DLP
 */
template <class _EC_Point_, class V, class _mat_, class FiniteField>
ZZ lasVegas(_EC_Point_ &P, _EC_Point_ &Q, ZZ ordP, ulong _p, const int offset, V *EC)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong kernelCnt = 0;
    ulong iterationCnt = 0;
    bool flag = false;
    bool flagArr[totalNumberOfProcessors];
    for (ulong i = 0; i < totalNumberOfProcessors; ++i)
        flagArr[i] = false;

    // Generate kernels unitll DLP is solved
    while (iterationCnt < conv<ulong>(SqrRoot(ordP)))
    {
        masterPrint(processorId) << "\n Iteration cnt :: " << (iterationCnt + 1) << endl;
        iterationCnt += totalNumberOfProcessors;
        if (!flag)
        {
            // Each processor generates a kernel
            double kerGenerationTime = GetTime();
            masterPrint(processorId) << "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            masterPrint(processorId) << "\n Generating Kernel's  ";
            cout.flush();

            SetSeed(to_ZZ(processorId + time(0)));
            genetateKernels<_EC_Point_, V, _mat_>(P, Q, ordP, _p, offset, EC);

            kernelCnt += totalNumberOfProcessors;
            masterPrint(processorId) << "\t Time :: " << (GetTime() - kerGenerationTime) << " seconds."
                                     << "   KernelCnt :: " << kernelCnt << endl;

            MPI_Barrier(MPI_COMM_WORLD);

            // Second part of the LasVegas algorithm to be implemented here
            // flag = principleDeviation<_mat_, FiniteField>(ordP);
            // flag = principleDeviation_parallel_8<_mat_, FiniteField>(ordP);
            // flag = principleDeviation_parallel<_mat_, FiniteField>(ordP);

            flag = principleDeviation_parallel_3<_mat_, FiniteField>(ordP);

            // flag = reverse_obliqueElimination<_mat_, FiniteField>(ordP);
            // flag = obliqueElimination<_mat_, FiniteField>(ordP);
            // schurComplement_serial<_mat_, FiniteField>(ordP);

            // flag = gaussianElimination_multiple<FiniteField, _mat_>(ordP);

            // flag = bruteForce_AllMinor_Parallel<_mat_, FiniteField>(ordP);

            if (flag)
                cout << "\n iterationCnt :: " << iterationCnt << "\t @ pID :: " << processorId << endl;
        }
        else
        {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // This is to make it work in case of serial execution
        MPI_Allgather(&flag, 1, MPI_C_BOOL, flagArr, 1, MPI_C_BOOL, MPI_COMM_WORLD);

        // bool ansFlag = true;
        // for (ulong i = 0; i < totalNumberOfProcessors; ++i)
        //     ansFlag = ansFlag & flagArr[i];

        // if (!flag)
        //     cout << "\n flag :: " << flag << "\t pId :: " << processorId << "\t ansFlag :: " << ansFlag << endl;

        // if (ansFlag)
        //     break;

    } // END : generate kernel untill ECDLP solved
    // cout << "\n Total [FINAL] iterationCnt :: " << iterationCnt << " @ pId :: " << processorId << endl;
    // for (ulong i = 0; i < totalNumberOfProcessors; ++i)
    //     cout << flagArr[i] << " ";
    // cout << endl;

    delete NodeName;

    return conv<ZZ>(0);
}
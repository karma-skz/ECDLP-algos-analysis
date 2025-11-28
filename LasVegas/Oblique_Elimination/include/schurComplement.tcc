#include "EC_lasVegas.tcc"
#include "constants.hpp"
#include "utils.hpp"
#include "containment.tcc"

#include <bits/stdc++.h>

void printPartitionData2x2(const partitionData_2x2 pD, const int processorId)
{
    cout << " Processor :: " << processorId << " >> i_start :: " << pD.i_start
         << "\t j_start :: " << pD.j_start << "\t quota :: " << pD.quota << endl;
}

/**
 * Return a matrix such that the returned matrix does not contain the first n
 * row and columns from the original matrix.
 * @param orgMat : original matrix
 * @param mat : resultant matrix
 * @param n : number of columns and rows not to be included in the resultant mat
 */
template <class _mat_>
void getSubMatrix(const _mat_ &orgMat, _mat_ &mat, ulong n)
{
    for (int i = 0; i < mat.NumRows(); i++)
    {
        for (int j = 0; j < mat.NumRows(); j++)
        {
            mat[i][j] = orgMat[i + n][j + n];
        }
    }
}

template <class _mat_>
void getSubMatrix_extended(const _mat_ &orgMat, _mat_ &mat, ulong n)
{
    for (int i = 0; i < mat.NumRows(); i++)
    {
        for (int j = 0; j < mat.NumRows(); j++)
        {
            mat[i][j] = orgMat[i + n][j + n];
        }
    }

    ulong mat_col = orgMat.NumRows() - mat.NumRows();

    for (ulong i = (orgMat.NumRows() - mat.NumRows()), row = 0; i < orgMat.NumRows(), row < mat.NumRows(); i++, ++row)
    {
        for (ulong j = (orgMat.NumCols() - mat_col), col = mat.NumRows(); j < orgMat.NumCols(), col < mat.NumCols(); j++, ++col)
        {
            mat[row][col] = orgMat[(i)][j];
        }
    }
}

//---------------------------------------------------

template <class _mat_>
void bCastMatrixSend(const _mat_ &mat)
{
    stringstream ss;
    ss << mat;

    std::string s_i = ss.str();
    int strLen = s_i.length() + 1; // +1; Maybe important.
    char *ker_str = new char[strLen];
    strcpy(ker_str, s_i.c_str());

    //Broadcast  size and the ker.
    MPI_Bcast(&strLen, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(ker_str, strLen, MPI_CHAR, 0, MPI_COMM_WORLD);

    ss.clear();
    delete ker_str;
}

template <class _mat_>
void bCastMatrixRecv(_mat_ &mat)
{
    int strLen;

    MPI_Bcast(&strLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char *strKer_Ci = new char[strLen];

    MPI_Bcast(strKer_Ci, strLen, MPI_CHAR, 0, MPI_COMM_WORLD);

    _mat_ hPrime;
    stringstream ss;

    ss << strKer_Ci;
    ss >> mat;
    delete strKer_Ci;
    ss.clear();
}

//---------------------------------------------------

/**
 * This function sends partition data to all Slaves.
 * sendPartitionDataToProcessors_2x2() sends to all (slaves+master)
 */
void sendPartitionDataToSlaveProcessors_2x2(partitionData_2x2 pD[], ulong totalNumberOfProcessors)
{
    for (int i = 1; i < totalNumberOfProcessors; ++i)
    {
        ulong arr[3];
        arr[0] = pD[i].i_start;
        arr[1] = pD[i].j_start;
        arr[2] = pD[i].quota;
        MPI_Send(arr, 3, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
    }
}

void receivepartitionDataFromMaster_2x2(partitionData_2x2 &pD, ulong processorId)
{
    ulong arr[3];
    MPI_Recv(arr, 3, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    pD.i_start = arr[0];
    pD.j_start = arr[1];
    pD.quota = arr[2];
}

// template <class _mat_, class FiniteField>
// void schurComplement_internal(_mat_ ker, ZZ ordP)
// {
//     int processorId, totalNumberOfProcessors;
//     char *NodeName = new char[1000];
//     MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

//     if (processorId == MASTER_NODE)
//     {
//         ulong kerColCnt = conv<ulong>(ker.NumCols());
//         ulong columnReduceConstant = 2;
//         ulong columnsToBeReduced = (kerColCnt / columnReduceConstant);

//         // BCast () & columnsToBeReduced
//         MPI_Bcast(&kerColCnt, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
//         MPI_Bcast(&columnReduceConstant, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

//         ulong cloumnReduceCount = 1;
//         while (cloumnReduceCount < columnsToBeReduced)
//         {
//             cout << "\n Before Gauss...\n";
//             gauss(ker, cloumnReduceCount);
//             cout << "\n After Gauss...\n";
//             cout << "\n ker :: \n"
//                  << ker << endl;
//             _mat_ hPrime;
//             hPrime.SetDims((ker.NumRows() - cloumnReduceCount), (ker.NumCols() - cloumnReduceCount));
//             cout << "\n Before getSubMatrix...\n";
//             getSubMatrix<_mat_>(ker, hPrime, cloumnReduceCount);
//             cout << "\n cloumnReduceCount :: " << cloumnReduceCount << "\t of columnsToBeReduced :: " << columnsToBeReduced
//                  << "\t ker.Col :: " << kerColCnt << "\t H.r :: " << hPrime.NumRows() << "\t H.c :: " << hPrime.NumCols() << endl;

//             ulong hPrime_Row = hPrime.NumRows();
//             // Temporary escape hatch
//             if (hPrime_Row <= totalNumberOfProcessors)
//             {
//                 cout << "\n hPrime.R :: " << hPrime.NumRows() << "\t totalNumberOfProcessors :: " << totalNumberOfProcessors << endl;
//                 int asdasda;
//                 cin >> asdasda;
//             }

//             // send hPrime to slave
//             // bCastMatrixSend(hPrime);

//             //Compute and Send partition data to all processors
//             partitionData_2x2 pD[totalNumberOfProcessors];
//             cout << "\n Before going in...\n";
//             // computePartitionData_2x2(hPrime_Row, pD);
//             {
//                 ulong r = hPrime_Row;
//                 cout << "\n 1.ONE.,..\n";

//                 // int processorId, totalNumberOfProcessors;
//                 cout << "\n 2.ONE.,..\n";

//                 // char *NodeName = new char[1000];
//                 cout << "\n 3.ONE.,..\n";

//                 // MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
//                 cout << "\n 4.ONE.,..\n";
//                 ZZ combos = nCr(r, 2);
//                 cout << "\n 5.ONE.,..\n";
//                 cout << "\n combos :: " << combos << "\t r :: " << r << "\t tnp :: " << totalNumberOfProcessors << endl;
//                 cout << "\n @@@@ :: " << combos / (conv<ZZ>(totalNumberOfProcessors)) << endl;
//                 ZZ numberOfCombosEachProcessorGets = conv<ZZ>(combos / (conv<ZZ>(totalNumberOfProcessors)));
//                 // ZZ numberOfCombosEachProcessorGets = conv<ZZ>(combos / conv<ZZ>(totalNumberOfProcessors));

//                 cout << "\n 6.ONE.,..\n";

//                 ZZ aaa = conv<ZZ>(numberOfCombosEachProcessorGets) * conv<ZZ>(totalNumberOfProcessors);
//                 ulong numberOfExtraCombos = conv<ulong>(combos - aaa);
//                 cout << "\n 7.ONE...\n";
//                 cout << "\n combos :: " << combos << "\t number-Of-combos-each-processor-gets :: " << numberOfCombosEachProcessorGets
//                      << "\t numberOfExtraCombos :: " << numberOfExtraCombos << endl;

//                 ulong cnt = 0;
//                 ulong pDCnt = 1;
//                 pD[0].i_start = 0;
//                 pD[0].j_start = 1;
//                 pD[0].quota = conv<ulong>(numberOfCombosEachProcessorGets);
//                 cout << "\n TWO.,..\n";
//                 ulong quota = conv<ulong>(numberOfCombosEachProcessorGets);
//                 ulong dimenson = r;
//                 for (ulong i = 0; i < dimenson; ++i)
//                 {
//                     for (ulong j = i + 1; j < dimenson; ++j)
//                     {
//                         cnt++;
//                         if (cnt == quota)
//                         {
//                             pD[pDCnt].i_start = i;
//                             pD[pDCnt].j_start = j;
//                             pD[pDCnt].quota = conv<ulong>(numberOfCombosEachProcessorGets);

//                             pDCnt++;
//                             cnt = 0;
//                         }
//                     }
//                 }
//                 pD[totalNumberOfProcessors - 1].quota += numberOfExtraCombos;
//                 // delete NodeName;
//                 cout << "\n Out of computePartitionData_2x2...\n";
//             }
//             cout << "\n Its here...\n";
//             // sendPartitionDataToSlaveProcessors_2x2(pD, totalNumberOfProcessors);

//             // resultData_2x2 rD;
//             // if (is_2by2_DeterminantZero<_mat_, FiniteField>(hPrime, pD[0], rD))
//             // {
//             //     cout << " @ZM @pId :: " << processorId << "\t r1 :: " << rD.row1 << "\t r2 :: " << rD.row2
//             //          << "\t c1 :: " << rD.col1 << "\t c2 :: " << rD.col2
//             //          << "\t cloumnReduceCount :: " << cloumnReduceCount << " of " << columnsToBeReduced << endl;
//             //     // MPI_Abort(MPI_COMM_WORLD, 73);
//             // }
//             ++cloumnReduceCount;
//         }
//     }
//     else
//     {
//         //BCast ker.NumCols() & columnsToBeReduceded
//         ulong kerColCnt, columnReduceConstant, cloumnReduceCount = 1;

//         MPI_Bcast(&kerColCnt, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
//         MPI_Bcast(&columnReduceConstant, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

//         ulong columnsToBeReduced = (kerColCnt / columnReduceConstant);
//         while (cloumnReduceCount < columnsToBeReduced)
//         {
//             //Receive matrix
//             // _mat_ hPrime;
//             // bCastMatrixRecv(hPrime);

//             // partitionData_2x2 pD;
//             // receivepartitionDataFromMaster_2x2(pD, processorId);

//             // resultData_2x2 rD;
//             // if (is_2by2_DeterminantZero<_mat_, FiniteField>(hPrime, pD, rD))
//             // {
//             //     cout << " #ZM @pId :: " << processorId << "\t r1 :: " << rD.row1 << "\t r2 :: " << rD.row2
//             //          << "\t c1 :: " << rD.col1 << "\t c2 :: " << rD.col2
//             //          << "\t cloumnReduceCount :: " << (cloumnReduceCount) << " of " << columnsToBeReduced << endl;
//             //     // MPI_Abort(MPI_COMM_WORLD, 73);
//             // }
//             ++cloumnReduceCount;
//         }

//     } //END::ELSE
//     MPI_Barrier(MPI_COMM_WORLD);
//     masterPrint(processorId) << "\n====================================================\n";
// }

// template <class _mat_, class FiniteField>
// void schurComplement(ZZ ordP)
// {
//     int processorId, totalNumberOfProcessors;
//     char *NodeName = new char[1000];
//     MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

//     ulong fileId = 0;

//     // Iterate over all the kernel files.
//     for (ulong i = 0; i < totalNumberOfProcessors; ++i)
//     {
//         masterPrint(processorId) << " Processing fileID :: " << i << endl;
//         _mat_ ker;
//         if (processorId == MASTER_NODE)
//         {
//             ker = getKernelFromFile<_mat_>(i, totalNumberOfProcessors);
//             if (ker.NumRows() != ker.NumCols())
//             {
//                 cout << "\n ker row != col ==> asd jkl \n";
//                 int lll;
//                 cin >> lll;
//             }
//         }
//         schurComplement_internal<_mat_, FiniteField>(ker, ordP);
//     } //END :: FOR LOOP ITERATING OVER KERNEL-FILES
// }

template <class _mat_, class FiniteField>
bool schurComplement(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong fileId = 0;

    // Iterate over all the kernel files.
    for (ulong i = 0; i < 1; ++i)
    {
        masterPrint(processorId) << " Processing fileID :: " << i << endl;
        double startTime = GetTime();
        if (processorId == MASTER_NODE)
        {
            _mat_ ker = getKernelFromFile<_mat_>(i, totalNumberOfProcessors);
            // _mat_ ker = makeKernelFromMatrix<_mat_, FiniteField>(ker_0);

            if (ker.NumRows() != ker.NumCols())
            {
                cout << "\n ker row != col ==> asd jkl \n";
                int lll;
                cin >> lll;
            }
            ulong kerColCnt = conv<ulong>(ker.NumCols());
            ulong columnReduceConstant = 2;
            ulong columnsToBeReduced = kerColCnt / columnReduceConstant;
            cout << "\n ONE...\n";
            //BCast ker.NumCols() & columnsToBeReduceded
            MPI_Bcast(&kerColCnt, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(&columnReduceConstant, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

            ulong cloumnReduceCount = 1;
            _mat_ orgMat = ker;
            while (cloumnReduceCount < columnsToBeReduced)
            {
                orgMat = ker;
                gauss(orgMat, cloumnReduceCount);

                _mat_ hPrime;
                hPrime.SetDims((orgMat.NumRows() - cloumnReduceCount), (orgMat.NumRows()));
                getSubMatrix<_mat_>(orgMat, hPrime, cloumnReduceCount);
                // send hPrime to slave
                bCastMatrixSend(hPrime);

                //Compute and Send partition data to all processors
                partitionData_2x2 pD[totalNumberOfProcessors];
                computePartitionData_2x2(hPrime.NumRows(), pD);
                sendPartitionDataToSlaveProcessors_2x2(pD, totalNumberOfProcessors);

                resultData_2x2 rD;
                if (is_2by2_DeterminantZero<_mat_, FiniteField>(hPrime, pD[0], rD))
                {
                    // cout << " ZM @pId :: " << processorId << "\t r1 :: " << rD.row1 << "\t r2 :: " << rD.row2 << "\t c1 :: " << rD.col1 << "\t c2 :: " << rD.col2 << "\t cloumnReduceCount :: " << cloumnReduceCount << endl;
                    // MPI_Abort(MPI_COMM_WORLD, 73);
                    return true;
                }
                ++cloumnReduceCount;
            }
        }
        else
        {
            //BCast ker.NumCols() & columnsToBeReduceded
            ulong kerColCnt, columnReduceConstant, cloumnReduceCount = 1;

            MPI_Bcast(&kerColCnt, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(&columnReduceConstant, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

            ulong columnsToBeReduced = kerColCnt / columnReduceConstant;
            while (cloumnReduceCount < columnsToBeReduced)
            {
                //Receive matrix
                _mat_ hPrime;
                bCastMatrixRecv(hPrime);

                partitionData_2x2 pD;
                receivepartitionDataFromMaster_2x2(pD, processorId);

                resultData_2x2 rD;
                if (is_2by2_DeterminantZero<_mat_, FiniteField>(hPrime, pD, rD))
                {
                    // cout << " ZM @pId :: " << processorId << "\t r1 :: " << rD.row1 << "\t r2 :: " << rD.row2 << "\t c1 :: " << rD.col1 << "\t c2 :: " << rD.col2 << "\t cloumnReduceCount :: " << (cloumnReduceCount) << endl;
                    // MPI_Abort(MPI_COMM_WORLD, 73);
                    return true;
                }
                ++cloumnReduceCount;
            }

        } //END::ELSE
        MPI_Barrier(MPI_COMM_WORLD);
        masterPrint(processorId) << "\n Time :: " << GetTime() - startTime << " sec. \n";
        masterPrint(processorId) << "\n====================================================\n";
    } //END :: FOR LOOP ITERATING OVER KERNEL-FILES
    return false;
}
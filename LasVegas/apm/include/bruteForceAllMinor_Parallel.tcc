#include "constants.hpp"
#include "utils.hpp"
#include "containment.tcc"

template <class _mat_, class FiniteField>
bool bruteForce_AllMinor_Parallel(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    //  All processors iterate over all kernel files.
    for (ulong fileId = 0; fileId < totalNumberOfProcessors; ++fileId)
    {
        const _mat_ mat = getKernelFromFile<_mat_>(fileId, totalNumberOfProcessors, ordP);
        ulong kerColCnt = conv<ulong>(mat.NumCols());
        masterPrint(processorId) << "\n Processing File :: " << fileId << "\t mat.r :: " << mat.NumRows() << "\t mat.c :: " << mat.NumCols() << endl;

        ulong matRows = mat.NumRows();
        ulong max_blockSize = matRows - 1;

        char fileName2[200];
        sprintf(fileName2, "output/tmp/ZM_%u_fId_%u.txt", processorId, fileId);
        ofstream fout(fileName2);
        fout << "Processing File :: " << fileName2 << endl;

        for (ulong blockSize = 2; blockSize < max_blockSize; ++blockSize)
        {
            double time_s = GetTime();

            masterPrint(processorId) << " Processing block size :: " << blockSize << " of " << max_blockSize;
            masterPrint(processorId).flush();

            // Each processor will figure out the combinations it has to look at
            // Last processor gets extra (leftover) combinations

            //-------------------------------------------------------------------------------------------------------------------
            ZZ totalRowCombinations = nCr(max_blockSize, blockSize);

            if (totalRowCombinations >= conv<ZZ>("18446744073709551615"))
            {
                cout << "\n nCr :: " << nCr(matRows - blockSize, 2) << endl;
                cout << "\n totalRowCombinations :: " << totalRowCombinations << endl;
                cout << "\n totalRowCombinations > 64 bits.... => bruteForce_AllMinor_Parallel() in bruteForceAllMinor_Parallel.tcc " << endl;
                cout << " The program will now terminate...\n";
                exit(0);
            }

            ZZ totalCombinations = totalRowCombinations * totalRowCombinations;
            ulong quota = conv<ulong>((totalCombinations / conv<ZZ>(totalNumberOfProcessors)));
            ulong extra = conv<ulong>((totalCombinations % conv<ZZ>(totalNumberOfProcessors)));

            ulong symbolVec_dim = max_blockSize;
            ulong symbolVec[symbolVec_dim];
            for (int i = 0; i < symbolVec_dim; ++i)
                symbolVec[i] = i;

            ulong rowCombinationIndex = conv<ulong>((processorId * quota) / conv<ulong>(totalRowCombinations));
            ulong colCombinationIndex = conv<ulong>((processorId * quota) % conv<ulong>(totalRowCombinations));

            ulong rowCombinationVec[blockSize], colCombinationVec[blockSize];

            // cout << "\n rowCombonationIndex :: " << rowCombinationIndex << "\t colCombinationIndex :: " << colCombinationIndex << endl;
            // cout << "\n totalCombinations :: " << totalCombinations << "\t quota :: " << quota << endl;

            get_kth_combination(symbolVec, symbolVec_dim, blockSize, rowCombinationIndex, rowCombinationVec);
            get_kth_combination(symbolVec, symbolVec_dim, blockSize, colCombinationIndex, colCombinationVec);

            if (processorId == totalNumberOfProcessors - 1)
                quota += extra;
            //-------------------------------------------------------------------------------------------------------------------

            ulong zeroMinorCnt = processAllSubMatricesOfDimension_parallel(blockSize, mat, fout, rowCombinationVec, colCombinationVec, quota);

            fout.close();

            MPI_Barrier(MPI_COMM_WORLD);
            double time_e = GetTime();
            masterPrint(processorId) << "\t Time :: " << (time_e - time_s) << " seconds.\n";
        }
    } //END :: for fileId
}
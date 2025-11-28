#include "constants.hpp"
#include "utils.hpp"
#include "containment.tcc"

// This function is in almostPrinciple_deviation_serial.tcc
template <class _mat_, class FiniteField>
bool makeAllAlmostPrincipleMinors(const _mat_ &minor, const _mat_ &mat, ulong blockSize, ulong block_I[]);

template <class _mat_>
bool twoDeviation(ulong blockSize, ulong block_I[], const _mat_ &mat, ulong complementSize, ulong complement_block_I[]);

/**
 * Each slave processors gets the same PM.
 * All slaves look at two deviations before and after this PM
 */
template <class _mat_, class FiniteField>
bool principleDeviation_parallel_3(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
    ulong const numberOfDeviations = 2;
    // Iterate over all the kernel files.
    for (ulong fileId = 0; fileId < totalNumberOfProcessors; ++fileId)
    {
        masterPrint(processorId) << " Processing fileId :: " << (fileId + 1) << "\t of " << totalNumberOfProcessors << endl;

        _mat_ ker = getKernelFromFile<_mat_>(fileId, totalNumberOfProcessors, ordP);

        const ulong matRows = ker.NumRows();
        const ulong max_blockSize = (matRows / 2) + 1;

        masterPrint(processorId) << "\n ker.r :: " << matRows << "\t ker.c :: " << ker.NumCols() << "\t max_blockSize :: " << max_blockSize << endl;

        for (ulong blockSize = 2; blockSize <= max_blockSize; ++blockSize)
        {
            masterPrint(processorId) << " Processing block size :: " << blockSize << "\t of " << max_blockSize << endl;

            const ZZ number_of_APM_half = nCr((matRows - blockSize), numberOfDeviations);
            const ZZ number_of_APM = number_of_APM_half * number_of_APM_half;

            ZZ quota = conv<ZZ>(number_of_APM / totalNumberOfProcessors);
            ulong extra_combo = conv<ulong>(number_of_APM % totalNumberOfProcessors);
            // last processor gets all extra APM
            if (processorId == MASTER_NODE)
                quota += extra_combo;

            // get start row, col combinations for each processor
            ulong symbolVec_dim = (matRows - blockSize);
            ulong symbolVec[symbolVec_dim];
            for (int i = 0; i < symbolVec_dim; ++i)
                symbolVec[i] = i;

            ulong rowCombinationIndex = conv<ulong>((processorId * quota) / conv<ulong>(number_of_APM_half));
            ulong colCombinationIndex = conv<ulong>((processorId * quota) % conv<ulong>(number_of_APM_half));

            ulong rowCombinationVecOrg[numberOfDeviations], colCombinationVecOrg[numberOfDeviations];

            get_kth_combination(symbolVec, symbolVec_dim, numberOfDeviations, rowCombinationIndex, rowCombinationVecOrg);
            get_kth_combination(symbolVec, symbolVec_dim, numberOfDeviations, colCombinationIndex, colCombinationVecOrg);
            MPI_Barrier(MPI_COMM_WORLD);

            ulong block_I[blockSize];
            initCombinations(block_I, blockSize);
            ulong PM_cnt = 1;
            ulong numberOf_PM = (max_blockSize - blockSize) + 1;
            double s_time, e_time;
            while (PM_cnt <= numberOf_PM)
            {
                s_time = GetTime();
                masterPrint(processorId) << " Processing PM(blockSize = " << blockSize << ") :: " << PM_cnt << " of " << numberOf_PM;
                masterPrint(processorId).flush();
                ++PM_cnt;

                // compute complement of block_I
                ulong complementSize = matRows - blockSize;
                ulong complement_block_I[complementSize];
                {
                    ulong cnt = 0;
                    ulong index = 0;
                    while (cnt < matRows)
                    {
                        if (cnt != block_I[0])
                            complement_block_I[index++] = cnt++;
                        else
                            cnt += blockSize;
                    }
                }

                // copy
                ulong rowCombinationVec[numberOfDeviations], colCombinationVec[numberOfDeviations];

                for (ulong i = 0; i < numberOfDeviations; ++i)
                {
                    rowCombinationVec[i] = rowCombinationVecOrg[i];
                    colCombinationVec[i] = colCombinationVecOrg[i];
                }

                // Process all APM
                ZZ APM_cnt = conv<ZZ>("0");
                // cout << "\n APM_cnt :: " << APM_cnt << "\t quota :: " << quota << "\t pId :: " << processorId << endl;

                while (APM_cnt < quota)
                {
                    _mat_ minor;
                    ulong APM_dims = blockSize + numberOfDeviations;
                    minor.SetDims(APM_dims, APM_dims);

                    ulong startRow = block_I[0] < complement_block_I[rowCombinationVec[0]] ? block_I[0] : complement_block_I[rowCombinationVec[0]];
                    ulong startCol = block_I[0] < complement_block_I[colCombinationVec[0]] ? block_I[0] : complement_block_I[colCombinationVec[0]];
                    if (startRow != startCol)
                    {
                        // Merge Block_I with row, col combo vec
                        ulong rowCombo[APM_dims];
                        {
                            ulong block_I_ptr = 0, vec_ptr = 0;
                            for (ulong i = 0; i < APM_dims; ++i)
                            {
                                if (block_I[block_I_ptr] <= complement_block_I[rowCombinationVec[vec_ptr]])
                                    rowCombo[i] = block_I[block_I_ptr++];
                                else
                                    rowCombo[i] = complement_block_I[rowCombinationVec[vec_ptr++]];

                                if (vec_ptr == numberOfDeviations)
                                {
                                    // copy block_I remaining to rowCombo
                                    for (ulong j = i + 1; j < APM_dims; ++j)
                                        rowCombo[j] = block_I[block_I_ptr++];
                                    break;
                                }
                                if (block_I_ptr == blockSize)
                                {
                                    for (ulong j = i + 1; j < APM_dims; ++j)
                                        rowCombo[j] = complement_block_I[rowCombinationVec[vec_ptr++]];
                                    break;
                                }
                            }
                        }

                        ulong colCombo[APM_dims];
                        {
                            ulong block_I_ptr = 0, vec_ptr = 0;
                            for (ulong i = 0; i < APM_dims; ++i)
                            {
                                if (block_I[block_I_ptr] <= complement_block_I[colCombinationVec[vec_ptr]])
                                    colCombo[i] = block_I[block_I_ptr++];
                                else
                                    colCombo[i] = complement_block_I[colCombinationVec[vec_ptr++]];

                                if (vec_ptr == numberOfDeviations)
                                {
                                    // copy block_I remaining to colCombo
                                    for (ulong j = i + 1; j < APM_dims; ++j)
                                        colCombo[j] = block_I[block_I_ptr++];
                                    break;
                                }
                                if (block_I_ptr == blockSize)
                                {
                                    for (ulong j = i + 1; j < APM_dims; ++j)
                                        colCombo[j] = complement_block_I[colCombinationVec[vec_ptr++]];
                                    break;
                                }
                            }
                        }

                        makeMatrixFromRowColCombination(rowCombo, colCombo, ker, minor);

                        if (IsZero(determinant(minor)))
                        {
                            cout << "\n ZM found...pId :: " << processorId << endl;
                            printCombination2(rowCombo, APM_dims);
                            printCombination2(colCombo, APM_dims);
                            printCombination2(complement_block_I, complementSize);
                            printCombination2(block_I, blockSize);
                            printCombination2(rowCombinationVec, numberOfDeviations);
                            printCombination2(colCombinationVec, numberOfDeviations);
                            cout << "\n minor :: \n " << minor << endl;
                            MPI_Abort(MPI_COMM_WORLD, 73);
                        }
                    }

                    ++APM_cnt;
                    if (!isLastCombination(colCombinationVec, numberOfDeviations, complementSize))
                    {
                        _getNextCombination(colCombinationVec, complementSize, numberOfDeviations);
                    }
                    else
                    {
                        if (isLastCombination(rowCombinationVec, numberOfDeviations, complementSize))
                            break;

                        initCombinations(colCombinationVec, numberOfDeviations);
                        _getNextCombination(rowCombinationVec, complementSize, numberOfDeviations);
                    }

                } // END :: while process all apm.

                e_time = GetTime();
                masterPrint(processorId) << "\t Time :: " << (e_time - s_time) << " seconds " << endl;
                MPI_Barrier(MPI_COMM_WORLD);
                if (!isLastCombination(block_I, blockSize, matRows))
                    _getNextCombination_continous(block_I, matRows, blockSize);
                else
                    break;

            } // END :: iteration over PM of blockSize

            // int a;
            // cin >> a;
            MPI_Barrier(MPI_COMM_WORLD);
        } // END :: for blockSize
    }     // END : for fileId
}

/**
 * Each slave processors get a different PM.
 * The slave looks at all the apm for this pm
 */
template <class _mat_, class FiniteField>
bool principleDeviation_parallel_2(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    // Iterate over all the kernel files.
    for (ulong fileId = 0; fileId < totalNumberOfProcessors; ++fileId)
    {
        masterPrint(processorId) << " Processing fileId :: " << (fileId + 1) << "\t of " << totalNumberOfProcessors << endl;

        _mat_ ker = getKernelFromFile<_mat_>(fileId, totalNumberOfProcessors, ordP);

        ulong matRows = ker.NumRows();
        ulong max_blockSize = (matRows / 2) + 1;

        masterPrint(processorId) << "\n ker.r :: " << matRows << "\t ker.c :: " << ker.NumCols() << "\t max_blockSize :: " << max_blockSize << endl;
        for (ulong blockSize = 2; blockSize < max_blockSize; ++blockSize)
        {
            double time_s = GetTime();
            masterPrint(processorId) << "\n Processing blockSize :: " << blockSize << " of " << max_blockSize << endl;

            ulong numberOfBlocks = (matRows - (blockSize - 1)) / 2;
            ulong numberOfBlocks_eachProcessor = numberOfBlocks / totalNumberOfProcessors;

            ulong blockStartCount = processorId * numberOfBlocks_eachProcessor;
            ulong blockEndCount = blockStartCount + numberOfBlocks_eachProcessor;

            ulong extraCombo = numberOfBlocks % totalNumberOfProcessors;

            if (extraCombo >= 1)
                if (processorId < extraCombo)
                {
                    if (processorId != MASTER_NODE)
                        blockStartCount = blockStartCount + processorId;
                    blockEndCount = blockEndCount + processorId + 1;
                }

            MPI_Barrier(MPI_COMM_WORLD);
            masterPrint(processorId) << " numberOfBlocks :: " << numberOfBlocks << "\t numberOfBlocks_eachProcessor :: " << numberOfBlocks_eachProcessor << endl;

            ulong block_I[blockSize];
            initCombinations(block_I, blockSize);
            ulong iterationCnt = 0;
            bool flag = false;
            while (!flag)
            {
                if (iterationCnt >= blockStartCount && iterationCnt < blockEndCount)
                {
                    _mat_ principleMinor;
                    principleMinor.SetDims(blockSize, blockSize);

                    makeMatrixFromRowColCombination(block_I, block_I, ker, principleMinor);

                    flag = makeAllAlmostPrincipleMinors<_mat_, FiniteField>(principleMinor, ker, blockSize, block_I);
                }

                if (!isLastCombination(block_I, blockSize, matRows))
                    _getNextCombination_continous(block_I, matRows, blockSize);
                else
                    break;

                ++iterationCnt;
            }

            double time_e = GetTime();
            masterPrint(processorId) << "\n Block-Size :: " << blockSize << "\t Time :: " << (time_e - time_s) << " Sec. \n";
        } // END : for

        MPI_Barrier(MPI_COMM_WORLD);
        // MPI_Abort(MPI_COMM_WORLD, 72);
    }
    return false;
}
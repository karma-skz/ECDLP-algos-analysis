#include "constants.hpp"
#include "utils.hpp"
#include "containment.tcc"

#include <array>
#include <vector>
#include <algorithm>
#include <random>

// This function is in almostPrinciple_deviation_serial.tcc
template <class _mat_, class FiniteField>
bool makeAllAlmostPrincipleMinors(const _mat_ &minor, const _mat_ &mat, ulong blockSize, ulong block_I[], ulong &dim, ulong row[], ulong col[]);

template <class _mat_, class FiniteField>
ZZ solveDLP_apm(ulong rowCombo[], ulong colCombo[], ulong dim, ulong fileId, const _mat_ &ker, ZZ ordP);

/**
 * @brief This is same as principleDeviation_parallel_3()
 */
template <class _mat_, class FiniteField>
void process_small(ZZ ordP, _mat_ ker, ulong fileId)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    const ulong matRows = ker.NumRows();
    const ulong numberOfParts = 6;
    const ulong max_blockSize = matRows / numberOfParts;
    const ulong blockStartDims = 2;
    const ulong numberOfDeviations_start = 2, numberOfDeviations_end = 3; // both inclusive

    masterPrint(processorId)
        << "\n ker.r :: " << matRows << "\t ker.c :: " << ker.NumCols() << "\t max_blockSize :: " << max_blockSize << endl;

    for (ulong blockSize = blockStartDims; blockSize <= max_blockSize; ++blockSize)
    {
        for (ulong numberOfDeviations = numberOfDeviations_start; numberOfDeviations <= numberOfDeviations_end; ++numberOfDeviations)
        {
            masterPrint(processorId) << " Processing block size :: " << blockSize << "\t of " << max_blockSize << "\t numberOfDeviations :: " << numberOfDeviations << endl;

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
            ulong numberOf_PM = matRows - blockSize;
            // ulong numberOf_PM = 10;
            double s_time;
            while (PM_cnt <= numberOf_PM)
            {
                s_time = GetTime();
                masterPrint(processorId) << " Processing PM(PM-size = " << blockSize << ") :: " << PM_cnt << " of " << numberOf_PM;
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

                _mat_ minor;
                ulong APM_dims = blockSize + numberOfDeviations;
                minor.SetDims(APM_dims, APM_dims);

                // Process all APM
                ZZ APM_cnt = conv<ZZ>("0");
                while (APM_cnt < quota)
                {
                    // Merge Block_I with row, col combo vec
                    ulong rowCombo[APM_dims], colCombo[APM_dims];
                    mergerVectors(APM_dims, blockSize, numberOfDeviations, block_I, complement_block_I, rowCombinationVec, rowCombo);
                    mergerVectors(APM_dims, blockSize, numberOfDeviations, block_I, complement_block_I, colCombinationVec, colCombo);

                    makeMatrixFromRowColCombination(rowCombo, colCombo, ker, minor);

                    if (IsZero(determinant(minor)))
                    {
                        cout << "\n [by6-APM] ZM found @ pId :: " << processorId << "\t PM-size :: " << blockSize << "\t APM-size :: "
                             << (blockSize + numberOfDeviations) << "\t PM_cnt :: " << (PM_cnt - 1) << " of " << numberOf_PM
                             << "\t  nod_start :: " << numberOfDeviations_start << "\t nod :: " << numberOfDeviations << endl;

                        printCombination2(rowCombo, APM_dims);
                        printCombination2(colCombo, APM_dims);

                        cout << "\n minor :: \n"
                             << minor << endl;

                        // Solve ECDLP here
                        // ZZ DLP = solveDLP_apm<_mat_, FiniteField>(rowCombo, colCombo, APM_dims, fileId, ker, ordP);
                        // cout << "\n DLP :: " << DLP << "\t pId :: " << processorId << "\t";
                        // cout.flush();

                        MPI_Abort(MPI_COMM_WORLD, 73);
                    }

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

                    ++APM_cnt;
                } // END :: while process all apm.

                masterPrint(processorId) << "\t Time :: " << (GetTime() - s_time) << " seconds " << endl;
                // MPI_Barrier(MPI_COMM_WORLD);

                if (!isLastCombination(block_I, blockSize, matRows))
                    _getNextCombination_continous(block_I, matRows, blockSize);
                else
                    break;

            } // END :: iteration over PM of blockSize

            MPI_Barrier(MPI_COMM_WORLD);
        } // END :: loop for numberOfDeviations
    }     // END :: loop for blockSize
}

/**
 * @brief Same as principleDeviation_parallel_8
 * Each slave processors gets the same PM. All processors processes APM for this PM.
 * pattern_4D
 */
template <class _mat_, class FiniteField>
void process_small_parallel_8(ZZ ordP, _mat_ ker, ulong fileId)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    const ulong matRows = ker.NumRows();
    const ulong numberOfParts = 6;
    const ulong max_blockSize = matRows / numberOfParts;
    const ulong blockStartDims = 2;
    const ulong numberOfDeviations_start = 4, numberOfDeviations_end = 5; // both inclusive

    masterPrint(processorId)
        << "\n ker.r :: " << matRows << "\t ker.c :: " << ker.NumCols() << "\t max_blockSize :: " << max_blockSize << endl;

    for (ulong blockSize = blockStartDims; blockSize <= max_blockSize; ++blockSize)
    {
        for (ulong numberOfDeviations = numberOfDeviations_start; numberOfDeviations <= numberOfDeviations_end; ++numberOfDeviations)
        {
            masterPrint(processorId) << " Processing block size :: " << blockSize << "\t of " << max_blockSize << "\t numberOfDeviations :: " << numberOfDeviations << endl;

            ulong s = 4; // segment or part size i.e this code works for 4-deviations only
            ulong s_arr[s];
            ulong arr[s - 1];
            {
                ulong t1 = (matRows - blockSize);
                ulong first_last_percentage = 22; // 22% first and last
                ulong first_last_value = (first_last_percentage * t1) / 100;
                ulong middle_two = (t1 - (first_last_value * 2)) / 2;

                s_arr[0] = first_last_value;
                s_arr[1] = middle_two;
                s_arr[2] = middle_two;
                s_arr[3] = first_last_value;

                if ((s_arr[0] + s_arr[1] + s_arr[2] + s_arr[3]) < t1)
                    s_arr[3] += (t1 - (s_arr[0] + s_arr[1] + s_arr[2] + s_arr[3]));
            }

            const ZZ number_of_APM_half = conv<ZZ>(s_arr[0] * s_arr[1] * s_arr[2] * s_arr[3]);
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

            ++rowCombinationIndex;
            ++colCombinationIndex;

            ulong rowCombinationVecOrg[numberOfDeviations], colCombinationVecOrg[numberOfDeviations];
            MPI_Barrier(MPI_COMM_WORLD);

            {
                arr[0] = s_arr[3];
                arr[1] = s_arr[2] * arr[0];
                arr[2] = s_arr[1] * arr[1];
            }

            get_kth_combination_seg4(symbolVec_dim, symbolVec, 4, s_arr, arr, rowCombinationIndex, rowCombinationVecOrg);
            get_kth_combination_seg4(symbolVec_dim, symbolVec, 4, s_arr, arr, colCombinationIndex, colCombinationVecOrg);

            // cout << "\n symbol-vec-dim :: " << symbolVec_dim << "\t row-com-index :: " << rowCombinationIndex << "\t col-com-index :: " << colCombinationIndex << "\t pId :: " << processorId << endl;
            // printCombination2(rowCombinationVecOrg, numberOfDeviations);
            // printCombination2(colCombinationVecOrg, numberOfDeviations);

            MPI_Barrier(MPI_COMM_WORLD);

            ulong block_I[blockSize];
            initCombinations(block_I, blockSize);
            ulong PM_cnt = 1;
            ulong numberOf_PM = matRows - blockSize;
            // ulong numberOf_PM = 10;
            double s_time;
            while (PM_cnt <= numberOf_PM)
            {
                s_time = GetTime();
                masterPrint(processorId) << " Processing PM(PM-size = " << blockSize << ") :: " << PM_cnt << " of " << numberOf_PM;
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

                _mat_ minor;
                ulong APM_dims = blockSize + numberOfDeviations;
                minor.SetDims(APM_dims, APM_dims);

                // Process all APM
                ZZ APM_cnt = conv<ZZ>("0");
                while (APM_cnt < quota)
                {
                    // cout << "\n one... @pId :: " << processorId << endl;
                    // Merge Block_I with row, col combo vec
                    ulong rowCombo[APM_dims], colCombo[APM_dims];
                    mergerVectors(APM_dims, blockSize, numberOfDeviations, block_I, complement_block_I, rowCombinationVec, rowCombo);
                    mergerVectors(APM_dims, blockSize, numberOfDeviations, block_I, complement_block_I, colCombinationVec, colCombo);

                    makeMatrixFromRowColCombination(rowCombo, colCombo, ker, minor);

                    if (IsZero(determinant(minor)))
                    {
                        cout << "\n [by6-APM] ZM found @ pId :: " << processorId << "\t PM-size :: " << blockSize << "\t APM-size :: "
                             << (blockSize + numberOfDeviations) << "\t PM_cnt :: " << (PM_cnt - 1) << " of " << numberOf_PM
                             << "\t  nod_start :: " << numberOfDeviations_start << "\t nod :: " << numberOfDeviations << endl;

                        printCombination2(rowCombo, APM_dims);
                        printCombination2(colCombo, APM_dims);

                        // Solve ECDLP here
                        ZZ DLP = solveDLP_apm<_mat_, FiniteField>(rowCombo, colCombo, APM_dims, fileId, ker, ordP);
                        cout << "\n DLP :: " << DLP << "\t pId :: " << processorId << "\t";
                        cout.flush();

                        MPI_Abort(MPI_COMM_WORLD, 73);
                    }

                    // isLastCombination_seg(v, n, v2, seg_size, s_arr, seg_size)
                    // isLastCombination_seg(complement_block_I, complementSize, colCombinationVec, s, s_arr, s);

                    if (!isLastCombination_seg(complement_block_I, complementSize, colCombinationVec, s, s_arr, s))
                    {
                        getNextCombination_seg(symbolVec, symbolVec_dim, colCombinationVec, s, s_arr, s);
                        // _getNextCombination(colCombinationVec, complementSize, numberOfDeviations);
                    }
                    else
                    {

                        if (isLastCombination_seg(complement_block_I, complementSize, rowCombinationVec, s, s_arr, s))
                            break;

                        initCombinations_seg(symbolVec, s_arr, s, colCombinationVec);
                        // initCombinations_seg(colCombinationVec, numberOfDeviations);
                        // _getNextCombination(rowCombinationVec, complementSize, numberOfDeviations);
                        getNextCombination_seg(symbolVec, symbolVec_dim, rowCombinationVec, s, s_arr, s);
                    }

                    ++APM_cnt;
                } // END :: while process all apm.

                masterPrint(processorId) << "\t Time :: " << (GetTime() - s_time) << " seconds " << endl;
                // MPI_Barrier(MPI_COMM_WORLD);

                if (!isLastCombination(block_I, blockSize, matRows))
                    _getNextCombination_continous(block_I, matRows, blockSize);
                else
                    break;

            } // END :: iteration over PM of blockSize

            MPI_Barrier(MPI_COMM_WORLD);
        } // END :: loop for numberOfDeviations
    }     // END :: loop for blockSize
}

/**
 * Pattern 4D.
 * Each processor gets a different PM and it processes all APM for this PM.
 */
template <class _mat_, class FiniteField>
bool process_small_parallel_7(ZZ ordP, _mat_ ker, ulong fileId)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    const ulong matRows = ker.NumRows();
    const ulong numberOfParts = 6;
    const ulong max_blockSize = matRows / numberOfParts;
    const ulong blockStartDims = 2;
    const ulong numberOfDeviations_start = 4, numberOfDeviations_end = 5; // both inclusive

    masterPrint(processorId)
        << "\n ker.r :: " << matRows << "\t ker.c :: " << ker.NumCols() << "\t max_blockSize :: " << max_blockSize << endl;

    for (ulong blockSize = blockStartDims; blockSize <= max_blockSize; ++blockSize)
    {
        for (ulong numberOfDeviations = numberOfDeviations_start; numberOfDeviations <= numberOfDeviations_end; ++numberOfDeviations)
        {
            masterPrint(processorId) << " Processing block size :: " << blockSize << "\t of " << max_blockSize << "\t numberOfDeviations :: " << numberOfDeviations << endl;

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
            ulong numberOf_PM = matRows - blockSize;
            // ulong numberOf_PM = 10;
            double s_time;
            while (PM_cnt <= numberOf_PM)
            {
                s_time = GetTime();
                masterPrint(processorId) << " Processing PM(PM-size = " << blockSize << ") :: " << PM_cnt << " of " << numberOf_PM;
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

                _mat_ minor;
                ulong APM_dims = blockSize + numberOfDeviations;
                minor.SetDims(APM_dims, APM_dims);

                // Process all APM
                ZZ APM_cnt = conv<ZZ>("0");
                while (APM_cnt < quota)
                {
                    if (isValidCombination_pattern4D(rowCombinationVec, colCombinationVec, numberOfDeviations, complementSize))
                    {
                        // Merge Block_I with row, col combo vec
                        ulong rowCombo[APM_dims], colCombo[APM_dims];
                        mergerVectors(APM_dims, blockSize, numberOfDeviations, block_I, complement_block_I, rowCombinationVec, rowCombo);
                        mergerVectors(APM_dims, blockSize, numberOfDeviations, block_I, complement_block_I, colCombinationVec, colCombo);

                        makeMatrixFromRowColCombination(rowCombo, colCombo, ker, minor);

                        if (IsZero(determinant(minor)))
                        {
                            cout << "\n [by6-APM] ZM found @ pId :: " << processorId << "\t PM-size :: " << blockSize << "\t APM-size :: "
                                 << (blockSize + numberOfDeviations) << "\t PM_cnt :: " << (PM_cnt - 1) << " of " << numberOf_PM
                                 << "\t  nod_start :: " << numberOfDeviations_start << "\t nod :: " << numberOfDeviations << endl;

                            printCombination2(rowCombo, APM_dims);
                            printCombination2(colCombo, APM_dims);

                            // Solve ECDLP here
                            ZZ DLP = solveDLP_apm<_mat_, FiniteField>(rowCombo, colCombo, APM_dims, fileId, ker, ordP);
                            cout << "\n DLP :: " << DLP << "\t pId :: " << processorId << "\t";
                            cout.flush();

                            // MPI_Abort(MPI_COMM_WORLD, 73);
                        }
                    }

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

                    ++APM_cnt;
                } // END :: while process all apm.

                masterPrint(processorId) << "\t Time :: " << (GetTime() - s_time) << " seconds " << endl;
                // MPI_Barrier(MPI_COMM_WORLD);

                if (!isLastCombination(block_I, blockSize, matRows))
                    _getNextCombination_continous(block_I, matRows, blockSize);
                else
                    break;

            } // END :: iteration over PM of blockSize

            MPI_Barrier(MPI_COMM_WORLD);
        } // END :: loop for numberOfDeviations
    }     // END :: loop for blockSize
}

/**
 * delete - odd row, columns indices
 */
template <class _mat_, class FiniteField>
_mat_ delete_removeOddIndex(ZZ ordP, const _mat_ &mat, ulong fileId)
{
    _mat_ newMat;

    ulong orgMatRow = mat.NumRows();
    ulong orgMatCol = mat.NumCols();

    newMat.SetDims(orgMatRow / 2, orgMatCol / 2);

    for (ulong i = 1, k = 0; i < orgMatRow; i += 2, ++k)
    {
        for (ulong j = 1, l = 0; j < orgMatCol; j += 2, ++l)
        {
            newMat[k][l] = mat[i][j];
        }
    }

    return newMat;
}

/**
 * delete - one third row, columns
 */
template <class _mat_, class FiniteField>
_mat_ delete_removeSome(ZZ ordP, const _mat_ &mat, ulong count)
{
    _mat_ newMat;

    ulong orgMatRow = mat.NumRows();
    ulong orgMatCol = mat.NumCols();

    newMat.SetDims(orgMatRow / 2, orgMatCol / 2);

    bool flag = true;
    for (ulong i = 1, k = 0; i < orgMatRow; i++)
    {
        for (ulong j = 1, l = 0; j < orgMatCol; j++)
        {
            if (flag)
            {
                newMat[k][l] = mat[i][j];
            }
        }
    }

    return newMat;
}

/**
 * Each slave processors gets the same PM.
 * All slaves look at two deviations before and after this PM
 */
template <class _mat_, class FiniteField>
bool principleDeviation_parallel_3_small(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
    // ulong numberOfDeviations = 3;
    // Iterate over all the kernel files.
    for (ulong fileId = 0; fileId < totalNumberOfProcessors; ++fileId)
    {
        masterPrint(processorId) << " Processing fileId :: " << (fileId + 1) << "\t of " << totalNumberOfProcessors << endl;

        _mat_ ker = getKernelFromFile<_mat_>(fileId, totalNumberOfProcessors, ordP);

        // --------------------------------------------------------------------------

        _mat_ newKer = delete_removeOddIndex<_mat_, FiniteField>(ordP, ker, fileId);

        // --------------------------------------------------------------------------

        process_small_parallel_7<_mat_, FiniteField>(ordP, newKer, fileId);

    } // END : loop for kernel files (i.e. fileId)

    return true;
}
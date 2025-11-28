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

template <class _mat_>
bool twoDeviation(ulong blockSize, ulong block_I[], const _mat_ &mat, ulong complementSize, ulong complement_block_I[]);

bool mergerVectors(ulong APM_dims, ulong blockSize, ulong numberOfDeviations, ulong block_I[],
                   ulong complement_block_I[], ulong combinationVec[], ulong finalCombo[])
{
    ulong block_I_ptr = 0, vec_ptr = 0;
    for (ulong i = 0; i < APM_dims; ++i)
    {
        if (block_I[block_I_ptr] <= complement_block_I[combinationVec[vec_ptr]])
            finalCombo[i] = block_I[block_I_ptr++];
        else
            finalCombo[i] = complement_block_I[combinationVec[vec_ptr++]];

        if (vec_ptr == numberOfDeviations)
        {
            // copy block_I remaining to finalCombo
            for (ulong j = i + 1; j < APM_dims; ++j)
                finalCombo[j] = block_I[block_I_ptr++];
            break;
        }
        if (block_I_ptr == blockSize)
        {
            for (ulong j = i + 1; j < APM_dims; ++j)
                finalCombo[j] = complement_block_I[combinationVec[vec_ptr++]];
            break;
        }
    }
}

template <class _mat_, class FiniteField>
ZZ solveDLP_apm(ulong rowCombo[], ulong colCombo[], ulong dim, ulong fileId, const _mat_ &ker, ZZ ordP)
{
    int processorId2, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId2, totalNumberOfProcessors, NodeName);
    delete NodeName;

    const ulong orgMatCol = (ker.NumCols() * 2);
    ulong k_rn = (orgMatCol / 2) - 1;
    ulong t_rn = k_rn + 2;

    // Make bool vector
    bool *vec = new bool[orgMatCol];
    for (ulong i = 0; i < orgMatCol; ++i)
        if (i <= k_rn)
            vec[i] = true;
        else
            vec[i] = false;

    for (int i = 0; i < dim; ++i)
        vec[colCombo[i]] = false;

    for (int i = 0; i < dim; ++i)
        vec[(orgMatCol - 1) - rowCombo[i]] = true;

    ZZ A, B;
    ZZ randomNumbers[orgMatCol];
    getRandomNumbersFromFile(fileId, totalNumberOfProcessors, randomNumbers, ordP);

    for (ulong i = 0; i < k_rn; ++i)
        if (vec[i])
            A += randomNumbers[i];

    for (ulong i = k_rn; i < orgMatCol; ++i)
        if (vec[i])
            B += randomNumbers[i];

    {
        // Calculate DLP
        ZZ_p::init(ordP);

        ZZ_p A_dlp = conv<ZZ_p>(A);
        ZZ_p B_dlp = conv<ZZ_p>(B);

        if (!IsZero(A_dlp) && !IsZero(B_dlp))
        {
            ZZ_p DLP = (A_dlp / B_dlp);
            return conv<ZZ>(DLP);
        }
        else
        {
            std::cerr << B_RED_START << "\n Something is wrong... A or B is zero in getDLP() ... A :: " << A_dlp << "\t B ::" << B_dlp << "\n"
                      << RESET_TERM;
            return conv<ZZ>("0");
        }
    }

    return conv<ZZ>("0");
}

// Function to compute the number of runs in an array of unsigned long integers
int numRuns(const ulong array[], size_t size)
{
    // Initialize the number of runs to 1
    int num_runs = 1;

    // Iterate through the array and count the number of runs
    for (size_t i = 1; i < size; i++)
    {
        // If the current element is not equal to the previous element, increment the number of runs
        if (array[i] != array[i - 1])
        {
            num_runs++;
        }
    }

    return num_runs;
}

// Function to shuffle an array of integers
void shuffleArray(ulong array[], size_t size)
{
    std::random_device rd;
    std::mt19937 g(size);

    // Shuffle the elements of the array
    for (size_t i = 0; i < size; i++)
    {
        // Generate a random index between i and size - 1
        std::uniform_int_distribution<size_t> dist(i, size - 1);
        size_t j = dist(g);

        // Swap the element at index i with the element at index j
        std::swap(array[i], array[j]);
    }
}

bool isValidCombination_pattern4D(ulong rowCombinationVec[], ulong colCombinationVec[], ulong numberOfDeviations, ulong dims)
{
    ulong segCount = 4;
    ulong seg[segCount - 1];

    ulong first_last_percentage = 22;
    ulong first_last_value = (first_last_percentage * dims) / 100;
    ulong middle_two = (dims - (first_last_value * 2)) / 2;

    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    seg[0] = first_last_value;
    seg[1] = seg[0] + middle_two;
    seg[2] = seg[1] + middle_two;

    masterPrint(processorId) << "seg[0] :: " << seg[0] << "\t seg[1] :: " << seg[1] << "\t seg[2] :: " << seg[2] << "\t seg[3] :: " << seg[3] << endl;

    if (rowCombinationVec[0] > seg[0])
        return false;

    // if (rowCombinationVec[0] < seg[1] || rowCombinationVec[1] > seg[1])
    //     return false;

    // if (rowCombinationVec[1] < seg[2] || rowCombinationVec[2] > seg[2])
    //     return false;

    // if (rowCombinationVec[2] < seg[3] || rowCombinationVec[3] < seg[3])
    //     return false;

    if (rowCombinationVec[3] < seg[3])
        return false;

    if (colCombinationVec[0] > seg[0])
        return false;

    // if (colCombinationVec[0] < seg[1] || colCombinationVec[1] > seg[1])
    //     return false;

    // if (colCombinationVec[1] < seg[2] || colCombinationVec[2] > seg[2])
    //     return false;

    // if (colCombinationVec[2] < seg[3] || colCombinationVec[3] < seg[3])
    //     return false;

    if (colCombinationVec[3] < seg[3])
        return false;

    return true;
}

/**
 * @brief segment size is assumed to be same as result vector size
 *
 * @param vecOrg
 * @param seg
 * @param segSize
 * @param resultVec
 */
void initCombinations_seg(ulong vecOrg[], ulong seg[], ulong segSize, ulong resultVec[])
{
    int start_index = 0;
    for (int i = 0; i < segSize; ++i)
    {
        resultVec[i] = vecOrg[start_index];
        start_index += seg[i];
    }
}

bool isLastCombination_seg(ulong vecOrg[], ulong vecOrg_size, ulong v2[], ulong v2_size, ulong seg[], ulong seg_size)
{
    ulong index = 0;

    for (ulong i = 0; i < seg_size; ++i)
    {
        index += seg[i];
        if (v2[i] != vecOrg[(index - 1)])
            return false;
    }

    return true;
}

/**
 * @brief
 * Assumption : vector v2 size is same as seg_size
 * @param vecOrg
 * @param vecOrg_size
 * @param v2
 * @param v2_size
 * @param seg
 * @param seg_size
 */
bool getNextCombination_seg(ulong vecOrg[], ulong vecOrg_size, ulong v2[], ulong v2_size, ulong seg[], ulong seg_size)
{
    // Make a vector having the last possible element for each segment
    ulong lastElement_Seg[seg_size];
    {
        ulong a = 0;
        for (ulong i = 0; i < seg_size; ++i)
        {
            a += seg[i];
            lastElement_Seg[i] = vecOrg[a - 1];
        }
    }

    for (int i = (v2_size - 1); i >= 0; --i)
    {
        if (v2[i] != lastElement_Seg[i])
        {
            v2[i] += 1;

            if (i != v2_size - 1)
                for (ulong j = i + 1; j < v2_size; ++j)
                {
                    v2[j] = lastElement_Seg[j] - seg[j] + 1;
                }
            return true;
        }
    }

    return false;
}

bool get_kth_combination_seg4(ulong n, ulong v[], ulong s, ulong s_arr[], ulong arr[], ulong ip, ulong ans[])
{
    if (ip <= 0)
        return false;
    --ip;

    // int ans[s];
    if (ip < arr[0])
    {
        ans[0] = 0;
        ans[1] = s_arr[0];
        ans[2] = s_arr[0] + s_arr[1];
        ans[3] = s_arr[0] + s_arr[1] + s_arr[2] + ip;
    }
    else if (ip < arr[1])
    {
        ans[0] = 0;
        ans[1] = s_arr[0];
        ans[2] = s_arr[0] + s_arr[1] + (ip / arr[0]);
        ans[3] = s_arr[0] + s_arr[1] + s_arr[2] + (ip % s_arr[3]);
    }
    else if (ip < arr[2])
    {
        ans[0] = 0;
        ans[1] = s_arr[0] + (ip / arr[1]) % s_arr[1];
        ans[2] = s_arr[0] + s_arr[1] + ((ip - arr[1]) / arr[0]) % s_arr[2];
        ans[3] = s_arr[0] + s_arr[1] + s_arr[2] + (ip % s_arr[3]);
    }
    else
    {
        ans[0] = ip / arr[2];
        ans[1] = s_arr[0] + (ip % arr[2]) / arr[1];
        ans[2] = s_arr[0] + s_arr[1] + (ip % arr[1]) / arr[0];
        ans[3] = s_arr[0] + s_arr[1] + s_arr[2] + (ip % s_arr[3]);
    }

    if ((ans[0] < s_arr[0]) && (ans[1] < (s_arr[0] + s_arr[1])) && (ans[2] < (s_arr[0] + s_arr[1] + s_arr[2])) && (ans[3] < (s_arr[0] + s_arr[1] + s_arr[2] + s_arr[3])))
    {
        // cout << ans[0] << "\t" << ans[1] << "\t" << ans[2] << "\t" << ans[3] << "\t ip :: " << (ip + 1) << endl;
        return true;
    }
    else
    {
        return false;
    }
}

/**
 * Each slave processors gets the same PM. All processors processes APM for this PM.
 * pattern_4D
 * 19th Jan. 2023
 */
template <class _mat_, class FiniteField>
bool principleDeviation_parallel_8(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    // Iterate over all the kernel files.
    for (ulong fileId = 0; fileId < totalNumberOfProcessors; ++fileId)
    {
        masterPrint(processorId) << " Processing fileId :: " << (fileId + 1) << "\t of " << totalNumberOfProcessors << endl;

        _mat_ ker = getKernelFromFile<_mat_>(fileId, totalNumberOfProcessors, ordP);

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
    }         // END : loop for kernel files (i.e. fileId)
}

/**
 * Pattern 4D.
 * Each processor gets a different PM and it processes all APM for this PM.
 * 6th Jan. 2023
 */
template <class _mat_, class FiniteField>
bool principleDeviation_parallel_7(ZZ ordP)
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
    }         // END : loop for kernel files (i.e. fileId)
}

/**
 * Each slave processors gets the same PM.
 * All slaves look at two deviations before and after this PM
 */
template <class _mat_, class FiniteField>
bool principleDeviation_parallel_shuffleComplement_6(ZZ ordP)
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

        const ulong matRows = ker.NumRows();
        const ulong numberOfParts = 6;
        const ulong max_blockSize = matRows / numberOfParts;
        const ulong blockStartDims = 2;
        const ulong numberOfDeviations_start = 2, numberOfDeviations_end = 2; // both inclusive

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

                    // shuffle complement_block_I
                    {
                        shuffleArray(complement_block_I, complementSize);
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

                            // Solve ECDLP here
                            ZZ DLP = solveDLP_apm<_mat_, FiniteField>(rowCombo, colCombo, APM_dims, fileId, ker, ordP);
                            cout << "\n DLP :: " << DLP << "\t pId :: " << processorId << "\t";
                            cout.flush();

                            // MPI_Abort(MPI_COMM_WORLD, 73);
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
    }         // END : loop for kernel files (i.e. fileId)
}

/**
 * Each slave processors gets the same PM.
 * Rows remain constant (i.e. AP) deviations in columns.
 * numberOfDeviations = blockSize.
 */
template <class _mat_, class FiniteField>
bool principleDeviation_parallel_5(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
    // ulong numberOfDeviations = 2;
    // Iterate over all the kernel files.
    for (ulong fileId = 0; fileId < totalNumberOfProcessors; ++fileId)
    {
        masterPrint(processorId) << " Processing fileId :: " << (fileId + 1) << "\t of " << totalNumberOfProcessors << endl;

        _mat_ ker = getKernelFromFile<_mat_>(fileId, totalNumberOfProcessors, ordP);

        const ulong matRows = ker.NumRows();
        const ulong numberOfParts = 6;
        const ulong max_blockSize = matRows / numberOfParts;
        const ulong blockStartDims = 2;

        masterPrint(processorId)
            << "\n ker.r :: " << matRows << "\t ker.c :: " << ker.NumCols() << "\t max_blockSize :: " << max_blockSize << endl;

        for (ulong blockSize = blockStartDims; blockSize < max_blockSize; ++blockSize)
        {
            for (ulong numberOfDeviations = 2; numberOfDeviations < 4; ++numberOfDeviations)
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
                ulong numberOf_PM = matRows - (blockSize + numberOfDeviations);
                double s_time, e_time;
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

                    ulong rowCombo[APM_dims];
                    {
                        for (ulong i = 0; i < blockSize; ++i)
                        {
                            rowCombo[i] = block_I[i];
                        }

                        for (int i = blockSize; i < APM_dims; ++i)
                            rowCombo[i] = rowCombo[i - 1] + 1;
                    }

                    // Process all APM
                    ZZ APM_cnt = conv<ZZ>("0");
                    while (APM_cnt < quota)
                    {
                        // Merge Block_I with col combo vec
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
                            cout << "\n by6-APM-3D ZM found @ pId :: " << processorId << "\t PM-size :: " << blockSize << "\t APM-size :: " << (blockSize + numberOfDeviations) << "\t PM_cnt :: " << (PM_cnt - 1) << " of " << numberOf_PM << endl;
                            printCombination2(rowCombo, APM_dims);
                            printCombination2(colCombo, APM_dims);
                            printCombination2(complement_block_I, complementSize);
                            printCombination2(block_I, blockSize);
                            printCombination2(rowCombinationVec, numberOfDeviations);
                            printCombination2(colCombinationVec, numberOfDeviations);
                            MPI_Abort(MPI_COMM_WORLD, 73);
                        }

                        if (!isLastCombination(colCombinationVec, numberOfDeviations, complementSize))
                            _getNextCombination(colCombinationVec, complementSize, numberOfDeviations);
                        else
                            break;

                        ++APM_cnt;
                    } // END :: while process all apm.

                    e_time = GetTime();

                    MPI_Barrier(MPI_COMM_WORLD);
                    masterPrint(processorId) << "\t Time :: " << (e_time - s_time) << " seconds " << endl;

                    if (!isLastCombination(block_I, blockSize, matRows))
                        _getNextCombination_continous(block_I, matRows, blockSize);
                    else
                        break;

                } // END :: iteration over PM of blockSize

                MPI_Barrier(MPI_COMM_WORLD);
            }
        } // END :: for blockSize
    }     // END : for fileId
}

/**
 * Each slave processors gets the same PM.
 * All slaves look at two deviations before and after this PM
 * This is similar to Totally positive matrix Initial Minors.
 */
template <class _mat_, class FiniteField>
bool principleDeviation_parallel_4(ZZ ordP)
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
        // const ulong max_blockSize = (matRows / 2) + 1;
        // const ulong max_blockSize = (matRows / 6);
        const ulong max_blockSize = matRows - 1;
        const ulong blockStartDims = 2;

        masterPrint(processorId) << "\n ker.r :: " << matRows << "\t ker.c :: " << ker.NumCols() << "\t max_blockSize :: " << max_blockSize << endl;

        for (ulong blockSize = blockStartDims; blockSize < max_blockSize; ++blockSize)
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

            ulong block_I[blockSize], block_J[blockSize];
            initCombinations(block_I, blockSize);
            initCombinations(block_J, blockSize);

            ulong PM_cnt = 1;
            ulong numberOf_PM = matRows - blockSize;
            double s_time, e_time;
            while (PM_cnt < numberOf_PM)
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

                ulong complement_block_J[complementSize];
                {
                    ulong cnt = 0;
                    ulong index = 0;
                    while (cnt < matRows)
                    {
                        if (cnt != block_J[0])
                            complement_block_J[index++] = cnt++;
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
                while (APM_cnt < quota)
                {
                    _mat_ minor;
                    ulong APM_dims = blockSize + numberOfDeviations;
                    minor.SetDims(APM_dims, APM_dims);

                    // ulong startRow = block_I[0] < complement_block_I[rowCombinationVec[0]] ? block_I[0] : complement_block_I[rowCombinationVec[0]];
                    // ulong startCol = block_I[0] < complement_block_I[colCombinationVec[0]] ? block_I[0] : complement_block_I[colCombinationVec[0]];
                    // if (startRow != startCol)
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
                            ulong block_J_ptr = 0, vec_ptr = 0;
                            for (ulong i = 0; i < APM_dims; ++i)
                            {
                                if (block_J[block_J_ptr] <= complement_block_I[colCombinationVec[vec_ptr]])
                                    colCombo[i] = block_I[block_J_ptr++];
                                else
                                    colCombo[i] = complement_block_J[colCombinationVec[vec_ptr++]];

                                if (vec_ptr == numberOfDeviations)
                                {
                                    // copy block_I remaining to colCombo
                                    for (ulong j = i + 1; j < APM_dims; ++j)
                                        colCombo[j] = block_J[block_J_ptr++];
                                    break;
                                }
                                if (block_J_ptr == blockSize)
                                {
                                    for (ulong j = i + 1; j < APM_dims; ++j)
                                        colCombo[j] = complement_block_J[colCombinationVec[vec_ptr++]];
                                    break;
                                }
                            }
                        }

                        makeMatrixFromRowColCombination(rowCombo, colCombo, ker, minor);

                        if (IsZero(determinant(minor)))
                        {
                            cout << "\n FULL-APM ZM found @ pId :: " << processorId << "\t PM-size :: " << blockSize << "\t APM-size :: " << (blockSize + numberOfDeviations) << "\t PM_cnt :: " << (PM_cnt - 1) << " of " << numberOf_PM << endl;

                            printCombination2(rowCombo, APM_dims);
                            printCombination2(colCombo, APM_dims);
                            printCombination2(complement_block_I, complementSize);
                            printCombination2(block_I, blockSize);
                            printCombination2(rowCombinationVec, numberOfDeviations);
                            printCombination2(colCombinationVec, numberOfDeviations);
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

                // while we do not get a combination with row or column index as 0
                while (1)
                {
                    if (!isLastCombination(block_J, blockSize, matRows))
                    {
                        _getNextCombination_continous(block_J, matRows, blockSize);
                    }
                    else
                    {
                        if (isLastCombination(block_I, blockSize, matRows))
                            break;

                        initCombinations(block_J, blockSize);
                        _getNextCombination_continous(block_I, matRows, blockSize);
                    }

                    if (block_I[0] != 0 || block_J[0] != 0)
                        continue;
                    else
                        break;
                }

            } // END :: iteration over PM of blockSize

            MPI_Barrier(MPI_COMM_WORLD);
        } // END :: for blockSize
    }     // END : for fileId
}

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
    // ulong numberOfDeviations = 3;
    // Iterate over all the kernel files.
    for (ulong fileId = 0; fileId < totalNumberOfProcessors; ++fileId)
    {
        masterPrint(processorId) << " Processing fileId :: " << (fileId + 1) << "\t of " << totalNumberOfProcessors << endl;

        _mat_ ker = getKernelFromFile<_mat_>(fileId, totalNumberOfProcessors, ordP);

        const ulong matRows = ker.NumRows();
        const ulong numberOfParts = 6;
        const ulong max_blockSize = matRows / numberOfParts;
        const ulong blockStartDims = 2;
        const ulong numberOfDeviations_start = 2, numberOfDeviations_end = 2; // both inclusive

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

                            // Solve ECDLP here
                            ZZ DLP = solveDLP_apm<_mat_, FiniteField>(rowCombo, colCombo, APM_dims, fileId, ker, ordP);
                            cout << "\n DLP :: " << DLP << "\t pId :: " << processorId << "\t";
                            cout.flush();

                            // MPI_Abort(MPI_COMM_WORLD, 73);
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
    }         // END : loop for kernel files (i.e. fileId)
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
        // ulong max_blockSize = (matRows / 2) + 1;
        const ulong numberOfParts = 6;
        const ulong max_blockSize = matRows / numberOfParts;
        const ulong blockStartDims = 2;

        masterPrint(processorId) << "\n ker.r :: " << matRows << "\t ker.c :: " << ker.NumCols() << "\t max_blockSize :: " << max_blockSize << endl;
        for (ulong blockSize = blockStartDims; blockSize < max_blockSize; ++blockSize)
        {
            double time_s = GetTime();
            masterPrint(processorId) << "\n Processing blockSize :: " << blockSize << " of " << max_blockSize << endl;

            ulong numberOfBlocks = matRows - blockSize;

            ulong numberOfBlocks_eachProcessor = 0;
            if (numberOfBlocks <= totalNumberOfProcessors)
            {
                numberOfBlocks_eachProcessor = 1;
                masterPrint(processorId) << " Number of PM's :: " << numberOfBlocks_eachProcessor << "\t Total number of Processors :: " << totalNumberOfProcessors << endl;
                masterPrint(processorId) << " [Warning] In-efficient CPU utilization...\n";
            }
            else
            {
                numberOfBlocks_eachProcessor = numberOfBlocks / totalNumberOfProcessors;
            }

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
            masterPrint(processorId) << " Total numberOf PM :: " << numberOfBlocks << "\t numberOf PM each Processor gets :: " << numberOfBlocks_eachProcessor << endl;

            ulong block_I[blockSize];
            initCombinations(block_I, blockSize);
            ulong iterationCnt = 0;
            bool flag = false;
            while (!flag)
            {
                // Condition for this processor to process PM assigned to it
                if (iterationCnt >= blockStartCount && iterationCnt < blockEndCount)
                {
                    _mat_ principleMinor;
                    principleMinor.SetDims(blockSize, blockSize);

                    makeMatrixFromRowColCombination(block_I, block_I, ker, principleMinor);

                    // TODO : this function should return row, col indices for ZM. Then compute DLP in the if condition below
                    ulong dim = 6;
                    ulong row[dim], col[dim];
                    flag = makeAllAlmostPrincipleMinors<_mat_, FiniteField>(principleMinor, ker, blockSize, block_I, dim, row, col);
                    if (flag)
                    {
                        ZZ DLP = solveDLP_apm<_mat_, FiniteField>(row, col, dim, fileId, ker, ordP);
                        cout << "\n DLP :: " << DLP << "\t pId :: " << processorId << "\t";
                        cout << " Time :: " << (GetTime() - time_s) << " Sec. \n";

                        MPI_Abort(MPI_COMM_WORLD, 73);
                    }
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

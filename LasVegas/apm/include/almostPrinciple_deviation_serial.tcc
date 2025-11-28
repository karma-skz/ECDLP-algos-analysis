#include "constants.hpp"
#include "utils.hpp"
#include "containment.tcc"

template <class _mat_>
bool oneDeviation(ulong blockSize, ulong block_I[], const _mat_ &mat, ulong complementSize, ulong complement_block_I[])
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong almostPrincipleMinor_row[blockSize + 1];
    ulong almostPrincipleMinor_col[blockSize + 1];

    for (ulong i = 0; i < blockSize; ++i)
    {
        almostPrincipleMinor_row[i] = block_I[i];
        almostPrincipleMinor_col[i] = block_I[i];
    }

    for (ulong row1_ptr = 0; row1_ptr < complementSize; ++row1_ptr)
    {
        almostPrincipleMinor_row[blockSize] = complement_block_I[row1_ptr];

        for (ulong col_ptr = 0; col_ptr < complementSize; col_ptr++)
        {
            bool flag = false;
            if ((complement_block_I[row1_ptr] < block_I[0]) && (complement_block_I[col_ptr] < block_I[0]))
                flag = true;

            if ((complement_block_I[row1_ptr] > block_I[blockSize - 1]) && (complement_block_I[col_ptr] > block_I[blockSize - 1]))
                flag = true;

            if (flag)
            {
                almostPrincipleMinor_col[blockSize] = complement_block_I[col_ptr];

                _mat_ almostPrincipleMinor;
                almostPrincipleMinor.SetDims(blockSize + 1, blockSize + 1);
                makeMatrixFromRowColCombination(almostPrincipleMinor_row, almostPrincipleMinor_col, mat, almostPrincipleMinor);

                if (IsZero(determinant(almostPrincipleMinor)))
                {
                    printCombination2(almostPrincipleMinor_row, blockSize + 1);
                    printCombination2(almostPrincipleMinor_col, blockSize + 1);

                    cout << "\n apm :: \n"
                         << almostPrincipleMinor << endl;
                    cout << "\n ### determinant is zero (1D) ...\t blockSize :: " << blockSize << "\t @ pId :: " << processorId << endl;
                    return true;
                }
            }
        }
    }
    return false;
}

template <class _mat_>
bool twoDeviation(ulong blockSize, ulong block_I[], const _mat_ &mat, ulong complementSize, ulong complement_block_I[])
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong almostPrincipleMinor_row[blockSize + 2];
    ulong almostPrincipleMinor_col[blockSize + 2];

    for (ulong i = 0; i < blockSize; ++i)
    {
        almostPrincipleMinor_row[i] = block_I[i];
        almostPrincipleMinor_col[i] = block_I[i];
    }

    for (ulong row1_ptr = 0; row1_ptr < complementSize; ++row1_ptr)
    {
        almostPrincipleMinor_row[blockSize] = complement_block_I[row1_ptr];

        for (ulong row2_ptr = row1_ptr + 1; row2_ptr < complementSize; ++row2_ptr)
        {
            if ((complement_block_I[row1_ptr] < block_I[0]) && (complement_block_I[row2_ptr] > block_I[blockSize - 1]))
                continue;

            almostPrincipleMinor_row[blockSize + 1] = complement_block_I[row2_ptr];

            for (ulong col1_ptr = 0; col1_ptr < complementSize; col1_ptr++)
            {
                almostPrincipleMinor_col[blockSize] = complement_block_I[col1_ptr];

                for (ulong col2_ptr = col1_ptr + 1; col2_ptr < complementSize; col2_ptr++)
                {
                    if ((complement_block_I[col1_ptr] < block_I[0]) && (complement_block_I[col2_ptr] > block_I[blockSize - 1]))
                        continue;

                    // These two conditions check if both are on the same side of the principal minor
                    if ((complement_block_I[row2_ptr] < block_I[0]) && (complement_block_I[col1_ptr] > block_I[blockSize - 1]))
                        continue;
                    if ((complement_block_I[row1_ptr] > block_I[0]) && (complement_block_I[col2_ptr] < block_I[0]))
                        continue;

                    almostPrincipleMinor_col[blockSize + 1] = complement_block_I[col2_ptr];

                    _mat_ almostPrincipleMinor;
                    almostPrincipleMinor.SetDims(blockSize + 2, blockSize + 2);
                    makeMatrixFromRowColCombination(almostPrincipleMinor_row, almostPrincipleMinor_col, mat, almostPrincipleMinor);

                    if (IsZero(determinant(almostPrincipleMinor)))
                    {
                        printCombination2(almostPrincipleMinor_row, blockSize + 2);
                        printCombination2(almostPrincipleMinor_col, blockSize + 2);

                        cout << "\n row1_ptr :: " << row1_ptr << "\t  block_I[0] :: " << block_I[0] << "\t row2_ptr :: " << row2_ptr << "\t block_I[blockSize - 1] :: " << block_I[blockSize - 1] << endl;
                        cout << "\n col1  :: " << col1_ptr << "\t col2 :: " << col2_ptr << endl;
                        cout << "\n apm :: \n"
                             << almostPrincipleMinor << endl;

                        cout << "\n ### determinant is zero (2D) ...\t blockSize :: " << blockSize << "\t @ pId :: " << processorId << endl;
                        return true;
                    }
                } // END:col2_ptr

            } // End:col1_ptr

        } // End:row2_ptr

    } // END:row1_ptr
    return false;
}

template <class _mat_>
bool threeDeviation(ulong blockSize, ulong block_I[], const _mat_ &mat, ulong complementSize, ulong complement_block_I[])
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong almostPrincipleMinor_row[blockSize + 3];
    ulong almostPrincipleMinor_col[blockSize + 3];

    for (ulong i = 0; i < blockSize; ++i)
    {
        almostPrincipleMinor_row[i] = block_I[i];
        almostPrincipleMinor_col[i] = block_I[i];
    }

    for (ulong row1_ptr = 0; row1_ptr < complementSize; ++row1_ptr)
    {
        almostPrincipleMinor_row[blockSize] = complement_block_I[row1_ptr];

        for (ulong row2_ptr = row1_ptr + 1; row2_ptr < complementSize; ++row2_ptr)
        {
            almostPrincipleMinor_row[blockSize + 1] = complement_block_I[row2_ptr];

            for (ulong row3_ptr = row2_ptr + 1; row3_ptr < complementSize; ++row3_ptr)
            {
                almostPrincipleMinor_row[blockSize + 2] = complement_block_I[row3_ptr];

                for (ulong col1_ptr = 0; col1_ptr < complementSize; col1_ptr++)
                {
                    almostPrincipleMinor_col[blockSize] = complement_block_I[col1_ptr];

                    for (ulong col2_ptr = col1_ptr + 1; col2_ptr < complementSize; col2_ptr++)
                    {
                        almostPrincipleMinor_col[blockSize + 1] = complement_block_I[col2_ptr];

                        for (ulong col3_ptr = col2_ptr + 1; col3_ptr < complementSize; col3_ptr++)
                        {
                            almostPrincipleMinor_col[blockSize + 2] = complement_block_I[col3_ptr];
                            _mat_ almostPrincipleMinor;
                            almostPrincipleMinor.SetDims(blockSize + 3, blockSize + 3);
                            makeMatrixFromRowColCombination(almostPrincipleMinor_row, almostPrincipleMinor_col, mat, almostPrincipleMinor);

                            if (IsZero(determinant(almostPrincipleMinor)))
                            {
                                cout << "\n apm :: \n " << almostPrincipleMinor << endl;
                                cout << "\n ### determinant is zero (3D) ...\t blockSize :: " << blockSize << "\t @ pId :: " << processorId << endl;
                                return true;
                            }
                        }
                    } // END:col2_ptr

                } // End:col1_ptr
            }
        } // End:row3_ptr

    } // END:row1_ptr
    return false;
}

template <class _mat_>
bool fourDeviation(ulong blockSize, ulong block_I[], const _mat_ &mat, ulong complementSize, ulong complement_block_I[])
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong almostPrincipleMinor_row[blockSize + 4];
    ulong almostPrincipleMinor_col[blockSize + 4];

    for (ulong i = 0; i < blockSize; ++i)
    {
        almostPrincipleMinor_row[i] = block_I[i];
        almostPrincipleMinor_col[i] = block_I[i];
    }

    for (ulong row1_ptr = 0; row1_ptr < complementSize; ++row1_ptr)
    {
        almostPrincipleMinor_row[blockSize] = complement_block_I[row1_ptr];

        for (ulong row2_ptr = row1_ptr + 1; row2_ptr < complementSize; ++row2_ptr)
        {
            almostPrincipleMinor_row[blockSize + 1] = complement_block_I[row2_ptr];

            for (ulong row3_ptr = row2_ptr + 1; row3_ptr < complementSize; ++row3_ptr)
            {
                almostPrincipleMinor_row[blockSize + 2] = complement_block_I[row3_ptr];

                for (ulong row4_ptr = row3_ptr + 1; row4_ptr < complementSize; ++row4_ptr)
                {
                    almostPrincipleMinor_row[blockSize + 3] = complement_block_I[row4_ptr];

                    for (ulong col1_ptr = 0; col1_ptr < complementSize; col1_ptr++)
                    {
                        almostPrincipleMinor_col[blockSize] = complement_block_I[col1_ptr];

                        for (ulong col2_ptr = col1_ptr + 1; col2_ptr < complementSize; col2_ptr++)
                        {
                            almostPrincipleMinor_col[blockSize + 1] = complement_block_I[col2_ptr];

                            for (ulong col3_ptr = col2_ptr + 1; col3_ptr < complementSize; col3_ptr++)
                            {
                                almostPrincipleMinor_col[blockSize + 2] = complement_block_I[col3_ptr];

                                for (ulong col4_ptr = col3_ptr + 1; col4_ptr < complementSize; col4_ptr++)
                                {
                                    almostPrincipleMinor_col[blockSize + 3] = complement_block_I[col4_ptr];

                                    _mat_ almostPrincipleMinor;
                                    almostPrincipleMinor.SetDims(blockSize + 3, blockSize + 3);
                                    makeMatrixFromRowColCombination(almostPrincipleMinor_row, almostPrincipleMinor_col, mat, almostPrincipleMinor);

                                    if (IsZero(determinant(almostPrincipleMinor)))
                                    {
                                        cout << "\n apm :: \n " << almostPrincipleMinor << endl;
                                        cout << "\n ### determinant is zero (4D) ...\t blockSize :: " << blockSize << "\t @ pId :: " << processorId << endl;
                                        printCombination2(almostPrincipleMinor_row, blockSize + 4);
                                        printCombination2(almostPrincipleMinor_col, blockSize + 4);
                                        return true;
                                    }
                                } // END : col4_ptr
                            }
                        } // END:col2_ptr

                    } // End:col1_ptr

                } // END :: row3_ptr
            }
        } // End:row3_ptr

    } // END:row1_ptr
    return false;
}

template <class _mat_>
bool fourDeviation_segmented(ulong blockSize, ulong block_I[], const _mat_ &mat, ulong complementSize, ulong complement_block_I[], ulong &rDim, ulong row[], ulong col[])
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong almostPrincipleMinor_row[blockSize + 4];
    ulong almostPrincipleMinor_col[blockSize + 4];

    for (ulong i = 0; i < blockSize; ++i)
    {
        almostPrincipleMinor_row[i] = block_I[i];
        almostPrincipleMinor_col[i] = block_I[i];
    }

    ulong first_last_percentage = 22;
    ulong first_last_value = (first_last_percentage * complementSize) / 100;
    ulong middle_two = (complementSize - (first_last_value * 2)) / 2;

    ulong r2_start = blockSize, r2_end = first_last_value;
    ulong r3_start = first_last_value, r3_end = (r2_end + middle_two);
    ulong r4_start = (r2_end + middle_two), r4_end = (r2_end + middle_two * 2);
    ulong r5_start = (r2_end + middle_two * 2), r5_end = complementSize;

    ulong c2_start = blockSize, c2_end = first_last_value;
    ulong c3_start = first_last_value, c3_end = (r2_end + middle_two);
    ulong c4_start = (r2_end + middle_two), c4_end = (r2_end + middle_two * 2);
    ulong c5_start = (r2_end + middle_two * 2), c5_end = complementSize;

    for (ulong r2 = r2_start; r2 < r2_end; ++r2)
    {
        almostPrincipleMinor_row[blockSize] = complement_block_I[r2];

        for (ulong r3 = r3_start + 1; r3 < r3_end; ++r3)
        {
            almostPrincipleMinor_row[blockSize + 1] = complement_block_I[r3];

            for (ulong r4 = r4_start + 1; r4 < r4_end; ++r4)
            {
                almostPrincipleMinor_row[blockSize + 2] = complement_block_I[r4];

                for (ulong r5 = r5_start + 1; r5 < r5_end; ++r5)
                {
                    almostPrincipleMinor_row[blockSize + 3] = complement_block_I[r5];

                    for (ulong c2 = c2_start; c2 < c2_end; c2++)
                    {
                        almostPrincipleMinor_col[blockSize] = complement_block_I[c2];

                        for (ulong c3 = c3_start; c3 < c3_end; c3++)
                        {
                            almostPrincipleMinor_col[blockSize + 1] = complement_block_I[c3];

                            for (ulong c4 = c4_start; c4 < c4_end; c4++)
                            {
                                almostPrincipleMinor_col[blockSize + 2] = complement_block_I[c4];

                                for (ulong c5 = c5_start; c5 < c5_end; c5++)
                                {
                                    almostPrincipleMinor_col[blockSize + 3] = complement_block_I[c5];

                                    _mat_ almostPrincipleMinor;
                                    almostPrincipleMinor.SetDims(blockSize + 4, blockSize + 4);
                                    makeMatrixFromRowColCombination(almostPrincipleMinor_row, almostPrincipleMinor_col, mat, almostPrincipleMinor);

                                    if (IsZero(determinant(almostPrincipleMinor)))
                                    {
                                        cout << "\n apm :: \n " << almostPrincipleMinor << endl;
                                        cout << "\n ### [rPattern-4D] determinant is zero (4D) ...\t PM-size :: " << blockSize << "\t @ pId :: " << processorId << endl;
                                        printCombination2(almostPrincipleMinor_row, blockSize + 4);
                                        printCombination2(almostPrincipleMinor_col, blockSize + 4);

                                        for (size_t i = 0; i < (blockSize + 4); i++)
                                        {
                                            row[i] = almostPrincipleMinor_row[i];
                                            col[i] = almostPrincipleMinor_col[i];
                                        }

                                        return true;
                                    }

                                } // END : col4_ptr
                            }     // END : col3_ptr
                        }         // END:col2_ptr
                    }             // End:col1_ptr

                } // END :: row4_ptr
            }     // END :: row3_ptr
        }         // End:row2_ptr
    }             // END:row1_ptr

    return false;
}

template <class _mat_, class FiniteField>
bool makeAllAlmostPrincipleMinors(const _mat_ &minor, const _mat_ &mat, ulong blockSize, ulong block_I[], ulong &dim, ulong row[], ulong col[])
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong orgMatRows = mat.NumRows();
    ulong minorMatRows = blockSize;

    ulong complementSize = orgMatRows - minorMatRows;
    ulong complement_block_I[complementSize];

    // Populate complement block I array
    ulong cnt = 0;
    ulong index = 0;
    while (cnt < orgMatRows)
    {
        if (cnt != block_I[0])
            complement_block_I[index++] = cnt++;
        else
            cnt += blockSize;
    }

    // if (oneDeviation<_mat_>(blockSize, block_I, mat, complementSize, complement_block_I))
    // {
    //     MPI_Abort(MPI_COMM_WORLD, 73);
    //     // return true;
    // }
    // if (twoDeviation(blockSize, block_I, mat, complementSize, complement_block_I))
    // {
    //     MPI_Abort(MPI_COMM_WORLD, 73);
    //     // return true;
    // }
    // if (threeDeviation(blockSize, block_I, mat, complementSize, complement_block_I))
    // {
    //     MPI_Abort(MPI_COMM_WORLD, 73);
    //     // return true;
    // }

    if (fourDeviation_segmented(blockSize, block_I, mat, complementSize, complement_block_I, dim, row, col))
    {
        // MPI_Abort(MPI_COMM_WORLD, 73);
        return true;
    }

    return false;
}

/**
 * @brief Serial version...
 *
 * @tparam _mat_
 * @tparam FiniteField
 * @param ordP
 * @return true
 * @return false
 */
template <class _mat_, class FiniteField>
bool principleDeviation(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    _mat_ mat = getKernelFromFile<_mat_>(processorId, totalNumberOfProcessors, ordP);

    ulong matRows = mat.NumRows();
    ulong max_blockSize = matRows / 2;

    for (ulong blockSize = 2; blockSize < max_blockSize; ++blockSize)
    {
        double time_s = GetTime();
        masterPrint(processorId) << "\n Processing block size :: " << blockSize << " of max_blockSize :: " << max_blockSize << endl;
        ulong block_I[blockSize];
        ulong block_I_complement[matRows - blockSize], block_J_complement[matRows - blockSize];

        initCombinations(block_I, blockSize);

        while (true)
        {
            _mat_ principleMinor;
            principleMinor.SetDims(blockSize, blockSize);

            makeMatrixFromRowColCombination(block_I, block_I, mat, principleMinor);

            if (makeAllAlmostPrincipleMinors<_mat_, FiniteField>(principleMinor, mat, blockSize, block_I))
                return true;

            if (!isLastCombination(block_I, blockSize, max_blockSize))
                _getNextCombination_continous(block_I, max_blockSize, blockSize);
            else
                break;
        }
        double time_e = GetTime();
        cout << "\n blockSize :: " << blockSize << "\t Time :: " << (time_e - time_s) << " Sec.\t pId :: " << processorId << endl;
    }
    return false;
}
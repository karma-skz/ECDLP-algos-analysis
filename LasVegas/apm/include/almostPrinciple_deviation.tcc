#include "constants.hpp"
#include "utils.hpp"
#include "containment.tcc"

template <class _mat_, class FiniteField>
FiniteField getDeterminant(const _mat_ &mat, _mat_ newMat, const _mat_ &metaMat, _mat_ rowMat, _mat_ colMat, FiniteField determinantVal)
{
    ulong matRow = mat.NumRows();

    newMat[0][matRow] = colMat[0][0];
    rowMat[0] = rowMat[0] - ((rowMat[0][0] / newMat[0][0]) * newMat[0]);
    for (ulong i = 1; i < matRow; i++)
    {
        for (ulong j = 0; j < i; ++j)
        {
            newMat[i][matRow] = colMat[i][0] - (metaMat[i][j] * colMat[j][0]);
            colMat[i][0] = newMat[i][matRow];
        }
        rowMat[0] = rowMat[0] - (rowMat[0][i] / newMat[i][i]) * newMat[i];
    }
    determinantVal *= rowMat[0][matRow];

    return determinantVal;
}

template <class _mat_, class FiniteField>
FiniteField getDeterminant_generic(_mat_ newMat, const _mat_ &metaMat, FiniteField determinantVal)
{
    ulong matRow = metaMat.NumRows();
    ulong matCol = metaMat.NumCols();

    ulong newMatRow = newMat.NumCols();

    for (ulong i = 1; i < matRow; ++i)
    {
        ulong pivotRow = 0;
        for (ulong j = 0; j < i; ++j)
        {
            for (ulong k = matRow; k < newMatRow; ++k)
                newMat[i][k] = newMat[i][k] - ((metaMat[i][j]) * (newMat[pivotRow][k]));

            ++pivotRow;
        }
    }

    gauss(newMat, newMat.NumRows() - 1);

    for (ulong i = matCol; i < newMat.NumRows(); ++i)
        determinantVal *= newMat[i][i];

    return determinantVal;
}

template <class _mat_>
void getComplementBlock(ulong blockSize, ulong block_I[], const _mat_ &mat, ulong complement_block_I[])
{
    ulong orgMatRows = mat.NumRows();

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
}

bool isCombinationValid(const ulong rowCombo[], const ulong colCombo[], ulong deviationOf, const ulong block_I[], ulong blockSize, const ulong complement_block_I[])
{
    if ((complement_block_I[rowCombo[0]] < block_I[0]) && (complement_block_I[rowCombo[deviationOf - 1]] > block_I[blockSize - 1]))
        return false;

    if ((complement_block_I[colCombo[0]] < block_I[0]) && (complement_block_I[colCombo[deviationOf - 1]] > block_I[blockSize - 1]))
        return false;

    if ((complement_block_I[rowCombo[deviationOf - 1]] < block_I[0]) && (complement_block_I[colCombo[0]] > block_I[blockSize - 1]))
        return false;

    if ((complement_block_I[rowCombo[0]] > block_I[blockSize - 1]) && (complement_block_I[colCombo[deviationOf - 1]] < block_I[0]))
        return false;

    return true;
}

template <class _mat_>
bool deviation_parallel_v2(ulong blockSize, ulong block_I[], const _mat_ &mat, ulong complementSize,
                           ulong complement_block_I[], ulong rowCombination_org[], ulong colCombination_org[], ulong quota)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong matRows = mat.NumRows();
    ulong deviationOf = 2;

    ulong rowCombo[deviationOf], colCombo[deviationOf];

    for (ulong i = 0; i < deviationOf; ++i)
    {
        rowCombo[i] = rowCombination_org[i];
        colCombo[i] = colCombination_org[i];
    }

    ulong apm_rows = blockSize + deviationOf;
    ulong almostPrincipleMinor_row[apm_rows];
    ulong almostPrincipleMinor_col[apm_rows];

    for (ulong i = 0; i < blockSize; ++i)
    {
        almostPrincipleMinor_row[i] = block_I[i];
        almostPrincipleMinor_col[i] = block_I[i];
    }

    ulong iterationCnt = 0;
    while (true)
    {
        for (ulong i = 0; i < deviationOf; ++i)
        {
            almostPrincipleMinor_row[blockSize + i] = complement_block_I[rowCombo[i]];
            almostPrincipleMinor_col[blockSize + i] = complement_block_I[colCombo[i]];
        }

        // if (isCombinationValid(rowCombo, colCombo, deviationOf, block_I, blockSize, complement_block_I))
        {
            _mat_ almostPrincipleMinor;
            almostPrincipleMinor.SetDims(apm_rows, apm_rows);
            makeMatrixFromRowColCombination(almostPrincipleMinor_row, almostPrincipleMinor_col, mat, almostPrincipleMinor);

            if (IsZero(determinant(almostPrincipleMinor)))
            {
                cout << "\n ######################################### \n";
                cout << "\n ### determinant is zero (2D) ...\t blockSize :: " << blockSize << "\t @ pId :: " << processorId << endl;
                printCombination2(almostPrincipleMinor_row, apm_rows);
                printCombination2(almostPrincipleMinor_col, apm_rows);

                cout << "\n apm :: \n"
                     << almostPrincipleMinor << endl;
                cout << "\n ######################################### \n";
                MPI_Abort(MPI_COMM_WORLD, 73);
            }
        }

        iterationCnt++;
        if (iterationCnt > quota)
            return false;

        if (!isLastCombination(colCombo, deviationOf, complementSize))
        {
            _getNextCombination(colCombo, complementSize, deviationOf);
        }
        else
        {
            if (isLastCombination(rowCombo, deviationOf, complementSize))
                break;

            initCombinations(colCombo, deviationOf);
            _getNextCombination(rowCombo, complementSize, deviationOf);
        }
    }

    return false;
}

template <class _mat_, class FiniteField>
bool deviation_parallel_v2_customDet(ulong blockSize, ulong block_I[], const _mat_ &mat, ulong complementSize,
                                     ulong complement_block_I[], ulong rowCombination_org[], ulong colCombination_org[], ulong quota)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong matRows = mat.NumRows();
    ulong deviationOf = 2;

    ulong rowCombo[deviationOf], colCombo[deviationOf];

    for (ulong i = 0; i < deviationOf; ++i)
    {
        rowCombo[i] = rowCombination_org[i];
        colCombo[i] = colCombination_org[i];
    }

    const ulong apm_rows = blockSize + deviationOf;
    ulong almostPrincipleMinor_row[apm_rows];
    ulong almostPrincipleMinor_col[apm_rows];

    for (ulong i = 0; i < blockSize; ++i)
    {
        almostPrincipleMinor_row[i] = block_I[i];
        almostPrincipleMinor_col[i] = block_I[i];
    }

    _mat_ principalMinor, metaMat;
    {
        principalMinor.SetDims(blockSize, blockSize);
        metaMat.SetDims(blockSize, blockSize);

        makeMatrixFromRowColCombination(block_I, block_I, mat, principalMinor);

        for (ulong j = 1; j < blockSize; ++j)
            metaMat[j][0] = principalMinor[j][0] / principalMinor[0][0];

        for (ulong i = 1; i < blockSize; ++i)
        {
            gauss(principalMinor, i);
            for (ulong j = (i + 1); j < blockSize; ++j)
                metaMat[j][i] = principalMinor[j][i] / principalMinor[i][i];
        }
    }

    FiniteField determinantVal;
    {
        if (typeid(FiniteField) == typeid(GF2E))
        {
            determinantVal = conv<FiniteField>("[1]");
        }
        else if (typeid(FiniteField) == typeid(ZZ_p))
        {
            determinantVal = conv<FiniteField>("1");
        }
        else
        {
            cout << "\n FiniteField (" << typeid(FiniteField).name() << ") not supported w.r.t. IDENTITY Element...";
            cout << "\n Exiting program from almostPrinciple_deviation.tcc => deviation_parallel_v2_customDet() :-( " << endl;
            exit(0);
        }
        for (ulong i = 0; i < blockSize; ++i)
            determinantVal *= principalMinor[i][i];
    }

    const ulong newMatDim = apm_rows;
    _mat_ newMat;
    newMat.SetDims(newMatDim, newMatDim);

    // Copy principal minor into newMat
    for (ulong i = 0; i < blockSize; i++)
        for (ulong j = 0; j < blockSize; j++)
            newMat[i][j] = principalMinor[i][j];

    ulong iterationCnt = 0;
    while (true)
    {
        // can it be done without this loop... can be deleted.. though it makes some thing easier for now....LoL
        for (ulong i = 0; i < deviationOf; ++i)
        {
            almostPrincipleMinor_row[blockSize + i] = complement_block_I[rowCombo[i]];
            almostPrincipleMinor_col[blockSize + i] = complement_block_I[colCombo[i]];
        }

        // if (isCombinationValid(rowCombo, colCombo, deviationOf, block_I, blockSize, complement_block_I))
        {
            // Copy deviation-row into newMat
            ulong cnt = 0;
            for (ulong i = blockSize; i < newMatDim; ++i)
            {
                for (ulong j = 0; j < newMatDim; ++j)
                    newMat[i][j] = mat[complement_block_I[rowCombo[cnt]]][almostPrincipleMinor_col[j]];
                ++cnt;
            }

            // copy deviation-col into newMat
            for (ulong i = 0; i < blockSize; ++i)
            {
                cnt = 0;
                for (ulong j = blockSize; j < newMatDim; ++j)
                    newMat[i][j] = mat[block_I[i]][complement_block_I[colCombo[cnt++]]];
            }
            // -----------------------------------------------------------------------

            FiniteField deter = getDeterminant_generic(newMat, metaMat, determinantVal);

            if (IsZero(deter))
            {
                cout << "\n ######################################### \n";
                cout << "\n ### determinant is zero (2D) ...\t blockSize :: " << blockSize << "\t @ pId :: " << processorId << endl;

                printCombination2(almostPrincipleMinor_row, apm_rows);
                printCombination2(almostPrincipleMinor_col, apm_rows);

                _mat_ almostPrincipleMinor;
                almostPrincipleMinor.SetDims(apm_rows, apm_rows);
                makeMatrixFromRowColCombination(almostPrincipleMinor_row, almostPrincipleMinor_col, mat, almostPrincipleMinor);
                cout << "\n apm :: \n"
                     << almostPrincipleMinor << endl;

                cout << "\n ######################################### \n";
                // MPI_Abort(MPI_COMM_WORLD, 73);
            }
        }

        iterationCnt++;
        if (iterationCnt > quota)
            return false;

        if (!isLastCombination(colCombo, deviationOf, complementSize))
        {
            _getNextCombination(colCombo, complementSize, deviationOf);
        }
        else
        {
            if (isLastCombination(rowCombo, deviationOf, complementSize))
                break;

            initCombinations(colCombo, deviationOf);
            _getNextCombination(rowCombo, complementSize, deviationOf);
        }
    }

    return false;
}

template <class _mat_, class FiniteField>
bool principleDeviation_parallel(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    //  All processors iterate over all kernel files.
    for (ulong fileId = 0; fileId < totalNumberOfProcessors; ++fileId)
    {
        _mat_ mat = getKernelFromFile<_mat_>(fileId, totalNumberOfProcessors, ordP);
        ulong kerColCnt = conv<ulong>(mat.NumCols());
        masterPrint(processorId) << "\n Processing File :: " << fileId << "\t mat.r :: " << mat.NumRows() << "\t mat.c :: " << mat.NumCols() << endl;

        ulong matRows = mat.NumRows();
        ulong max_blockSize = matRows / 2;

        for (ulong blockSize = 2; blockSize < max_blockSize; ++blockSize)
        {
            masterPrint(processorId) << " Processing block size :: " << blockSize << " of " << max_blockSize;
            masterPrint(processorId).flush();

            // Each processor will figure out the combinations it has to look at
            // Last processor gets extra (leftover) combinations
            // This is for 2-Deviations

            //-------------------------------------------------------------------------------------------------------------------
            ZZ totalRowCombinations = nCr(matRows - blockSize, 2);

            if (totalRowCombinations >= conv<ZZ>("18446744073709551615"))
            {
                cout << "\n nCr :: " << nCr(matRows - blockSize, 2) << endl;
                cout << "\n totalRowCombinations :: " << totalRowCombinations << endl;
                cout << "\n totalRowCombinations > 64 bits.... => principleDeviation_parallel() in almostPrinciple_deviation.tcc " << endl;
                cout << " The program will now terminate...\n";
                exit(0);
            }

            ZZ totalCombinations = totalRowCombinations * totalRowCombinations;
            ulong quota = conv<ulong>((totalCombinations / conv<ZZ>(totalNumberOfProcessors)));
            ulong extra = conv<ulong>((totalCombinations % conv<ZZ>(totalNumberOfProcessors)));

            ulong symbolVec_dim = (matRows - blockSize);
            ulong symbolVec[symbolVec_dim];
            for (int i = 0; i < symbolVec_dim; ++i)
                symbolVec[i] = i;

            ulong rowCombinationIndex = conv<ulong>((processorId * quota) / conv<ulong>(totalRowCombinations));
            ulong colCombinationIndex = conv<ulong>((processorId * quota) % conv<ulong>(totalRowCombinations));

            ulong rowCombinationVec[2], colCombinationVec[2];

            get_kth_combination(symbolVec, symbolVec_dim, 2, rowCombinationIndex, rowCombinationVec);
            get_kth_combination(symbolVec, symbolVec_dim, 2, colCombinationIndex, colCombinationVec);

            if (processorId == totalNumberOfProcessors - 1)
                quota += extra;
            //-------------------------------------------------------------------------------------------------------------------

            ulong block_I[blockSize];
            initCombinations(block_I, blockSize);

            ulong complementSize = matRows - blockSize;
            ulong block_I_complement[complementSize];

            double time_s = GetTime();
            // Iterate over principal minors
            while (!isLastCombination(block_I, blockSize, matRows))
            {
                getComplementBlock(blockSize, block_I, mat, block_I_complement);

                bool flag = deviation_parallel_v2<_mat_>(blockSize, block_I, mat, complementSize, block_I_complement, rowCombinationVec, colCombinationVec, quota);

                // MPI_BCAST flag
                // if atleast one flag is true -> STOP

                _getNextCombination_continous(block_I, matRows, blockSize);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double time_e = GetTime();
            masterPrint(processorId) << "\t Time :: " << (time_e - time_s) << " seconds.\n";
        }
    } //END :: for fileId
}
#include "EC_lasVegas.tcc"
#include "constants.hpp"
#include "utils.hpp"

template <class _mat_>
ZZ getDlp_2(const _mat_ ker, const long rowIndex, const ulong k_randomNums, const ulong t_randomNums,
            ZZ *PQ_randomNumbers, ZZ ordP)
{

    ZZ dlp_A = conv<ZZ>(0), dlp_B = conv<ZZ>(0);
    for (ulong k = 0; k < (k_randomNums); ++k)
    {
        if (!(IsZero(ker[rowIndex][k])))
        {
            dlp_A += PQ_randomNumbers[k];
        }
    }

    for (ulong k = k_randomNums; k < (k_randomNums + t_randomNums); ++k)
    {
        if (!(IsZero(ker[rowIndex][k])))
        {
            dlp_B += PQ_randomNumbers[k];
        }
    }

    {
        // ZZ p_old;
        ZZ_pPush push;
        ZZ_p::init(conv<ZZ>(ordP));
        ZZ_p A = conv<ZZ_p>(dlp_A);
        ZZ_p B = conv<ZZ_p>(dlp_B);

        if (!IsZero(A) && !IsZero(B))
        {
            ZZ_p DLP = (A / B);
            return conv<ZZ>(DLP);
        }
        else
        {
            std::cerr << B_RED_START << "\n Something is wrong... A or B is zero in getDLP() ...\n"
                      << RESET_TERM;
            return conv<ZZ>("0");
        }
    }
}

void computePartitionData_2x2(ulong r, partitionData_2x2 pD[])
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ZZ combos = nCr(r, 2);
    ZZ numberOfCombosEachProcessorGets = conv<ZZ>(combos / (conv<ZZ>(totalNumberOfProcessors)));
    ulong numberOfExtraCombos = conv<ulong>((combos)-conv<ZZ>(numberOfCombosEachProcessorGets * totalNumberOfProcessors));

    ulong cnt = 0;
    ulong pDCnt = 1;
    pD[0].i_start = 0;
    pD[0].j_start = 1;
    pD[0].quota = conv<ulong>(numberOfCombosEachProcessorGets);

    ulong quota = conv<ulong>(numberOfCombosEachProcessorGets);
    ulong dimenson = r;
    for (ulong i = 0; i < dimenson; ++i)
    {
        for (ulong j = i + 1; j < dimenson; ++j)
        {
            cnt++;
            if (cnt == quota)
            {
                pD[pDCnt].i_start = i;
                pD[pDCnt].j_start = j;
                pD[pDCnt].quota = conv<ulong>(numberOfCombosEachProcessorGets);

                pDCnt++;
                cnt = 0;
            }
        }
    }
    pD[totalNumberOfProcessors - 1].quota += numberOfExtraCombos;
    delete NodeName;
}

void sendPartitionDataToProcessors_2x2(ulong r)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    partitionData_2x2 pD[totalNumberOfProcessors];
    computePartitionData_2x2(r, pD);

    for (int i = 0; i < totalNumberOfProcessors; ++i)
    {
        ulong arr[3];
        arr[0] = pD[i].i_start;
        arr[1] = pD[i].j_start;
        arr[2] = pD[i].quota;
        MPI_Send(arr, 3, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
    }
    delete NodeName;
}

void receivepartitionData_2x2(partitionData_2x2 &pD, ulong processorId)
{
    ulong arr[3];
    MPI_Recv(arr, 3, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    pD.i_start = arr[0];
    pD.j_start = arr[1];
    pD.quota = arr[2];
}

template <class T>
bool isDependenceFound(T *arr, ulong count, ulong &col1, ulong &col2)
{
    for (ulong i = 0; i < count; ++i)
    {
        for (ulong j = i + 1; j < count; ++j)
        {
            if (arr[i] == arr[j])
            {
                col1 = i;
                col2 = j;
                return true;
            }
        }
    }
    return false;
}

/**
 * This is same as isDeterminantOfSubMatrixZero_Rainbow2 in EC_impl.tcc
 * The only difference being that this implementation is for parallel env.
 * @param mat
 * @param pD
 * @param rD
 * @return true or false
 */
template <class _mat_, class FiniteField>
bool is_2by2_DeterminantZero(const _mat_ &mat, const partitionData_2x2 &pD, resultData &rD)
{
    double s_time = GetTime();
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
    delete NodeName;

    ulong A = pD.i_start;
    ulong B = pD.j_start;

    ulong n = mat.NumRows();
    ulong mat_col = mat.NumCols();
    ulong cnt = 0;
    ulong depdencyCol1, depdencyCol2;
    for (ulong row1 = A; row1 < n; ++row1)
    {
        for (ulong row2 = B; row2 < n; ++row2)
        {
            FiniteField result[mat_col];

            for (ulong col = 0; col < mat_col; ++col)
            {
                if (IsZero(mat[row2][col]) || IsZero(mat[row1][col]))
                {
                    // TODO : update the row and col for which ever cell contains zero element
                    // either of them can contain
                    // rD.col1 = col;
                    // rD.col2 = col;
                    // rD.row1 = row1;
                    // rD.row2 = row2;
                    cout << "\n mat[row1][col] :: " << mat[row1][col] << "\t mat[row2][col] :: " << mat[row2][col] << endl;
                    // cout << mat[row1] << "\n"
                    //      << mat[row2] << endl;
                    return true;
                }
                else
                {
                    result[col] = mat[row1][col] / mat[row2][col];
                }
            }

            if (isDependenceFound(result, mat_col, depdencyCol1, depdencyCol2))
            {
                rD.col1 = depdencyCol1;
                rD.col2 = depdencyCol2;
                rD.row1 = row1;
                rD.row2 = row2;

                double e_time = GetTime();
                v_cout << "\n Solved by processor Id ::" << processorId << "\t on node :: " << NodeName << " \t Time :: " << (e_time - s_time) << " seconds \n";
                cout << "\n ZM @pId :: " << processorId << "\t row1 :: " << row1 << "\t row2 :: "
                     << row2 << "\t col1 :: " << depdencyCol1 << "\t col2 :: " << depdencyCol2 << endl;

                return true;
            }

            cnt++;
            if (cnt == pD.quota)
            {
                double e_time = GetTime();
                v_cout << "\n Iterations completed by processorId :: " << processorId << "\t on node :: " << NodeName << "\t Time :: " << (e_time - s_time) << " seconds \n";

                return false;
            }
        }
        A = row1 + 1;
        B = A + 1;
    }
    return false;
}

template <class _mat_>
void getNorhtWestElementLocation(const _mat_ &minor, const _mat_ &mat, ulong &i1, ulong &j1)
{
    for (int i = 0; i < mat.NumRows(); i++)
    {
        for (int j = 0; j < mat.NumCols(); j++)
        {
            if (mat[i][j] == minor[0][0])
            {
                i1 = i;
                j1 = j;
                return;
            }
        }
    }
}

template <class _mat_>
void getIndexForMinor(const _mat_ &minor, const _mat_ &mat, ulong I[], ulong J[])
{
    ulong minor_row = minor.NumRows();
    ulong index = 0;

    // Get north-west element position
    ulong i1 = 0, j1 = 0;
    getNorhtWestElementLocation(minor, mat, i1, j1);

    // getting columns
    for (ulong i = j1; i < mat.NumCols(); i++)
    {
        for (ulong j = 0; j < minor.NumCols(); j++)
        {
            if (mat[i1][i] == minor[0][j])
            {
                J[index++] = i;
                break;
            }
        }
    }

    index = 0;
    // getting rows
    for (ulong i = i1; i < mat.NumRows(); i++)
    {
        for (ulong j = 0; j < minor.NumRows(); j++)
        {
            if (mat[i][J[minor_row - 1]] == minor[j][minor.NumCols() - 1])
            {
                I[index++] = i;
                break;
            }
        }
    }
}

template <class _mat_, class FintieField>
ZZ solveDLP(const _mat_ &mat, ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
    delete NodeName;
    ulong rowIndex = mat.NumRows() - 1;

    ulong orgMatCol = mat.NumCols();
    ulong k_rn = (orgMatCol / 2) - 1;
    ulong t_rn = k_rn + 2;
    ZZ randomNumbers[(orgMatCol * 2)];

    getRandomNumbersFromFile(processorId, totalNumberOfProcessors, randomNumbers);

    ZZ DLP = getDlp_2(mat, rowIndex, k_rn, t_rn, randomNumbers, ordP);
    return DLP;
}

template <class _mat_, class FintieField>
ZZ solveDLP_rowIndex(const _mat_ &mat, ZZ ordP, ulong rowIndex)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
    delete NodeName;

    ulong orgMatCol = mat.NumCols();
    ulong k_rn = (orgMatCol / 2) - 1;
    ulong t_rn = k_rn + 2;
    ZZ randomNumbers[(orgMatCol * 2)];

    getRandomNumbersFromFile(processorId, totalNumberOfProcessors, randomNumbers, ordP);

    ZZ DLP = getDlp_2(mat, rowIndex, k_rn, t_rn, randomNumbers, ordP);
    return DLP;
}

template <class _mat_, class FiniteField>
_mat_ gaussian_extended(const _mat_ &orgMat, ulong J[], ulong len)
{
    _mat_ mat = orgMat;
    for (ulong i = 0; i < mat.NumRows(); i++)
    {
        for (ulong j = (i + 1); j < mat.NumRows(); j++)
        {
            if (IsZero(mat[i][J[i]]))
                continue;

            FiniteField ele = mat[j][J[i]] / mat[i][J[i]];

            for (ulong k = j; k < mat.NumRows(); ++k)
                mat[k] = mat[k] - (ele * mat[i]);
        }
    }
    return mat;
}

template <class _mat_>
_mat_ getExtendedMinor(const _mat_ &orgMat, ulong I[], ulong len)
{
    _mat_ extendedMinor;
    extendedMinor.SetDims(len, orgMat.NumCols());

    for (ulong i = 0; i < len; ++i)
    {
        extendedMinor[i] = orgMat[I[i]];
    }

    return extendedMinor;
}

template <class _mat_, class FiniteField>
bool elimination_minor(const _mat_ &minor, const _mat_ &orgMat, ZZ ordP)
{
    // Get position of this minor in the orgMat
    ulong I[minor.NumRows()], J[minor.NumRows()];
    getIndexForMinor(minor, orgMat, I, J);
    _mat_ extendeMinor = getExtendedMinor(orgMat, I, minor.NumRows());
    _mat_ ge_mat = gaussian_extended<_mat_, FiniteField>(extendeMinor, J, minor.NumRows());

    // cout << "\n before-ge_mat :: \n"
    //      << ge_mat << endl;

    for (int i = 0; i < extendeMinor.NumRows(); ++i)
    {
        ulong extendeMinor_zeroCnt = 0, geMat_zeroCnt = 0;
        for (ulong j = 0; j < extendeMinor.NumCols(); ++j)
        {
            if (IsZero(extendeMinor[i][j]))
                ++extendeMinor_zeroCnt;
            if (IsZero(ge_mat[i][j]))
                ++geMat_zeroCnt;
        }
    }

    ZZ DLP = solveDLP<_mat_, FiniteField>(ge_mat, ordP);
    if (ENABLE_LOGGING)
    {
        std::ostringstream ss;
        ss << " !!!-DLP :: " << DLP;
        Logger::getInstance()->info(ss);
    }

    if (DLP != conv<ZZ>("9343"))
    {
        return false;
    }
    return true;
}

void initCombinations_biggerMinor(ulong vec[], ulong prev_dim, ulong dimension, ulong hPrime_rows)
{
    ulong index = 0;
    ulong numberOfElements = prev_dim; // As we are looking for 2x2 minors in H'

    for (ulong i = numberOfElements; i < dimension; ++i)
    {
        while (true)
        {
            bool flag = true;
            for (ulong j = 0; j < numberOfElements; ++j)
            {
                if (index == vec[j])
                {
                    flag = false;
                    break;
                }
            }
            if (flag)
            {
                vec[i] = index;
                ++index;
                ++numberOfElements;
                break;
            }
            ++index;
            if (index > hPrime_rows)
                cout << "\n This is something not very good...\n";
        }
    }
}

/**
 * vec is divided into two parts. Static|Dynamic
 * If a valid Dynamic part is found the two are merged
 * to form a new vec
 */
void getNextCombination_biggerMinor(ulong vec[], ulong prev_dim, ulong dimension, ulong hPrime_rows)
{
    ulong s_s = prev_dim;        // 2x2 minor in H', static
    ulong d_s = dimension - s_s; // dynamic

    ulong d_arr[d_s];
    for (ulong i = s_s; i < dimension; ++i)
        d_arr[i - s_s] = vec[i];

    while (true)
    {
        _getNextCombination(d_arr, hPrime_rows, d_s);

        // Merge conditionally
        bool flag = true;
        for (ulong i = 0; i < d_s; ++i)
        {
            for (ulong j = 0; j < s_s; ++j)
            {
                if (d_arr[i] == vec[j])
                {
                    flag = false;
                    break;
                }
            }
            if (!flag)
                break;
        }
        // If the combination is valid => Merge static and dynamic parts
        if (flag)
        {
            for (ulong i = s_s; i < dimension; ++i)
                vec[i] = d_arr[i - s_s];
            break;
        }
    }
}

/**
 * vec is divided into two parts. Static|Dynamic
 * If the dynamic part has the last combination
 * then return true else check if there exists
 * a next combinatio if yes return true else false
 */
bool isLastCombination_biggerMinor(const ulong vec[], ulong prev_dim, ulong dimension, ulong hPrime_rows)
{
    ulong s_s = prev_dim;        // static
    ulong d_s = dimension - s_s; // dynamic

    ulong d_arr[d_s];
    for (ulong i = s_s; i < dimension; ++i)
        d_arr[i - s_s] = vec[i];

    if (isLastCombination(d_arr, d_s, hPrime_rows))
    {
        return true;
    }
    else
    {
        // If this is not the last combination check if the next combination can
        // be conditionally mearged with the static part
        ulong d_arr2[d_s];
        for (ulong i = 0; i < d_s; ++i)
            d_arr2[i] = d_arr[i];

        while (true)
        {
            _getNextCombination(d_arr2, hPrime_rows, d_s);

            if (isLastCombination(d_arr2, d_s, hPrime_rows))
                return true;

            // Check if d_arr2 can be merged conditionally
            bool flag = true;
            for (ulong i = 0; i < d_s; ++i)
            {
                for (ulong j = 0; j < s_s; ++j)
                {
                    if (d_arr2[i] == vec[j])
                    {
                        flag = false;
                        break;
                    }
                }
                if (!flag)
                    break;
            }
            // If the combination is valid => there exists a next combination
            // i.e. this is not the last combination
            if (flag)
                return false;
        }
    }
}

void getBinaryString(const ulong vec[], bool binaryString[], const ulong dimension)
{
    for (ulong i = 0; i < dimension; ++i)
        binaryString[vec[i]] = 1;
}

ulong hammingDistance(const bool binStr1[], const bool binStr2[], ulong matRow)
{
    ulong distance = 0;

    for (ulong i = 0; i < matRow; ++i)
        if (binStr1[i] != binStr2[i])
            ++distance;

    return distance;
}

ulong getHammingDistance_vec(const ulong I[], const ulong J[], const ulong dimension, ulong matRows)
{
    bool I_binString[matRows], J_binString[matRows];
    getBinaryString(I, I_binString, dimension);
    getBinaryString(J, J_binString, dimension);

    return hammingDistance(I_binString, J_binString, matRows);
}

void writeCombinationToLog(const ulong I_ext[], const ulong J_ext[], const ulong dimension)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
    delete NodeName;

    // std::ostringstream ss;
    for (ulong i = 0; i < dimension; ++i)
        cout << I_ext[i] << "\t";

    cout << "\n";

    for (ulong i = 0; i < dimension; ++i)
        cout << J_ext[i] << "\t";

    cout << "## dim :: " << dimension << " ## pId :: " << processorId << endl;

    // Logger::getInstance()->info(ss);
}

template <class _mat_>
bool processBiggerMinorOfDimension(const ulong I[], const ulong J[], const ulong prev_dim, const ulong dimension, const _mat_ &hPrime)
{
    if (dimension >= hPrime.NumRows())
        return false;

    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
    delete NodeName;

    ulong I_ext[dimension];
    ulong J_ext[dimension];

    for (ulong i = 0; i < prev_dim; i++)
    {
        I_ext[i] = I[i];
        J_ext[i] = J[i];
    }

    const ulong hPrime_rows = hPrime.NumRows();
    initCombinations_biggerMinor(I_ext, prev_dim, dimension, hPrime_rows);
    initCombinations_biggerMinor(J_ext, prev_dim, dimension, hPrime_rows);

    ulong whileCnt = 0, zeroMinorCnt = 0;
    ulong whileLimit = (hPrime.NumRows() - dimension) * (hPrime.NumRows() - dimension);

    while (true)
    {
        _mat_ minor;
        minor.SetDims(dimension, dimension);

        makeMatrixFromRowColCombination(I_ext, J_ext, hPrime, minor);

        if (IsZero(determinant(minor)))
        {
            ulong HD = getHammingDistance_vec(I_ext, J_ext, dimension, hPrime.NumRows());
            if (HD == 0)
            {
                int qwe123;
                cin >> qwe123;
            }
            cout << "\n " << B_WHITW_START << dimension << "x" << dimension << " Zero minor found...@ pID :: "
                 << processorId << B_CYAN_START << "\t Ham-Dist :: " << HD << RESET_TERM << endl;

            writeCombinationToLog(I_ext, J_ext, dimension);
            {
                ulong new_dim = dimension + 1;
                ulong end_new_dim = new_dim + 10;

                // while (new_dim < hPrime.NumRows() && new_dim < end_new_dim)
                // {
                //     // cout << " Procecssing new_dimension :: " << new_dim << endl;
                //     bool flag2 = processBiggerMinorOfDimension(I_ext, J_ext, dimension, new_dim, hPrime);
                //     if (flag2)
                //         return true;
                //     ++new_dim;
                // }
            }
            zeroMinorCnt++;
            return true;
            // break;
        }

        if (!isLastCombination_biggerMinor(J_ext, prev_dim, dimension, hPrime_rows))
        {
            getNextCombination_biggerMinor(J_ext, prev_dim, dimension, hPrime_rows);
        }
        else
        {
            if (isLastCombination_biggerMinor(I_ext, prev_dim, dimension, hPrime_rows))
                break;

            initCombinations_biggerMinor(J_ext, prev_dim, dimension, hPrime_rows);
            getNextCombination_biggerMinor(I_ext, prev_dim, dimension, hPrime_rows);
        }
        whileCnt++;
    }
    return false;
}

template <class _mat_>
void processBiggerMinors(_mat_ hPrime, _mat_ orgmat, ulong org_row1, ulong org_row2, ulong org_col1, ulong org_col2)
{
    ulong I[2];
    ulong J[2];

    if (org_row1 < org_row2)
    {
        I[0] = org_row1;
        I[1] = org_row2;
    }
    else
    {
        I[0] = org_row2;
        I[1] = org_row1;
    }
    if (org_col1 < org_col2)
    {
        J[0] = org_col1;
        J[1] = org_col2;
    }
    else
    {
        J[0] = org_col2;
        J[1] = org_col1;
    }

    ulong dimension = 3;
    while (dimension < 10)
    {
        ZZ expectedCombos = nCr((hPrime.NumRows() - 2), dimension);
        cout << "\n Procecssing dimension :: " << dimension << "\t expected-Combos :: " << expectedCombos;
        cout << " hPrime.R :: " << hPrime.NumRows() << "\t hPrime.C :: " << hPrime.NumCols() << endl;
        double s_time = GetTime();
        processBiggerMinorOfDimension(I, J, 2, dimension, hPrime);
        double e_time = GetTime();
        cout << "\n Time :: " << e_time - s_time << " sec. \n";
        dimension += 1;
    }
}

/**
 * This is same as isDeterminantOfSubMatrixZero_Rainbow2 in EC_impl.tcc
 * The only difference being that this implementation is for parallel env.
 * Function only for Schur Complement Serial. To test the number of zeros
 * when a minor is found.
 * => This is a implementation for END-2-END for schur complement.
 * => Also this checks that only one zero is added to a row after row transformation
 * @param mat
 * @param pD
 * @param rD
 * @return true or false
 */
template <class _mat_, class FiniteField>
bool is_2by2_DeterminantZero_2(const _mat_ mat, const partitionData_2x2 &pD, resultData &rD, _mat_ orgMat, ZZ ordP)
{
    double s_time = GetTime();
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
    delete NodeName;

    ulong A = pD.i_start;
    ulong B = pD.j_start;

    ulong n = mat.NumRows();
    ulong mat_col = mat.NumCols();
    ulong cnt = 0;
    ulong depdencyCol1, depdencyCol2;
    for (ulong row1 = A; row1 < n; ++row1)
    {
        for (ulong row2 = B; row2 < n; ++row2)
        {
            FiniteField result[mat_col];

            for (ulong col = 0; col < mat_col; ++col)
            {
                if (IsZero(mat[row2][col]) || IsZero(mat[row1][col]))
                {
                    // TODO : update the row and col for which ever cell contains zero element
                    // either of them can contain
                    rD.col1 = col;
                    rD.col2 = col;
                    rD.row1 = row1;
                    rD.row2 = row2;
                    cout << "\n mat[row1][col] :: " << mat[row1][col] << "\t mat[row2][col] :: " << mat[row2][col] << endl;
                    cout << mat[row1] << "\n"
                         << mat[row2] << endl;
                    return true;
                }
                else
                {
                    result[col] = mat[row1][col] / mat[row2][col];
                }
            }

            if (isDependenceFound(result, mat_col, depdencyCol1, depdencyCol2))
            {
                rD.col1 = depdencyCol1;
                rD.col2 = depdencyCol2;
                rD.row1 = row1;
                rD.row2 = row2;

                double e_time = GetTime();
                v_cout << "\n Solved by processor Id ::" << processorId << "\t on node :: " << NodeName << " \t Time :: " << (e_time - s_time) << " seconds \n";
                cout << "\n##########################################################\n";
                cout << "\n !ZM @pId :: " << processorId << "\t row1 :: " << row1 << "\t row2 :: "
                     << row2 << "\t col1 :: " << depdencyCol1 << "\t col2 :: " << depdencyCol2 << endl;

                {
                    // Testing minors which this 2x2-minor is a sub matrix of...
                    // ulong offset = orgMat.NumRows() - mat.NumRows();
                    // ulong org_row1 = row1 + offset;
                    // ulong org_row2 = row2 + offset;
                    // ulong org_col1 = depdencyCol1 + offset;
                    // ulong org_col2 = depdencyCol2 + offset;
                    // processBiggerMinors(mat, orgMat, org_row1, org_row2, org_col1, org_col2);

                    processBiggerMinors(mat, orgMat, row1, row2, depdencyCol1, depdencyCol2);
                }

                if (0)
                {
                    // Elimination Testing....
                    _mat_ twoMinor;
                    twoMinor.SetDims(2, 2);
                    twoMinor[0][0] = mat[row1][depdencyCol1];
                    twoMinor[0][1] = mat[row1][depdencyCol2];

                    twoMinor[1][0] = mat[row2][depdencyCol1];
                    twoMinor[1][1] = mat[row2][depdencyCol2];
                    cout << "\n twoMinor :: \n"
                         << twoMinor << endl;
                    cout << "\n determinant(twoMinor) :: " << determinant(twoMinor) << endl;

                    _mat_ detMat, E;
                    ulong principalMinor_row = orgMat.NumRows() - mat.NumRows();
                    ulong principalMinor_col = principalMinor_row;
                    detMat.SetDims(principalMinor_row + 2, principalMinor_col + 2);
                    E.SetDims(principalMinor_row, principalMinor_col);

                    ulong principalMinor_IndexCount = principalMinor_row + 2;
                    ulong rowIndex_I[principalMinor_IndexCount], colIndex_J[principalMinor_IndexCount];
                    for (int i = 0; i < principalMinor_IndexCount; ++i)
                    {
                        rowIndex_I[i] = i;
                        colIndex_J[i] = i;
                    }

                    makeMatrixFromRowColCombination(rowIndex_I, colIndex_J, orgMat, E);

                    rowIndex_I[principalMinor_row] = row1 + principalMinor_row;
                    rowIndex_I[principalMinor_row + 1] = row2 + principalMinor_row;

                    colIndex_J[principalMinor_col] = principalMinor_col + depdencyCol1;
                    colIndex_J[principalMinor_col + 1] = principalMinor_col + depdencyCol2;

                    makeMatrixFromRowColCombination(rowIndex_I, colIndex_J, orgMat, detMat);
                    cout << "\n Solving Schur Complement DLP..\n";
                    // This will solve the DLP in M
                    bool flag = elimination_minor<_mat_, FiniteField>(detMat, orgMat, ordP);

                    // Process all minors in M by eliminating rows/cols
                    _mat_ minor = eliminationTest<_mat_, FiniteField>(detMat, orgMat, ordP);

                    cout << "\n-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-\n";

                    if (ENABLE_LOGGING)
                        LOG_INFO("Processing E");

                    cout << "\n Processing E :: \n";
                    _mat_ minor1 = eliminationTest<_mat_, FiniteField>(E, orgMat, ordP);
                }

                return true;
            }

            cnt++;
            if (cnt == pD.quota)
            {
                double e_time = GetTime();
                cout << "\n Iterations completed by processorId :: " << processorId << "\t on node :: " << NodeName << "\t Time :: " << (e_time - s_time) << " seconds \n";

                return false;
            }
        }
        A = row1 + 1;
        B = A + 1;
    }
    return false;
}

template <class _mat_, class FiniteField>
void searchAllTwoMinors(ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);
    delete NodeName;

    ulong fileId = 0;
    // Iterate over all the kernel files.
    for (int i = 0; i < 1; ++i)
    {
        masterPrint(processorId) << " Processing fileID :: " << fileId << endl;
        _mat_ ker = getKernelFromFile<_mat_>(fileId, totalNumberOfProcessors);

        if (processorId == MASTER_NODE)
        {
            long rowIndex = -1;
            if (isKernelHaving_r_Zeros<_mat_>(ker, ulong(ker.NumCols() / 2), rowIndex))
            {
                cout << "\n Row has r-zeros zeros @ row :: " << rowIndex << endl;

                ZZ PQ_randomNumbers[ker.NumCols()];
                getRandomNumbersFromFile(i, totalNumberOfProcessors, PQ_randomNumbers);

                ZZ DLP = getDlp(ker, rowIndex, ((ker.NumCols() / 2) - 1), ((ker.NumCols() / 2) + 1), PQ_randomNumbers, ordP);
                cout << "\n \t\t\t\t\t\t\t\t DLP :: " << DLP << "\t ordP :: " << ordP << endl;

                // @TODO Design graceful termination
                MPI_Abort(MPI_COMM_WORLD, 10);
            }

            // Send to all processors partition data
            sendPartitionDataToProcessors_2x2(ker.NumRows());
        }

        // Receive partition data from MASTER
        partitionData_2x2 pD;
        resultData_2x2 rD;
        receivepartitionData_2x2(pD, processorId);
        MPI_Barrier(MPI_COMM_WORLD);

        if (is_2by2_DeterminantZero<_mat_, FiniteField>(ker, pD, rD))
        {
            // cout << "\n ZERO MINOR IN RAINBOW2...\n";
            // MPI_Abort(MPI_COMM_WORLD, 73);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        ++fileId;
    }
}

// _mat_ mat2;
// mat2.SetDims(2, 2);

// mat2[0][0] = mat[row1][depdencyCol1];
// mat2[0][1] = mat[row1][depdencyCol2];

// mat2[1][0] = mat[row2][depdencyCol1];
// mat2[1][1] = mat[row2][depdencyCol2];

// FiniteField a, b, c, d;

// a = mat2[0][0];
// b = mat2[0][1];

// c = mat2[1][0];
// d = mat2[1][1];

// FiniteField ans1 = a / c;
// _mat_ tmpMat;
// tmpMat.SetDims(1, orgMat.NumCols());
// tmpMat[0] = orgMat[row1 + principalMinor_row] - (ans1 * orgMat[row2 + principalMinor_row]);

// ulong zeroCntOrgMat = 0;
// ulong zeroCntProcessed = 0;
// for (int i = 0; i < orgMat.NumCols(); ++i)
// {
//     if (IsZero(orgMat[row1 + principalMinor_row][i]))
//         zeroCntOrgMat++;

//     if (IsZero(tmpMat[0][i]))
//         zeroCntProcessed++;
// }

// cout << B_WHITW_START << " #-of-zeros :: " << (zeroCntProcessed - zeroCntOrgMat) << RESET_TERM << endl;
// if ((zeroCntProcessed - zeroCntOrgMat) > 2)
// {
//     cout << "\n Un-Wanted number of zeros...\n";
//     int asd123;
//     cin >> asd123;
// }
// else
// {
//     //Get RandomNumber from file.
//     _mat_ new_orgMat;
//     new_orgMat = orgMat;

//     new_orgMat[row1 + principalMinor_row] = tmpMat[0];
//     ulong rowIndex = row1 + principalMinor_row;

//     ulong orgMatCol = orgMat.NumCols();
//     ulong k_rn = (orgMatCol / 2) - 1;
//     ulong t_rn = k_rn + 2;
//     ZZ randomNumbers[(orgMatCol * 2)];

//     getRandomNumbersFromFile(processorId, totalNumberOfProcessors, randomNumbers);

//     ZZ DLP = getDlp_2(new_orgMat, rowIndex, k_rn, t_rn, randomNumbers, ordP);
//     cout << "\n DLP :: " << DLP << endl;

//     // Testing Theorem 3.
//     ulong cnt_th3 = 0;
//     for (ulong i = 0; i < principalMinor_IndexCount; ++i)
//     {
//         if (IsZero(tmpMat[0][colIndex_J[i]]))
//         {
//             cnt_th3++;
//         }
//         else
//         {
//             cout << "\n Non-Zero element @ :: " << i << "\t tmpMat[0][" << colIndex_J[i] << "] :: " << tmpMat[0][colIndex_J[i]] << endl;
//             // int qaasd123;
//             // cin >> qaasd123;
//         }
//     }
//     cout << "\n tmp Zero position cnt :: " << cnt_th3 << endl;
// }

// {
// Testing extended-elimination-test
// if ((orgMat.NumRows() - mat.NumRows()) > 10)
// {
//     cout << "\n In HERE ... (orgMat.NumRows() - mat.NumRows()) :: "
//          << (orgMat.NumRows() - mat.NumRows()) << endl;

//     _mat_ detMat;
//     ulong principalMinor_row = orgMat.NumRows() - mat.NumRows();
//     ulong principalMinor_col = principalMinor_row;
//     detMat.SetDims(principalMinor_row + 2, principalMinor_col + 2);

//     ulong principalMinor_IndexCount = principalMinor_row + 2;
//     ulong rowIndex_I[principalMinor_IndexCount], colIndex_J[principalMinor_IndexCount];
//     for (int i = 0; i < principalMinor_IndexCount; ++i)
//     {
//         rowIndex_I[i] = i;
//         colIndex_J[i] = i;
//     }

//     ulong row1, row2, col1, col2;
//     row1 = (orgMat.NumRows() - mat.NumRows()) - 10;
//     row2 = row1;

//     col1 = (orgMat.NumRows() - mat.NumRows()) - 10;
//     col2 = col1;

//     rowIndex_I[principalMinor_row] = 21 + principalMinor_row;
//     rowIndex_I[principalMinor_row + 1] = 22 + principalMinor_row;

//     colIndex_J[principalMinor_col] = principalMinor_col + 15;
//     colIndex_J[principalMinor_col + 1] = principalMinor_col + 18;

//     for (int i = 0; i < principalMinor_IndexCount; i++)
//     {
//         for (int j = 0; j < principalMinor_IndexCount; j++)
//         {
//             detMat[i][j] = mat[rowIndex_I[i]][colIndex_J[j]];
//         }
//     }

//     eliminationTest(detMat);
// } //END :: If
// }
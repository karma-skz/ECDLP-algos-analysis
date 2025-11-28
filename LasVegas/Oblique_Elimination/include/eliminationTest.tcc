#include "containment.tcc"
#include "Logger.hpp"

void printCombination(const ulong vec[], ulong dimension)
{
    for (int i = 0; i < dimension; i++)
    {
        cout << vec[i] << "\t";
    }
    cout << endl;
}

void getNewIndexes(const ulong rowCombination[], ulong new_RowCombination[], ulong new_Combination, int mat_row)
{
    ulong index = 0;
    for (int j = 0; j < mat_row; j++)
    {
        bool flag = true;
        for (int i = 0; i < (mat_row - new_Combination); ++i)
        {
            if (j == rowCombination[i])
            {
                flag = false;
                break;
            }
        }
        if (flag)
        {
            new_RowCombination[index] = j;
            index++;
        }
    }
}

template <class _mat_>
void makeMatrixFromRowColCombination_2(ulong rowCombination[], ulong colCombination[], \ 
                                        const _mat_ &mat,
                                       _mat_ &minor, ulong rowCol_dimension)
{
    ulong newCombinationSize = minor.NumRows();
    ulong new_RowCombination[newCombinationSize];
    ulong new_ColCombination[newCombinationSize];

    getNewIndexes(rowCombination, new_RowCombination, newCombinationSize, mat.NumRows());
    getNewIndexes(colCombination, new_ColCombination, newCombinationSize, mat.NumRows());

    makeMatrixFromRowColCombination(new_RowCombination, new_ColCombination, mat, minor);
}

/**
 * Returns true if atleast one column has all zero elements.
 * Returns false if non of the column is all zero.
 */
template <class _mat_>
bool isAnyColumnZero(const _mat_ &minor)
{
    for (ulong col = 0; col < minor.NumCols(); col++)
    {
        bool flag = true;
        for (ulong row = 0; row < minor.NumRows(); row++)
        {
            if (!IsZero(minor[row][col]))
            {
                flag = false;
                break;
            }
        }
        if (flag)
        {
            return true;
        }
    }
    return false;
}

template <class _mat_>
bool isAnyRowColumnZero(const _mat_ &minor)
{
    bool colFlag = isAnyColumnZero(minor);
    bool rowFlag = isAnyColumnZero(transpose(minor));

    // if (IsZero(minor[1][1]))
    //     return true;

    return (colFlag || rowFlag);
}

template <class _mat_, class FiniteField>
void getRowSum(const _mat_ &mat, FiniteField arr[])
{
    for (ulong i = 0; i < mat.NumRows(); i++)
    {
        for (ulong j = 0; j < mat.NumCols(); j++)
        {
            arr[i] += mat[i][j];
        }
    }
}

/**
 * The input matrix is singular (i.e. determinant = 0)
 */
template <class _mat_, class FiniteField>
_mat_ eliminationTest(_mat_ mat, const _mat_ &orgMat, ZZ ordP)
{
    ulong mat_row = mat.NumRows();
    ulong mat_col = mat.NumCols();

    ulong dimension = 1;

    if (mat_row < 3)
    {
        cout << "\n In Elimination test ... matrix size TOOOOO small... " << mat_row << endl;
        _mat_ minor;
        minor.SetDims((mat_row - dimension), (mat_row - dimension));
        return minor;
    }

    while (mat_col > (dimension + 1))
    {
        ulong rowCombination[dimension];
        ulong colCombination[dimension];

        cout << "\n dimension :: " << dimension << "\t sub-matrix-dim :: " << (mat_row - dimension)
             << "\t mat-dim :: " << mat.NumRows() << endl;

        initCombinations(rowCombination, dimension);
        initCombinations(colCombination, dimension);
        ulong detCnt = 0, non_detCnt = 0, whileCnt = 0;
        while (whileCnt < TEN_MILLION)
        {
            ++whileCnt;
            // Make minor
            _mat_ minor;
            minor.SetDims((mat_row - dimension), (mat_row - dimension));

            makeMatrixFromRowColCombination_2(rowCombination, colCombination, mat, minor, dimension);
            if (!isAnyRowColumnZero(minor))
            {
                ulong newCombinationSize = minor.NumRows();
                ulong new_RowCombination[newCombinationSize];
                ulong new_ColCombination[newCombinationSize];

                getNewIndexes(rowCombination, new_RowCombination, newCombinationSize, mat.NumRows());
                getNewIndexes(colCombination, new_ColCombination, newCombinationSize, mat.NumRows());

                // Test minor determinant
                if (IsZero(determinant(minor)))
                {
                    if (elimination_minor<_mat_, FiniteField>(minor, orgMat, ordP))
                    {
                        cout << "\n minor :: \n"
                             << minor << endl;
                        cout << "\n determinant(minor) :: " << determinant(minor) << endl;

                        FiniteField arr[minor.NumRows()];
                        getRowSum<_mat_, FiniteField>(minor, arr);
                        if (ENABLE_LOGGING)
                        {
                            std::ostringstream ss;
                            ss << " Row-Sum :: ";
                            for (ulong i = 0; i < minor.NumRows(); i++)
                                ss << arr[i] << "\t";
                            LOG_INFO(ss);
                        }

                        printCombination(new_RowCombination, newCombinationSize);
                        printCombination(new_ColCombination, newCombinationSize);

                        cout << "\n ENTER :: ";
                        int asd123;
                        cin >> asd123;
                        ++detCnt;
                    }
                    else
                    {
                        if (ENABLE_LOGGING)
                        {
                            LOG_INFO("Minor Determinant not valid as cannot solve DLP...");
                            std::ostringstream ss1, ss2;
                            ss1 << " Row ";
                            ss2 << " Col ";
                            for (ulong i = 0; i < newCombinationSize; ++i)
                            {
                                ss1 << new_RowCombination[i] << "\t";
                                ss2 << new_ColCombination[i] << "\t";
                            }
                            LOG_INFO(ss1);
                            LOG_INFO(ss2);
                        }
                        ++non_detCnt;
                    }
                }
                else
                {
                    if (ENABLE_LOGGING)
                        LOG_INFO("Minor Determinant not Zero...");
                    ++non_detCnt;
                }
            }

            // Get Next Combibnation
            if (!isLastCombination(colCombination, dimension, mat_row))
            {
                _getNextCombination(colCombination, mat_row, dimension);
            }
            else
            {
                if (isLastCombination(rowCombination, dimension, mat_row))
                {
                    break;
                }
                initCombinations(colCombination, dimension);
                _getNextCombination(rowCombination, mat_row, dimension);
            }
        } // END : while
        cout << " zero-detCnt :: " << detCnt << "\t non_detCnt :: " << non_detCnt << "\t whileCnt :: " << whileCnt << endl;
        cout << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
        //-------------------------------------------------------
        dimension++;
    }
    _mat_ minor;
    minor.SetDims((mat_row - dimension), (mat_row - dimension));
    return minor;
}
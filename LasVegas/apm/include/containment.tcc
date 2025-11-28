#ifndef CONTAINMENT_HPP
#define CONTAINMENT_HPP

#include <cstdlib>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <mutex>

#include <NTL/ZZ_p.h>
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/mat_GF2E.h>
#include <NTL/mat_lzz_p.h>
#include <NTL/mat_GF2.h>
#include <NTL/mat_ZZ.h>

#include "RowCol.hpp"
#include "constants.hpp"

std::mutex mtx;

using namespace std;
using namespace NTL;

// These function's are at the end of this file
bool getNextCombination_partWise(ulong dimension, ulong r, ulong mainVec[], const ulong segLenVec[]);
bool isLastCombination_partWise(const ulong vec[], const ulong segLenVec[], ulong r, ulong dimension);

void writeCombinationToFile(ulong rowCombination[], ulong colCombination[], ofstream &foutDispMat, ulong dimension)
{
    foutDispMat << dimension << endl;
    for (int i = 0; i < dimension; ++i)
    {
        foutDispMat << rowCombination[i] << "\t";
    }
    foutDispMat << endl;
    for (int i = 0; i < dimension; ++i)
    {
        foutDispMat << colCombination[i] << "\t";
    }
    foutDispMat << "\n###\n";
    foutDispMat.flush();
}

void _getNextCombination(ulong currentVec[], ulong n, ulong r)
{
    ulong max_vector[r];
    ulong nextVec[r];
    max_vector[r - 1] = n - 1;
    for (int i = r - 2; i >= 0; --i)
    {
        max_vector[i] = max_vector[i + 1] - 1;
    }
    if (currentVec[r - 1] != max_vector[r - 1])
    {
        nextVec[r - 1] = currentVec[r - 1] + 1;
        // copy current vector to next vector
        for (int i = 0; i < r - 1; i++)
        {
            nextVec[i] = currentVec[i];
        }
    }
    else
    {
        int index = -1;
        for (int i = r - 1; i >= 0; --i)
        {
            if (currentVec[i] != max_vector[i])
            {
                index = i;
                break;
            }
        } // END::FOR
        nextVec[index] = currentVec[index] + 1;
        for (int i = 0; i < r; ++i)
        {
            if (i < index)
            {
                nextVec[i] = currentVec[i];
            }
            else if (i > index)
            {
                nextVec[i] = nextVec[i - 1] + 1;
            }
        }
    }
    // Copy nextVect to currentVec
    for (int i = 0; i < r; i++)
    {
        currentVec[i] = nextVec[i];
    }
}

bool isLastCombination(const ulong combination[], const ulong dimension, const ulong n)
{
    ulong start = n - dimension;
    for (int i = 0; i < dimension; ++i)
        if (combination[i] != (start + i))
            return false;

    return true;
}

void initCombinations(ulong vec[], const ulong dimension)
{
    for (int i = 0; i < dimension; i++)
        vec[i] = i;
}

void printCombination2(const ulong vec[], ulong dimension)
{
    for (int i = 0; i < dimension; i++)
    {
        cout << vec[i] << "\t";
    }
    cout << endl;
}

template <class _mat_>
void makeMatrixFromRowColCombination(const ulong rowCombination[], const ulong colCombination[], const _mat_ &mat, _mat_ &minor)
{
    ulong minorRow = minor.NumRows();
    ulong minorCol = minor.NumCols();
    for (int i = 0; i < minorRow; i++)
    {
        for (int j = 0; j < minorCol; j++)
        {
            minor[i][j] = mat[rowCombination[i]][colCombination[j]];
        }
    }
}

/**
 * Return true if a minor with determinant zero is found, and false o.w.
 * @param dimension
 * @param mat
 * @param foutDispMat
 * @return
 */
template <class _mat_>
ulong processAllSubMatricesOfDimension(ulong dimension, const _mat_ &mat, ofstream &foutDispMat)
{
    const ulong numRows = mat.NumRows();
    const ulong numCols = mat.NumCols();

    // < : non-square & square matrices, <= only for square matrices
    if (numRows < dimension)
    {
        cout << "\n Invalid dimension in processAllSubMatricesOfDimension() \n";
        return 0;
    }

    ulong rowCombination[dimension];
    ulong colCombination[dimension];
    initCombinations(rowCombination, dimension);
    initCombinations(colCombination, dimension);
    ulong whileCnt = 0;
    ulong zeroMinorCnt = 0;

    while (true)
    {
        _mat_ minor;
        minor.SetDims(dimension, dimension);

        makeMatrixFromRowColCombination(rowCombination, colCombination, mat, minor);

        if (IsZero(determinant(minor)))
        {
            zeroMinorCnt++;
            writeCombinationToFile(rowCombination, colCombination, foutDispMat, dimension);
        }

        if (!isLastCombination(colCombination, dimension, numCols))
        {
            _getNextCombination(colCombination, numCols, dimension);
        }
        else
        {
            if (isLastCombination(rowCombination, dimension, numRows))
            {
                foutDispMat << "\n whileCnt :: " << whileCnt << "\t zeroMinorCnt :: " << zeroMinorCnt << endl;
                break;
            }
            initCombinations(colCombination, dimension);
            _getNextCombination(rowCombination, numRows, dimension);
        }
        whileCnt++;
    }

    return zeroMinorCnt;
}

void initCombinations_partWise(ulong vec[], ulong r, const ulong segLenVec[])
{
    vec[0] = 0;
    vec[1] = segLenVec[0];
    for (int i = 2; i < r; ++i)
        vec[i] = segLenVec[i - 2] + segLenVec[i - 1];
}

/**
 * This function makes a vec such that this vector does not have
 * the elements from mainVec.
 *
 * @param vec : resultant vector
 * @param dimension : size of vec @param
 * @param mainVec : the elements of this vector will not be present in the resultant vector
 * @param r : size of the mainVec @param
 * @param numRows : upper bound for the elements that can be present in the vec @param
 */
void makeVectorFrom_partWiseSegCombination(ulong vec[], ulong dimension, const ulong mainVec[], ulong r, ulong numRows)
{
    ulong mainVec_ptr = 0;
    ulong vecCnt = 0;
    for (ulong i = 0; i < numRows; ++i)
    {
        if (mainVec_ptr < r)
            if (i != mainVec[mainVec_ptr])
                vec[vecCnt++] = i;
            else
                ++mainVec_ptr;
        else
            vec[vecCnt++] = i;
    }
}

/**
 * Return true if a minor with determinant zero is found, and false o.w.
 * Same as previous function just row combinations are done part wise.
 */
template <class _mat_>
ulong processAllSubMatricesOfDimension_partWise(ulong dimension, const _mat_ &mat, ofstream &foutDispMat)
{
    const ulong numRows = mat.NumRows();
    if (numRows <= dimension)
    {
        cout << "\n Invalid dimension in processAllSubMatricesOfDimension() \n";
        return 0;
    }

    ulong r = numRows - dimension;
    r = 3; // temporary - to be delated later :-(
    ulong segLenVec[r];
    segLenVec[0] = 5;                                     // 5
    segLenVec[1] = numRows * 0.6;                         // 65
    segLenVec[2] = numRows - segLenVec[1] - segLenVec[0]; // 90

    cout << "\n seg-1 :: " << segLenVec[0] << "\t seg-2 :: " << segLenVec[1] << "\t seg-3 :: " << segLenVec[2] << endl;

    ulong mainVec[r];
    initCombinations_partWise(mainVec, r, segLenVec);

    ulong rowCombination[dimension];
    makeVectorFrom_partWiseSegCombination(rowCombination, dimension, mainVec, r, numRows);

    ulong colCombination[dimension];
    initCombinations(colCombination, dimension);

    ulong whileCnt = 0;
    ulong zeroMinorCnt = 0;

    while (true)
    {
        int aa;
        cin >> aa;
        _mat_ minor;
        minor.SetDims(dimension, dimension);

        makeMatrixFromRowColCombination(rowCombination, colCombination, mat, minor);

        // cout << "\n determinant :: " << determinant(minor) << "\t row :: " << minor.NumRows() << endl;
        // for (ulong i = 0; i < r; ++i)
        //     cout << mainVec[i] << "\t";
        // cout << endl;
        // printCombination2(rowCombination, dimension);

        if (IsZero(determinant(minor)))
        {
            zeroMinorCnt++;
            writeCombinationToFile(rowCombination, colCombination, foutDispMat, dimension);
        }

        if (!isLastCombination(colCombination, dimension, numRows))
        {
            // TODO: combine the following two functions to make it more clean.
            // if (getNextCombination_partWise(numRows, r, mainVec, segLenVec))
            // {
            //     cout << "\n in if...\n";
            //     makeVectorFrom_partWiseSegCombination(rowCombination, dimension, mainVec, r, numRows);
            // }
            // else
            // {
            //     cout << "\n in else...\n";
            // }
            getNextCombination_partWise(numRows, r, mainVec, segLenVec);
            makeVectorFrom_partWiseSegCombination(rowCombination, dimension, mainVec, r, numRows);
        }
        else
        {
            if (isLastCombination_partWise(mainVec, segLenVec, r, numRows))
            {
                foutDispMat << "\n whileCnt :: " << whileCnt << "\t zeroMinorCnt :: " << zeroMinorCnt << endl;
                break;
            }
            initCombinations(colCombination, dimension);

            getNextCombination_partWise(numRows, r, mainVec, segLenVec);
            makeVectorFrom_partWiseSegCombination(rowCombination, dimension, mainVec, r, numRows);
        }

        whileCnt++;
    }

    return zeroMinorCnt;
}

/**
 * Return true if a minor with determinant zero is found, and false o.w.
 * @param dimension
 * @param mat
 * @param foutDispMat
 * @return
 */
template <class _mat_>
ulong processAllSubMatricesOfDimension_parallel(ulong dimension, const _mat_ &mat, ofstream &foutDispMat, ulong rowCombinationVec[], ulong colCombinationVec[], ulong quota)
{
    const ulong numRows = mat.NumRows();

    if (numRows <= dimension)
    {
        cout << "\n Invalid dimension in processAllSubMatricesOfDimension() \n";
        return 0;
    }

    ulong rowCombination[dimension];
    ulong colCombination[dimension];

    for (ulong i = 0; i < dimension; ++i)
    {
        rowCombination[i] = rowCombinationVec[i];
        colCombination[i] = colCombinationVec[i];
    }

    ulong whileCnt = 0;
    ulong zeroMinorCnt = 0;

    while (whileCnt < quota)
    {
        _mat_ minor;
        minor.SetDims(dimension, dimension);

        makeMatrixFromRowColCombination(rowCombination, colCombination, mat, minor);

        if (IsZero(determinant(minor)))
        {
            zeroMinorCnt++;
            writeCombinationToFile(rowCombination, colCombination, foutDispMat, dimension);
        }

        if (!isLastCombination(colCombination, dimension, numRows))
        {
            _getNextCombination(colCombination, numRows, dimension);
        }
        else
        {
            if (isLastCombination(rowCombination, dimension, numRows))
                break;

            initCombinations(colCombination, dimension);
            _getNextCombination(rowCombination, numRows, dimension);
        }
        whileCnt++;
    }

    foutDispMat << "\n Total minors processed :: " << whileCnt << "\t zeroMinorCnt :: " << zeroMinorCnt << endl;

    return zeroMinorCnt;
}

/**
 * Return true if a minor with determinant zero is found, and false o.w.
 * @param dimension
 * @param mat
 * @param foutDispMat
 * @return
 */
template <class _mat_, class FiniteField>
ulong processAllSubMatricesOfDimension_ALL_Det(ulong dimension, const _mat_ &mat, ofstream &foutDispMat)
{
    const ulong numRows = mat.NumRows();

    if (numRows <= dimension)
    {
        cout << "\n Invalid dimension in processAllSubMatricesOfDimension() \n";
        return 0;
    }

    ulong rowCombination[dimension];
    ulong colCombination[dimension];
    initCombinations(rowCombination, dimension);
    initCombinations(colCombination, dimension);
    ulong whileCnt = 0;
    ulong zeroMinorCnt = 0;

    const ulong modulus_p = conv<ulong>(ZZ_p::modulus());
    ulong detArr[modulus_p] = {0};
    ulong avg = 0;
    while (true)
    {
        _mat_ minor;
        minor.SetDims(dimension, dimension);

        makeMatrixFromRowColCombination(rowCombination, colCombination, mat, minor);

        ulong det = conv<ulong>(determinant(minor));
        detArr[det] = detArr[det] + 1;

        // if (det == 0)
        // {
        //     cout << "\n S ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n";
        //     cout << "\n det :: " << det << "\t determinant(minor) :: " << determinant(minor) << endl;
        //     for (ulong i = 0; i < modulus_p; ++i)
        //     {
        //         cout << detArr[i] << "\t";
        //     }
        //     cout << "\n E ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n";
        // int asd123;
        // cin >> asd123;
        // }

        if (det == 0)
            zeroMinorCnt++;

        if (!isLastCombination(colCombination, dimension, numRows))
        {
            _getNextCombination(colCombination, numRows, dimension);
        }
        else
        {
            if (isLastCombination(rowCombination, dimension, numRows))
            {
                break;
            }
            initCombinations(colCombination, dimension);
            _getNextCombination(rowCombination, numRows, dimension);
        }
        whileCnt++;
    }

    // cout << "\n S ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n";
    // cout << zeroMinorCnt << endl;
    // for (ulong i = 0; i < modulus_p; ++i)
    // {
    //     cout << detArr[i] << "\t";
    //     avg += detArr[i];
    // }
    // cout << "\n avg :: " << avg / modulus_p << "\t sum :: " << avg << endl;
    // cout << "\n E ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n";

    return zeroMinorCnt;
}

/**
 * Gives the next continous combination for the vector
 * Paramater n is not used as of now
 */
void _getNextCombination_continous(ulong currentVec[], ulong n, ulong r)
{
    for (ulong i = 0; i < r; ++i)
    {
        currentVec[i] = currentVec[i] + 1;
    }
}

/**
 * Return true if a minor with determinant zero is found, and false o.w.
 * Only consider continous rows
 * @param dimension
 * @param mat
 * @param foutDispMat
 * @return
 */
template <class _mat_>
ulong processAllSubMatricesOfDimension_continousRow(ulong dimension, const _mat_ &mat, ofstream &foutDispMat)
{
    const ulong numRows = mat.NumRows();

    if (numRows <= dimension)
    {
        cout << "\n Invalid dimension in processAllSubMatricesOfDimension() \n";
        return 0;
    }

    ulong rowCombination[dimension];
    ulong colCombination[dimension];
    initCombinations(rowCombination, dimension);
    initCombinations(colCombination, dimension);
    ulong whileCnt = 0;
    ulong zeroMinorCnt = 0;

    // Make Matrix and check the determinant
    ulong rowCnt = 0;
    ulong colCnt = 0;
    while (true)
    {
        _mat_ minor;
        minor.SetDims(dimension, dimension);

        makeMatrixFromRowColCombination(rowCombination, colCombination, mat, minor);

        if (IsZero(determinant(minor)))
        {
            zeroMinorCnt++;
            writeCombinationToFile(rowCombination, colCombination, foutDispMat, dimension);
        }

        if (!isLastCombination(colCombination, dimension, numRows))
        {
            _getNextCombination(colCombination, numRows, dimension);
            colCnt++;
        }
        else
        {
            if (isLastCombination(rowCombination, dimension, numRows))
            {
                foutDispMat << "\n whileCnt :: " << whileCnt << "\t zeroMinorCnt :: " << zeroMinorCnt << endl;
                break;
            }
            initCombinations(colCombination, dimension);
            colCnt = 0;

            _getNextCombination_continous(rowCombination, numRows, dimension);
            rowCnt++;
        }
        whileCnt++;
    }

    return zeroMinorCnt;
}

/**
 * Return true if a minor with determinant zero is found, and false o.w.
 * Only consider first k continous rows
 * @param dimension
 * @param mat
 * @param foutDispMat
 * @return
 */
template <class _mat_>
ulong processAllSubMatricesOfDimension_FirstContinousRow(ulong dimension, const _mat_ &mat)
{
    const ulong numRows = mat.NumRows();

    if (numRows <= dimension)
    {
        cout << "\n Invalid dimension in processAllSubMatricesOfDimension() \n";
        return 0;
    }

    ulong rowCombination[dimension];
    ulong colCombination[dimension];

    // initCombinations(rowCombination, dimension);

    for (ulong i = 0; i < dimension; ++i)
    {
        rowCombination[i] = i + 51;
    }

    // can have random col combinatio here...
    initCombinations(colCombination, dimension);

    cout << "\n Initial Row-Col :: \n ";
    printCombination2(rowCombination, dimension);
    printCombination2(colCombination, dimension);
    cout << endl;

    ZZ whileCnt = conv<ZZ>("0");
    ulong zeroMinorCnt = 0;

    // make Matrix and check the determinant
    ulong rowCnt = 0;
    ulong colCnt = 0;
    bool flag = false;
    while (1)
    {
        _mat_ minor;
        minor.SetDims(dimension, dimension);

        makeMatrixFromRowColCombination(rowCombination, colCombination, mat, minor);

        if (IsZero(determinant(minor)))
        {
            zeroMinorCnt++;
            cout << "\n";
            printCombination2(rowCombination, dimension);
            printCombination2(colCombination, dimension);
            cout << "\n Minor found... DLP should be solved...\n";
            flag = true;
            break;
        }

        if (!isLastCombination(colCombination, dimension, numRows))
            _getNextCombination(colCombination, numRows, dimension);
        else
            break;

        whileCnt++;
    }

    return flag;
}

/**
 * Return true if a minor with determinant zero is found, and false o.w.
 * Same as the original one... dddd...
 * @param dimension
 * @param mat
 * @param foutDispMat
 * @return
 */
template <class _mat_>
ulong processAllSubMatricesOfDimension_MaximalMinor(ulong dimension, const _mat_ &mat)
{
    const ulong numRows = mat.NumRows();

    if (numRows <= dimension)
    {
        cout << "\n Invalid dimension in processAllSubMatricesOfDimension() \n";
        return 0;
    }

    ulong rowCombination[dimension];
    ulong colCombination[dimension];

    for (ulong i = 0; i < dimension; ++i)
    {
        rowCombination[i] = i;
        colCombination[i] = i;
    }

    cout << "\n Initial Row-Col :: \n ";
    printCombination2(rowCombination, dimension);
    printCombination2(colCombination, dimension);
    cout << endl;

    ulong whileCnt = 0;
    ulong zeroMinorCnt = 0;

    while (true)
    {
        _mat_ minor;
        minor.SetDims(dimension, dimension);
        makeMatrixFromRowColCombination(rowCombination, colCombination, mat, minor);

        if (IsZero(determinant(minor)))
        {
            zeroMinorCnt++;
            cout << "\n Zero minor - row:col \n";
            printCombination2(rowCombination, dimension);
            printCombination2(colCombination, dimension);
            // exit(0);
            // return zeroMinorCnt;
        }

        if (!isLastCombination(colCombination, dimension, numRows))
        {
            _getNextCombination(colCombination, numRows, dimension);
        }
        else
        {
            if (isLastCombination(rowCombination, dimension, numRows))
            {
                cout << "\n whileCnt :: " << whileCnt << "\t zeroMinorCnt :: " << zeroMinorCnt << endl;
                break;
                return zeroMinorCnt;
            }
            initCombinations(colCombination, dimension);
            _getNextCombination(rowCombination, numRows, dimension);
        }
        whileCnt++;
    }

    return zeroMinorCnt;
}

template <class _mat_>
int isMinorPresent(const _mat_ &mat, ulong start, ulong end)
{
    ofstream fout;
    fout.open("out.txt");
    bool flag = true;
    for (int i = start; i <= (end); i++)
    {
        cout << " Processing minors of dim :: " << i << "\t mat_row :: " << mat.NumRows();
        cout.flush();
        double sTime = GetTime();
        ulong count = processAllSubMatricesOfDimension(i, mat, fout);

        if (count)
        {
            cout << "\t count :: " << count << "\t time :: " << GetTime() - sTime << " Sec. \n";
            // return true;
            flag = true;
        }
        else
        {
            flag = false;
        }
    }

    return int(flag);
}

template <class _mat_, class FiniteField>
int allDeterminant_emumeration(const _mat_ &mat, ulong start, ulong end)
{
    ofstream fout;
    fout.open("out.txt");
    bool flag = true;
    for (int i = start; i <= (end); i++)
    {
        cout << "\n Processing minors of dim :: " << i << "\t mat_row :: " << mat.NumRows();
        cout.flush();
        double sTime = GetTime();
        ulong count = processAllSubMatricesOfDimension_ALL_Det<_mat_, FiniteField>(i, mat, fout);

        if (count)
        {
            cout << "\t ZM count :: " << count << "\t time :: " << GetTime() - sTime << " Sec. \n";
            flag = true;
        }
        else
        {
            flag = false;
        }
    }

    return int(flag);
}

// ***************************************************************************
// Containment using RowCol class

/***
 * Return the Row/Col indices for the first minor of given dimension.
 * TODO : Make use of RowCol class insted of array of type ulong.
 */
template <class _mat_>
RowCol getFirstMinor(const _mat_ &mat, ulong dimension)
{
    const ulong numRows = mat.NumRows();

    if (numRows <= dimension)
    {
        cout << "\n Invalid dimension in processAllSubMatricesOfDimension() \n";
        return RowCol(0);
    }

    ulong rowCombination[dimension];
    ulong colCombination[dimension];
    initCombinations(rowCombination, dimension);
    initCombinations(colCombination, dimension);

    RowCol rc(dimension);
    for (ulong i = 0; i < dimension; ++i)
    {
        rc.rows[i] = rowCombination[i];
        rc.cols[i] = colCombination[i];
    }

    RowCol rc2 = getNextMinor(mat, rc);
    return rc2;
}

template <class _mat_>
RowCol getNextMinor(const _mat_ &mat, const RowCol &rc)
{
    const ulong numRows = mat.NumRows();

    if (numRows <= rc.dimension)
    {
        cout << "\n Invalid dimension in processAllSubMatricesOfDimension() \n";
        return RowCol(0);
    }

    ulong rowCombination[rc.dimension];
    ulong colCombination[rc.dimension];

    for (ulong i = 0; i < rc.dimension; ++i)
    {
        rowCombination[i] = rc.rows[i];
        colCombination[i] = rc.cols[i];
    }

    // Get the next combination
    {
        if (!isLastCombination(colCombination, rc.dimension, numRows))
        {
            _getNextCombination(colCombination, numRows, rc.dimension);
        }
        else
        {
            if (isLastCombination(rowCombination, rc.dimension, numRows))
            {
                return RowCol(0);
            }
            initCombinations(colCombination, rc.dimension);
            _getNextCombination(rowCombination, numRows, rc.dimension);
        }
    }

    while (true)
    {
        _mat_ minor;
        minor.SetDims(rc.dimension, rc.dimension);

        makeMatrixFromRowColCombination(rowCombination, colCombination, mat, minor);

        if (IsZero(determinant(minor)))
        {
            RowCol rc2(rc.dimension);
            for (ulong i = 0; i < rc.dimension; ++i)
            {
                rc2.rows[i] = rowCombination[i];
                rc2.cols[i] = colCombination[i];
            }
            return rc2;
        }

        if (!isLastCombination(colCombination, rc.dimension, numRows))
        {
            _getNextCombination(colCombination, numRows, rc.dimension);
        }
        else
        {
            if (isLastCombination(rowCombination, rc.dimension, numRows))
            {
                RowCol rc(0);
                return rc;
            }
            initCombinations(colCombination, rc.dimension);
            _getNextCombination(rowCombination, numRows, rc.dimension);
        }
    }

    return RowCol(0);
}

// Delete this function.
void getKth_combination(ulong n, ulong r, ulong k, ulong combination[])
{
    // Use mutex/semaphore - so as to be thread safe - Hopefully
    mtx.lock();

    // Write n, r, k to a file
    ofstream fout(Kth_combination_InFile);
    if (!fout)
    {
        cout << "\n Unable to create/access out.txt file... \n";
        exit(0);
    }

    fout << n << " " << r << " " << k << endl;
    fout.close();

    // Execute the python script
    system("python3 ./include/getKth_combination.py");

    // Wait for a second or less - for the python script to execute
    usleep(9343);

    // Read the combination from the file
    ifstream fin(Kth_combination_OutFile);
    if (!fin)
    {
        cout << "\n Unable to open/access in.txt file...\n";
        exit(0);
    }

    for (int i = 0; i < r; ++i)
        fin >> combination[i];

    fin.close();
    mtx.unlock();
}

/**
 * https://github.com/more-itertools/more-itertools/blob/master/more_itertools/recipes.py
 * Function to get the kth combination.
 * @param : vec : a vector of all symbols
 * @param : n : n from nCr
 * @param : r : r from nCr
 * @param : index : value of k
 * @param : result_vec : answer is returned in this vector of size 'r'
 * n : lenght of vec
 */
bool get_kth_combination(ulong vec[], ulong n, ulong r, ZZ index, ulong result_vec[])
{
    const ulong vec_len = n;
    const ulong result_vec_len = r;

    if (r < 0 || r > n)
    {
        cerr << "\n get_kth_combination :: Value error : invalid value of r :: " << r << endl;
        return false;
    }
    ZZ c = conv<ZZ>("1");
    ulong k = r < (n - r) ? r : (n - r); //    min(r, n - r);

    for (int i = 1; i < k + 1; ++i)
    {
        c = c * conv<ZZ>(floor((n - k + i)));
        c = c / conv<ZZ>(i);
    }

    if (index < 0)
        index += c;

    if (index < 0 || index >= c)
    {
        cerr << "\n get_kth_combination :: Index error : invalid value for index :: " << index << endl;
        return false;
    }

    ulong result_vec_cnt = 0;
    while (r)
    {
        c = (c * r) / n;
        n = n - 1;
        r = r - 1;

        while (index >= c)
        {
            index -= c;
            c = c * (n - r) / n;
            n = n - 1;
        }

        result_vec[result_vec_cnt] = vec[vec_len - n - 1];
        ++result_vec_cnt;
    }

    return true;
}

/**
 * https://github.com/more-itertools/more-itertools/blob/master/more_itertools/recipes.py
 * Function to get the kth combination.
 * @param : vec : a vector of all symbols
 * @param : n : n from nCr
 * @param : r : r from nCr
 * @param : index : value of k
 * @param : result_vec : answer is returned in this vector of size 'r'
 * n : lenght of vec
 */
bool get_kth_combination(ulong vec[], ulong n, ulong r, ulong _index, ulong result_vec[])
{
    const ulong vec_len = n;
    const ulong result_vec_len = r;

    ZZ index = conv<ZZ>(_index);

    if (r < 0 || r > n)
    {
        cerr << "\n get_kth_combination :: Value error : invalid value of r :: " << r << endl;
        return false;
    }
    ZZ c = conv<ZZ>("1");
    ulong k = r < (n - r) ? r : (n - r); //    min(r, n - r);

    for (int i = 1; i < k + 1; ++i)
    {
        c = c * conv<ZZ>(floor((n - k + i)));
        c = c / conv<ZZ>(i);
    }

    if (index < 0)
        index += c;

    if (index < 0 || index >= c)
    {
        cerr << "\n get_kth_combination :: Index error : invalid value for index :: " << index << endl;
        return false;
    }

    ulong result_vec_cnt = 0;
    while (r)
    {
        c = (c * r) / n;
        n = n - 1;
        r = r - 1;

        while (index >= c)
        {
            index -= c;
            c = c * (n - r) / n;
            n = n - 1;
        }

        result_vec[result_vec_cnt] = vec[vec_len - n - 1];
        ++result_vec_cnt;
    }

    return true;
}

/**
 * This function returns true if the given vector i.e. @param vec is the last combination
 * vector possible. Returns false O.W.
 *
 * TODO: Can this be done in a better way? i.e. re-implement it to be more efficient ???
 *
 */
bool isLastCombination_partWise(const ulong vec[], const ulong segLenVec[], ulong r, ulong dimension)
{
    ulong lastElement_Seg[r]; // also the end of the segment

    ulong orgMainVec[r];
    orgMainVec[0] = 0;
    orgMainVec[1] = segLenVec[0];
    for (int i = 2; i < r; ++i)
        orgMainVec[i] = segLenVec[i - 2] + segLenVec[i - 1];

    for (ulong i = 0; i < r; ++i)
        lastElement_Seg[i] = orgMainVec[i + 1] - 1;

    lastElement_Seg[r - 1] = dimension - 1;

    for (ulong i = 0; i < r; ++i)
        if (lastElement_Seg[i] != vec[i])
            return false;

    return true;
}

/**
 * This function takes a vector as an input and three segments giving back the next
 * combination w.r.t. the the segments or parts.
 *
 * @param dimension : n  (n from nCr )
 * @param r : r (r from nCr)
 * @param mainVec : The next vector for this given vector is returned in the same vector
 * @param segLenVec : provides the length for the three segments
 *
 * @return : false if the given (input) vector is the last possible vector, true o.w.
 */
bool getNextCombination_partWise(ulong dimension, ulong r, ulong mainVec[], const ulong segLenVec[])
{
    ulong lastElement_Seg[r]; // Also the end of the segment

    ulong orgMainVec[r];
    orgMainVec[0] = 0;
    orgMainVec[1] = segLenVec[0];
    for (int i = 2; i < r; ++i)
        orgMainVec[i] = segLenVec[i - 2] + segLenVec[i - 1];

    for (ulong i = 0; i < r; ++i)
        lastElement_Seg[i] = orgMainVec[i + 1] - 1;

    lastElement_Seg[r - 1] = dimension - 1;

    bool flag = true;
    int i = r - 1;
    while (i >= 0)
    {
        if (mainVec[i] != lastElement_Seg[i])
        {
            mainVec[i] += 1;
            flag = false;
            break;
        }

        // logic to reset
        if (mainVec[i] == lastElement_Seg[i])
        {
            for (ulong j = i; j < r; ++j)
            {
                mainVec[j] = orgMainVec[j];
            }
        }

        --i;
    }

    return flag;
}

#endif
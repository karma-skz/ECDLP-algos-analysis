#include <iostream>
#include <sstream>
#include <iomanip>

#include <NTL/ZZ.h>
#include <NTL/mat_GF2E.h>
#include <NTL/vec_ZZ_p.h>

#include "lasVegasUtils.hpp"
#include "EC_GF2E_Point.hpp"
#include "EC_GF2E.hpp"
#include "utils.hpp"

using namespace std;
using namespace NTL;

ulong generateRandomN(ulong sizeOfArray, ZZ *randomNumbers, ZZ p, ulong *weightVectorSize) {

    ulong max = 0;
    ulong i = 0;

    while (i < sizeOfArray) {
        bool flag = true;
        ZZ tmp = conv<ZZ>(RandomWord()) % p;

        for (long j = 0; j < i; ++j) {
            if (randomNumbers[j] == tmp) {
                flag = false;
                break;
            }
        }
        if (flag) {
            randomNumbers[i] = tmp;

            if (weightVectorSize[i] > max)
                max = weightVectorSize[i];
            ++i;
        }
    }

    return max;
}

/**
 * Works for d <= 834
 * @param d
 */
void generateWeightedVector(ulong weight, ulong weightVector[][3]) {

    ulong totalElements = ((weight + 1)*(weight + 2)) / 2;

    for (ulong i = 0; i < totalElements; ++i) {
        weightVector[i][0] = 0;
        weightVector[i][1] = 0;
        weightVector[i][2] = 0;
    }

    //fill first column
    ulong tmp_cnt1 = weight;
    ulong tmp_cnt2 = 0;
    ulong tmp_cnt3 = weight;

    for (ulong i = (totalElements - 1); i > 0; --i) {
        weightVector[i][0] = tmp_cnt1;

        if (tmp_cnt1 == 0) {
            tmp_cnt3--;
            tmp_cnt1 = tmp_cnt3;
        } else {
            tmp_cnt1--;
        }
    }

    //fill second column
    tmp_cnt1 = weight;
    tmp_cnt2 = 0;
    tmp_cnt3 = weight;

    for (ulong i = (totalElements - 1); i > 0; --i) {
        weightVector[i][1] = tmp_cnt2;

        if (tmp_cnt2 == tmp_cnt3) {
            tmp_cnt3--;
            tmp_cnt2 = 0;
        } else {
            tmp_cnt2++;
        }
    }

    //fill third column
    tmp_cnt3 = 1;
    tmp_cnt1 = weight + 1;
    tmp_cnt2 = 0;

    weightVector[0][2] = weight;
    for (ulong i = (totalElements - 1); i > 0; --i) {

        weightVector[i][2] = tmp_cnt2;
        tmp_cnt1--;

        if (tmp_cnt1 == 0) {
            tmp_cnt2++;
            tmp_cnt1 = weight - tmp_cnt3 + 1;
            tmp_cnt3++;
        }
    }
}

void generateRandomNumbers2(ulong k, ZZ randomNumbers[], ZZ p) {

    ulong randomNumberCnt = 0;

    ulong someCnt = 0;
    while (randomNumberCnt < k) {
        someCnt++;
        bool flag = true;
        ZZ random_integer;

        random_integer = conv<ZZ>(conv<ZZ>(RandomWord()) % p);

        if (IsOdd(random_integer)) {
            for (ulong i = 0; i < randomNumberCnt; ++i) {
                if (randomNumbers[i] == random_integer) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                randomNumbers[randomNumberCnt] = random_integer;
                randomNumberCnt++;
            }
        }
    }
    randomNumbers[0] = 1;
    randomNumbers[1] = (p - 1);
}

/**
 * This function permutes the first k_randomNumber and keeps the last t_random cols as it is.  
 * @param ker
 * @param newKer
 * @param colIndexArray
 */
void swapKernelColsSeperatePQ(const mat_GF2E ker, mat_GF2E & newKer, ulong *colIndexArray, ulong k_random, ulong t_random) {

    ulong ker_row = ker.NumRows();
    ulong ker_col = ker.NumCols();

    ZZ P_randomNumbers[k_random];
    generateRandomNumbers2(k_random, P_randomNumbers, conv<ZZ>(k_random));

    for (ulong i = 0; i < k_random; ++i) {
        colIndexArray[i] = conv<long>(P_randomNumbers[i]);
    }

    for (ulong i = k_random; i < ker_col; ++i) {
        colIndexArray[i] = i;
    }

    newKer.SetDims(ker_row, ker_col);

    for (ulong j = 0; j < ker_col; ++j) {
        ulong indexJ = colIndexArray[j];
        for (ulong i = 0; i < ker_row; ++i) {
            newKer[i][j] = ker[i][indexJ];
        }
    }
}

void swapKernelCols(const mat_GF2E ker, mat_GF2E &newKer, ulong colIndexArray[]) {

    ulong ker_row = ker.NumRows();
    ulong ker_col = ker.NumCols();
    newKer.SetDims(ker_row, ker_col);

    for (ulong j = 0; j < ker_col; ++j) {
        ulong indexJ = conv<long>(colIndexArray[j]);
        for (ulong i = 0; i < ker_row; ++i) {
            newKer[i][j] = ker[i][indexJ];
        }
    }
}

void generateRandomColumnPermutation(ulong k, ulong *randomNumbers, ZZ p) {

    ulong randomNumberCnt = 0;

    ulong someCnt = 0;
    while (randomNumberCnt < k) {
        someCnt++;
        bool flag = true;
        ZZ random_integer;

        random_integer = conv<ZZ>(conv<ZZ>(RandomWord()) % p);

        for (ulong i = 0; i < randomNumberCnt; ++i) {
            if (randomNumbers[i] == random_integer) {
                flag = false;
                break;
            }
        }
        if (flag) {
            randomNumbers[randomNumberCnt] = conv<long>(random_integer);
            randomNumberCnt++;
        }
    }
}

void printMatrix1(const mat_GF2E mat) {

    for (ulong i = 0; i < mat.NumRows(); ++i) {
        for (ulong j = 0; j < mat.NumCols(); ++j) {
            string tmpStr;
            std::stringstream ss;
            ss << mat[i][j]._GF2E__rep.xrep;
            ss>>tmpStr;
            std::cout << std::left \
                     << std::setw(10) \
                     << std::setfill(' ')
                    << tmpStr << "  ";
        }
        std::cout << "\n";
    }
}

void printVector1(vec_GF2E v) {
    for (ulong i = 0; i < v.length(); ++i) {
        cout << v[i]._GF2E__rep.xrep << " ";
    }
}

/**
 * Function to swap ith and jth column of a matrix.
 * @param i : first columns
 * @param j : second column
 * @param mat : Input Matrix
 */
mat_GF2E swapTwoCols(ulong i, ulong j, const mat_GF2E mat, mat_GF2E newMat) {

    newMat = mat;

    //Copy j-th column into i-th column
    for (ulong row = 0; row < newMat.NumRows(); ++row)
        newMat[row][i] = mat[row][j];

    //Copy i-th column into j-th column
    for (ulong row = 0; row < newMat.NumRows(); ++row)
        newMat[row][j] = mat[row][i];

    return newMat;
}

/**
 * Function to swap i-th and j-th column of a matrix.
 * The input matrix is modified,
 * @param i : first columns
 * @param j : second column
 */
void swapTwoCols(ulong i, ulong j, mat_GF2E &mat) {

    vec_GF2E tmpVec, tmpVec2;
    tmpVec.SetLength(mat.NumRows());
    tmpVec2.SetLength(mat.NumRows());

    //Copy i-th column in tmp vector
    for (ulong row = 0; row < mat.NumRows(); ++row)
        tmpVec[row] = mat[row][i];

    //Copy j-th column into i-th column
    for (ulong row = 0; row < mat.NumRows(); ++row) {
        mat[row][i] = mat[row][j];
    }

    //Copy tmp vector to j-th column
    for (ulong row = 0; row < mat.NumRows(); ++row)
        mat[row][j] = tmpVec[row];
}

/**
 * Function to swap ith  jth and kth column of a matrix.
 * @param i : first columns
 * @param j : second column
 * @param k : third column
 * @param mat : Input Matrix
 */
mat_GF2E swapThreeCols(ulong i, ulong j, ulong k, const mat_GF2E mat, mat_GF2E newMat) {

    newMat = mat;
    //Copy j-th column into i-th column
    for (ulong row = 0; row < newMat.NumRows(); ++row) {
        newMat[row][i] = mat[row][j];
        newMat[row][j] = mat[row][i];
        newMat[row][k] = mat[row][i];
    }

    //    int a;
    //    std::cin >> a;

    return newMat;
}

mat_GF2E swapFourCols(ulong i, ulong j, ulong k, ulong l, const mat_GF2E mat, mat_GF2E newMat) {

    newMat = mat;
    //Copy j-th column into i-th column
    for (ulong row = 0; row < newMat.NumRows(); ++row) {
        newMat[row][i] = mat[row][j];
        newMat[row][j] = mat[row][i];
        newMat[row][k] = mat[row][i];
    }

    int a;
    std::cin >> a;

    return newMat;
}

/**
 * Function to check if the given matrix has r zeros in on of its row.
 * If r zeros are found in a row the input variable rowIndex is assigned the 
 * row number and the function returns true.
 * 
 * @param mat : Matrix to test for r zeros
 * @param r : Value for r
 * @param rowIndex : If r zeros are found, this variable has the row number.
 * @return : True if matrix has r zeros else False.
 */
ulong isMatrixHaving_R_Zeros(const mat_GF2E mat, const ulong r, ulong &rowIndex) {

    ulong zeroCnt = 0;
    rowIndex = 0;

    for (ulong i = 0; i < mat.NumRows(); ++i) {
        zeroCnt = 0;
        for (ulong j = 0; j < mat.NumCols(); ++j) {
            if (IsZero(mat[i][j])) {
                zeroCnt++;
            }
        }

        if (zeroCnt == r) {
            rowIndex = i;
            return true;
        }
    }
    return false;
}

/**
 * Function to create all permutations of length three.
 * @param n : The size of Symmetric group Sn
 * @param arr : All permutations a stored in this 2D array. 
 *              Each row is a Permutation.
 */
void createLengthThreePermutation(ulong n, ulong arr[][3]) {
    ulong cnt = 0;
    for (ulong i = 0; i < n; ++i) {
        for (ulong j = 0; j < n; ++j) {
            if (i != j) {
                for (ulong k = 0; k < n; ++k) {
                    if (k != j && k != i) {
                        arr[cnt][0] = i;
                        arr[cnt][1] = j;
                        arr[cnt][2] = k;
                        cnt++;
                    }
                }
            }
        }
    }
}

/**
 * Returns 1 in case of an accident.
 * @param M
 * @param P
 * @param Q
 * @param k_randomNums
 * @param t_randomNums
 * @param PQ_randomNumbers
 * @param weightedVector_arr
 * @return : 1 in case of an Accident.
 */
int EC_GF2E::generateMatrix(mat_GF2E &M, EC_GF2E_Point P, EC_GF2E_Point Q, \
        ulong k_randomNums, ulong t_randomNums, ZZ *PQ_randomNumbers, ulong weightedVector_arr[][3]) {

    // <editor-fold defaultstate="collapsed" desc="Creating the first k = 3n-1 rows of M">
    for (ulong i = 0; i < k_randomNums; ++i) {
        EC_GF2E_Point P1;
        this->scalarMultiplicationDA(P, conv<ZZ>(PQ_randomNumbers[i]), P1);

        if ((P1.x == Q.x) && (P1.y == Q.y)) {
            return 1;
        }

        for (ulong j = 0; j < M.NumCols(); ++j) {
            M[i][j] = power(P1.x, weightedVector_arr[j][0]) * power(P1.y, weightedVector_arr[j][1]);
        }
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Creating Q rows">
    for (ulong i = k_randomNums; i < (k_randomNums + t_randomNums); ++i) {

        EC_GF2E_Point P1, P2;
        this->pointNegation(Q, P1);

        this->scalarMultiplicationDA(P1, conv<ZZ>(PQ_randomNumbers[i]), P2);

        if ((P2.x == Q.x) && (P2.y == Q.y)) {
            return 1;
        }

        for (ulong j = 0; j < M.NumCols(); ++j) {
            M[i][j] = power(P2.x, weightedVector_arr[j][0]) * power(P2.y, weightedVector_arr[j][1]);
        }
    }

    return 0;
    // </editor-fold>
}

void second_GE1(mat_GF2E &mat) {

    const ulong n = mat.NumRows();
    const long mat_col = mat.NumCols();
    try {
        ulong curRow = 1;
        for (ulong k = n; k < ((2 * n) - 1); ++k) {
            GF2E tmp = mat[curRow - 1][k];

            for (ulong i = curRow; i < n; ++i) {
                GF2E factor = mat[i][k];
                for (ulong j = 0; j < mat_col; ++j) {

                    if ((!IsZero(tmp)) && (!IsZero(factor))) {
                        if (!IsZero(mat[curRow - 1][j])) {
                            if (IsOne(tmp)) {
                                mat[i][j] = (factor * mat[curRow - 1][j]) + (mat[i][j]);

                            } else {
                                mat[i][j] = ((factor / tmp) * mat[curRow - 1][j]) + (mat[i][j]);
                            }
                        }
                    }
                }
            }
            curRow++;
        }
    } catch (...) {
        std::cerr << "\n Exception ";
    }
}

/**
 * Function to copy first row number of rows and columns from kernel to new mat
 * @param mat : Output matrix
 * @param ker : Input matrix
 */
void getMatrixFromKernel(mat_GF2E &mat, const mat_GF2E &ker) {

    ulong row = ker.NumRows();
    mat.SetDims(row, row);

    for (ulong i = 0; i < row; ++i) {
        for (ulong j = 0; j < row; ++j) {
            mat[i][j] = ker[i][j];
        }
    }
}

void getMatrixFromKernel(mat_GF2E &mat, const mat_GF2E &ker, ulong col) {

    ulong row = ker.NumRows();
    mat.SetDims(row, col);

    for (ulong i = 0; i < row; ++i) {
        for (ulong j = 0; j < col; ++j) {
            mat[i][j] = ker[i][j];
        }
    }
}

bool getCondensedMatrix(const mat_GF2E &mat, mat_GF2E &minor, long &row, long &col) {

    long rowMinor = minor.NumRows();
    long colMinor = minor.NumRows();

    for (long i = 0; i < rowMinor; ++i) {
        for (long j = 0; j < colMinor; ++j) {
            minor[i][j] = (mat[i][j] * mat[i + 1][j + 1]) -(mat[i][j + 1] * mat[i + 1][j]);
            if (IsZero(minor[i][j])) {
                row = i;
                col = j;
                return true;
            }
        }
    }
    return false;
}

void internalOfMatrix(const mat_GF2E &mat, mat_GF2E &resMat) {

    resMat.SetDims((mat.NumRows() - 2), (mat.NumRows() - 2));

    for (ulong i = 0; i < resMat.NumRows(); ++i) {
        for (ulong j = 0; j < resMat.NumCols(); ++j) {
            resMat[i][j] = mat[i + 1][j + 1];
        }
    }
}

void divideMinorByInternalOfMatrix(mat_GF2E &mat, const mat_GF2E &internalMat) {

    if ((mat.NumCols() != internalMat.NumCols()) || (mat.NumCols() != internalMat.NumCols())) {
        cerr << "\n divideMinorByInternalOfMatrix : Error => Dimension miss match...\n";
        return;
    }

    for (ulong i = 0; i < mat.NumRows(); ++i) {
        for (ulong j = 0; j < mat.NumCols(); ++j) {
            mat[i][j] = mat[i][j] / internalMat[i][j];
        }
    }
}

/**
 * @HighLevelAlgorithm
 *  Step 1: Copy the rxr sub-matrix of the given matrix to a new matrix
 *  Step 2: Apply Dodgson's condensation to the new matrix
 *  Step 3: If a sub-matrix with determinant zero is found return true.
 *  Step 4: If non of the sub-matrices have determinant zero return false.
 * 
 * Function to look for a sub-matrix with determinant zero from the given matrix
 * @param ker : The given matrix
 * @return : True if a sub-matrix with determinant zero is found.          
 */
bool isKernelHavingSubMatrixWithDeterminantZero(const mat_GF2E &ker, long &return_col, long &return_row, long &return_loopCnt) {
    mat_GF2E mat, condensedMat, internalOfMat, tmpInternalOfMat;
    ulong r = ker.NumRows();
    mat.SetDims(r, r);

    for (ulong i = 0; i < r; ++i) {
        for (ulong j = 0; j < r; ++j) {
            mat[i][j] = ker[i][j];
        }
    }

    condensedMat.SetDims(r - 1, r - 1);

    long row = 0, col = 0;
    //1. A-int = internal(mat)
    internalOfMatrix(mat, internalOfMat);

    //2. minor = condensed(mat)
    if (getCondensedMatrix(mat, condensedMat, row, col)) {
        return_row = row;
        return_col = col;
        return_loopCnt = 0;
        return true;
    }

    long loopCnt = 1;
    mat.SetDims(condensedMat.NumRows(), condensedMat.NumCols());

    while (1) {

        mat = condensedMat;
        condensedMat.SetDims(mat.NumRows() - 1, mat.NumCols() - 1);

        if (getCondensedMatrix(mat, condensedMat, row, col)) {
            return_row = row;
            return_col = col;
            return_loopCnt = loopCnt;
            return true;
        }

        internalOfMatrix(mat, tmpInternalOfMat);
        divideMinorByInternalOfMatrix(condensedMat, internalOfMat);

        internalOfMat.SetDims(tmpInternalOfMat.NumRows(), tmpInternalOfMat.NumCols());
        internalOfMat = tmpInternalOfMat;

        ++loopCnt;

        if (loopCnt > (r - 2)) {
            return false;
        }
    }
}

/**
 * @HighLevelAlgorithm
 *  Step 1: 
 *  Step 2: Apply Dodgson's condensation to the matrix
 *  Step 3: If a sub-matrix with determinant zero is found return true.
 *  Step 4: If non of the sub-matrices have determinant zero return false.
 * 
 * Function to look for a sub-matrix with determinant zero from the given matrix
 * @param ker : The given matrix
 * @return : True if a sub-matrix with determinant zero is found.          
 */
bool isMatrixHavingSubMatrixWithDeterminantZero(const mat_GF2E &matOrg, long &return_col, long &return_row, long &return_loopCnt) {

    mat_GF2E mat, condensedMat, internalOfMat, tmpInternalOfMat;

    mat = matOrg;
    ulong r = mat.NumRows();

    condensedMat.SetDims(r - 1, r - 1);

    long row = 0, col = 0;
    //1. A-int = internal(mat)
    internalOfMatrix(mat, internalOfMat);

    //2. minor = condensed(mat)
    if (getCondensedMatrix(mat, condensedMat, row, col)) {
        return_row = row;
        return_col = col;
        return_loopCnt = 0;

        return true;
    }

    long loopCnt = 1;
    mat.SetDims(condensedMat.NumRows(), condensedMat.NumCols());

    while (1) {

        mat = condensedMat;
        condensedMat.SetDims(mat.NumRows() - 1, mat.NumCols() - 1);

        if (getCondensedMatrix(mat, condensedMat, row, col)) {
            return_row = row;
            return_col = col;
            return_loopCnt = loopCnt;

            return true;
        }

        internalOfMatrix(mat, tmpInternalOfMat);
        divideMinorByInternalOfMatrix(condensedMat, internalOfMat);

        internalOfMat.SetDims(tmpInternalOfMat.NumRows(), tmpInternalOfMat.NumCols());
        internalOfMat = tmpInternalOfMat;

        ++loopCnt;

        if (loopCnt > (r - 2)) {
            return false;
        }
    }
}

ulong getNumberOfOnesInMatrix(const mat_GF2E &mat) {

    ulong cnt = 0;

    for (ulong i = 0; i < mat.NumRows(); ++i) {
        for (int j = 0; j < mat.NumCols(); j++) {
            if (IsOne(mat[i][j])) {
                cnt++;
            }
        }
    }

    return cnt;
}

ulong getNumberOfSameElements(const mat_GF2E &mat) {

    ulong cnt = 0;
    ulong n = mat.NumCols();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > i) {
                for (ulong k = 0; k < n; ++k) {
                    if (mat[i][k] == mat[j][k]) {
                        cnt++;
                    }
                }
            }
        }
    }
    return cnt;
}
//==============================================================================
//Rainbow 3 Start

ZZ nCr(ulong n, ulong r) {

    ZZ numerator = factorial(conv<ZZ>(n));
    ZZ denominator = factorial(conv<ZZ>(n - r)) * factorial(conv<ZZ>(r));
    ZZ factorial = numerator / denominator;

    return factorial;
}

template <class T>
void getVectorMatrix_3(T &vectorMatrix, const T &mat, ulong numberOfRows) {

    ulong n = mat.NumCols();
    vectorMatrix.SetDims(numberOfRows, n);

    ulong newMatrixRowCnt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > i) {
                for (ulong k = 0; k < n; ++k) {
                    vectorMatrix[newMatrixRowCnt][k] = mat[i][k] + mat[j][k];
                }
                newMatrixRowCnt++;
            }
        }
    }
}

template <class T>
void getVectorMatrix_4(T &vectorMatrix, const T &mat, ulong numberOfRows) {

    ulong n = mat.NumCols();
    vectorMatrix.SetDims(numberOfRows, n);

    ulong newMatrixRowCnt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j > i) {
                for (ulong k = 0; k < n; ++k) {
                    vectorMatrix[newMatrixRowCnt][k] = mat[i][k] + mat[j][k];
                }
                newMatrixRowCnt++;
            }
        }
    }
}

bool isSubVectorPresentInVector(const vec_GF2E &v, const vec_GF2E &matRow) {

    ulong numberOfElements = matRow.length();
    ulong threeCnt = 0;
    for (ulong i = 0; i < numberOfElements; ++i) {
        GF2E element = v[i];
        for (ulong j = 0; j < numberOfElements; ++j) {
            //            cout << " element :: " << element._GF2E__rep.xrep << "\t matRow[j] :: " << matRow[j]._GF2E__rep.xrep << endl;
            if (element == matRow[j]) {
                threeCnt++;
                if (threeCnt == 3) {
                    return true;
                }
                break;
            }
        }
    }
    //    cout << "\n threeCnt :: " << threeCnt << endl;
    return false;
}

bool isVectorPresentInMatrix(const vec_GF2E &v, const mat_GF2E &mat) {

    for (ulong i = 0; i < mat.NumRows(); ++i) {
        vec_GF2E matRow;
        matRow = mat[i];
        if (isSubVectorPresentInVector(v, matRow)) {
            return true;
        }
    }
    return false;
}

bool isDependencyFound_Rainbow3(const mat_GF2E &mat) {

    //    cout << "\n in isDependencyFound_Rainbow3 ...\n";
    mat_GF2E vectorMatrix;
    ulong dim = conv<ulong>(nCr(mat.NumRows(), 2));
    vectorMatrix.SetDims(dim, dim);

    getVectorMatrix_3(vectorMatrix, mat, dim);

    vec_GF2E v;
    for (ulong i = 0; i < vectorMatrix.NumRows(); ++i) {
        v = vectorMatrix[i];
        if (isVectorPresentInMatrix(v, mat)) {
            return true;
        }
    }

    //    cout << "\n out isDependencyFound_Rainbow3 ...\n";
    return false;
}

//Rainbow 3 - End
//==============================================================================

bool isDependencyFound_Rainbow4(const mat_GF2E &mat) {

    //    cout << "\n in isDependencyFound_Rainbow3 ...\n";
    mat_GF2E vectorMatrix;
    ulong dim = conv<ulong>(nCr(mat.NumRows(), 3));

    cout << "\n dim :: " << dim << endl;

    vectorMatrix.SetDims(dim, dim);

    /*
        getVectorMatrix_3(vectorMatrix, mat, dim);

        vec_GF2E v;
        for (ulong i = 0; i < vectorMatrix.NumRows(); ++i) {
            v = vectorMatrix[i];
            if (isVectorPresentInMatrix(v, mat)) {
                return true;
            }
        }
     */

    //    cout << "\n out isDependencyFound_Rainbow3 ...\n";
    return false;
}


/* 
 * File:   EC_impl.tcc
 * Author: abdullah
 *
 * Created on 20 June, 2018, 4:59 PM
 */

#ifndef EC_IMPL_TCC
#define EC_IMPL_TCC

#include <bits/stdc++.h>

#include "mpi.h"

#include "EC_utils.hpp"
#include "lasVegasUtils.hpp"

using namespace std;

typedef struct resultData {
    ulong row1;
    ulong row2;
    ulong col1;
    ulong col2;
} resultData_2x2;

/**
 * This function tries to find a vector with r-zeros in the given matrix
 * @param ker : The matrix
 * @param r : number of zeros, expected.
 * @param rowIndex : If a vector with r-zeors is found this variable is assigned the row index
 * @return : True if a vector with r-zeors is found, false otherwise.
 */
template <class T>
bool isKernelHaving_r_Zeros(const T ker, const ulong r, long &rowIndex) {

    ulong zeroCnt = 0;
    for (ulong i = 0; i < ker.NumRows(); ++i) {
        zeroCnt = 0;
        for (ulong j = 0; j < ker.NumCols(); ++j) {
            if (IsZero(ker[i][j])) {
                zeroCnt++;
            }
        }

        if (zeroCnt == r) {
            rowIndex = i;
            cout << "\n row with " << r << " zeros found at :: " << rowIndex << endl;
            return true;
        }
    }

    return false;
}

/**
 * This function computes DLP, once a vector with r-zeors is found.
 * @param ker : The kernel Matrix
 * @param rowIndex : Index of row in the kernel with r-zeroes
 * @param k_randomNums : Number of random numbers for EC point P
 * @param t_randomNums : Number of random numbers for EC point Q
 * @param PQ_randomNumbers : Array with random numbers
 * @param ordP : order of the EC point P
 * @return : DLP
 */
template <class T>
ZZ getDlp(const T ker, const long rowIndex, const ulong k_randomNums, const ulong t_randomNums, \
        ZZ *PQ_randomNumbers, ZZ ordP) {

    ZZ dlp_A = conv<ZZ>(0), dlp_B = conv<ZZ>(0);
    for (ulong k = 0; k < (k_randomNums); ++k) {
        if (!(IsZero(ker[rowIndex][k]))) {
            dlp_A += PQ_randomNumbers[k];
            //            cout << "\n A - PQ_randomNumbers[k] :: " << PQ_randomNumbers[k] << "\t k :: " << k << endl;
        }
    }

    for (ulong k = k_randomNums; k < (k_randomNums + t_randomNums); ++k) {
        if (!(IsZero(ker[rowIndex][k]))) {
            dlp_B += PQ_randomNumbers[k];
            //            cout << "\n B - PQ_randomNumbers[k] :: " << PQ_randomNumbers[k] << "\t k :: " << k << endl;
        }
    }
    //    cout << "\n dlp_A :: " << dlp_A << "\t dlp_B :: " << dlp_B << endl;

    ZZ_p::init(conv<ZZ>(ordP));
    ZZ_p A = conv<ZZ_p>(dlp_A);
    ZZ_p B = conv<ZZ_p>(dlp_B);

    if (!IsZero(A) && !IsZero(B)) {
        ZZ_p DLP = (A / B);
        return conv<ZZ>(DLP);
    } else {
        std::cerr << B_RED_START << "\n Something is wrong... A or B is zero in getDLP() ...\n" << RESET_TERM;
        return conv<ZZ>("0");
    }
}

/**
 * Performs Gaussian Elimination on the second half of the matrix
 * @param arr : The matrix to be reduced
 */
template <class T, class U>
void second_GE1(T &mat) {

    const ulong n = mat.NumRows();
    const long mat_col = mat.NumCols();
    try {
        ulong curRow = 1;
        for (ulong k = n; k < ((2 * n) - 1); ++k) {

            U tmp = mat[curRow - 1][k];

            for (ulong i = curRow; i < n; ++i) {
                U factor = mat[i][k];
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

template <class T>
void permute_matrix_row(T &M, string permutation) {

    T tmp_M;
    tmp_M.SetDims(M.NumRows(), M.NumCols());

    cout << "\n Permutation :: " << permutation << endl;

    for (ulong i = 0; i < permutation.length(); ++i) {
        tmp_M[i] = M[(int(permutation[i]) - 65)];
    }

    if (permutation.length() < M.NumRows()) {
        for (ulong i = permutation.length(); i < M.NumCols(); ++i) {
            tmp_M[i] = M[i];
        }
    }

    M = tmp_M;
}

template <class T>
void permute_matrix_col(T &M, string permutation) {

    T tmp_M, M_transpose;
    transpose(M_transpose, M);

    tmp_M.SetDims(M_transpose.NumRows(), M_transpose.NumCols());

    cout << "\n Permutation :: " << permutation << endl;

    for (ulong i = 0; i < permutation.length(); ++i) {
        tmp_M[i] = M_transpose[(int(permutation[i]) - 65)];
    }

    if (permutation.length() < M_transpose.NumRows()) {
        for (ulong i = permutation.length(); i < M.NumCols(); ++i) {
            tmp_M[i] = M_transpose[i];
        }
    }

    transpose(M, tmp_M);
}

template <class T, class U>
ZZ lasVegas(const U& P, const U& Q, ZZ ordP) {

    int processorId, numberOfProcessors;
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);

    ZZ iterationCnt;

    const ulong n = 2;
    const ulong r = 3 * n;

    ZZ sqrt_ordP = SqrRoot(ordP);

    const ulong k_randomNums = (3 * n) - 1, t_randomNums = r + 1;
    const ulong mat_row = r + r, mat_col = ((n + 1)*(n + 2)) / 2;

    ZZ PQ_randomNumbers[(k_randomNums + t_randomNums)];

    ulong weightedVector_arr[mat_col][3];
    generateWeightedVector(n, weightedVector_arr);

    if (processorId == MASTER_NODE) {
        generateRandomNumbers2(mat_row, PQ_randomNumbers, ordP);

        // send to all processors
        cout << "\n Sending random numbers :: ";
        for (ulong i = 0; i < mat_row; ++i)
            cout << PQ_randomNumbers[i] << "\t";
        cout << endl;

        stringstream ss;
        for (ulong i = 0; i < mat_row; ++i)
            ss << PQ_randomNumbers[i] << "\t";

        std::string randomNUM_s = ss.str();
        const char* randomNum_string = randomNUM_s.c_str();
        int randomNUM_s_Len = strlen(randomNum_string);

        for (uint32_t i = 1; i < numberOfProcessors; ++i) {
            cout << "\n Sending to :: " << i << endl;
            MPI_Send(&randomNUM_s_Len, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(randomNum_string, randomNUM_s_Len, MPI_CHAR, i, 1, MPI_COMM_WORLD);
        }

    } else {
        //receive random numbers from MASTER
        MPI_Status *status;
        int randomNumLen;
        stringstream ss2;

        cout << "\n Receiving @ processor :: " << processorId << endl;
        MPI_Recv(&randomNumLen, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, status);
        char* strRandomNum = new char[randomNumLen];
        MPI_Recv(strRandomNum, randomNumLen, MPI_INT, 0, 1, MPI_COMM_WORLD, status);

        ss2 << strRandomNum;

        for (ulong i = 0; i < mat_row; ++i)
            ss2 >> PQ_randomNumbers[i];

        if (processorId == 1) {
            cout << "\n Receiving @ 1 random numbers :: ";
            for (ulong i = 0; i < mat_row; ++i)
                cout << PQ_randomNumbers[i] << "\t";
            cout << endl;
        }
        ss2.clear();
        delete []strRandomNum;
    }

}

template <class T>
bool isDependenceFound(T *arr, ulong count, ulong &col1, ulong &col2) {
    for (ulong i = 0; i < count; ++i) {
        for (ulong j = i + 1; j < count; ++j) {
            if (arr[i] == arr[j]) {
                col1 = i;
                col2 = j;
                return true;
            }
        }
    }
    return false;
}

template <class T, class U>
bool isDeterminantOfSubMatrixZero_Rainbow2(T mat, resultData_2x2 &rD) {

    ulong k = 0;
    U arr[mat.NumCols()];
    ulong row = mat.NumRows();
    ulong col = mat.NumCols();

    while (1) {
        for (int i = k + 1; i < row; i++) {

            for (int j = 0; j < col; j++) {
                arr[j] = mat[k][j] / mat[i][j];
            }

            if (isDependenceFound(arr, col, rD.col1, rD.col2)) {
                rD.row1 = k;
                rD.row2 = i;
                return true;
            }
        }

        if (k == (col - 2))
            break;
        k++;
    }

    return false;
}

template <class T, class U>
bool isDeterminantOfSubMatrixZero_Rainbow2(T mat, resultData_2x2 &rD, ulong col1) {

    ulong k = 0;
    ulong row = mat.NumRows();
    ulong col = mat.NumCols();
    U arr[col];

    while (1) {
        for (int i = k + 1; i < row; i++) {

            for (int j = 0; j < col; j++) {
                arr[j] = mat[k][j] / mat[i][j];
            }
            if (isDependenceFound(arr, col, rD.col1, rD.col2)) {
                rD.row1 = k;
                rD.row2 = i;
                return true;
            }
        }

        if (k == (col - 2))
            break;
        k++;
    }

    return false;
}

template <class T, class U>
bool isDeterminantOfSubMatrixZero_Rainbow2_modified(T mat, ulong &depdencyCol1, ulong &depdencyCol2, ulong &depedencyRow1, ulong &depedencyRow2) {


    ulong k = 0;
    U arr[mat.NumCols()];
    ulong row = mat.NumRows();
    ulong col = mat.NumCols();

    while (1) {
        for (int i = k + 1; i < row; i++) {
            ulong arrCnt = 0;
            for (int j = 0; j < col; j++) {

                if (!IsZero(mat[k][j]) && !IsZero(mat[i][j]))
                    arr[arrCnt++] = mat[k][j] / mat[i][j];

            }
            if (isDependenceFound(arr, arrCnt, depdencyCol1, depdencyCol2)) {

                depedencyRow1 = k;
                depedencyRow2 = i;
                return true;
            }
        }
        if (k == (col - 2))
            break;
        k++;
    }

    return false;
}

/**
 * Function to take in a kernel and rows and columns having dependency and 
 * return the DLP.
 * @param ker
 * @param k_randomNums
 * @param t_randomNums
 * @param PQ_randomNumbers
 * @param ordP
 * @param rD
 * @return DLP
 */
template <class T>
ZZ getDlp_R2(const T &ker, ulong k_randomNums, ulong t_randomNums, ZZ PQ_randomNumbers[], ZZ ordP, const resultData_2x2 &rD) {

    T ker2;
    ker2 = ker;

    ulong row1_OnePosition = (ker.NumCols() - 1) - rD.row1;
    ulong row2_OnePosition = (ker.NumCols() - 1) - rD.row2;

    ker2[rD.row1][rD.col1] = 0;
    ker2[rD.row1][rD.col2] = 0;
    ker2[rD.row1][row2_OnePosition] = 1;

    ker2[rD.row2][rD.col1] = 0;
    ker2[rD.row2][rD.col2] = 0;
    ker2[rD.row2][row1_OnePosition] = 1;

    //    cout << "\n ker[row] :: \n" << ker2[rD.row1] << endl;
    ZZ DLP = getDlp(ker2, rD.row1, k_randomNums, t_randomNums, PQ_randomNumbers, ordP);
    return DLP;
}
#endif /* EC_IMPL_TCC */
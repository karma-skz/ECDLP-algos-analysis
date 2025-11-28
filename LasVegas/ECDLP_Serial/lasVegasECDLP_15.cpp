#include "EC_GF2E.hpp"
#include "dlp_input_2m.hpp"
#include "EC_GF2E_Point.hpp"
#include "EC_impl.tcc"

#include <iomanip>
#include <NTL/matrix.h>
#include <NTL/mat_GF2E.h>

/**
 * LasVegas algorithm with function
 * High Level Algorithm
 *   Step 1 : Generate random numbers, use them to generate a matrix
 *   Step 2 : Compute kernel
 *   Step 3 : Make a matrix using the non-identity part of the kernel
 *   Step 4 : Use rainbow2 algorithm to check if something happens :-( :-)
 *   
 * @param P : Point P
 * @param Q : Point Q i.e. Q = mP
 * @param ordP : Order of P
 * @return : DLP
 */

ZZ EC_GF2E::lasVegasECDLP_15(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, const int offset) {

    int processorId;
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);

    ZZ iterationCnt;

    const ulong n = offset * p;
    const ulong r = 3 * n;

    const ulong k_randomNums = (3 * n) - 1, t_randomNums = r + 1;
    const ulong mat_row = r + r, mat_col = ((n + 1) * (n + 2)) / 2;

    if (processorId == 0) {
        cout << "\n k_randomNums :: " << k_randomNums << "\t t_randomNums :: " << t_randomNums << endl;
        cout << "\n mat_row :: " << mat_row << "\t mat_col :: " << mat_col << endl;
    }

    ZZ PQ_randomNumbers[(k_randomNums + t_randomNums)];

    ulong weightedVector_arr[mat_col][3];
    generateWeightedVector(n, weightedVector_arr);
    ulong accidentCnt = 0;

    while (1) {
        iterationCnt++;
        double s_time = GetTime();
        generateRandomNumbers2(mat_row, PQ_randomNumbers, ordP);

        mat_GF2E M;
        M.SetDims(mat_row, mat_col);
        double time_MStart = GetTime();
        int result = this->generateMatrix(M, P, Q, k_randomNums, t_randomNums, PQ_randomNumbers, weightedVector_arr);
        double time_MEnd = GetTime();
        if (result == 1) {
            iterationCnt--;
            accidentCnt++;
            continue;
        }

        //        int a;
        //        cout << "\n M :: \n";
        //        printMatrix1(M);
        //        cin >> a;

        mat_GF2E ker;
        double start_kTime = GetTime();
        kernel(ker, M);
        double end_kTime = GetTime();

        //        cout << "\n ker :: \n";
        //        printMatrix1(ker);

        if (ker.NumRows() == 0)
            continue;

        if (ker.NumRows() < r) {
            iterationCnt--;
            continue;
        }

        long rowIndex = -1;
        if (isKernelHaving_r_Zeros<mat_GF2E>(ker, r, rowIndex)) {
            //cout << "\n Solved by " << B_WHITW_START << "1" << RESET_TERM << " G.E. in iteration::" << B_RED_START << iterationCnt << RESET_TERM << endl;
            return getDlp(ker, rowIndex, k_randomNums, t_randomNums, PQ_randomNumbers, ordP);
        }

        mat_GF2E mat;
        getMatrixFromKernel(mat, ker, ker.NumRows());

        //cout << "\n ker[0] :: " << ker[0] << endl << endl << endl << endl << endl;
        //cout << "\n mat[0] :: " << mat[0] << endl;
        //        cout << "\n after get-matrix...mat.row :: " << mat.NumRows() << "\t mat.col :: " << mat.NumCols() << endl;
        //        cout << "\n ker.row :: " << ker.NumRows() << "\t ker.col :: " << ker.NumCols() << endl;

        resultData_2x2 rD;
        if (isDeterminantOfSubMatrixZero_Rainbow2<mat_GF2E, GF2E>(mat, rD)) {

            mat_GF2E tmpMat;
            tmpMat.SetDims(2, 2);
            tmpMat[0][0] = mat[rD.row1][rD.col1];
            tmpMat[0][1] = mat[rD.row1][rD.col2];
            tmpMat[1][0] = mat[rD.row2][rD.col1];
            tmpMat[1][1] = mat[rD.row2][rD.col2];

            //cout<<"\n A :: "<<mat.NumCols()<<"\t B :: "<< (mat_row-mat.NumCols())<<endl;
            ZZ DLP = getDlp_R2<mat_GF2E>(ker, mat.NumCols(), (mat_row - mat.NumCols()), PQ_randomNumbers, ordP, rD);
            //cout<<"\n determinant :: "<<determinant(tmpMat)<<endl;
            cout << " DLP :: " << DLP << "\t iteration :: " << iterationCnt << "\t processorId :: " << processorId;
            cout << "\t ker.row :: " << ker.NumRows() << "\t ker.col :: " << ker.NumCols() << "\t mat.row :: " << mat.NumRows() << "\t mat.col :: " << mat.NumCols() << endl;
            cout << "\n ker[0] :: \n"
                    << ker[0] << "\t mat[0] :: \n " << mat[0] << "\t k_time :: " << (end_kTime - start_kTime) << endl;

            return conv<ZZ>(iterationCnt);
        }
        cout << " Iteration Cnt :: " << iterationCnt << "\t Time :: " << GetTime() - s_time << "\t processorId :: " << processorId << "\t k_time :: " << (end_kTime - start_kTime)
                << "\t mat_time :: " << (time_MEnd - time_MStart) << endl;
    } //END:While

    //cout << "\n iterationCnt :: " << iterationCnt << endl;
    return conv<ZZ>(iterationCnt);
}
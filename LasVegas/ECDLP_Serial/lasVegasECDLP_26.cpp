#include "EC_GF2E.hpp"
#include "dlp_input_2m.hpp"
#include "EC_GF2E_Point.hpp"
#include "EC_impl.tcc"
#include "utils.hpp"

#include <iomanip>
#include <NTL/matrix.h>
#include <NTL/mat_GF2E.h>
#include <mpi/mpi.h>
#include <NTL/mat_GF2.h>
#include <NTL/lzz_p.h>

#define verbose 0
#define print_v  if(verbose) cout 

typedef struct data_1 {
    ulong i_start;
    ulong j_start;
    ulong quota;
} partitionData_2x2_1;

void printPartitionData2x2_1(const partitionData_2x2_1 pD, const int processorId) {
    print_v << " Processor :: " << processorId << " >> i_start :: " << pD.i_start << "\t j_start :: " << pD.j_start <<
            "\t quota :: " << pD.quota << endl;
}

/**
 * This is same as isDeterminantOfSubMatrixZero_Rainbow2 in EC_impl.tcc
 * The only difference being that this implementation is for parallel env.
 * @param mat
 * @param pD
 * @param rD
 * @return true or false
 */
bool is_2by2_DeterminantZero_1(const mat_GF2E &mat, const partitionData_2x2_1 &pD, resultData &rD, int processorId1) {

    double s_time = GetTime();
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char [1000];
    int NodeNameLen;

    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &totalNumberOfProcessors);
    MPI::Get_processor_name(NodeName, NodeNameLen);

    ulong A = pD.i_start;
    ulong B = pD.j_start;

    ulong n = mat.NumRows();
    ulong cnt = 0;
    ulong depdencyCol1, depdencyCol2;
    for (ulong row1 = A; row1 < n; ++row1) {
        for (ulong row2 = B; row2 < n; ++row2) {

            GF2E result[n];

            for (ulong col = 0; col < n; ++col)
                result[col] = mat[row1][col] / mat[row2][col];

            if (isDependenceFound(result, n, depdencyCol1, depdencyCol2)) {

                rD.col1 = depdencyCol1;
                rD.col2 = depdencyCol2;
                rD.row1 = row1;
                rD.row2 = row2;

                double e_time = GetTime();
                print_v << "\n Solved by processor Id ::" << processorId << "\t on node :: " << NodeName << " \t Time :: " << (e_time - s_time) << " seconds \n";
                print_v << "\n row1 :: " << row1 << "\t row2 :: " << row2 << "\t col1 :: " << depdencyCol1 << "\t col2 :: " << depdencyCol2 << endl;
                return true;
            }

            cnt++;
            if (cnt == pD.quota) {
                double e_time = GetTime();
                print_v << "\n Iterations completed by processorId :: " << processorId << "\t on node :: " << NodeName << "\t Time :: " << (e_time - s_time) << " seconds \n";

                return false;
            }
        }
        A = row1 + 1;
        B = A + 1;
    }
}

void sendPartitionDataToProcessor1_1(const partitionData_2x2_1 &pD, const ulong sendToProcessor) {
    ulong arr[3];
    arr[0] = pD.i_start;
    arr[1] = pD.j_start;
    arr[2] = pD.quota;

    MPI_Send(arr, 3, MPI_UNSIGNED_LONG, sendToProcessor, 0, MPI_COMM_WORLD);
}

void receiveDataOnProcessor1_1(partitionData_2x2_1 &pD, int processorId) {
    ulong arr[3];
    MPI_Recv(arr, 3, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    pD.i_start = arr[0];
    pD.j_start = arr[1];
    pD.quota = arr[2];
}

void makeKernelFromMat_1(const mat_GF2E &mat, mat_GF2E &ker) {

    ulong row = mat.NumRows();
    ulong col = mat.NumCols();

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            ker[i][j] = mat[i][j];
        }
    }

    int onePosition = ker.NumCols() - 1;
    for (int i = 0; i < row; i++) {
        ker[i][onePosition] = 1;
        onePosition--;
    }

}

void EC_GF2E::generateKernels(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, const int offset) {

    int processorId, totalNumberOfProcessors;
    ZZ iterationCnt;

    char *NodeName = new char [1000];
    int NodeNameLen;

    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &totalNumberOfProcessors);
    MPI::Get_processor_name(NodeName, NodeNameLen);

    int numberOfProcessors = totalNumberOfProcessors - 1;
    ZZ sqrt_ordP = SqrRoot(ordP);
    ulong sqrt_p = SqrRoot(p);
    ulong accidentCnt = 0;

    const ulong n = offset * p;
    const ulong r = 3 * n;

    if (processorId == 0) {
        print_v << "\n sqrt_p :: " << sqrt_p << "\t n :: " << n << "\t r :: " << r << endl;
    }

    const ulong k_randomNums = (3 * n) - 1, t_randomNums = r + 1;
    const ulong mat_row = r + r, mat_col = ((n + 1)*(n + 2)) / 2;
    ZZ PQ_randomNumbers[(k_randomNums + t_randomNums)];

    ulong weightedVector_arr[mat_col][3];

    generateWeightedVector(n, weightedVector_arr);
    generateRandomNumbers2(mat_row, PQ_randomNumbers, ordP);

    mat_GF2E M;
    M.SetDims(mat_row, mat_col);

    print_v << "\n Generating matrix M ...";
    cout.flush();
    double m_start = GetTime();
    int result = this->generateMatrix(M, P, Q, k_randomNums, t_randomNums, PQ_randomNumbers, weightedVector_arr);
    double m_end = GetTime();
    print_v << "[Done] \t time :: " << (m_end - m_start) << endl;

    //    if (result == 1) {
    //        iterationCnt--;
    //        accidentCnt++;
    //        continue;
    //    }

    mat_GF2E ker;

    print_v << "\n Computing Kernel ...";
    cout.flush();
    double k_start = GetTime();
    kernel(ker, M);
    double k_end = GetTime();
    print_v << "[Done] \t time :: " << (k_end - k_start) << endl;

    //    if (ker.NumRows() == 0)
    //        continue;
    //
    //    if (ker.NumRows() < r) {
    //        iterationCnt--;
    //        continue;
    //    }
    //    long rowIndex = -1;
    //    if (isKernelHaving_r_Zeros<mat_GF2E>(ker, r, rowIndex)) {
    //        print_v << "\n Solved by " << B_WHITW_START << "1" << RESET_TERM << " iteration::" << B_RED_START << iterationCnt << RESET_TERM << endl;
    //        return getDlp(ker, rowIndex, k_randomNums, t_randomNums, PQ_randomNumbers, ordP);
    //    }

    char *fileName = new char[200];
    sprintf(fileName, "kernel/p_%u_%u.txt", processorId, numberOfProcessors);

    ofstream fout(fileName);
    fout << k_randomNums << "\n" << t_randomNums << endl;

    for (ulong i = 0; i < (k_randomNums + t_randomNums); ++i) {
        fout << PQ_randomNumbers[i] << "\t";
    }
    fout << endl;

    fout << ker << endl;
    fout << (m_end - m_start) << endl;
    fout << (k_end - k_start) << endl;
    fout << processorId << endl;
    fout << NodeName << endl;

    fout.close();
}

/**
 * LasVegas algorithm with function
 * High Level Algorithm
 *   Step 1 : Generate random numbers, use them to generate a matrix
 *   Step 2 : Compute kernel
 *   Step 3 : @ThorsParallelALgorithm : Generate kernels on each core
 *   Step 4 : Each processor stores the generated kernel in a separate file
 *   Step 5 : After kernels are generated root processor reads kernels and runs
 *            Rainbow-2 on the kernel.
 *   Step 6 : If solution is found stop else read the next kernel
 * 
 * @param P : Point P
 * @param Q : Point Q i.e. Q = mP
 * @param ordP : Order of P
 * @return : DLP
 */
ZZ EC_GF2E::lasVegasECDLP_26(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, const int offset) {
    
    ZZ iterationCnt;
    ulong localOffset = offset;
    ulong numberOfKernelsGenerated = 0;

    while (1) {

        iterationCnt++;
        int processorId, totalNumberOfProcessors;

        char *NodeName = new char [1000];
        int NodeNameLen;

        MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
        MPI_Comm_size(MPI_COMM_WORLD, &totalNumberOfProcessors);
        MPI::Get_processor_name(NodeName, NodeNameLen);

        if (processorId == 0) {
            cout << "\n Generating " << totalNumberOfProcessors << " kernels...";
            cout.flush();
        }

        double iterationTimeStart, iterationTimeEnd;
        iterationTimeStart = GetTime();

        double k_generationTimeStart = GetTime();
        generateKernels(P, Q, ordP, localOffset);
        MPI_Barrier(MPI_COMM_WORLD);
        double k_generationTimeEnd = GetTime();

        if (processorId == 0) {
            cout << "...[Done]\t Time :: " << (k_generationTimeEnd - k_generationTimeStart) << " Secs." << endl;
        }

        int dlpSolvedFlag = 0;
        ZZ DLP;

        ulong fileId = 0;
        double e_time, s_time;

        numberOfKernelsGenerated += (totalNumberOfProcessors);
        s_time = GetTime();

        while (1) {
            if (processorId == 0) {

                char *fileName = new char[200];
                sprintf(fileName, "kernel/p_%u.txt", fileId);
                sprintf(fileName, "kernel/p_%u_%u.txt", fileId, (totalNumberOfProcessors - 1));

                ifstream fin;
                fin.open(fileName);

                ulong k_randomNums, t_randomNums;
                fin >> k_randomNums;
                fin >> t_randomNums;
                ZZ PQ_randomNumbers[(k_randomNums + t_randomNums)];

                for (ulong i = 0; i < (k_randomNums + t_randomNums); ++i) {
                    fin >> PQ_randomNumbers[i];
                }
                mat_GF2E ker;
                fin>>ker;

                double m_time, k_time;
                int pId;
                string pName;
                fin >> m_time;
                fin >> k_time;
                fin >> pId;
                fin >> pName;

                s_time = GetTime();
                cout << " Processing File :: " << fileName << "\t m_time :: " << std::setw(6) << m_time << "\t k_time :: " << std::setw(6) <<
                        k_time << "\t offset :: " << (localOffset) << "\t pId :: " << pId << "\t Node-Name :: " << std::setw(6) << pName;

                // <editor-fold defaultstate="collapsed" desc="BCast mat to all processors...">

                mat_GF2E mat;
                getMatrixFromKernel(mat, ker);

                stringstream ss;
                ss << ker;

                std::string s_i = ss.str();
                int strLen = s_i.length() + 1; // +1; Maybe important.
                char *ker_str = new char[strLen];
                strcpy(ker_str, s_i.c_str());

                //Broadcast  size and the ker.
                MPI_Bcast(&strLen, 1, MPI_INT, processorId, MPI_COMM_WORLD);
                MPI_Bcast(ker_str, strLen, MPI_CHAR, processorId, MPI_COMM_WORLD);

                ss.clear();
                delete ker_str;

                //Broadcasting random numbers
                ;
                stringstream ss2;
                for (ulong i = 0; i < (k_randomNums + t_randomNums); ++i)
                    ss2 << PQ_randomNumbers[i] << "\t";

                string randomNUM_s = ss2.str();
                int randomNUM_s_Len = randomNUM_s.length() + 1;
                char *randomNum_string = new char[randomNUM_s_Len];
                strcpy(randomNum_string, randomNUM_s.c_str());

                MPI_Bcast(&randomNUM_s_Len, 1, MPI_INT, processorId, MPI_COMM_WORLD);
                MPI_Bcast(randomNum_string, randomNUM_s_Len, MPI_CHAR, processorId, MPI_COMM_WORLD);

                // </editor-fold>
                // <editor-fold defaultstate="collapsed" desc="Compute Partition Data">

                ulong numberOfProcessors = totalNumberOfProcessors - 1;
                ZZ combos = nCr(mat.NumRows(), 2);
                ZZ numberOfCombosEachProcessorGets = conv<ZZ>(combos / (conv<ZZ>(numberOfProcessors)));
                ulong numberOfExtraCombos = conv<ulong>((combos) - conv<ZZ>(numberOfCombosEachProcessorGets * numberOfProcessors));

                print_v << "\n Total-Combinations :: " << combos;
                print_v << "\t numberOfCombosEachProcessorGets :: " << numberOfCombosEachProcessorGets;
                print_v << "\t numberOfExtraCombos :: " << numberOfExtraCombos << endl;

                partitionData_2x2_1 pD[numberOfProcessors + 1];

                ulong cnt = 0;
                ulong pDCnt = 1;
                pD[0].i_start = 0;
                pD[0].j_start = 1;
                pD[0].quota = conv<ulong>(numberOfCombosEachProcessorGets);

                print_v << "\n Offset :: " << offset << endl;
                print_v << "\n Computing Partition Data ....";
                cout.flush();
                double s_time3 = GetTime();

                ulong quota = conv<ulong>(numberOfCombosEachProcessorGets);
                ulong fCnt = 0;
                ulong dimenson = mat.NumRows();
                for (ulong i = 0; i < dimenson; ++i) {
                    for (ulong j = i + 1; j < dimenson; ++j) {
                        cnt++;
                        fCnt++;
                        if (cnt == quota) {
                            pD[pDCnt].i_start = i;
                            pD[pDCnt].quota = conv<ulong>(numberOfCombosEachProcessorGets);
                            pD[pDCnt].j_start = j;

                            pDCnt++;
                            cnt = 0;
                        }
                    }
                }
                pD[pDCnt - 1].quota += numberOfExtraCombos;
                double e_time3 = GetTime();

                print_v << " [Done] \t Time :: " << (e_time3 - s_time3) << endl;
                // </editor-fold>

                for (ulong i = 1; i <= numberOfProcessors; ++i) {
                    sendPartitionDataToProcessor1_1(pD[(i - 1)], i);
                }

                print_v << "\n Data from Master sent to all slaves, Iteration :: " << iterationCnt << endl;

                fin.close();
                delete fileName;
            } else {

                int strLen;
                MPI_Bcast(&strLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
                char* strKer_Ci = new char[strLen];
                MPI_Bcast(strKer_Ci, strLen, MPI_CHAR, 0, MPI_COMM_WORLD);

                mat_GF2E ker;
                stringstream ss2, ss;

                ss << strKer_Ci;
                ss >> ker;
                delete strKer_Ci;
                ss.clear();

                //Recv RandomNumber
                int randomNumLen;
                MPI_Bcast(&randomNumLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
                char* strRandomNum = new char[randomNumLen];
                MPI_Bcast(strRandomNum, randomNumLen, MPI_CHAR, 0, MPI_COMM_WORLD);

                ss2 << strRandomNum;

                const ulong k_randomNums = (ker.NumCols() / 2) - 1;
                const ulong t_randomNums = (ker.NumCols() / 2) + 1;

                ZZ PQ_randomNumbers[(k_randomNums + t_randomNums)];
                for (ulong i = 0; i < (k_randomNums + t_randomNums); ++i)
                    ss2 >> PQ_randomNumbers[i];

                delete strRandomNum;
                ss2.clear();

                mat_GF2E mat;
                getMatrixFromKernel(mat, ker);

                ulong numberOfProcessors = totalNumberOfProcessors - 1;
                for (ulong i = 1; i <= numberOfProcessors; ++i) {
                    if (processorId == i) {
                        partitionData_2x2_1 pD;
                        resultData_2x2 rD;
                        receiveDataOnProcessor1_1(pD, processorId);
                        if (is_2by2_DeterminantZero_1(mat, pD, rD, processorId)) {
                            GF2E det = (mat[rD.row1][rD.col1] * mat[rD.row2][rD.col2]) - (mat[rD.row1][rD.col2] * mat[rD.row2][rD.col1]);
                            DLP = getDlp_R2<mat_GF2E>(ker, k_randomNums, t_randomNums, PQ_randomNumbers, ordP, rD);

                            cout << "\t time :: " << (GetTime() - s_time) << endl;
                            cout << " DLP :: " << DLP << "\t";
                            cout << "iterationCnt :: " << iterationCnt << "\n";
                            cout << "\n Total number of kernels generated :: " << numberOfKernelsGenerated << endl;
                            
                            dlpSolvedFlag = 1;
                            MPI_Abort(MPI_COMM_WORLD, 10);
                        }
                    }
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            if (processorId == 0) {
                e_time = GetTime();
                cout << "\t time :: " << (e_time - s_time) << endl;
            }
            fileId++;

            if (fileId > (totalNumberOfProcessors - 1))
                break;
        }

        iterationTimeEnd = GetTime();
        if (processorId == 0) {
            cout << "\n iteration :: " << iterationCnt << "  Kernel-Set-Iteration-Time :: " << (iterationTimeEnd - iterationTimeStart) << " sec. \n";
            cout << "\n=======================================================\n";
        }
    }
    return conv<ZZ>(iterationCnt);
}
/* 
 * File:   main.cpp
 * Author: abdullah
 *
 * Created on 28 November, 2017, 11:12 AM
 */

#include <mpi/mpi.h>

#include "dlp_input_2m.hpp"
#include "dlp_input.hpp"
#include "EC_GF2E.hpp"
#include "EC_utils.hpp"
#include "utils.hpp"

#include <bits/stdc++.h>

using namespace std;
using namespace NTL;

const ulong TEN_MILLION = 10000000;
const ulong ONE_MILLION = 1000000;
const ulong HUNDRED_THOUSAND = 100000;
const ulong TEN_THOUSAND = 10000;
const ulong ONE_THOUSAND = 1000;
const ulong ONE_HUNDRED = 100;
const ulong THIRTY = 30;
const ulong FOURTY = 40;
const ulong FIFTY = 50;

const ulong MASTER = 0;

void ECDLP_GF2EX() {
    int processorId, numberOfProcessors;

    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    string fileName = "newInput/2_30.txt";

    dlp_input_2m dd(fileName);

    if (dd.data[0].p > 80) {
        if (processorId == MASTER) {
            cerr << B_RED_START << "\nUnable to solve for fields greater than equal to 80 bits.\n\n" << RESET_TERM;
        }
        return;
    }

    // <editor-fold defaultstate="collapsed" desc="Initialization">
    EC_GF2E E1(dd.data[0].p, dd.data[0].irrd, dd.data[0].a, dd.data[0].b);
    EC_GF2E_Point P, Q;

    P.x._GF2E__rep = dd.data[0].Px;
    P.y._GF2E__rep = dd.data[0].Py;

    Q.x._GF2E__rep = dd.data[0].Qx;
    Q.y._GF2E__rep = dd.data[0].Qy;
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Print Input at Master Processor">
    if (processorId == MASTER) {
        masterCout(processorId) << "\n Field Size :: 2^" << dd.data[0].p << endl;
        masterPrint(processorId) P.printPoint1("\n P ");
        masterPrint(processorId) Q.printPoint1("\t Q ");
        masterCout(processorId) << "\n\n Ord :: " << dd.data[0].ordP << "\t sqrt(Ord) :: " << SqrRoot(dd.data[0].ordP);
        masterCout(processorId) << "\t m :: " << dd.data[0].e << endl;
    }
    // </editor-fold>

    ZZ iterationCnt = E1.lasVegasECDLP_26(P, Q, dd.data[0].ordP, 3);

}

int main(int argc, char** argv) {

    MPI_Init(NULL, NULL);

    int numberOfProcessors;
    int processorId;

    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);

    ECDLP_GF2EX();

    MPI_Finalize();

    return 0;
}
/*
 * File:   main.cpp
 * Author: abdullah
 *
 * Created on 28 November, 2017, 11:12 AM
 */

#include "dlp_input_2m.hpp"
#include "dlp_input.hpp"
#include "EC_GF2E.hpp"
#include "EC_ZZp.hpp"
#include "EC_ZZp_Point.hpp"
#include "constants.hpp"
#include "lasVegas.tcc"
#include "containment.tcc"

#include <mpi.h>
#include <bits/stdc++.h>

using namespace std;
using namespace NTL;

const ulong numberOfInputs = 1;
const ulong numberOfKernelsToGenerate = 100;
// These two function can be converted into template.
// Re-write these two function in a master-slave model.
void fun_ZZp()
{
    int processorId, numberOfProcessors;

    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    string fileName = "input/3.txt";

    dlp_input dd(fileName);
    masterPrint(processorId) << "\n Number of inputs :: " << dd.numberOfInputs << endl;

    for (int i = 0; i < 1; ++i)
    {
        // char *fileName_o = new char[200];
        // sprintf(fileName_o, "output/p_%u_%u_%d.txt", processorId, numberOfProcessors, i);
        // freopen(fileName_o, "w", stdout);

        if (dd.numberOfInputs < i)
            break;

        EC_ZZp EC(dd.data[i].p, dd.data[i].a, dd.data[i].b, dd.data[i].ordP);
        EC_ZZp_Point P, Q;

        P.x = conv<ZZ_p>(dd.data[i].Px);
        P.y = conv<ZZ_p>(dd.data[i].Py);

        Q.x = conv<ZZ_p>(dd.data[i].Qx);
        Q.y = conv<ZZ_p>(dd.data[i].Qy);

        ulong numberOfBits = NumBits(dd.data[i].ordP);
        if (processorId == MASTER_NODE)
        {
            masterPrint(processorId) << "\n Field Size :: " << dd.data[i].p << endl;
            P.printPoint("\n P ");
            Q.printPoint("\t Q ");
            masterPrint(processorId) << "\n\n Ord :: " << dd.data[i].ordP << "\t sqrt(Ord) :: " << SqrRoot(dd.data[i].ordP);
            masterPrint(processorId) << "\t m :: " << dd.data[i].e << "\t num-Of-Bits :: " << numberOfBits << endl;
        }

        // ZZ ans = hypothesis_A<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(P, Q, dd.data[i].ordP, numberOfBits, 1, EC.address());

        ZZ ans = lasVegas<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(P, Q, dd.data[i].ordP, numberOfBits, 1, EC.address());

        // makeKernelDB<EC_ZZp_Point, EC_ZZp, mat_ZZ_p, ZZ_p>(P, Q, dd.data[i].ordP, numberOfBits,
        //                                                    1, EC.address(), numberOfKernelsToGenerate);

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void fun_GF2EX()
{
    int processorId, numberOfProcessors;

    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    string fileName = "input/2_50.txt";

    dlp_input_2m dd(fileName);

    for (int i = 0; i < 1; ++i)
    {
        // char *fileName = new char[200];
        // sprintf(fileName, "output/p_%u_%u_%d.txt", processorId, numberOfProcessors, i);
        // freopen(fileName, "w", stdout);

        masterPrint(processorId) << "\n Processing input :: " << (i + 1) << "\t of " << dd.numberOfInputs << endl;
        if (dd.numberOfInputs < i)
            break;

        EC_GF2E EC(dd.data[i].p, dd.data[i].irrd, dd.data[i].a, dd.data[i].b);
        EC_GF2E_Point P, Q;

        P.x._GF2E__rep = dd.data[i].Px;
        P.y._GF2E__rep = dd.data[i].Py;

        Q.x._GF2E__rep = dd.data[i].Qx;
        Q.y._GF2E__rep = dd.data[i].Qy;

        if (processorId == MASTER_NODE)
        {
            cout << "\n Field Size :: 2^" << dd.data[i].p << endl;
            P.printPoint1("\n P ");
            Q.printPoint1("\t Q ");
            cout << "\n\n Ord :: " << dd.data[i].ordP << "\t sqrt(Ord) :: " << SqrRoot(dd.data[i].ordP);
            cout << "\t m :: " << dd.data[i].e << endl;
        }
        ulong numberOfBits = NumBits(dd.data[i].ordP);
        ZZ ans = lasVegas<EC_GF2E_Point, EC_GF2E, mat_GF2E, GF2E>(P, Q, dd.data[i].ordP, numberOfBits, 1, EC.address());
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void inverseTest_ZZp()
{
    ulong p = 1009;
    ZZ_p::init(conv<ZZ>(p));

    string fileName = "kernel_DB/1009/kernel_p_1009_2.txt";
    ifstream kernelFile(fileName);
    cout << "\n fileName :: " << fileName << endl;

    mat_ZZ_p mat, invMat;
    kernelFile >> mat;
    invMat = inv(mat);

    cout << "\n mat :: \n"
         << mat << endl;

    cout << "\n invMat :: \n"
         << invMat << endl;

    isMinorPresent(mat, 2, 11);
    cout << "\n++++++++++++++++++++++++++++++++++++\n";
    isMinorPresent(invMat, 2, 11);
}

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    int numberOfProcessors;
    int processorId;

    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);

    // processBigMinors_parallel();
    // processBiggerMinors();

    // LU_Circular_PrincipleMinorTest();
    // gaussianElimination_multiple();

    fun_ZZp();
    // fun_GF2EX();

    MPI_Finalize();

    return 0;
}

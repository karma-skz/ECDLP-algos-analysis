#include <iostream>
#include <fstream>

#include <mpi.h>

#include <NTL/mat_ZZ_p.h>
#include <NTL/ZZ_p.h>

// #include "EC_lasVegas.tcc"
#include "playground.tcc"

#include "MPI_utils.hpp"
#include "constants.hpp"

using namespace NTL;
using namespace std;

void initCombinations(ulong[], ulong);
void printCombination2(const ulong[], ulong);
bool isLastCombination(const ulong[], ulong, ulong);
void _getNextCombination(ulong[], ulong, ulong);

template <class _mat_, class FiniteField>
_mat_ makeKernelFromMatrix(const _mat_ &ker);

template <class _mat_>
void getSubMatrix_extended(const _mat_ &orgMat, _mat_ &mat, ulong n);

template <class _mat_>
int isMinorPresent(const _mat_ &mat, ulong start, ulong end);

void testDeterminant()
{
    ulong p = 33554393;
    ZZ_p::init(conv<ZZ>(p));

    string fileName = "kernel_DB/25_29/25/kernel_c1_25_1.txt";

    ifstream fin;
    fin.open(fileName);
    if (!fin)
    {
        cerr << "\n Error opening file :: " << fileName << endl;
        exit(0);
    }

    mat_ZZ_p orgMat, ker;
    fin >> orgMat;
    mat_ZZ_p ext_orgMat = makeKernelFromMatrix<mat_ZZ_p, ZZ_p>(orgMat);

    ulong columnsRecudeCnt = 0;
    while (columnsRecudeCnt < 10)
    {
        cout << " column-reduce-cnt :: " << columnsRecudeCnt << endl;
        ker = orgMat;
        gauss(orgMat, columnsRecudeCnt);

        mat_ZZ_p hPrime;
        hPrime.SetDims(ker.NumRows() - columnsRecudeCnt, ker.NumRows() - columnsRecudeCnt);
        getSubMatrix_extended(orgMat, hPrime, columnsRecudeCnt);

        isMinorPresent(orgMat, 3, 3);

        ++columnsRecudeCnt;
    }
}

void processSmallAlgoTest(ulong n, ulong dimension, ulong numberOfRandomNumbers, const ZZ PQ_randomNumbers[])
{
    // compute addition of random numbers
    ZZ leftSide, rightSide;

    for (ulong i = 0; i < n; ++i)
        leftSide += PQ_randomNumbers[i];

    for (ulong i = n; i < numberOfRandomNumbers; ++i)
        rightSide += PQ_randomNumbers[i];

    // Get all combination of the range we are intrested in.
    ulong vector[dimension];
    initCombinations(vector, dimension);

    ulong rowNumber = 6;
    ulong row_vec[rowNumber];

    for (ulong i = 0; i < rowNumber; ++i)
    {
        row_vec[i] = (n - 1) - i;
    }

    ulong whileCnt = 0;
    while (!isLastCombination(vector, dimension, n))
    {

        _getNextCombination(vector, n, dimension);
        whileCnt++;
    }
    cout << "\n while cnt :: " << whileCnt << endl;
}

void shortAlgoTest()
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong p = 33554393;

    ulong fileId = 1;

    while (fileId <= 10)
    {
        char *fileName = new char[200];
        sprintf(fileName, "kernel_DB/25_29/25/kernel_c1_25_%u_RN.txt", fileId);

        masterPrint(processorId) << "\n fileName :: " << fileName << endl;

        ZZ_p::init(conv<ZZ>(p));

        ifstream fin;
        fin.open(fileName);
        if (!fin)
        {
            cerr << "\n Error opening file :: " << fileName << endl;
            exit(0);
        }
        cout << "\n Processing file :: " << fileName << endl;
        ulong numberOfRandomNumbers = 0;
        fin >> numberOfRandomNumbers;
        cout << "\n number of random numbers :: " << numberOfRandomNumbers << endl;

        ZZ PQ_randomNumbers[numberOfRandomNumbers];
        for (ulong i = 0; i < numberOfRandomNumbers; ++i)
        {
            fin >> PQ_randomNumbers[i];
        }
        fin.close();

        ulong dimension = 3; // we look for dimension x dimension minors
        ulong n = numberOfRandomNumbers / 2;

        for (ulong i = dimension; i < 6; ++i)
        {
            cout << "\n Processing dimension :: " << i << endl;
            processSmallAlgoTest(n, i, numberOfRandomNumbers, PQ_randomNumbers);
        }

        // ##########################################
        delete fileName;
        break;
    } // END : while(fileId)
}
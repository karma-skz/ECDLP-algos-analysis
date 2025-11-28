#include "MPI_utils.hpp"

#include <NTL/ZZ.h>

#include <fstream>

using namespace NTL;
using namespace std;

void sort_ZZ(ZZ a[], ulong n)
{
    ZZ temp;
    for (int i = 0; i < n; i++)
    {
        for (int j = 1 + i; j < n; j++)
        {
            if (a[i] > a[j])
            {
                temp = a[i];
                a[i] = a[j];
                a[j] = temp;
            }
        }
    }
}

// void getRandomNumbersFromFile(ulong fileId, ulong totalNumberOfProcessors, ZZ PQ_randomNumbers[])
// {
//     char *fileName = new char[200];
//     sprintf(fileName, "randomNumbers/p_%u_%u.txt", fileId, totalNumberOfProcessors);

//     ifstream randomFile(fileName);

//     ulong numberOfRandomNumbers;
//     randomFile >> numberOfRandomNumbers;
//     for (ulong i = 0; i < numberOfRandomNumbers; ++i)
//     {
//         randomFile >> PQ_randomNumbers[i];
//     }
//     randomFile.close();
//     delete fileName;
// }

void saveRandomNumberToFile(char *fileName, ZZ PQ_randomNumbers[], ulong numberOfRandomNumbers)
{
    ofstream randomFile(fileName);

    randomFile << numberOfRandomNumbers << "\n";
    for (ulong i = 0; i < numberOfRandomNumbers; ++i)
    {
        randomFile << PQ_randomNumbers[i] << "\t";
    }
    randomFile.close();
}

/**
 * Function to generate unique random numbers using NTL.
 * @param k : specifies the number of random numbers
 * @param randomNumbers : The output is returned in this array
 * @param p : Random numbers are generated modulp p
 */
void generateRandomNumbers(ulong k, ZZ randomNumbers[], ZZ ordP)
{
    ulong randomNumberCnt = 0;

    ulong someCnt = 0;
    while (randomNumberCnt < k)
    {
        someCnt++;
        bool flag = true;
        ZZ random_integer;

        random_integer = conv<ZZ>(conv<ZZ>(RandomWord()) % ordP);
        // ZZ randomBits = RandomBnd(ordP);
        // random_integer = GenPrime_ZZ(ulong(log2(conv<ulong>(randomBits))));

        for (ulong i = 0; i < randomNumberCnt; ++i)
        {
            if (randomNumbers[i] == random_integer)
            {
                flag = false;
                break;
            }
        }
        if (flag)
        {
            randomNumbers[randomNumberCnt] = random_integer;
            randomNumberCnt++;
        }
    }
    sort_ZZ(randomNumbers, k);
}

/**
 * Function to generate random numbers.
 * Unique random numbers are generated for each processor.
 * Numbers for each processor is saved in a seperate file
 * in a directory named randomNumbers
 *
 * @TODO: replace folder Name with a constant (constant.hpp)
 */
void generateRandomNumbersForProcessors(ulong numberOfRandomNumbers, ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];

    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    // Generate random numbers for each processor and store them in files
    for (int i = 0; i < totalNumberOfProcessors; ++i)
    {
        ZZ PQ_randomNumbers[numberOfRandomNumbers];
        // SetSeed(to_ZZ(i + time(0)));
        generateRandomNumbers(numberOfRandomNumbers, PQ_randomNumbers, ordP);

        ulong numberOfBits = NumBits(ordP);
        char *randomNumberfileName = new char[200];
        sprintf(randomNumberfileName, "randomNumbers/p_%d_%u_%u.txt", i, totalNumberOfProcessors, numberOfBits);
        saveRandomNumberToFile(randomNumberfileName, PQ_randomNumbers, numberOfRandomNumbers);
    }
}

/**
 * Function to read random numbers for the processor calling this function.
 * The random numbers are filled up in the given array.
 * @TODO: replace folder Name with a constant (constant.hpp)
 */
void getRandomNumbersFromFile(int processorId, int totalNumberOfProcessors, ZZ PQ_randomNumbers[], ZZ ordP)
{
    ulong numberOfBits = NumBits(ordP);
    char *randomNumberfileName = new char[200];
    sprintf(randomNumberfileName, "randomNumbers/p_%d_%u_%u.txt", processorId, totalNumberOfProcessors, numberOfBits);

    ifstream randomFileInput(randomNumberfileName);

    ulong numberOfRN;
    randomFileInput >> numberOfRN;
    for (ulong i = 0; i < numberOfRN; ++i)
    {
        randomFileInput >> PQ_randomNumbers[i];
    }
    randomFileInput.close();
}
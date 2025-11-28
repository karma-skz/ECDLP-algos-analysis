/* 
 * File:   EC_GF2E.hpp
 * Author: abdullah
 *
 * Created on 15 December, 2017, 11:28 AM
 */

#ifndef EC_GF2E_HPP
#define EC_GF2E_HPP

#include <NTL/GF2X.h>
#include <NTL/GF2XFactoring.h>

#include <NTL/GF2E.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ.h>
#include <NTL/mat_GF2E.h>

#include "EC_GF2E_Point.hpp"
#include "EC_utils.hpp"
#include "dlp_input_2m.hpp"

#include "EC_impl.tcc"

#include <iostream>

using namespace NTL;

class EC_GF2E {
public:

    ulong p;
    GF2X irrd;
    GF2E a1, a2, a3, a4, a6;
    GF2E discriminant;

    EC_GF2E(ulong);
    EC_GF2E(ulong, GF2X, GF2X);
    EC_GF2E(ulong, GF2X, GF2X, GF2X);
    void generateRandomCurve();

    void printCurve();
    void printCurve1();

    EC_GF2E_Point generateRandomPoint();

    bool isPointValid(const EC_GF2E_Point &);
    GF2E getDiscriminant(const GF2E &, const GF2E &);

    void pointAddition_Doubling(const EC_GF2E_Point&, const EC_GF2E_Point&, EC_GF2E_Point &);
    EC_GF2E_Point pointDoubling(const EC_GF2E_Point &P);
    void scalarMultiplication_Basic(const EC_GF2E_Point&, ZZ, EC_GF2E_Point &);
    void pointNegation(const EC_GF2E_Point&, EC_GF2E_Point&);

    void scalarMultiplicationDA(const EC_GF2E_Point&, ZZ, EC_GF2E_Point &);

    int generateMatrix(mat_GF2E &, EC_GF2E_Point, EC_GF2E_Point, ulong, ulong, ZZ *, ulong a[][3]);

    ZZ lasVegasECDLP_1(const EC_GF2E_Point &P, const EC_GF2E_Point & Q);
    ZZ lasVegasECDLP_3(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ);
    ZZ lasVegasECDLP_3_CountNoOfTries(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ);
    ZZ lasVegasECDLP_InternalSwap_Parallel(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ);
    ZZ lasVegasECDLP_ExternalSwap_Parallel(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ);

    ZZ lasVegasECDLP_ExtSwp_Serial(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_5(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_6(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_7(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);

    ZZ lasVegasECDLP_8(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_9(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_10(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_11(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_12(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_13(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_14(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_15(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, int);
    ZZ lasVegasECDLP_16(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_17(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_18(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_19(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_20(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_21(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_22(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_23(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_24(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP);
    ZZ lasVegasECDLP_25(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, int);

    ZZ lasVegasECDLP_26(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, int);
    void generateKernels(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, int);

    //@ISSC_Algorithm
    ZZ lasVegasECDLP_27(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, int);
    ZZ lasVegasECDLP_28(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, int);
    
    ZZ lasVegasECDLP_29(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, int);
    ZZ lasVegasECDLP_30(EC_GF2E_Point &P, EC_GF2E_Point &Q, ZZ ordP, int);
    
    friend EC_GF2E_Point operator+(const EC_GF2E_Point &P, const EC_GF2E_Point & Q);
};

EC_GF2E_Point operator+(const EC_GF2E_Point &P, const EC_GF2E_Point &Q);

//These function are in lasVegasUtils.cpp
ulong generateRandomN(ulong sizeOfArray, ZZ *randomNumbers, ZZ p, ulong *weightVectorSize);
void generateRandomNumbers2(ulong k, ZZ randomNumbers[], ZZ ordP);
void generateRandomColumnPermutation(ulong k, ulong *randomNumbers, ZZ p);

void generateWeightedVector(ulong weight, ulong weightVector[][3]);

void swapKernelColsSeperatePQ(const mat_GF2E ker, mat_GF2E & newKer, ulong *colIndexArray, ulong k_random, ulong t_random);
void swapKernelCols(const mat_GF2E ker, mat_GF2E & newKer, ulong colIndexArray[]);

void printMatrix1(const mat_GF2E mat);
void printVector1(vec_GF2E v);

mat_GF2E swapTwoCols(ulong i, ulong j, const mat_GF2E mat, mat_GF2E newMat);
void swapTwoCols(ulong i, ulong j, mat_GF2E &mat);
mat_GF2E swapThreeCols(ulong i, ulong j, ulong k, const mat_GF2E mat, mat_GF2E newMat);
mat_GF2E swapFourCols(ulong i, ulong j, ulong k, ulong l, const mat_GF2E mat, mat_GF2E newMat);

ulong isMatrixHaving_R_Zeros(const mat_GF2E mat, const ulong r, ulong &rowIndex);
void createLengthThreePermutation(ulong n, ulong arr[][3]);

//ZZ getDlp(const mat_GF2E, const long, const ulong, const ulong, ZZ *, ZZ);
//bool isKernelHaving_r_Zeros(const mat_GF2E, const ulong, long &);
void second_GE(mat_GF2E &);

void getMatrixFromKernel(mat_GF2E &, const mat_GF2E &);
void getMatrixFromKernel(mat_GF2E &, const mat_GF2E &, ulong col);
bool getCondensedMatrix(const mat_GF2E &, mat_GF2E &, long &, long &);
void internalOfMatrix(const mat_GF2E &, mat_GF2E &);
void divideMinorByInternalOfMatrix(mat_GF2E &, const mat_GF2E &);
bool isKernelHavingSubMatrixWithDeterminantZero(const mat_GF2E &, long &, long &, long &);
bool isMatrixHavingSubMatrixWithDeterminantZero(const mat_GF2E &, long &, long &, long &);

ulong getNumberOfSameElements(const mat_GF2E &);
ulong getNumberOfOnesInMatrix(const mat_GF2E &);

ZZ nCr(ulong n, ulong r);

//Rainbow3
bool isDependencyFound_Rainbow3(const mat_GF2E &);
bool isVectorPresentInMatrix(const vec_GF2E &, const mat_GF2E &);
bool isSubVectorPresentInVector(const vec_GF2E &, const vec_GF2E &);
template <class T> void getVectorMatrix_3(T &, const T &, ulong);
template <class T> void getVectorMatrix_4(T &, const T &, ulong);

//Rainbow4
bool isDependencyFound_Rainbow4(const mat_GF2E &);



#endif /* EC_GF2E_HPP */
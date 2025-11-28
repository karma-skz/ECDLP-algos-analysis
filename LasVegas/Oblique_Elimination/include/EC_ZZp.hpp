/* 
 * File:   EC_ZZp.hpp
 * Author: abdullah
 *
 * Created on 6 March, 2018, 10:50 AM
 */

#ifndef EC_ZZP_HPP
#define EC_ZZP_HPP

#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/mat_ZZ_p.h>

#include "EC_ZZp_Point.hpp"
// #include "utils.hpp"

using namespace NTL;

// #define verbosePrint 0
// #define v_print if(verbosePrint)

class EC_ZZp
{
public:
    ZZ p;
    ZZ_p a4, a6;
    ZZ_p discriminant;

    EC_ZZp(ZZ);
    EC_ZZp(ZZ, ZZ, ZZ);
    EC_ZZp(ZZ, ZZ, ZZ, ZZ);
    EC_ZZp *address()
    {
        return this;
    }

    void generateRandomCurve();
    EC_ZZp_Point generateRandomPoint();

    void printCurve();
    bool isPointValid(const EC_ZZp_Point &P);
    void pointAddition_Doubling(const EC_ZZp_Point &, const EC_ZZp_Point &, EC_ZZp_Point &);

    void scalarMultiplication_Basic(const EC_ZZp_Point &, ZZ, EC_ZZp_Point &);
    void scalarMultiplicationDA(const EC_ZZp_Point &, ZZ, EC_ZZp_Point &);

    void pointNegation(const EC_ZZp_Point &, EC_ZZp_Point &);
    ZZ order(const EC_ZZp_Point &);

    int generateMatrix(mat_ZZ_p &, EC_ZZp_Point P, EC_ZZp_Point Q,
                       ulong k_randomNums, ulong t_randomNums, ZZ *PQ_randomNumbers, ulong weightedVector_arr[][3]);

    int generateMatrix2(mat_ZZ_p &M, EC_ZZp_Point P, EC_ZZp_Point Q,
                        ulong k_randomNums, ulong t_randomNums, ZZ *PQ_randomNumbers, ulong weightedVector_arr[][3]);

    int generateMatrix_Random(mat_ZZ_p &, EC_ZZp_Point P, EC_ZZp_Point Q,
                              ulong k_randomNums, ulong t_randomNums, ZZ *PQ_randomNumbers, ulong weightedVector_arr[][3]);

    ZZ lasVegasECDLP_1(const EC_ZZp_Point &P, const EC_ZZp_Point &Q, ZZ);
    ZZ lasVegasECDLP_2(const EC_ZZp_Point &P, const EC_ZZp_Point &Q, ZZ);
    ZZ lasVegasECDLP_3(const EC_ZZp_Point &P, const EC_ZZp_Point &Q, ZZ);

    ZZ generateKernels_ZZp(EC_ZZp_Point &P, EC_ZZp_Point &Q, ZZ ordP, const int offset);
};

void printMatrix1(const mat_ZZ_p);

#endif /* EC_ZZP_HPP */
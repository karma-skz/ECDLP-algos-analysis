#ifndef EC_LASVEGAS_TCC
#define EC_LASVEGAS_TCC

#include <NTL/ZZ.h>
#include "Logger.hpp"
using namespace CPlusPlusLogging;

template <class T, class U, class W>
void genetateKernels(T &P, T &Q, ZZ ordP, const int offset, U *obj);

template <class T>
bool isKernelHaving_r_Zeros(const T ker, const ulong r, long &rowIndex);

template <class U, class V, class W>
int generateMatrix(U &M, V P, V Q,
                   ulong k_randomNums, ulong t_randomNums, ZZ *PQ_randomNumbers, ulong weightedVector_arr[][3], W *obj);

template <class T>
ZZ getDlp(const T ker, const long rowIndex, const ulong k_randomNums, const ulong t_randomNums,
          ZZ *PQ_randomNumbers, ZZ ordP);

template <class U>
void saveKernelToFile(const U &ker, ZZ);

#include "EC_lasVegas_impl.tcc"
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------

template <class _mat_>
_mat_ eliminationTest(_mat_ mat);

template <class _mat_, class FiniteField>
bool elimination_minor(const _mat_ &minor, const _mat_ &orgMat, ZZ ordP);

#include "eliminationTest.tcc"

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------

template <class T>
bool isDependenceFound(T *arr, ulong count, ulong &col1, ulong &col2);

template <class W, class>
void searchAllTwoMinors(ZZ);

template <class _mat_, class FiniteField>
bool is_2by2_DeterminantZero(const _mat_ &mat, const partitionData_2x2 &pD, resultData &rD);

template <class _mat_, class FiniteField>
bool is_2by2_DeterminantZero_2(const _mat_ mat, const partitionData_2x2 &pD, resultData &rD, _mat_ orgMat, ZZ ordP);

#include "searchAllTwoMinors.tcc"
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
template <class _mat_>
void getSubMatrix_extended(const _mat_ &orgMat, _mat_ &mat, ulong n);

template <class _mat_>
void getSubMatrix(const _mat_ &orgMat, _mat_ &mat, ulong n);

template <class _mat_, class FiniteField>
bool schurComplement(ZZ ordP);
#include "schurComplement.tcc"

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------

template <class _mat_, class FiniteField>
void elimination_minor(const _mat_ &minor, const _mat_ &orgMat, ZZ ordP);

template <class _mat_, class FiniteField>
_mat_ makeKernelFromMatrix(const _mat_ &ker);

template <class _mat_, class FiniteField>
bool schurComplement_serial(ZZ ordP);
#include "schurComplement_serial.tcc"

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------

template <class _mat_, class FiniteField>
bool obliqueElimination(ZZ ordP);
#include "obliqueElimination.tcc"

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------

#endif /* EC_LASVEGAS */

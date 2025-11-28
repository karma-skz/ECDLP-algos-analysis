#ifndef UTILS_HPP
#define UTILS_HPP

#include <NTL/ZZ.h>

NTL::ZZ nCr(ulong n, ulong r);
NTL::ZZ factorial(NTL::ZZ n);
NTL::ZZ factorial_iterative(NTL::ZZ n);

#endif
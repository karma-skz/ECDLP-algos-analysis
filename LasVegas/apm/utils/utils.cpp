#include "utils.hpp"

using namespace NTL;

ZZ nCr(ulong n, ulong r)
{
    ZZ numerator = factorial_iterative(conv<ZZ>(n));
    ZZ denominator = factorial_iterative(conv<ZZ>(n - r)) * factorial_iterative(conv<ZZ>(r));
    ZZ factorial = numerator / denominator;

    return factorial;
}

ZZ factorial(ZZ n)
{
    if (n == 0)
        return conv<ZZ>(1);
    else
        return (n * factorial(n - 1));
}

ZZ factorial_iterative(ZZ n)
{
    ZZ res = conv<ZZ>("1"), i;
    for (i = 2; i <= n; i++)
        res *= i;
    return res;
}
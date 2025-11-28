#include "utils.hpp"

#define MASTER_NODE 0

using namespace NTL;
using namespace std;

ZZ factorial(ZZ n) {

    if (n == 0)
        return conv<ZZ>(1);
    else
        return (n * factorial(n - 1));
}

//void permute_randomNumbers(ZZ *PQ_randomNumbers, string permutation) {
//
//    ZZ tmp[permutation.length()];
//
//    for (ulong i = 0; i < permutation.length(); ++i) {
//        tmp[i] = PQ_randomNumbers[i];
//    }
//
//    for (ulong i = 0; i < permutation.length(); ++i) {
//        PQ_randomNumbers[i] = tmp[(int(permutation[i]) - 65)];
//    }
//}
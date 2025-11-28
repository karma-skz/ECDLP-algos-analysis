#include "utils.hpp"

using namespace NTL;
using namespace std;

void permute_randomNumbers(ZZ *PQ_randomNumbers, string permutation) {

    ZZ tmp[permutation.length()];

    for (ulong i = 0; i < permutation.length(); ++i) {
        tmp[i] = PQ_randomNumbers[i];
    }

    for (ulong i = 0; i < permutation.length(); ++i) {
        PQ_randomNumbers[i] = tmp[(int(permutation[i]) - 65)];
    }
}
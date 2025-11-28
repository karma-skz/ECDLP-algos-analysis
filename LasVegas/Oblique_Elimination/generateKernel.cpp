// #include "EC_GF2E.hpp"
// #include "EC_GF2E_Point.hpp"
// #include "MPI_utils.hpp"
// #include "constants.hpp"

// #include <NTL/mat_GF2.h>

// /**
//  * Function to execute Las-Vegas algorithm and compute the kernel 
//  * The kernels will be stores in a directory named "kernel"
//  * @param P : Point P
//  * @param Q : Point Q i.e. Q = mP
//  * @param ordP : Order of P
//  */
// template <class T, class U>
// void genetateKernels(T &P, T &Q, ZZ ordP, const int offset, U *obj)
// {
//     int processorId, totalNumberOfProcessors;
//     char *NodeName = new char[1000];

//     MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

//     const ulong n = offset * obj->p;
//     const ulong r = 3 * n;

//     const ulong k_randomNums = (3 * n) - 1, t_randomNums = r + 1;
//     const ulong mat_row = r + r, mat_col = ((n + 1) * (n + 2)) / 2;

//     // Generate Weight Vector
//     ulong weightedVector_arr[mat_col][3];
//     generateWeightedVector(n, weightedVector_arr);

//     ZZ PQ_randomNumbers[(k_randomNums + t_randomNums)];

//     if (processorId == MASTER_NODE)
//         generateRandomNumbersForProcessors(mat_row, ordP);

//     MPI_Barrier(MPI_COMM_WORLD);
//     //Each processor reads random numbers and generates a matrix
//     getRandomNumbersFromFile(processorId, totalNumberOfProcessors, PQ_randomNumbers);

//     mat_GF2E M, ker;
//     M.SetDims(mat_row, mat_col);
//     int result = obj->generateMatrix(M, P, Q, k_randomNums, t_randomNums, PQ_randomNumbers, weightedVector_arr);
//     kernel(ker, M);

//     saveKernelToFile(ker);
// }
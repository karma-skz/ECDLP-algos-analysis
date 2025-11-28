#ifndef EC_LASVEGAS_IMPL_TCC
#define EC_LASVEGAS_IMPL_TCC

#include <NTL/ZZ.h>
#include "MPI_utils.hpp"
#include "constants.hpp"

#include <typeinfo>

typedef struct partitionData
{
    ulong i_start;
    ulong j_start;
    ulong quota;
} partitionData_2x2;

typedef struct resultData
{
    ulong row1;
    ulong row2;
    ulong col1;
    ulong col2;
} resultData_2x2;

/**
 * Function to execute Las-Vegas algorithm and compute the kernel
 * The kernels will be stores in a directory named "kernel"
 * @param P : Point P
 * @param Q : Point Q i.e. Q = mP
 * @param ordP : Order of P
 */
template <class T, class U, class W>
void genetateKernels(T &P, T &Q, ZZ ordP, ulong _p, const int offset, U *EC)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];

    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    const ulong n = _p;
    const ulong r = 3 * n; // variable k is used instead of r in the paper
    masterPrint(processorId) << "\t n :: " << n << "\t _p :: " << _p << "\t r :: " << r;

    const ulong k_randomNums = (r - 1), t_randomNums = (r + 1);
    const ulong mat_row = r + r, mat_col = ((n + 1) * (n + 2)) / 2;

    // Generate Weight Vector
    ulong weightedVector_arr[mat_col][3];
    generateWeightedVector(n, weightedVector_arr);

    ZZ PQ_randomNumbers[(k_randomNums + t_randomNums)];

    if (processorId == MASTER_NODE)
        generateRandomNumbersForProcessors(mat_row, ordP);

    MPI_Barrier(MPI_COMM_WORLD);
    // Each processor reads random numbers and generates a matrix
    getRandomNumbersFromFile(processorId, totalNumberOfProcessors, PQ_randomNumbers, ordP);

    W M, ker;
    M.SetDims(mat_row, mat_col);
    int result = generateMatrix(M, P, Q, k_randomNums, t_randomNums, PQ_randomNumbers, weightedVector_arr, EC);
    kernel(ker, M);

    saveKernelToFile<W>(ker, ordP);
    delete NodeName;
}

/**
 * Returns 1 in case of an accident.
 * @param M
 * @param P
 * @param Q
 * @param k_randomNums
 * @param t_randomNums
 * @param PQ_randomNumbers
 * @param weightedVector_arr
 * @return : 1 in case of an Accident.
 */
template <class U, class V, class W>
int generateMatrix(U &M, V P, V Q,
                   ulong k_randomNums, ulong t_randomNums, ZZ *PQ_randomNumbers, ulong weightedVector_arr[][3], W *EC)
{

    // <editor-fold defaultstate="collapsed" desc="Creating the first k = 3n-1 rows of M">
    for (ulong i = 0; i < k_randomNums; ++i)
    {
        V P1;
        EC->scalarMultiplicationDA(P, conv<ZZ>(PQ_randomNumbers[i]), P1);

        if ((P1.x == Q.x) && (P1.y == Q.y))
        {
            return 1;
        }

        for (ulong j = 0; j < M.NumCols(); ++j)
        {
            M[i][j] = power(P1.x, weightedVector_arr[j][0]) * power(P1.y, weightedVector_arr[j][1]);
        }
    }
    // </editor-fold>

    // <editor-fold defaultstate="collapsed" desc="Creating Q rows">
    for (ulong i = k_randomNums; i < (k_randomNums + t_randomNums); ++i)
    {
        V P1, P2;
        EC->pointNegation(Q, P1);

        EC->scalarMultiplicationDA(P1, conv<ZZ>(PQ_randomNumbers[i]), P2);

        if ((P2.x == Q.x) && (P2.y == Q.y))
        {
            return 1;
        }

        for (ulong j = 0; j < M.NumCols(); ++j)
        {
            M[i][j] = power(P2.x, weightedVector_arr[j][0]) * power(P2.y, weightedVector_arr[j][1]);
        }
    }

    return 0;
    // </editor-fold>
}

/**
 * This function tries to find a vector with r-zeros in the given matrix
 * @param ker : The matrix
 * @param r : number of zeros, expected.
 * @param rowIndex : If a vector with r-zeors is found this variable is assigned the row index
 * @return : True if a vector with r-zeors is found, false otherwise.
 */
template <class T>
bool isKernelHaving_r_Zeros(const T ker, const ulong r, long &rowIndex)
{
    ulong zeroCnt = 0;
    for (ulong i = 0; i < ker.NumRows(); ++i)
    {
        zeroCnt = 0;
        for (ulong j = 0; j < ker.NumCols(); ++j)
        {
            if (IsZero(ker[i][j]))
            {
                zeroCnt++;
            }
        }

        if (zeroCnt == r)
        {
            rowIndex = i;
            return true;
        }

        // This is to report when unwanted entries become zero.
        if ((ker.NumRows() - 1) != zeroCnt)
        {
            cout << "\n UNUSUAL NUMBER OF ZERO's DETECTED (isKernelHaving_r_Zeros) @ row :: " << i;
            cout << "\n Expected Number of zeros :: " << (ker.NumRows() - 1) << " found :: " << zeroCnt << endl;
            cout << "\n mat_row :: " << ker[i] << endl;
            MPI_Abort(MPI_COMM_WORLD, 50);
        }
    }

    return false;
}

/**
 * This function computes DLP, once a vector with r-zeors is found.
 * @param ker : The kernel Matrix
 * @param rowIndex : Index of row in the kernel with r-zeroes
 * @param k_randomNums : Number of random numbers for EC point P
 * @param t_randomNums : Number of random numbers for EC point Q
 * @param PQ_randomNumbers : Array with random numbers
 * @param ordP : order of the EC point P
 * @return : DLP
 */
template <class T>
ZZ getDlp(const T ker, const long rowIndex, const ulong k_randomNums, const ulong t_randomNums,
          ZZ *PQ_randomNumbers, ZZ ordP)
{

    ZZ dlp_A = conv<ZZ>(0), dlp_B = conv<ZZ>(0);
    for (ulong k = 0; k < (k_randomNums); ++k)
    {
        if (!(IsZero(ker[rowIndex][k])))
        {
            dlp_A += PQ_randomNumbers[k];
        }
    }

    for (ulong k = k_randomNums; k < (k_randomNums + t_randomNums); ++k)
    {
        if (!(IsZero(ker[rowIndex][k])))
        {
            dlp_B += PQ_randomNumbers[k];
        }
    }

    // not sure if this works
    // maybe try to use macro
    if (typeid(T) == typeid(mat_ZZ_p))
        ZZ_pPush push;

    ZZ_p::init(conv<ZZ>(ordP));

    ZZ_p A = conv<ZZ_p>(dlp_A);
    ZZ_p B = conv<ZZ_p>(dlp_B);

    if (!IsZero(A) && !IsZero(B))
    {
        ZZ_p DLP = (A / B);
        return conv<ZZ>(DLP);
    }
    else
    {
        std::cerr << B_RED_START << "\n Something is wrong... A or B is zero in getDLP() ...\n"
                  << RESET_TERM;
        return conv<ZZ>("0");
    }
}

template <class _mat_>
_mat_ getNonIdentityMatrixFromKernel(_mat_ ker)
{
    _mat_ nonIdentityKernel;
    nonIdentityKernel.SetDims(ker.NumRows(), ker.NumCols() / 2);
    for (int i = 0; i < ker.NumRows(); ++i)
    {
        for (int j = 0; j < (ker.NumCols() / 2); ++j)
        {
            nonIdentityKernel[i][j] = ker[i][j];
        }
    }
    return nonIdentityKernel;
}

/**
 *
 * @TODO: foldername and template
 */
template <class _mat_>
void saveKernelToFile(const _mat_ &ker, ZZ ordP)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];

    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong numberOfBits = NumBits(ordP);
    char *fileName = new char[200];
    sprintf(fileName, "kernel/p_%u_%u_%u.txt", processorId, totalNumberOfProcessors, numberOfBits);

    ofstream fout(fileName);
    _mat_ nonIdentityKernel = getNonIdentityMatrixFromKernel(ker);

    fout << nonIdentityKernel << endl;
    fout << processorId << endl;
    fout << NodeName << endl;

    fout.close();
    delete fileName;
    delete NodeName;
}

template <class _mat_>
_mat_ getKernelFromFile(ulong fileId, ulong totalNumberOfProcessors, ZZ ordP)
{
    ulong numberOfBits = NumBits(ordP);
    char *fileName = new char[200];
    sprintf(fileName, "kernel/p_%u_%u_%u.txt", fileId, totalNumberOfProcessors, numberOfBits);
    ifstream fin;
    fin.open(fileName);
    if (!fin)
    {
        cerr << "\n Unable to open file :: " << fileName << endl;
        exit(0);
    }

    _mat_ ker;
    fin >> ker;
    fin.close();

    delete fileName;
    return ker;
}

#endif /* EC_LASVEGAS_IMPL */
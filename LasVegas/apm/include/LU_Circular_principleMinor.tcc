#include "EC_lasVegas.tcc"
#include "constants.hpp"
#include "utils.hpp"
#include "containment.tcc"

#include <bits/stdc++.h>

mat_ZZ_p get_kth_principleMinor(const mat_ZZ_p &mat, ulong k)
{
    mat_ZZ_p minor;
    minor.SetDims(k, k);

    for (ulong i = 0; i < k; ++i)
        for (ulong j = 0; j < k; ++j)
            minor[i][j] = mat[i][j];

    return minor;
}

void checkAll_principleMinors(const mat_ZZ_p &mat)
{
    ulong k = 1;
    while (k < mat.NumRows())
    {
        // get k-th minor
        mat_ZZ_p minor = get_kth_principleMinor(mat, k);
        // cout << " determinant :: " << determinant(minor) << endl;
        if (IsZero(determinant(minor)))
        {
            cout << "\n k :: " << k << endl;
            cout << " minor :: \n"
                 << minor << endl;
            // check determinant
            cout << " determinant :: " << determinant(minor) << endl;
            int a;
            cin >> a;
        }
        ++k;
    }
}

void LU_Circular_PrincipleMinorTest()
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong p = 536870909;

    ulong fileId = 1;

    while (fileId < 10)
    {
        char *fileName = new char[200];
        sprintf(fileName, "kernel_DB/25_29/29/kernel_c1_29_%u.txt", fileId);

        masterPrint(processorId) << " fileName :: " << fileName << endl;

        ZZ_p::init(conv<ZZ>(p));

        ifstream fin;
        fin.open(fileName);
        if (!fin)
        {
            cerr << "\n Error opening file :: " << fileName << endl;
            exit(0);
        }

        mat_ZZ_p ker;
        fin >> ker;

        ulong rowCnt = 0;
        while (rowCnt < ker.NumRows())
        {
            // cout << " processing row :: " << rowCnt << endl;
            // perform circular swap
            ker = circulatShiftMatrixRow(ker);

            // check all principle minors
            checkAll_principleMinors(ker);

            ++rowCnt;
        }

        ++fileId;
    }
}
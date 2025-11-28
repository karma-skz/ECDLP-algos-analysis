#include <mpi.h>
#include <bits/stdc++.h>

#include <NTL/mat_ZZ_p.h>

// #include "containment.tcc"

using namespace std;
using namespace NTL;

// void inverseTest_ZZp()
// {
//     ulong p = 1009;
//     ZZ_p::init(conv<ZZ>(p));

//     ifstream kernelFile("kernel_DB/kernel_p_1009_1.txt");

//     mat_ZZ_p mat, invMat;
//     kernelFile >> mat;
//     invMat = inv(mat);

//     cout << "\n mat :: \n"
//          << mat << endl;

//     cout << "\n invMat :: \n"
//          << invMat << endl;

//     isMinorPresent(mat, 2, 3);
// }
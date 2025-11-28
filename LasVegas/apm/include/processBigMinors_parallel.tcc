#include "EC_lasVegas.tcc"
#include "containment.tcc"

void processBigMinors_parallel()
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong p = 33554393;

    ulong fileId = 1;

    while (fileId <= 10)
    {
        char *fileName = new char[200];
        sprintf(fileName, "kernel_DB/25_29/25/kernel_c1_25_%u.txt", fileId);

        masterPrint(processorId) << "\n fileName :: " << fileName << endl;

        ZZ_p::init(conv<ZZ>(p));

        ifstream fin;
        fin.open(fileName);
        if (!fin)
        {
            cerr << "\n Error opening file :: " << fileName << endl;
            exit(0);
        }

        mat_ZZ_p orgMat, ker;
        fin >> orgMat;
        mat_ZZ_p ext_orgMat = makeKernelFromMatrix<mat_ZZ_p, ZZ_p>(orgMat);

        ulong maxReduceCnt = orgMat.NumCols();
        masterPrint(processorId) << " Mat.Row :: " << orgMat.NumRows() << "\t Mat.Col :: " << orgMat.NumCols() << endl;

        ulong columnsRecudeCnt = 0;
        while (columnsRecudeCnt < maxReduceCnt)
        {
            ker = ext_orgMat;
            gauss(ker, columnsRecudeCnt);
            masterPrint(processorId) << "\n col reduce cnt :: " << columnsRecudeCnt << endl;
            mat_ZZ_p hPrime;
            hPrime.SetDims(ker.NumRows() - columnsRecudeCnt, ker.NumRows() - columnsRecudeCnt);
            getSubMatrix_extended(ker, hPrime, columnsRecudeCnt);

            ulong dimension = hPrime.NumRows() - 3;
            // ulong dimension = 2;
            while (dimension > 65)
            {
                masterPrint(processorId) << "\n [S]-######################################################## \n";
                char fileName2[200];
                sprintf(fileName2, "output/prime_25_d_%u_pId_%u_fId_%u.txt", dimension, processorId, fileId);
                ofstream fout(fileName2);
                fout << "Processing File :: " << fileName << endl;
                masterPrint(processorId) << " Processing dimension :: " << dimension << endl;

                // Based of processor Id get the starting row/col combination
                ulong rowCombinationVec[dimension], colCombinationVec[dimension];
                ZZ totalRowCombinations = (factorial(conv<ZZ>(hPrime.NumRows()))) / (factorial(conv<ZZ>(hPrime.NumRows() - dimension)) * (factorial(conv<ZZ>(dimension))));
                ZZ totalCombinations = totalRowCombinations * totalRowCombinations;
                ulong quota = conv<ulong>((totalCombinations / conv<ZZ>(totalNumberOfProcessors)));

                ulong rowCombinationNumber = ulong(processorId * quota) / conv<ulong>(totalRowCombinations);
                ulong colCombinationNumber = (processorId * quota) % conv<ulong>(totalRowCombinations);

                // Add the last few remaining combination to the last processor
                if (processorId == totalNumberOfProcessors - 1)
                    quota += conv<ulong>(totalRowCombinations) - (quota * totalNumberOfProcessors);

                // sleep(processorId * processorId * processorId);

                ulong symbolVec_dim = hPrime.NumRows();
                ulong symbolVec[symbolVec_dim];
                for (int i = 0; i < symbolVec_dim; ++i)
                    symbolVec[i] = i;

                get_kth_combination(symbolVec, symbolVec_dim, dimension, rowCombinationNumber, rowCombinationVec);
                get_kth_combination(symbolVec, symbolVec_dim, dimension, colCombinationNumber, colCombinationVec);

                // cout << "\n processorId :: " << processorId << endl;
                // printCombination2(rowCombinationVec, dimension);
                // printCombination2(colCombinationVec, dimension);
                // cout << "\n ++++++++++++++++++++++++++++++++++++ \n";
                // MPI_Barrier(MPI_COMM_WORLD);
                processAllSubMatricesOfDimension_parallel(dimension, hPrime, fout, rowCombinationVec, colCombinationVec, quota);

                fout.close();
                --dimension;
                MPI_Barrier(MPI_COMM_WORLD);
                masterPrint(processorId) << "\n [E]-######################################################## \n";
                break;
            } // END : while(diension)

            columnsRecudeCnt++;
            // break;
        } // END : while(columnsRecudeCnt < maxReduceCnt)
        ++columnsRecudeCnt;
        ++fileId;
    } // END : while(fileId)
}
#include "EC_lasVegas.tcc"
#include "containment.tcc"

void processBiggerMinors()
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong p = 33554393;

    char *fileName = new char[200];
    sprintf(fileName, "kernel_DB/25_29/25/kernel_c1_25_%u.txt", (processorId + 1));

    cout << "\n fileName :: " << fileName << endl;

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

    ulong columnsRecudeCnt = 0;
    while (columnsRecudeCnt < maxReduceCnt)
    {
        ker = ext_orgMat;
        gauss(ker, columnsRecudeCnt);
        cout << "\n col reduce cnt :: " << columnsRecudeCnt << endl;
        mat_ZZ_p hPrime;
        hPrime.SetDims(ker.NumRows() - columnsRecudeCnt, ker.NumRows() - columnsRecudeCnt);
        getSubMatrix_extended(ker, hPrime, columnsRecudeCnt);

        ulong dimension = hPrime.NumRows() - 3;
        while (1)
        {
            cout << "\n [S]-######################################################## \n";
            char fileName2[200];
            sprintf(fileName2, "output/dimension_%u_%u_%u.txt", dimension, processorId, columnsRecudeCnt);
            ofstream fout(fileName2);
            fout << "Processing File :: " << fileName << endl;
            cout << "Processing dimension :: " << dimension << endl;
            processAllSubMatricesOfDimension_partWise(dimension, hPrime, fout);
            fout.close();
            --dimension;
            cout << "\n [E]-######################################################## \n";
            break;
        }

        columnsRecudeCnt++;
        break;
    }

    columnsRecudeCnt++;
}

// if (ENABLE_LOGGING)
// {
//     std::ostringstream ss;
//     ss << " !!!-DLP :: " << DLP;
//     Logger::getInstance()->info(ss);
// }
// ulong p = 44923183;
// string fileName = "kernel_DB/26/kernel_26_3.txt";
// ulong p = 32749;
// string fileName = "kernel_DB/new/kernel_15_2.txt";
// ulong p = 2029;
// string fileName = "kernel_DB/kernel_11_3.txt";
// ulong p = 101;
// string fileName = "kernel_DB/new/kernel_7_1.txt";

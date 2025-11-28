#include "EC_lasVegas.tcc"
#include "containment.tcc"

/**
 * Return true if a minor with determinant zero is found, and false o.w.
 * @param dimension
 * @param mat
 * @param foutDispMat
 * @return 
 */
template <class _mat_>
ulong processAllSubMatricesOfDimension_modified(ulong I_prime[], ulong J_prime[], ulong dimension_prime, const _mat_ &mat, ulong dimension)
{
    const ulong numRows = dimension_prime;

    if (numRows < dimension)
    {
        cout << "\n Invalid dimension in processAllSubMatricesOfDimension() numRows :: " << numRows << "\t dimension :: " << dimension << endl;
        return 0;
    }

    ulong rowCombination[dimension];
    ulong colCombination[dimension];
    initCombinations(rowCombination, dimension);
    initCombinations(colCombination, dimension);
    ulong whileCnt = 1;
    ulong zeroMinorCnt = 0;

    ulong rowCombination_modified[dimension];
    ulong colCombination_modified[dimension];

    while (true)
    {
        for (int i = 0; i < dimension; ++i)
        {
            rowCombination_modified[i] = I_prime[rowCombination[i]];
            colCombination_modified[i] = J_prime[rowCombination[i]];
        }

        _mat_ minor;
        minor.SetDims(dimension, dimension);

        makeMatrixFromRowColCombination(rowCombination_modified, colCombination_modified, mat, minor);

        if (IsZero(determinant(minor)))
        {
            cout << "\n Minor found...\n";
            cout << "\n minor :: \n " << minor << endl;
            cout << "\n row/col :: \n";
            printCombination2(rowCombination_modified, dimension);
            printCombination2(colCombination_modified, dimension);
            zeroMinorCnt++;
        }

        if (!isLastCombination(colCombination, dimension, numRows))
        {
            _getNextCombination(colCombination, numRows, dimension);
        }
        else
        {
            if (isLastCombination(rowCombination, dimension, numRows))
            {
                cout << "\n whileCnt :: " << whileCnt << "\t zeroMinorCnt :: " << zeroMinorCnt << endl;
                break;
            }
            initCombinations(colCombination, dimension);
            _getNextCombination(rowCombination, numRows, dimension);
        }
        whileCnt++;
    }

    return zeroMinorCnt;
}

void fun(ulong I_prime[], ulong J_prime[], ulong dimension_prime, const mat_ZZ_p &orgMat, const mat_ZZ_p &minor)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong start = (minor.NumRows() - 1) - processorId;
    ulong end = 2;

    bool flag = true;
    for (ulong i = start; i >= (end);)
    {
        cout << "\n Processing minors of dim :: " << i << "\t mat_row :: " << orgMat.NumRows() << " @ processor :: " << processorId << endl;
        cout.flush();
        double sTime = GetTime();
        // ulong count = processAllSubMatricesOfDimension_modified(I_prime, J_prime, dimension_prime, orgMat, i);
        char *fileName = new char[200];
        sprintf(fileName, "output/elimination_dim_%u.txt", i);
        ofstream fout(fileName);
        ulong somehting = processAllSubMatricesOfDimension(i, minor, fout);
        fout.close();
        i = i - totalNumberOfProcessors;
    }
}

/**
 * External Minors : Those minors that are outside a given minors.
 */
void processExternalMinors()
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    ulong p = 33554393;

    char *fileName = new char[200];
    sprintf(fileName, "kernel_DB/25_29/25/kernel_c1_25_4.txt");

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

    // ---------------------------------------------------
    // Read minor indices from file

    ifstream fin1("output/3/dimension_72_3_0.txt");

    if (!fin1)
    {
        cerr << "\n Unable to open file minor dimension file....\n";
        exit(0);
    }

    string buffer;
    getline(fin1, buffer);

    while (true)
    {
        ulong dimension;
        fin1 >> dimension;

        if (fin1.eof())
            break;

        cout << "\n Input large minor dimension :: " << dimension << endl;

        ulong I[dimension], J[dimension];
        for (int i = 0; i < dimension; ++i)
            fin1 >> I[i];

        for (int i = 0; i < dimension; ++i)
            fin1 >> J[i];

        mat_ZZ_p minor;
        minor.SetDims(dimension, dimension);
        makeMatrixFromRowColCombination(I, J, orgMat, minor);

        ulong dimension_prime = orgMat.NumRows() - dimension;
        ulong I_prime[dimension_prime], J_prime[dimension_prime];
        ulong I_prime_cnt = 0, J_prime_cnt = 0;

        // -------------- @TODO : Convert this into a function ---------------
        for (size_t i = 0; i < orgMat.NumRows(); ++i)
        {
            bool flag = true;
            for (size_t j = 0; j < dimension; ++j)
            {
                if (I[j] == i)
                {
                    flag = false;
                    break;
                }
            }

            if (flag)
                I_prime[I_prime_cnt++] = i;

            bool flag_J = true;
            for (size_t j = 0; j < dimension; ++j)
            {
                if (J[j] == i)
                {
                    flag_J = false;
                    break;
                }
            }

            if (flag_J)
                J_prime[J_prime_cnt++] = i;
        }
        // -------------------------------------------------

        fun(I_prime, J_prime, dimension_prime, orgMat, minor);

        cout << "\n I/J_ prime :: \n";
        printCombination2(I_prime, dimension_prime);
        printCombination2(J_prime, dimension_prime);

        string hashHash;
        fin1 >> hashHash;

        cout << "\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n";
    }
    fin1.close();
}
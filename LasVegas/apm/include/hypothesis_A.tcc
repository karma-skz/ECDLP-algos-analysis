#include "EC_lasVegas.tcc"
#include "EC_GF2E.hpp"
#include "EC_ZZp.hpp"
#include "EC_GF2E_Point.hpp"

#include "MPI_utils.hpp"

#include <iomanip>
#include <NTL/matrix.h>
#include <NTL/mat_GF2E.h>

#include "constants.hpp"

vec_ZZ_p getRandomUniqueVec(ulong vectorSize, ZZ_p e)
{
    vec_ZZ_p vec;
    vec.SetLength(vectorSize);

    ulong randomNumberCnt = 0;
    while (randomNumberCnt < vectorSize)
    {
        bool flag = true;
        ZZ_p tmp = random_ZZ_p();

        if (IsZero(tmp))
            continue;

        if (tmp == e)
            continue;

        for (ulong i = 0; i < randomNumberCnt; ++i)
        {
            if (vec[i] == tmp)
            {
                flag = false;
                break;
            }
        }

        if (flag)
        {
            vec[randomNumberCnt] = tmp;
            randomNumberCnt++;
        }
    }

    return vec;
}

bool subVec_addition(EC_ZZp_Point P, EC_ZZp_Point Q, vec_ZZ_p vec, ulong chose, EC_ZZp *EC)
{
    // cout << "\n in subVec_addition ...\n";
    ulong comboVec[chose];
    initCombinations(comboVec, chose);
    ulong cnt = 0;

    while (1)
    {
        ZZ_p sum;
        for (ulong i = 0; i < chose; i++)
            sum += vec[comboVec[i]];

        EC_ZZp_Point tmpPoint;
        EC->scalarMultiplicationDA(P, conv<ZZ>(sum), tmpPoint);

        // if (sum == conv<ZZ_p>("438"))
        // {
        //     cout << "\n Bingo...........\n";
        //     int as;
        //     cin >> as;
        // }
        // check if the addition produces point Q
        if (tmpPoint.x == Q.x && tmpPoint.y == Q.y)
        {
            cout << "\n sum :: " << sum << "\t chose :: " << chose << "\t vec-Size ::" << vec.length() << endl;
            cout << "\n tmpPt :: " << tmpPoint.x << ":" << tmpPoint.y << "\t Q ::" << Q.x << ":" << Q.y << endl;
            return true;
        }
        else
        {
            ;
            // cout << "\n P :: " << P.x << ":" << P.y << "\t tmpPoint :: " << tmpPoint.x << ":" << tmpPoint.y << "\t Q : " << Q.x << ":" << Q.y << endl;
        }

        if (!isLastCombination(comboVec, chose, vec.length()))
            _getNextCombination(comboVec, vec.length(), chose);
        else
            break;
    }

    return false;
}

/**
 * @param P : Point P
 * @param Q : Point Q i.e. Q = mP
 * @param ordP : Order of P
 * @return : DLP
 */
template <class _EC_Point_, class _EC_ZZp, class _mat_, class FiniteField>
ZZ hypothesis_A(_EC_Point_ &P, _EC_Point_ &Q, ZZ ordP, ulong _p, const int offset, _EC_ZZp *EC)
{
    int processorId, totalNumberOfProcessors;
    char *NodeName = new char[1000];
    MPI_get_MPI_rank_size_name(processorId, totalNumberOfProcessors, NodeName);

    cout << "\n in hypothesis_A :: ordP :: " << ordP << endl;

    // ###############################################################################
    // ZZ_pPush push;
    // ZZ_p::init(conv<ZZ>("1009"));

    // Iterate over vectorSize
    ulong vectorStartSize = 19;
    ulong vectorEndSize = 24;

    for (ulong vectorSize = vectorStartSize; vectorSize < vectorEndSize; ++vectorSize)
    {
        cout << "\n Processing vectorSize :: " << vectorSize << endl;
        // Generate cnt number of vectors of size vectorSize
        vec_ZZ_p vec;
        vec.SetLength(vectorSize);

        ulong cnt = 1000;
        for (ulong i = 0; i < cnt; i++)
        {
            vec = getRandomUniqueVec(vectorSize, conv<ZZ_p>("0"));

            // Check if this random vector has a sub-vector such that the addition produces Q
            for (ulong subVecSize = 13; subVecSize <= (vectorSize - 1); ++subVecSize)
            {
                if (subVec_addition(P, Q, vec, subVecSize, EC))
                {
                    cout << "\n something +ve happened...\n";
                    return conv<ZZ>("123");
                }
            }
        }
    }
    return conv<ZZ>("999");
}

bool subVec_addition_kthCombo(ZZ_p ele, vec_ZZ_p vec, ulong chose, ulong comboVec[], ulong quota)
{
    int processorId, totalNumberOfProcessors;
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &totalNumberOfProcessors);

    ulong cnt = 0;

    while (cnt < quota)
    {
        ZZ_p sum;
        for (ulong i = 0; i < chose; i++)
            sum += vec[comboVec[i]];

        if (sum == ele)
        {
            cout << "\n sum :: " << sum << "\t ele :: " << ele << "\t cnt :: " << cnt << "\t vec-size :: " << vec.length() << "\t sub-vec-size :: " << chose << "\t pId :: " << processorId << endl;
            cout.flush();
            usleep(100);
            return true;
        }

        if (!isLastCombination(comboVec, chose, vec.length()))
            _getNextCombination(comboVec, vec.length(), chose);
        else
            break;
        ++cnt;
    }

    return false;
}

bool processCombinations(ZZ_p e, vec_ZZ_p vec, ulong vectorSize, ulong subVectorSize)
{
    int processorId, totalNumberOfProcessors;
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &totalNumberOfProcessors);

    for (ulong subVecSize = subVectorSize; subVecSize < (vectorSize - 1); ++subVecSize)
    {
        // get quota for each processor
        ZZ totalCombinations = nCr(vectorSize, subVecSize);
        ulong quota = conv<ulong>((totalCombinations / conv<ZZ>(totalNumberOfProcessors)));
        ulong extra = conv<ulong>((totalCombinations % conv<ZZ>(totalNumberOfProcessors)));

        if (processorId == totalNumberOfProcessors - 1)
            quota += extra;

        // get start combination
        ulong symbolVec_dim = vectorSize;
        ulong symbolVec[symbolVec_dim];
        for (int i = 0; i < symbolVec_dim; ++i)
            symbolVec[i] = i;

        ulong combinationVec[subVecSize];
        ulong combinationIndex = ((conv<ulong>(totalCombinations)) * processorId) / (totalNumberOfProcessors);
        if (processorId == 0)
            combinationIndex = 0;

        get_kth_combination(symbolVec, vectorSize, subVecSize, combinationIndex, combinationVec);

        // search
        if (subVec_addition_kthCombo(e, vec, subVecSize, combinationVec, quota))
            return true;
    }
    return false;
}

void hypothesisi_A_parallel()
{
    int processorId, totalNumberOfProcessors;
    MPI_Comm_rank(MPI_COMM_WORLD, &processorId);
    MPI_Comm_size(MPI_COMM_WORLD, &totalNumberOfProcessors);

    ulong p = 35184372088639;

    ZZ_p::init(conv<ZZ>(p));
    usleep(10);
    ulong vectorSize = 39;
    ulong max_vectorSize = 45;
    ulong start_subVec_Size = 24;

    ulong numberOfSubVectors = 1000;

    ZZ_p e;

    if (processorId == 0)
        e = random_ZZ_p();

    while (vectorSize <= max_vectorSize)
    {
        vec_ZZ_p vec;
        vec.SetLength(vectorSize);
        ulong subVecSize = start_subVec_Size;

        if (processorId == 0)
        {
            bool position = 1;
            if ((conv<ZZ>(e)) < (p / 2))
                position = 0;

            cout << "\n !e :: " << e << " (" << NumBits(conv<ZZ>(e)) << "b) \t p :: " << p << " (" << NumBits(p) << "b) \t vec-size :: " << vectorSize << "\t position :: " << position << "\t sub-vec-size :: " << subVecSize << endl;

            ulong cnt = 0; // number of vectors to test
            while (cnt < numberOfSubVectors)
            {
                // generate random vector with unique elements
                vec = getRandomUniqueVec(vectorSize, e);

                // send e, vec and vectorSize to slave processors
                stringstream ss;
                ss << e;
                std::string s_i = ss.str();
                int strLen = s_i.length() + 1; // +1; Maybe important.
                char *e_str = new char[strLen];
                strcpy(e_str, s_i.c_str());

                MPI_Bcast(&strLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(e_str, strLen, MPI_CHAR, 0, MPI_COMM_WORLD);
                ss.clear();
                delete e_str;

                // Bcast vector
                stringstream ss2;
                ss2 << vec;
                std::string s_i_2 = ss2.str();
                int strLen2 = s_i_2.length() + 1; // +1; Maybe important.
                char *e_str_2 = new char[strLen2];
                strcpy(e_str_2, s_i_2.c_str());

                MPI_Bcast(&strLen2, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(e_str_2, strLen2, MPI_CHAR, 0, MPI_COMM_WORLD);
                ss2.clear();
                delete e_str_2;
                ++cnt;

                if (processCombinations(e, vec, vectorSize, subVecSize))
                {
                    cout << "\n vec-cnt :: " << cnt << endl;
                    MPI_Abort(MPI_COMM_WORLD, 73);
                }
            }
        }
        else
        {
            int cnt = 0;
            while (cnt < numberOfSubVectors)
            {
                // receive data from master processor

                // receive e
                int strLen;
                MPI_Bcast(&strLen, 1, MPI_INT, 0, MPI_COMM_WORLD);
                char *strKer_Ci = new char[strLen];
                MPI_Bcast(strKer_Ci, strLen, MPI_CHAR, 0, MPI_COMM_WORLD);
                stringstream ss;
                ss << strKer_Ci;
                ss >> e;
                delete strKer_Ci;
                ss.clear();

                // Recv vector
                int strLen2;
                MPI_Bcast(&strLen2, 1, MPI_INT, 0, MPI_COMM_WORLD);
                char *strKer_Ci2 = new char[strLen2];
                MPI_Bcast(strKer_Ci2, strLen2, MPI_CHAR, 0, MPI_COMM_WORLD);

                stringstream ss2;
                ss2 << strKer_Ci2;
                ss2 >> vec;

                delete strKer_Ci2;
                ss.clear();
                ++cnt;

                if (processCombinations(e, vec, vectorSize, subVecSize))
                {
                    cout << "\n vec-cnt :: " << cnt << endl;
                    MPI_Abort(MPI_COMM_WORLD, 73);
                }
            }
        } // end else

        ++vectorSize;
    } // end of while
}
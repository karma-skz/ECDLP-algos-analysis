using namespace std;
#include <bits/stdc++.h>


/**
 * Works for d <= 834
 * @param d
 */
void generateWeightedVector(ulong weight, ulong weightVector[][3]) {

    ulong totalElements = ((weight + 1) * (weight + 2)) / 2;

    for (ulong i = 0; i < totalElements; ++i) {
        weightVector[i][0] = 0;
        weightVector[i][1] = 0;
        weightVector[i][2] = 0;
    }

    //fill first column
    ulong tmp_cnt1 = weight;
    ulong tmp_cnt2 = 0;
    ulong tmp_cnt3 = weight;

    for (ulong i = (totalElements - 1); i > 0; --i) {
        weightVector[i][0] = tmp_cnt1;

        if (tmp_cnt1 == 0) {
            tmp_cnt3--;
            tmp_cnt1 = tmp_cnt3;
        } else {
            tmp_cnt1--;
        }
    }

    //fill second column
    tmp_cnt1 = weight;
    tmp_cnt2 = 0;
    tmp_cnt3 = weight;

    for (ulong i = (totalElements - 1); i > 0; --i) {
        weightVector[i][1] = tmp_cnt2;

        if (tmp_cnt2 == tmp_cnt3) {
            tmp_cnt3--;
            tmp_cnt2 = 0;
        } else {
            tmp_cnt2++;
        }
    }

    //fill third column
    tmp_cnt3 = 1;
    tmp_cnt1 = weight + 1;
    tmp_cnt2 = 0;

    weightVector[0][2] = weight;
    for (ulong i = (totalElements - 1); i > 0; --i) {

        weightVector[i][2] = tmp_cnt2;
        tmp_cnt1--;

        if (tmp_cnt1 == 0) {
            tmp_cnt2++;
            tmp_cnt1 = weight - tmp_cnt3 + 1;
            tmp_cnt3++;
        }
    }
}


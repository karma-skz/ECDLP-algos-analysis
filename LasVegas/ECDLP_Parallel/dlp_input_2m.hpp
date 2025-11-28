/* 
 * File:   dlp_input.hpp
 * Author: abdullah
 *
 * Created on 23 December, 2017, 9:59 PM
 */

#ifndef DLP_INPUT_2M_HPP
#define DLP_INPUT_2M_HPP

#include <fstream>
#include <iostream>
#include <vector>

#include <NTL/ZZ.h>

#include "EC_GF2E.hpp"
#include "EC_GF2E_Point.hpp"
#include "EC_utils.hpp"

using namespace NTL;
using namespace std;

typedef struct id {
    ulong p;
    ZZ e;
    GF2X a, b;
    GF2X Px, Py, Pz;
    GF2X Qx, Qy, Qz;
    GF2X irrd;
    ZZ ordP, ordQ;
} inputData;

class dlp_input_2m {
public:
    std::vector<inputData> data;

    dlp_input_2m(string fileName);

    void printInputData(ulong index);
    void printInputData1(ulong index);
    
};

#endif /* DLP_INPUT_HPP */
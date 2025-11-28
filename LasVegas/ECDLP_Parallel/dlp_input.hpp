/* 
 * File:   dlp_input.hpp
 * Author: abdullah
 *
 * Created on 25 December, 2017, 7:12 PM
 */

#ifndef DLP_INPUT_HPP
#define DLP_INPUT_HPP

#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>

#include "dlp_input_2m.hpp"

typedef struct ZZp1 {
    ZZ p;
    ZZ e;
    ZZ a, b;

    ZZ Px, Py, Pz;
    ZZ Qx, Qy, Qz;
    ZZ ordP, ordQ;
} inputData_ZZp;

class dlp_input {
public:

    std::vector<inputData_ZZp> data;

    dlp_input(string);
};

#endif /* DLP_INPUT_HPP */
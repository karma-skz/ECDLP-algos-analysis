/* 
 * File:   EC_GF2E_Point.hpp
 * Author: abdullah
 *
 * Created on 15 December, 2017, 11:35 AM
 */

#ifndef EC_GF2E_POINT_HPP
#define EC_GF2E_POINT_HPP

#include <iostream>

#include <NTL/GF2X.h>
#include <NTL/GF2XFactoring.h>
#include <NTL/GF2E.h>
#include <NTL/ZZ_p.h>

using namespace NTL;
using namespace std;

class EC_GF2E_Point {
public:
    GF2E x;
    GF2E y;
    GF2E z;

    EC_GF2E_Point(GF2E, GF2E);
    EC_GF2E_Point();

    void printPoint(string msg);
    void printPoint1(string msg);
};

#endif /* EC_GF2E_POINT_HPP */


/* 
 * File:   EC_ZZp_Point.hpp
 * Author: abdullah
 *
 * Created on 6 March, 2018, 11:14 AM
 */

#ifndef EC_ZZP_POINT_HPP
#define EC_ZZP_POINT_HPP

#include <NTL/ZZ_p.h>

using namespace NTL;

class EC_ZZp_Point {
public:
    ZZ_p x;
    ZZ_p y;
   
    ZZ_p z;

    EC_ZZp_Point(ZZ_p, ZZ_p);
    EC_ZZp_Point();

    void printPoint(std::string msg);
};

#endif /* EC_ZZP_POINT_HPP */
#include "EC_ZZp_Point.hpp"
#include "EC_ZZp.hpp"

using namespace std;

EC_ZZp_Point::EC_ZZp_Point(ZZ_p x, ZZ_p y) {
    this->x = x;
    this->y = y;
    this->z = 1;
}

EC_ZZp_Point::EC_ZZp_Point() {
    this->z = 1;
}

void EC_ZZp_Point::printPoint(std::string msg) {
    std::cout << msg << " :: [" << this->x << " : " << this->y << " : " << this->z << "]";
}
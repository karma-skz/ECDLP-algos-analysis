#include "EC_GF2E_Point.hpp"
#include "EC_GF2E.hpp"

using namespace std;

EC_GF2E_Point::EC_GF2E_Point() {
    this->z = 1;
}

EC_GF2E_Point::EC_GF2E_Point(GF2E x, GF2E y) {
    this->x = x;
    this->y = y;
    this->z = 1;
}

void EC_GF2E_Point::printPoint(string msg = "") {
    cout << msg << " :: [" << this->x << " : " << this->y << " : " << this->z << "]";
}

void EC_GF2E_Point::printPoint1(string msg = " ") {
    cout << msg << " :: [" << x._GF2E__rep.xrep << " : " << y._GF2E__rep.xrep << " : " << z._GF2E__rep.xrep << "]";
}

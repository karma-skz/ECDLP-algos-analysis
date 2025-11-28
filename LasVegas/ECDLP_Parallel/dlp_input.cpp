#include <iostream>

#include "dlp_input.hpp"

using namespace std;

dlp_input::dlp_input(string fileName) {

    ifstream fin(fileName);
    inputData_ZZp tmpData;
    string tmp;

    while (!fin.eof()) {

        fin >> tmpData.p;
        fin >> tmpData.e;

        fin >> tmpData.a;
        fin >> tmpData.b;

        fin >> tmpData.Px;
        fin >> tmpData.Py;
        tmpData.Pz = conv<ZZ>("1");

        fin >> tmpData.Qx;
        fin >> tmpData.Qy;
        tmpData.Qz = conv<ZZ>("1");

        fin >> tmpData.ordP;
        tmpData.ordQ = tmpData.ordP;

        fin>>tmp;
        if (tmp == "#") {
            this->data.push_back(tmpData);
        }
    }

}
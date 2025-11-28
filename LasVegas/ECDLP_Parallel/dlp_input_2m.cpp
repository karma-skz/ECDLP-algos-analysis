#include <fstream>

#include "dlp_input_2m.hpp"

dlp_input_2m::dlp_input_2m(string fileName) {

    ifstream fin(fileName);

    if (!fin) {
        cerr << B_RED_START << "Unable to open file : " << fileName << " \n" << RESET_TERM;
        exit(1);
    }

    while (!fin.eof()) {
        inputData tmpData;
        tmpData.Px.SetLength(1);
        tmpData.Py.SetLength(1);
        tmpData.Qx.SetLength(1);
        tmpData.Qy.SetLength(1);
        tmpData.a.SetLength(1);
        tmpData.b.SetLength(1);

        string tmp;
        fin >> tmpData.p;
        fin >> tmpData.a;
        fin >> tmpData.b;
        fin >> tmpData.Px;
        fin >> tmpData.Py;
        fin >> tmpData.Pz;
        
        fin >> tmpData.Qx;
        fin >> tmpData.Qy;
        fin >> tmpData.Qz;

        fin >> tmpData.ordP;
        fin >> tmpData.ordQ;

        fin >> tmpData.e;
        fin >> tmpData.irrd;
        fin>>tmp;

        if (tmp == "#") {
            this->data.push_back(tmpData);
        }
    }
    fin.close();
}

void dlp_input_2m::printInputData(ulong index = 0) {

    cout << "\n p :: " << this->data[index].p << endl;
    cout << "\n irrd :: " << this->data[index].irrd << endl;
    cout << " a :: " << this->data[index].a << "\t b :: " << this->data[index].b << endl;
    cout << " Px :: " << this->data[index].Px << "\t Py :: " << this->data[index].Py << endl;
    cout << " Qx :: " << this->data[index].Qx << "\t Qy :: " << this->data[index].Qy << endl;
    cout << " e :: " << this->data[index].e << "\t ord(P) :: " << this->data[index].ordP << endl;
}

void dlp_input_2m::printInputData1(ulong index = 0) {

    cout << "\n p :: " << this->data[index].p << endl;
    cout << "\n irrd :: " << this->data[index].irrd.xrep << endl;
    cout << " a :: " << this->data[index].a.xrep << "\t b :: " << this->data[index].b.xrep << endl;
    cout << " Px :: " << this->data[index].Px.xrep << "\t Py :: " << this->data[index].Py.xrep << endl;
    cout << " Qx :: " << this->data[index].Qx.xrep << "\t Qy :: " << this->data[index].Qy.xrep << endl;
    cout << " e :: " << this->data[index].e << "\t ord(P) :: " << this->data[index].ordP;
    cout << "\t ord(Q) :: " << this->data[index].ordQ << endl;
}
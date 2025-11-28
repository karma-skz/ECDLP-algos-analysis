#include <iostream>

#include "EC_GF2E.hpp"

using namespace std;

void EC_GF2E::generateRandomCurve() {

    a2._GF2E__rep = random_GF2X(p); //randomElement()
    a6._GF2E__rep = random_GF2X(p); //randomElement()
    //    a6._GF2E__rep.xrep.rep[0] = NTL::RandomWord() % this->p; //randomElement()

    a1._GF2E__rep.xrep.rep[0] = 1;
    a3._GF2E__rep.xrep.rep[0] = 0;
    a4._GF2E__rep.xrep.rep[0] = 0;

    a1._GF2E__rep.normalize();
    a2._GF2E__rep.normalize();
    a3._GF2E__rep.normalize();
    a4._GF2E__rep.normalize();
    a6._GF2E__rep.normalize();
}

EC_GF2E::EC_GF2E(ulong p, GF2X irrd, GF2X a, GF2X b) {

    this->p = p;

    GF2E::init(irrd);

    a1._GF2E__rep.xrep.SetLength(1);
    a2._GF2E__rep.xrep.SetLength(1);
    a3._GF2E__rep.xrep.SetLength(1);
    a4._GF2E__rep.xrep.SetLength(1);
    a6._GF2E__rep.xrep.SetLength(1);

    a2._GF2E__rep = a;
    a6._GF2E__rep = b;

    a1._GF2E__rep.xrep.rep[0] = 1;
    a3._GF2E__rep.xrep.rep[0] = 0;
    a4._GF2E__rep.xrep.rep[0] = 0;

    a1._GF2E__rep.normalize();
    a2._GF2E__rep.normalize();
    a3._GF2E__rep.normalize();
    a4._GF2E__rep.normalize();
    a6._GF2E__rep.normalize();
}

EC_GF2E::EC_GF2E(ulong p, GF2X a, GF2X b) {

    this->p = p;

    irrd = BuildSparseIrred_GF2X(p);
    GF2E::init(irrd);

    a1._GF2E__rep.xrep.SetLength(1);
    a2._GF2E__rep.xrep.SetLength(1);
    a3._GF2E__rep.xrep.SetLength(1);
    a4._GF2E__rep.xrep.SetLength(1);
    a6._GF2E__rep.xrep.SetLength(1);

    a2._GF2E__rep = a;
    a6._GF2E__rep = b;

    a1._GF2E__rep.xrep.rep[0] = 1;
    a3._GF2E__rep.xrep.rep[0] = 0;
    a4._GF2E__rep.xrep.rep[0] = 0;

    a1._GF2E__rep.normalize();
    a2._GF2E__rep.normalize();
    a3._GF2E__rep.normalize();
    a4._GF2E__rep.normalize();
    a6._GF2E__rep.normalize();
}

EC_GF2E::EC_GF2E(ulong p) {
    this->p = p;

    //    irrd = BuildIrred_GF2X(p);
    irrd = BuildSparseIrred_GF2X(p);
    GF2E::init(irrd);

    a1._GF2E__rep.xrep.SetLength(1);
    a2._GF2E__rep.xrep.SetLength(1);
    a3._GF2E__rep.xrep.SetLength(1);
    a4._GF2E__rep.xrep.SetLength(1);
    a6._GF2E__rep.xrep.SetLength(1);

    generateRandomCurve();
}

GF2E EC_GF2E::getDiscriminant(const GF2E &x, const GF2E &y) {
    return GF2E(1);
}

EC_GF2E_Point EC_GF2E::generateRandomPoint() {

    EC_GF2E_Point P;

    P.x._GF2E__rep.SetLength(1);
    P.y._GF2E__rep.SetLength(1);
    P.z._GF2E__rep.SetLength(1);

    while (1) {
        EC_GF2E_Point P;

        P.x = random_GF2E();
        P.y = random_GF2E();

        this->discriminant = getDiscriminant(P.x, P.y);
        if (this->discriminant == 0)
            continue;

        if (isPointValid(P))
            return P;
    }
}

void EC_GF2E::printCurve() {
    cout << "Elliptic Curve Defined by \n y^2 + " << this->a1 << "xy + " << this->a3 << "y = "
            "x^3 + " << this->a2 << "x^2 + " << this->a4 << "x + " << this->a6 << " over a "
            "Fintie Field of size 2^" << this->p << endl;
}

void EC_GF2E::printCurve1() {
    cout << "Elliptic Curve Defined by \n y^2 + " << this->a1._GF2E__rep.xrep << "xy + " << this->a3._GF2E__rep.xrep << "y = "
            "x^3 + " << this->a2._GF2E__rep.xrep << "x^2 + " << this->a4._GF2E__rep.xrep << "x + " << this->a6._GF2E__rep.xrep << " over a "
            "Fintie Field of size 2^" << this->p << endl;
}

bool EC_GF2E::isPointValid(const EC_GF2E_Point &P) {

    // Check if P lies on the curve.
    // Equation of the curve is 
    // y^2*Z + X*Y*Z = X^3 + a2*X^2*Z + a6Z^3  (with Z = 1)

    // Check point at infinity
    if (P.x == 0 && P.y == 1 && P.z == 0) {
        return true;
    }

    GF2E ans = (P.y * P.y) + (P.x * P.y) -(P.x * P.x * P.x) - (this->a2 * P.x * P.x) - a6;

    if ((ans == 0))
        return true;
    else
        return false;
}

void EC_GF2E::pointAddition_Doubling(const EC_GF2E_Point &P, const EC_GF2E_Point &Q, EC_GF2E_Point &ans) {

    v_cout << "\n in pointAddition_Doubling...\n";

    //case one : Point at infinity (0:1:0). i.e. checking of z is ZERO
    if (P.x == 0 && P.y == 1 && P.z == 0) {
        v_cout << "\n return pointAddition_Doubling...\n";
        ans = Q;
        return;
    }

    if (Q.x == 0 && Q.y == 1 && Q.z == 0) {
        v_cout << "\n return pointAddition_Doubling...\n";
        ans = P;
        return;
    }

    if ((P.x == Q.x) && (P.y == Q.y)) {
        //Case Double...
        v_cout << "\n Case Double...\n";
        GF2E a, b, c, d, e;

        a = P.x * P.x;
        b = a + P.y * P.z;
        c = P.x * P.z;
        d = c * c;
        e = (b * b) + (b * c) + (this->a2 * d);

        ans.x = (c * e);
        ans.y = (((b + c) * e) + (a * a * c));
        ans.z = c * d;

        if (ans.z == 0 && ans.x == 0) {
            ans.x = 0;
            ans.y = 1;
            ans.z = 0;
        } else if (ans.z != 0) {
            ans.x /= ans.z;
            ans.y /= ans.z;
            ans.z = 1;
        }
        v_cout << "\n return pointAddition_Doubling...\n";
        return;
    } else if ((P.x == Q.x)) {
        v_cout << "\n Case Inverse...\n";

        ans.x = 0;
        ans.y = 1;
        ans.z = 0;

        v_cout << "\n return pointAddition_Doubling...\n";
        return;
    } else {
        v_cout << "\n Case Addition...\n";
        //        EC_GF2E_Point ans;
        GF2E a, b, c, d, e;

        a = P.y * Q.z + P.z * Q.y;
        b = P.x * Q.z + P.z * Q.x;
        c = b * b;
        d = P.z * Q.z;
        e = (a * a + a * b + this->a2 * c) * d + (b * c);

        ans.z = (b * b * b) * d;
        ans.x = (b * e) / ans.z;
        ans.y = (c * (a * P.x + P.y * b) * Q.z + (a + b) * e) / ans.z;
        ans.z = 1;

        v_cout << "\n return pointAddition_Doubling...\n";
        return;
    }
    v_cout << "\n LAST return this is Should not happen... \n";
}

void EC_GF2E::scalarMultiplication_Basic(const EC_GF2E_Point &P, ZZ scalar, EC_GF2E_Point &ans) {

    //Handel case when scalar = 0; return point at infinity (0:1:0)


    ZZ cnt = conv<ZZ>("2");
    pointAddition_Doubling(P, P, ans);

    while (cnt < scalar) {
        EC_GF2E_Point tmp_p;

        pointAddition_Doubling(P, ans, tmp_p);

        ans.x = tmp_p.x;
        ans.y = tmp_p.y;
        ans.z = tmp_p.z;

        cnt++;
    }
    return;
}

/**
 * Q = -P
 * @param P
 * @param Q
 */
void EC_GF2E::pointNegation(const EC_GF2E_Point &P, EC_GF2E_Point &Q) {
    EC_GF2E_Point ans;

    ans.x = P.x;
    ans.y = (P.x + P.y);
    ans.z = 1;

    Q.x = ans.x;
    Q.y = ans.y;
    Q.z = 1;
    return;
}

/**
 * 
 * @param 
 * @param 
 * @param 
 */
void EC_GF2E::scalarMultiplicationDA(const EC_GF2E_Point &P, ZZ e, EC_GF2E_Point &Q) {

    ulong numOfBits = NumBits(e);

    Q.x = 0;
    Q.y = 1;
    Q.z = 0;

    for (long i = (numOfBits - 1); i >= 0; --i) {
        bool b = bit(e, i);
        EC_GF2E_Point tmpP;

        // Q = 2Q
        pointAddition_Doubling(Q, Q, tmpP);
        Q.x = tmpP.x;
        Q.y = tmpP.y;
        Q.z = tmpP.z;

        if (b) {
            EC_GF2E_Point tmpP1;
            pointAddition_Doubling(Q, P, tmpP1);
            Q.x = tmpP1.x;
            Q.y = tmpP1.y;
            Q.z = tmpP1.z;
        }
    }
}
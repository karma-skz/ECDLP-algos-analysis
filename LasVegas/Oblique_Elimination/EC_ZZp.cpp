#include "EC_ZZp.hpp"
#include "EC_GF2E_Point.hpp"
#include "constants.hpp"

using namespace std;

void EC_ZZp::generateRandomCurve() {
    this->a4 = random_ZZ_p();
    this->a6 = random_ZZ_p();
}

EC_ZZp::EC_ZZp(ZZ p) {

    this->p = p;
    ZZ_p::init(this->p);
    generateRandomCurve();
}

EC_ZZp::EC_ZZp(ZZ p, ZZ a4, ZZ a6, ZZ OrdP) {
    this->p = p;
    ZZ_p::init(this->p);

    this->a4 = conv<ZZ_p>(a4);
    this->a6 = conv<ZZ_p>(a6);
}

void EC_ZZp::printCurve() {
    cout << "Elliptic Curve Defined by y^2 = x^3 + " << this->a4 << "*x + " << this->a6 << " over "
            "Finite Field of size " << this->p << endl;
}

bool EC_ZZp::isPointValid(const EC_ZZp_Point &P) {

    // Check if P lies on the curve.
    // Equation of the curve is 
    // y^2 = X^3 + a4*X + a6  (with Z = 1)

    // Check point at infinity
    if (P.x == 0 && P.y == 1 && P.z == 0) {
        return true;
    }

    ZZ_p ans = (P.x * P.x * P.x) + (this->a4 * P.x) + this->a6 - (P.y * P.y);

    if ((ans == 0))
        return true;
    else
        return false;
}

void EC_ZZp::pointAddition_Doubling(const EC_ZZp_Point &P, const EC_ZZp_Point &Q, EC_ZZp_Point &ans) {

    v_print cout << "\n in pointAddition_Doubling...\n";

    //case one : Point at infinity (0:1:0). i.e. checking of z is ZERO
    if (P.x == 0 && P.y == 1 && P.z == 0) {
        v_print cout << "\n return pointAddition_Doubling...\n";
        ans = Q;
        return;
    }

    if (Q.x == 0 && Q.y == 1 && Q.z == 0) {
        v_print cout << "\n return pointAddition_Doubling...\n";
        ans = P;
        return;
    }

    if ((P.x == Q.x) && (P.y == Q.y)) {
        //Case Double...
        v_print cout << "\n Case Double...\n";
        // P = (X1 : Y1 : Z1)

        ZZ_p A, B, C, D;

        A = this->a4 + 3 * (P.x * P.x); // a4 * Z2^2 + 3 X1^2
        B = P.y; // Y1 * Z1
        C = P.x * P.y * B; //X1 * Y1 * B
        D = (A * A) - (8 * C); //A^2 - 8C

        ans.x = 2 * B * D;
        ans.y = (A * (4 * C - D)) - ((8 * P.y * P.y)*(B * B)); // A(4C-D) - 8Y1^2 * B^2
        ans.z = 8 * (B * B * B);

        ans.x /= ans.z;
        ans.y /= ans.z;
        ans.z = 1;

        return;

    } else if ((P.x == Q.x)) {
        v_print cout << "\n Case Inverse...\n";

        ans.x = 0;
        ans.y = 1;
        ans.z = 0;

        v_print cout << "\n return pointAddition_Doubling...\n";
        return;
    } else {
        v_print cout << "\n Case Addition...\n";

        // P = (X1,:Y1:Z1)  Q = (X2:Y2:Z2)

        ZZ_p A, B, C;

        A = Q.y - P.y; // Y2*Z1 - Y1*Z2
        B = Q.x - P.x; // X2*Z1 - X1*Z2
        C = (A * A) - (B * B * B) - 2 * (B * B) * P.x; //A^2*Z1*Z2 - B^3 - 2*B^2*X1*Z2

        ans.x = B*C;
        ans.y = (A * ((B * B * P.x) - C)) - ((B * B * B) * P.y); // A(B^2*X1*Z2 -C) - B^3 * Y1 * Z2
        ans.z = (B * B * B); // B^3 * Z1 * Z2

        ans.x /= ans.z;
        ans.y /= ans.z;
        ans.z = 1;

        return;
    }
    v_print cout << "\n LAST return this is Should not happen... \n";
}

/**
 * This is scalar multiplication of points in prime field
 * @param P : Base point in the prime field
 * @param e : Exponent
 * @param Q : Ans point
 */
void EC_ZZp::scalarMultiplicationDA(const EC_ZZp_Point &P, ZZ e, EC_ZZp_Point &Q) {

    ulong numOfBits = NumBits(e);

    Q.x = 0;
    Q.y = 1;
    Q.z = 0;

    for (long i = (numOfBits - 1); i >= 0; --i) {
        bool b = bit(e, i);
        EC_ZZp_Point tmpP;

        // Q = 2Q
        pointAddition_Doubling(Q, Q, tmpP);
        Q.x = tmpP.x;
        Q.y = tmpP.y;
        Q.z = tmpP.z;

        if (b) {
            EC_ZZp_Point tmpP1;
            pointAddition_Doubling(Q, P, tmpP1);
            Q.x = tmpP1.x;
            Q.y = tmpP1.y;
            Q.z = tmpP1.z;
        }
    }

}

void EC_ZZp::scalarMultiplication_Basic(const EC_ZZp_Point &P, ZZ e, EC_ZZp_Point &ans) {

    ZZ cnt = conv<ZZ>("2");
    pointAddition_Doubling(P, P, ans);

    while (cnt < e) {
        EC_ZZp_Point tmp_p;

        pointAddition_Doubling(P, ans, tmp_p);

        ans.x = tmp_p.x;
        ans.y = tmp_p.y;
        ans.z = tmp_p.z;

        cnt++;
    }
    return;
}

EC_ZZp_Point EC_ZZp::generateRandomPoint() {

    EC_ZZp_Point P1;

    while (1) {

        P1.x = random_ZZ_p();
        P1.y = random_ZZ_p();

        //        this->discriminant = getDiscriminant(P.x, P.y);
        //        if (this->discriminant == 0)
        //            continue;

        if (isPointValid(P1))
            return P1;
    }
}

void EC_ZZp::pointNegation(const EC_ZZp_Point &P, EC_ZZp_Point &Q) {
    EC_ZZp_Point ans;

    ans.x = P.x;
    ans.y = -P.y;
    ans.z = 1;

    Q.x = ans.x;
    Q.y = ans.y;
    Q.z = 1;
    return;
}

ZZ EC_ZZp::order(const EC_ZZp_Point& P) {
    
    ZZ cnt = conv<ZZ>(1);

    EC_ZZp_Point P_tmp;

    P_tmp.x = P.x;
    P_tmp.y = P.y;
    P_tmp.z = P.z;

    while (1) {
        
        if (P.x == 0 && P.y == 1 && P.z == 0) {
            return cnt;
        }

        EC_ZZp_Point P2;

        pointAddition_Doubling(P_tmp, P_tmp, P2);
        cnt++;

        P_tmp.x = P2.x;
        P_tmp.y = P2.y;
        P_tmp.z = P2.z;
        cout << "\n cnt :: " << cnt;
        P2.printPoint("\tP2 :: ");
        P_tmp.printPoint("\t\tP_tmp :: ");

        if (cnt > 2000)
            break;
    }
    return conv<ZZ>(0);
}
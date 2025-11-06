# Elliptic Curve Cryptography (ECC) and Discrete Logarithm Problem (DLP) Theory

## What is Elliptic Curve Cryptography (ECC)?

We are given an elliptic curve defined by the equation:
$$ y^2 \equiv x^3 + ax + b (mod p) $$

Points on the curve $+ a$ point at infinity forms the group.
Suppose G is a point on the curve.

a ,b, p, G are all public parameters.

Now, we choose a private key d, which is a random integer in the interval [1, n-1], where n is the order of the point G, such that, 
- Public key Q is computed as Q = dG = G + G + ... + G (d times)

### How does addition of two points work?

- Let P = (x1, y1) and Q = (x2, y2) be two points on the elliptic curve.
- Extrapolate the line joining P and Q, the third point of intersection of this line with the curve is R = (x3, y3). Then we say, R = P + Q.

In our case, first we take the tangent line at G. The point of intersection will be G + G = 2G. Similarly, we can find 3G as G + 2G, and so on.
```
2G = G + G
3G = G + 2G
4G = 2G + 2G
5G = 4G + G
6G = 3G + 3G
i.e, for even k: kG = (k/2)G + (k/2)G
     for odd k: kG = ((k-1)/2)G + ((k+1)/2)G
```

(note : G = O + G, where O is the point at infinity which acts as some point of group which along with G forms the first line which we extrapolate to find the next point on the curve)

Let, R = dG = (xR, yR)

So , Public key Q = R = (xR, yR)
We have to keep d secret.

## What is Discrete Logarithm Problem (DLP)?

The Discrete Logarithm Problem (DLP) is the problem of finding the integer d, given the points G and Q on the elliptic curve, such that Q = dG.

Now, every group has something called the "order" of the group (n). The order of a point G on an elliptic curve is the smallest positive integer n such that nG = O, where O is the point at infinity.

n must be a prime so that all mod values are uniformly distributed (primitive root exists only for prime mod values).

So, d is in the range [1, n-1].

Intuition :
1. Finding Q = dG is easy (just keep adding G, d times).
2. But, given G and Q, finding d such that Q = dG is hard, thats the ECDLP.

## Toy example of ECC and DLP

Suppose, p = 17, a = 2, b = 2, G = (5,1), Q = (10,6)

So, E : y^2 = x^3 + 2x + 2 (mod 17)

This curve has 19 points (including point at infinity).

We have to find d such that Q = dG.

After calculating multiples of G, we get the following table:
| k | kG        |
|---|-----------|
| 1 | (5, 1)    |
| 2 | (6, 3)    |
| 3 | (10, 6)   |   
| 4 | (3, 1)    |
| 5 | (9, 16)   |
| 6 | (16, 5)   |
| 7 | (0, 6)    | 
| 8 | (13, 7)   |
| 9 | (7, 6)    |
| 10| (8, 3)    |
| 11| (16, 12)  |
| 12| (9, 1)    |
| 13| (3, 16)   |
| 14| (10, 11)  |
| 15| (6, 14)   |
| 16| (5, 16)   |
| 17| O         | # point at infinity (identity element)


From the table, we can see that Q = (10, 6) corresponds to k = 3.
Thus, d = 3.

## Practical value range:

For bitcoin, secp256k1 curve is used.

symbol    |  bit-size | example |
----------|-----------|------------------
p         |   256     | 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
a         |    0      | 0
b         |    7      | 7
G         |   256     | (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798, 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
n         |   256     | 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
h         |    1      | 1   


### Why Ellyptic Curve Cryptography, not RSA?

- For ECC, public key size (value of p) : 256 bits (32 bytes), private key size (value of d) : 256 bits (32 bytes)

- For RSA, public key size : 2048 bits (256 bytes), private key size : 2048 bits (256 bytes)

- For same level of security, ECC requires much smaller key sizes compared to RSA.

--- 

import numpy as np
from numpy.polynomial import Polynomial
import argparse


def DFT(f: np.array) -> np.array:
    """
    Convert polynomial to DFT
    :param f: Polynomial f(x)
    :return: DFT of polynomial f(x)
    """
    if f.size == 1:
        return f

    n = f.size
    f0 = [f[2 * i] for i in range(0, n // 2)]
    f1 = [f[2 * i + 1] for i in range(0, n // 2)]
    resultDFT0 = DFT(np.array(f0))
    resultDFT1 = DFT(np.array(f1))

    resultDFT = np.zeros(n, dtype = 'complex_')
    angle = 2 * np.pi / n
    w = 1
    wn = complex(np.cos(angle), np.sin(angle))
    for i in range(0, n // 2):
        resultDFT[i] = resultDFT0[i] + w * resultDFT1[i]
        resultDFT[i + n // 2] = resultDFT0[i] - w * resultDFT1[i]
        w *= wn
    return resultDFT

def invDFT(data: np.array) -> np.array:
    """
    Convert DFT to polynomial
    :param data: DFT of polynomial f
    :return: Polynomial f(x)
    """
    if data.size == 1:
        return data
    n = data.size
    p0 = invDFT(np.array([data[2 * i] for i in range(0, n // 2)]))
    p1 = invDFT(np.array([data[2 * i + 1] for i in range(0, n // 2)]))

    coeff = np.zeros(n, dtype='complex_')
    angle = -2 * np.pi / n
    w = 1
    wn = complex(np.cos(angle), np.sin(angle))
    for i in range(0, n // 2):
        coeff[i] = (p0[i] + w * p1[i]) / 2
        coeff[i + n // 2] = (p0[i] - w * p1[i]) / 2
        w *= wn
    return coeff

def FFT(a: np.array, b: np.array) -> np.array:
    """
    FFT algorithm to multiply two polynomials
    :param a: Polynomial a(x)
    :param b: Polynomial b(x)
    :return: Polynomial c(x) = a(x) * b(x)
    """
    EPS = 1e-10
    deg = 2 ** (int(np.log2(a.size + b.size - 1)) + 1)
    a.resize(deg, refcheck=False)
    b.resize(deg, refcheck=False)
    result = invDFT(np.multiply(DFT(a), DFT(b)))
    result = [value.real for value in result]
    while result[-1] < EPS:
        result.pop()
    return result

def trivial(a: np.array, b: np.array):
    n = a.size
    m = b.size
    c = np.zeros(n + m)
    for i in range(0, n):
        for j in range(0, m):
            c[i + j] += a[i] * b[j]
    return c

def main():
    parser = argparse.ArgumentParser(description="Program excuted FFT and trivial algorithms to multiply two polynomial")
    parser.add_argument('-a', '--algo')
    args = parser.parse_args()

    a = np.array(list(map(int, input().split())))
    b = np.array(list(map(int, input().split())))

    # print(f'a(x) = {Polynomial(a)}')
    # print(f'b(x) = {Polynomial(b)}')
    if args.algo == 'trivial':
        print("Trivial", end="")
        # print(f'Trivial: a(x) * b(x) = {Polynomial(trivial(a, b))}')

    if args.algo == 'FFT':
        print("FFT", end="")
        Polynomial(FFT(a, b))
        # print(f'FFT :    a(x) * b(x) = {Polynomial(FFT(a, b))}')
    # print(f'Diff :                 {Polynomial(FFT(a, b)) - Polynomial(a) * Polynomial(b)}')


if __name__ == '__main__':
    main()
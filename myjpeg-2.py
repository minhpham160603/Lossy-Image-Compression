#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:17:05 2022

@author: minhpham1606
"""

from math import *
import random
import time

'''Week 1
------------------'''

def ppm_tokenize(stream):
    for line in stream:
        tmp = line.strip().split()
        for item in tmp:
            if item == '#':
                break
            yield item
            
#with open('test0.ppm') as stream:
#	for token in ppm_tokenize(stream):
#		print(token)

def ppm_load(stream):
    code = [c for c in ppm_tokenize(stream)]
    for i in range(1, len(code)):
        code[i] = int(code[i])
    w, h = code[1], code[2]
    img = []
    line = []
    for i in range(4, len(code), 3):
        if len(line) < w:
            line.append(tuple(code[i:i + 3]))
        if len(line) == w:
            img.append(line.copy())
            line = []
    return w, h, img
            
#with open('test1.ppm') as stream:
#    w, h, img = ppm_load(stream)
#print(w)
#print(h)
#print(img)

def ppm_save(w, h, img, output):
    output.write('P3\n')
    output.write(f'{w} {h}\n')
    output.write('255\n')
    for line in img:
        for pixel in line:
            output.write(f'{pixel[0]} {pixel[1]} {pixel[2]}\n')
    
        
#with open('test1.ppm', 'w') as output:
#    ppm_save(w, h, img, output)
    
'''Part 2'''
def RGB2YCbCr(r, g, b):
    Y = round(0.299*r + 0.587*g + 0.114*b)
    Cb = round(128 - 0.168736*r - 0.331264*g + 0.5*b)
    Cr = round(128 + 0.5*r - 0.418688*g - 0.081312*b)
    l = [Y, Cb, Cr]
    for i in range(3):
        if l[i] < 0:
            l[i] = 0
        elif l[i] > 255:
            l[i] = 255
    return tuple(l)
    
def YCbCr2RGB(Y, Cb, Cr):
    r = round(Y + 1.402*(Cr - 128))
    g = round(Y - 0.344136*(Cb - 128) - 0.714136*(Cr - 128))
    b = round(Y + 1.772*(Cb - 128))
    l = [r, g, b]
    for i in range(3):
        if l[i] < 0:
            l[i] = 0
        elif l[i] > 255:
            l[i] = 255
    return tuple(l)

def img_RGB2YCbCr(img):
    w, h = len(img[0]), len(img)
    Y, Cb, Cr = [[0]*w for _ in range(h)], [[0]*w for _ in range(h)], [[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            r, g, b = img[i][j]
            Y[i][j], Cb[i][j], Cr[i][j] = RGB2YCbCr(r, g, b)
    return Y, Cb, Cr


def img_YCbCr2RGB(channels):
    Y, Cb, Cr = channels 
    w, h = len(Y[0]), len(Y)
    img = [[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            img[i][j] = YCbCr2RGB(Y[i][j], Cb[i][j], Cr[i][j])
    return img

#print(img_YCbCr2RGB(Y, Cb, Cr))

'''Subsampling'''
def subsampling(w, h, C, a, b):
    #i height, j width
    # b width, a height
    res = []
    for i in range(0, h, a):
        row = []
        for j in range(0, w, b):
            block = []
            bw, bh = min(w, j + b) - j, min(i + a, h) - i #block width and block height
            for k in range(i, min(i + a, h)):
                line = C[k][j: min(w, j + b)]
                block.append(sum(line))
            row.append(round(sum(block)/(bh*bw)))
        res.append(row)
    return res
    
def extrapolate(w, h, C, a, b):
    res = [[0]*w for _ in range(h)]
    for i in range(0, h):
        for j in range(0, w):
            res[i][j] = C[i//b][j//a]
    return res

'''Block spliting'''
def block_splitting(w, h, C):
    i = 0 #i -> h, j -> w
    tmp = []
    while i < h:
        j = 0
        while j < w:
            if j < w - 8 and i < h - 8:
                tmp = [[C[i][j] for j in range(j, j + 8)] for i in range(i, i + 8)]
            elif j >= w - 8 and i < h - 8:
                tmp = []
                for k in range(i, i + 8):
                    line = C[k][j:] + [C[k][-1]]*(8 - (w - j)) #filling with dummy pixel
                    tmp.append(line)
            elif j < w - 8 and i >= h - 8:
                tmp = [C[i][j:j + 8] for i in range(i, h)] + [C[-1][j: j + 8]]*(8 - (h - i))
            else:
                tmp = []
                for k in range(i, h):
                    line = C[k][j:] + [C[k][-1]]*(8 - (w - j)) #filling with dummy pixel
                    tmp.append(line)
                tmp += [tmp[-1]]*(8 - (h - i))
            yield tmp
            j += 8
        i += 8
            
            
C = [
    [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    [ 2,  3,  4,  5,  6,  7,  8,  9, 10,  1],
    [ 3,  4,  5,  6,  7,  8,  9, 10,  1,  2],
    [ 4,  5,  6,  7,  8,  9, 10,  1,  2,  3],
    [ 5,  6,  7,  8,  9, 10,  1,  2,  3,  4],
    [ 6,  7,  8,  9, 10,  1,  2,  3,  4,  5],
    [ 7,  8,  9, 10,  1,  2,  3,  4,  5,  6],
    [ 8,  9, 10,  1,  2,  3,  4,  5,  6,  7],
    [ 9, 10,  1,  2,  3,  4,  5,  6,  7,  8],
]
#m = block_splitting(10, 9, C)
#print(m[2])

#m = subsampling(10, 9, C, 3, 2)
#n = extrapolate(10, 9, m, 3, 2)
#print(subsampling(10, 9, n, 3, 2))

'''Week 2
-----------------------------'''

def DCT(v):
    n = len(v)
    v_hat = [0]*n
    for i in range(n):
        for j in range(n):
            delta = 1
            if i == 0:
                delta = 1/sqrt(2)
            v_hat[i] += delta*sqrt(2/n)*v[j]*cos((pi/n)*(j + 1/2)*i)
    return v_hat

def IDCT(v):
    '''Inverse DCT
    We deduce from v_hat = v*C^T that v = v_hat*C, since C^(-1) = C^T
    To avoid run nest foor loops twice we calculate v_out directly base on the matrix mul'''
    n = len(v)
    v_out = [0]*n
    for j in range(n):
        for i in range(n):
            delta = 1
            if i == 0:
                delta = 1/sqrt(2)
            v_out[j] += delta*sqrt(2/n)*v[i]*cos((pi/n)*(j + 1/2)*i)
    return v_out

def test1():
    '''Check IDCT'''
    v = [
        float(random.randrange(-10**5, 10**5))
        for _ in range(random.randrange(1, 128))
    ]
    v2 = IDCT(DCT(v))
    assert (all(isclose(v[i], v2[i]) for i in range(len(v))))

def transpose(A):
    m, n = len(A), len(A[0])
    A_t = [[0]*m for _ in range(n)]
    for j in range(n):
        for i in range(m):
            A_t[j][i] = A[i][j] 
    return A_t

def DCT2(m, n, A):
    # Apply DCT for each row in A
    B = A.copy()
    for i in range(m):
        B[i] = DCT(A[i])
    #Create a transpose matrix
    B = transpose(B)
    #Apply DCT for each row in the transpose matrix
    for i in range(n):
        B[i] = DCT(B[i])
    #Transpose the matrix back to A
    B = transpose(B)
    return B

Atest = [
  [ 140,  144,  147,  140,  140,  155,  179,  175],
  [ 144,  152,  140,  147,  140,  148,  167,  179],
  [ 152,  155,  136,  167,  163,  162,  152,  172],
  [ 168,  145,  156,  160,  152,  155,  136,  160],
  [ 162,  148,  156,  148,  140,  136,  147,  162],
  [ 147,  167,  140,  155,  155,  140,  136,  162],
  [ 136,  156,  123,  167,  162,  144,  140,  147],
  [ 148,  155,  136,  155,  152,  147,  147,  136],
]
            
#res = DCT2(len(Atest), len(Atest[0]), Atest)
A_result = [
  [1210.000,  -17.997,   14.779,   -8.980,   23.250,   -9.233,  -13.969,  -18.937],
  [  20.538,  -34.093,   26.330,   -9.039,  -10.933,   10.731,   13.772,    6.955],
  [ -10.384,  -23.514,   -1.854,    6.040,  -18.075,    3.197,  -20.417,   -0.826],
  [  -8.105,   -5.041,   14.332,  -14.613,   -8.218,   -2.732,   -3.085,    8.429],
  [  -3.250,    9.501,    7.885,    1.317,  -11.000,   17.904,   18.382,   15.241],
  [   3.856,   -2.215,  -18.167,    8.500,    8.269,   -3.608,    0.869,   -6.863],
  [   8.901,    0.633,   -2.917,    3.641,   -1.172,   -7.422,   -1.146,   -1.925],
  [   0.049,   -7.813,   -2.425,    1.590,    1.199,    4.247,   -6.417,    0.315],
]


def IDCT2(m, n, A):
    B = A.copy()
    for i in range(m):
        B[i] = IDCT(A[i])
    B = transpose(B)
    for i in range(n):
        B[i] = IDCT(B[i])
    return transpose(B)
    
def test2():
    '''Check IDCT2'''
    m = random.randrange(1, 128)
    n =  random.randrange(1, 128)
    A = [
    [
        float(random.randrange(-10**5, 10**5))
        for _ in range(n)
    ]
    for _ in range(m)
    ]
    A2 = IDCT2(m, n, DCT2(m, n, A))
    assert (all(isclose(A[i][j], A2[i][j])
    for i in range(m) for j in range(n)))
    

'''8 x 8 DCT-II Transform'''
def redalpha(i):
    p = i//16 #number of times of pi
    k = i - p*16
    if k > 8:
        k = 16 - k
        p += 1
    s = (-1)**p
    return s, k
    
def ncoeff8(i, j):
    return redalpha(i*(2*j + 1))

M8 = [
    [ncoeff8(i, j) for j in range(8)]
    for i in range(8)
]

def M8_to_str(M8):
    def for1(s, i):
        return f"{'+' if s >= 0 else '-'}{i:d}"

    return "\n".join(
        " ".join(for1(s, i) for (s, i) in row)
        for row in M8
    )
    
#print(M8_to_str(M8))

def alpha(i):
    '''Calculate alpha and divide by 2 to reduce multiplication'''
    if i == 0:
        i = 4
    return cos(i*pi/16)/2

# DCT Chen

def DCT_Chen_1D(v):
    '''
    Apply 1D DCT to a vector
    22 multiplications per row'''
    u = [0]*8
    for i in [1, 3, 5, 7]: #the elements that there's no overlap mul to reuse: 16 muls
        for j in range(4):
            s, id_alpha = ncoeff8(i, j)
            u[i] += (v[j] + (-1)*v[7 - j])*s*alpha(id_alpha) #literal expression
    #The elements which have overlap multiplications  
    #Element 0: 1 mul
    u[0] = alpha(0)*sum(v) 
    #Element 2: 2 muls
    u[2] = alpha(2)*(v[0] + v[7] - v[3] - v[4]) + alpha(6)*(v[1] + v[6] - v[2] - v[5])
    #Element 4: 1 mul
    u[4] = alpha(4)*(v[0] + v[7] - v[1] - v[6] - v[2] - v[5] + v[3] + v[4])
    #Element 6: 2 muls
    u[6] = alpha(6)*(v[0] + v[7] - v[3] - v[4]) - alpha(2)*(v[1] + v[6] - v[2] - v[5])
    return u

def DCT_Chen(A):
    for r in range(8): #loop through row of A
        A[r] = DCT_Chen_1D(A[r])
    A = transpose(A)
    for r in range(8): #loop through row of A
        A[r] = DCT_Chen_1D(A[r])
    return transpose(A)

#Omega = [[s*alpha(k)/2 for s, k in M8[r]] for r in range(0, 8, 2)] #precalculate Omega/2
#Theta = [[s*alpha(k)/2 for s, k in M8[r]] for r in range(1, 8, 2)] #precalculate Theta/2

def IDCT_Chen_1D(v):
    '''We calculate [v0, v2, v4, v6]*Omega and [v1, v3, v5, v7]*Theta separately'''
    cache = dict()
    # [v0, v2, v4, v6]*Omega
    ve_Omega = [0]*4
    for i in range(4):
            for j in range(0, 8, 2):
                omega, id_alpha = ncoeff8(j, i)
                name = f'v{j}_a{id_alpha}'
                if name not in cache:
                    cache[name] = v[j]*alpha(id_alpha)
                ve_Omega[i] += cache[name]*omega
    # [v1, v3, v5, v7]*Theta
    vo_Theta = [0]*4
    for i in range(4):
            for j in range(1, 8, 2):
                omega, id_alpha = ncoeff8(j, i)
                name = f'v{j}_a{id_alpha}'
                if name not in cache:
                    cache[name] = v[j]*alpha(id_alpha)
                vo_Theta[i] += cache[name]*omega
    ans1 = [ve_Omega[i] + vo_Theta[i] for i in range(4)]
    ans2 = [ve_Omega[j] - vo_Theta[j] for j in range(3, -1, -1)]
    return ans1 + ans2

def IDCT_Chen(A):
    for _ in range(2):
        for r in range(8):
            A[r] = IDCT_Chen_1D(A[r])
        A = transpose(A)
    return A

#x = DCT_Chen(Atest)
def test3():
    '''Compare DCT2 vs DCT_Chen'''
    A = [
    [
        float(random.randrange(-10**5, 10**5))
        for _ in range(8)
    ]
    for _ in range(8)
    ]
    A1 = DCT2(8, 8, A.copy())
    A2 = DCT_Chen(A.copy())
    assert (all(isclose(A1[i][j], A2[i][j])
    for i in range(8) for j in range(8)))
    
def test4():
    '''Check IDCT_Chen and DCT_Chen'''
    A = [
    [
        float(random.randrange(-10**5, 10**5))
        for _ in range(8)
    ]
    for _ in range(8)
    ]
    A2 = IDCT_Chen(DCT_Chen(A.copy()))
    assert (all(isclose(A[i][j], A2[i][j])
    for i in range(8) for j in range(8)))

            
def quantization(A, Q):
    return [[round(A[i][j]/Q[i][j]) for j in range(8)] for i in range(8)]

def quantizationI(A, Q):
    return [[A[i][j]*Q[i][j] for j in range(8)] for i in range(8)]
    
#Luminance
LQM = [
  [16, 11, 10, 16,  24,  40,  51,  61],
  [12, 12, 14, 19,  26,  58,  60,  55],
  [14, 13, 16, 24,  40,  57,  69,  56],
  [14, 17, 22, 29,  51,  87,  80,  62],
  [18, 22, 37, 56,  68, 109, 103,  77],
  [24, 35, 55, 64,  81, 104, 113,  92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103,  99],
]

#Chrominance
CQM = [
  [17, 18, 24, 47, 99, 99, 99, 99],
  [18, 21, 26, 66, 99, 99, 99, 99],
  [24, 26, 56, 99, 99, 99, 99, 99],
  [47, 66, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
]

def Qmatrix(isY, phi):
    Q = []; Sphi = 0
    if phi < 50:
        Sphi = round(5000/phi)
    else:
        Sphi = 200 - 2*phi
    if isY:
        Q = LQM
    else:
        Q = CQM
    return [[ceil((50 + Sphi*Q[i][j])/100) for j in range(8)] for i in range(8)]

'''Week 3'''
def zigzag(A):
    i = 0; j = 0
    isUp = True
    while i < 8 and j < 8:
        yield A[i][j]
        if isUp:
            if i != 0 and j != 7:
                i -= 1
            else:
                if j == 7:
                    i += 1
                    j -= 1
                isUp = False
            j += 1
        else:
            if j != 0 and i != 7:
                j -= 1
            else:
                if i == 7:
                    i -= 1
                    j += 1
                isUp = True
            i += 1
        
Ctest = [
    [ 1,  2,  3,  4,  5,  6,  7,  8],
    [ 2,  3,  4,  5,  6,  7,  8,  9],
    [ 3,  4,  5,  6,  7,  8,  9, 10],
    [ 4,  5,  6,  7,  8,  9, 10,  1],
    [ 5,  6,  7,  8,  9, 10,  1,  2],
    [ 6,  7,  8,  9, 10,  1,  2,  3],
    [ 7,  8,  9, 10,  1,  2,  3,  4],
    [ 8,  9, 10,  1,  2,  3,  4,  5],
]

#print([x for x in zigzag(C)])

Ctest1 = [
    [ 1,  2,  0,  0,  0,  0,  7,  8],
    [ 2,  3,  0,  5,  6,  0,  8,  9],
    [ 3,  4,  0,  6,  0,  8,  9, 10],
    [ 0,  5,  0,  0,  0,  9, 10,  1],
    [ 5,  6,  0,  8,  9, 0,  1,  2],
    [ 6,  7,  8,  9, 0,  1,  2,  3],
    [ 7,  8,  9, 10,  0,  0,  3,  0],
    [ 8,  9, 10,  1,  2,  3,  4,  5],
]


def rle0(g):
    count = 0
    for n in g:
        if n == 0:
            count += 1
        else:
            yield(count, n)
            count = 0
            
g = [x for x in zigzag(Ctest1)] 
        
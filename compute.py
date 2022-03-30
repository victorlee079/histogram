import numpy as np
import math
from scipy import signal as sg

Rx = math.radians(1)
Ry = math.radians(2)
Rz = math.radians(1.5)

# Rotation
R = np.array([[1, -Rz, Ry],
              [Rz, 1, -Rx],
              [-Ry, Rx, 1]])

Kint = np.array([[500, 0, 500],
                 [0, 525, 495],
                 [0, 0, 1]])
# Model
M = np.array([[0.5],
              [0.5],
              [2.5]])

# Reference Point
P = np.array([[0.4],
              [0.6],
              [2.4]])

# Translation
T = np.array([[0.2],
              [0.1],
              [0.4]])

# New Model
print("***** M' *****")
Mn = R.dot(M + T - P) + P
print(Mn)

print("***** Projection *****")
x = Kint.dot(Mn)
print(x / x[-1])

# Convolution

# Image
S = np.array([[2, 5, 2],
              [1, 7, 4],
              [2, 5, 1]])

# Mask
m = np.array([[5, 5, 2],
              [2, -8, 1],
              [2, 6, 3]])

# Fully overlapped
print("***** Convolution *****")
# mode="valid" for fully overlapped
cS = sg.convolve2d(S, m)
print(cS)

# Color
rgb_max = 255

R = 45
G = 20
B = 10

nR = R / rgb_max
nG = G / rgb_max
nB = B / rgb_max

max_channel = max(R, G, B)
min_channel = min(R, G, B)

H1 = 0
if R == max_channel:
    H1 = (G - B) / (max_channel - min_channel)
if G == max_channel:
    H1 = 2 + (B - R) / (max_channel - min_channel)
if B == max_channel:
    H1 = 4 + (R - G) / (max_channel - min_channel)
H = H1 * 60
C = (max_channel - min_channel) / rgb_max
I = (1 / 3) * (nR + nG + nB)
V = max_channel / rgb_max
L = (1 / 2) * (max_channel + min_channel) / rgb_max

S_HSV = 0 if C == 0 else C / V

S_HSL = 0
if C == 0:
    S_HSL = 0
elif L > 1 / 2:
    S_HSL = C / (2 - 2 * L)
elif L <= 1 / 2:
    S_HSL = C / (2 * L)

S_HSI = 0 if C == 0 else 1 - min_channel / rgb_max / I

print("***** HSV *****")
print(H)
print(S_HSV)
print(V)

print("***** HSI *****")
print(H)
print(S_HSI)
print(I)

print("***** HSL *****")
print(H)
print(S_HSL)
print(L)

# Eigen Value
print("***** Eigen Values *****")
harris_matrix = np.array([[12, 12],
                          [12, 12]])
evs = np.linalg.eigvals(harris_matrix)
print(evs)

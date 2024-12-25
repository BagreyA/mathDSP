import numpy as np
import matplotlib.pyplot as plt

M = 66
h_a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
k = np.arange(0, M, 1)  

h_p = np.exp(-1j * k * ((2 * np.pi) / M) * ((M - 1) / 2))


H = h_a * h_p


plt.figure(1)
plt.plot(h_a)
plt.title('Amplitude Signal h_a')

plt.figure(2)
plt.plot(h_p)
plt.title('Exponential Signal h_p')

plt.figure(3)
plt.plot(H)
plt.title('Combined Signal h')

plt.show()
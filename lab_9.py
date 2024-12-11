import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2/60, 4000)
f_s = 400
Ts = 1/f_s
t_n = np.arange(0, 2/60, Ts)

x1 = np.cos(2 * np.pi * 60 * t + np.pi/3)
x2 = np.cos(2 * np.pi * 340 * t + np.pi/3)
x3 = np.cos(2 * np.pi * 460 * t + np.pi/3)

x4 = np.cos(2 * np.pi * 60 * t_n + np.pi/3)
x5 = np.cos(2 * np.pi * 340 * t_n + np.pi/3)
x6 = np.cos(2 * np.pi * 460 * t_n + np.pi/3)

t1 = np.linspace(-10, 10, 251)
h = (np.sin(np.pi * t1)) / (np.pi * t1)
h = np.nan_to_num(h, nan=1)

print(h)

plt.figure(figsize=[10, 5])
plt.plot(t, x1, label="x1")
plt.stem(t_n, x4, label="x4", linefmt='g', markerfmt='go', basefmt="k")
plt.title('cos(2π60t + π/3)')
plt.legend()
plt.show()

plt.figure(figsize=[10, 5])
plt.plot(t, x2, label="x2")
plt.stem(t_n, x5, label="x5", linefmt='g', markerfmt='go', basefmt="k")
plt.title('cos(2π340t + π/3)')
plt.legend()
plt.show()

plt.figure(figsize=[10, 5])
plt.plot(t, x3, label="x3")
plt.stem(t_n, x6, label="x6", linefmt='g', markerfmt='go', basefmt="k")
plt.title('cos(2π460t + π/3)')
plt.legend()
plt.show()

plt.figure(figsize=[10, 5])
plt.xlim(-10, 10)
plt.plot(t1, h)
plt.title('sin(πt1) / (πt1)')
plt.show()

y1 = np.convolve(x1, h, mode='same')
y2 = np.convolve(x2, h, mode='same')
y3 = np.convolve(x3, h, mode='same')

plt.figure(figsize=[10, 5])
plt.plot(y1, label="y1 = x1 * h")
plt.title('Свертка x1 с h')
plt.legend()
plt.show()

plt.figure(figsize=[10, 5])
plt.plot(y2, label="y2 = x2 * h")
plt.title('Свертка x2 с h')
plt.legend()
plt.show()

plt.figure(figsize=[10, 5])
plt.plot(y3, label="y3 = x3 * h")
plt.title('Свертка x3 с h')
plt.legend()
plt.show()

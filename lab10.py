import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

# ===== Задание 1 =====
fc1 = 10
fs1 = 32 * fc1
t1 = np.arange(0, 2, 1/fs1)
x1 = np.cos(2 * np.pi * fc1 * t1)

fc2 = 50
fs2 = 32 * fc2
t2 = np.arange(0, 2, 1/fs2)
x2 = np.cos(2 * np.pi * fc2 * t2)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t1, x1, label=f'fc={fc1} Hz')
plt.xlabel('$t=nT_s$')
plt.ylabel('$x[n]$')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t2, x2, label=f'fc={fc2} Hz')
plt.xlabel('$t=nT_s$')
plt.ylabel('$x[n]$')
plt.legend()
plt.tight_layout()
plt.show()

# ===== Задание 2 =====
N = 256
X = np.fft.fft(x1, N)
fs = fs1
df = fs / N
k = np.arange(N)
frequencies = k * df

signal_index = int(fc1 / df)
print(f"Шаг частоты Δf: {df:.2f} Гц")
print(f"Частота сигнала {fc1} Гц соответствует точке ДПФ с индексом {signal_index}.")

# ===== Задание 3 =====
fc_new = 20
x_new = np.cos(2 * np.pi * fc_new * t1)
X_new = np.fft.fft(x_new, N) / N

signal_index_new = int(fc_new / df)
print(f"Частота сигнала {fc_new} Гц соответствует точке ДПФ с индексом {signal_index_new}.")

# ===== Задание 4 =====
N_new = 512
X_new_res = np.fft.fft(x1, N_new) / N_new
df_new = fs / N_new

signal_index_res = int(fc1 / df_new)
print(f"Новая точность Δf: {df_new:.2f} Гц")
print(f"Частота сигнала {fc1} Гц соответствует точке ДПФ с индексом {signal_index_res}.")

# ===== Задание 5 =====
fc3 = 15
x_sum = x1 + np.cos(2 * np.pi * fc3 * t1)
X_sum = np.fft.fft(x_sum, N) / N

plt.figure()
plt.stem(k * df, np.abs(X_sum))
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.title('ДПФ суммы сигналов')
plt.show()

# ===== Задание 6 =====
X_custom = np.array([0, 0, 1j, 0, 0, 0, 0, 0])  # Задан спектр
x_reconstructed = np.fft.ifft(X_custom, 8).real  # ОДПФ

# Визуализация
plt.figure()
plt.stem(np.arange(len(x_reconstructed)), x_reconstructed)
plt.xlabel('Отсчеты')
plt.ylabel('Амплитуда')
plt.title('Сигнал, восстановленный по ОДПФ')
plt.show()


X_custom_16 = np.zeros(16, dtype=complex)
X_custom_16[2] = 2 - 1j
x_reconstructed_16 = np.fft.ifft(X_custom_16).real

plt.figure()
plt.stem(np.arange(len(x_reconstructed_16)), x_reconstructed_16)
plt.xlabel('Отсчеты')
plt.ylabel('Амплитуда')
plt.title('Сигнал, восстановленный по ОДПФ (16 точек)')
plt.show()

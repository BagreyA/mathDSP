import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, get_window, convolve
from scipy.fftpack import fft

# Параметры сигнала
f1 = 80  # Частота 1
f2 = 150  # Частота 2
Ts = 1e-3  # Шаг времени
fs = 1 / Ts  # Частота дискретизации
t = np.arange(0, 100) * Ts  # Временной массив
s = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)  # Исходный сигнал

# Построение спектра исходного сигнала
plt.figure(1)
sp = fft(s)
freqs = np.arange(0, fs, fs / len(s))
plt.plot(freqs[:len(s)//2], np.abs(sp[:len(s)//2]))
plt.title("Модуль спектра исходного сигнала")
plt.xlabel("Частота в герцах [Hz]")
plt.ylabel("Модуль спектра")
plt.grid()

# Параметры фильтра
fc = 78
wc = 2 * np.pi * fc / fs
M = 25  # Порядок фильтра
n = np.arange(-M, M + 1)  # Индексы фильтра

# Импульсная характеристика идеального фильтра
h = np.sinc(2 * fc / fs * n)
h[M] = 2 * fc / fs  # Центральное значение

# Построение импульсной характеристики фильтра
plt.figure(2)
plt.stem(n, h)
plt.title("Импульсная характеристика фильтра h(n)")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid()

# Построение частотной характеристики фильтра
plt.figure(3)
w, hf = freqz(h, 1, worN=8000, fs=fs)
plt.plot(w, 20 * np.log10(np.abs(hf)))
plt.title("Частотная характеристика фильтра H(jw)")
plt.xlabel("Частота в герцах [Hz]")
plt.ylabel("Амплитуда [dB]")
plt.grid()

# Фильтрация сигнала через свертку
y = np.convolve(s, h, mode='same')

# Построение спектра выходного сигнала
plt.figure(4)
yf = fft(y)
plt.plot(freqs[:len(y)//2], np.abs(yf[:len(y)//2]))
plt.title("Модуль спектра выходного сигнала")
plt.xlabel("Частота в герцах [Hz]")
plt.ylabel("Модуль спектра")
plt.grid()

# Оконные функции
windows = {
    "Rectangular": np.ones(len(n)),  # Прямоугольное окно
    "Hamming": get_window("hamming", len(n)),
    "Hann": get_window("hann", len(n)),
    "Blackman": get_window("blackman", len(n)),
}

# Частичное и полное перекрытие
overlap_modes = {"Full": 1.0, "Partial": 0.5}  # Полное и частичное перекрытие


# Анализ сигналов с перекрытием
plt.figure(6, figsize=(12, 8))
for i, (overlap_mode, overlap_ratio) in enumerate(overlap_modes.items()):
    for j, (window_name, window) in enumerate(windows.items()):
        # Формирование окна
        h_windowed = h * window

        # Перекрытие
        if overlap_mode == "Full":
            y_filtered = np.convolve(s, h_windowed, mode='same')
        elif overlap_mode == "Partial":
            # Умножение центральной части
            center = len(t) // 2
            window_length = len(h_windowed)
            start = max(0, center - window_length // 2)
            end = min(len(s), center + window_length // 2)
            s_windowed = np.copy(s)
            s_windowed[start:end] *= window[:end - start]
            y_filtered = np.convolve(s_windowed, h_windowed, mode='same')

        # Построение спектра результата
        yf_filtered = fft(y_filtered)
        plt.subplot(len(overlap_modes), len(windows), i * len(windows) + j + 1)
        plt.plot(freqs[:len(y_filtered)//2], np.abs(yf_filtered[:len(y_filtered)//2]))
        plt.title(f"{window_name}\n{overlap_mode} Overlap")
        plt.xlabel("Частота [Hz]")
        plt.ylabel("Амплитуда")
        plt.grid()

plt.tight_layout()
plt.show()

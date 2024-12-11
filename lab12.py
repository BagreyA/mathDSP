import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Заданные параметры
F1 = 600  # Частота среза, Гц
fd = 4800  # Частота дискретизации, Гц

# Нормированная частота среза
wc = 2 * np.pi * F1 / fd

# Функция для расчета импульсной характеристики
def calculate_impulse_response(wc, N):
    n = np.arange(-N, N + 1)
    h = np.sinc(wc * n / np.pi)  # sinc функция для корректного расчета
    return h, n

# Построение графиков для N = 7 и N = 15
for N in [7, 15]:
    # Расчет импульсной характеристики
    h, n = calculate_impulse_response(wc, N)

    # Частотная характеристика
    w, H = freqz(h, worN=8000, fs=fd)

    # График импульсной характеристики
    plt.figure()
    plt.stem(n, h)
    plt.title(f"Импульсная характеристика (N={N})")
    plt.xlabel("n")
    plt.ylabel("h(n)")
    plt.grid()

    # График АЧХ
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(H)))
    plt.title(f"АЧХ фильтра (N={N})")
    plt.xlabel("Частота, Гц")
    plt.ylabel("Амплитуда, дБ")
    plt.grid()

    # График ФЧХ
    plt.figure()
    plt.plot(w, np.unwrap(np.angle(H)))
    plt.title(f"ФЧХ фильтра (N={N})")
    plt.xlabel("Частота, Гц")
    plt.ylabel("Фаза, рад")
    plt.grid()

plt.show()

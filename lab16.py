import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

# Задание параметров системы
f_s = 8 * 10**3  # Частота дискретизации
T_s = 1 / f_s
t = np.arange(0, 0.04, T_s)  # Временной вектор

# Исходный сигнал
s_t = 5 * np.cos(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 2500 * t)

# Децимация
M = 3  # Фактор децимации
s_t_decimated = s_t[::M]  # Сигнал после децимации
t_decimated = t[::M]  # Обновляем временные метки децимированного сигнала

# Интерполяция
L = 3  # Фактор интерполяции
num_samples = len(s_t_decimated) * L  # Число образцов после интерполяции
s_t_interpolated = resample(s_t_decimated, num_samples)  # Интерполируем сигнал

# Функция для кодирования Хэмминга
def hamming_encode(data):
    """
    Кодирование данных с использованием кода Хэмминга (7, 4).
    :param data: Список бинарных данных для кодирования.
    :return: Закодированные данные.
    """
    encoded_data = []
    
    # Проходимся по входным данным блоками по 4 бита
    for i in range(0, len(data), 4):
        block = data[i:i+4]
        if len(block) < 4:
            # Если блок меньше 4 бит, дополняем нулями
            block += [0] * (4 - len(block))
        
        # Создаём 7-битный блок: первые 4 бита - данные
        encoded_block = [0] * 7
        encoded_block[0:4] = block  # Копируем данные в первые 4 бита
        
        # Вычисляем контрольные биты
        encoded_block[6] = (encoded_block[0] + encoded_block[1] + encoded_block[3]) % 2  # P1
        encoded_block[5] = (encoded_block[0] + encoded_block[2] + encoded_block[3]) % 2  # P2
        encoded_block[4] = (encoded_block[1] + encoded_block[2] + encoded_block[3]) % 2  # P3
        
        # Добавляем кодированный блок в результат
        encoded_data.extend(encoded_block)
    
    return np.array(encoded_data)

# Пример входных данных в бинарном формате
binary_data = [1, 0, 1, 1, 0, 1, 0, 0]

# Кодирование данных
encoded_data = hamming_encode(binary_data)

# Визуализация результатов
plt.figure(figsize=(12, 8))

# Исходный сигнал
plt.subplot(4, 1, 1)
plt.title('Исходный сигнал')
plt.plot(t, s_t)
plt.grid()

# Децимированный сигнал
plt.subplot(4, 1, 2)
plt.title('Децимированный сигнал')
plt.stem(t_decimated, s_t_decimated)
plt.grid()

# Интерполированный сигнал
plt.subplot(4, 1, 3)
plt.title('Интерполированный сигнал')
new_t = np.linspace(0, t_decimated[-1], num_samples)
plt.plot(new_t, s_t_interpolated)
plt.grid()

# Закодированные данные по Хэммингу
plt.subplot(4, 1, 4)
plt.title('Закодированные данные по Хэммингу')
plt.stem(range(len(encoded_data)), encoded_data)
plt.grid()

# Общая настройка графиков
plt.tight_layout()
plt.show()

# Вывод закодированных данных
print("Закодированные данные по Хэммингу:")
print(encoded_data)

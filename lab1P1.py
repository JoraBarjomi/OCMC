import matplotlib.pyplot as plt
import numpy as np
from fft import myRfft

# 13) Оцените влияние разрядности АЦП на спектр сигнала. Для этого
#     нужно написать функцию, которая бы округляла значения отсчетов сигнала,
#     заданного в вашем варианте, до какого-то числа, определяемого разрядностью
#     АЦП. Допустим если у АЦП всего 3 разряда, то диапазон возможных
#     дискретных значений амплитуд временных отсчетов сигнала – это 0..7 (то есть,
#     7=2^3-1). Все значения больше 7 округляются до 7. Для результирующего
#     дискретного сигнала требуется выполнить прямое преобразование Фурье.
#     Сравнить полученный спектр со спектром исходной синусоиды, отсчеты
#     которой не подвергались квантованию по уровню. Вывести среднюю ошибку
#     квантования для случаев, когда разрядность АЦП равна 3/4/5/6.
def adc(y: np.ndarray, raz=3):
    levels = (2**raz) - 1
    y_min, y_max = np.min(y), np.max(y)
    y_norm = (y - y_min) / (y_max - y_min)
    y_quant = np.round(y_norm * levels)
    y_quant = np.clip(y_quant, 0, levels)
    y_restored = y_quant / levels * (y_max - y_min) + y_min
    return y_restored

f = 26
t = np.linspace(0, 1, 10000)
y = 4 * np.cos(2 * np.pi * f * t  +  (np.pi / 12))

print("\n\n")

# 2) Определить максимальную частоту в спектре данного сигнала.
print("Максимальную частоту в спектре данного сигнала: ", f)
# 3) Определить минимальную необходимую частоту 
#    дискретизации полученного сигнала (теорема Котельникова).
print("Минимальную необходимую частоту дискретизации: ", f * 2)

print("\n\n")

# 4) Оцифровать сигнал с полученной частотой дискретизации, выбрав
#   требуемое число отсчетов сигнала на длительности 1 секунда, сохранить
#   полученные значения в массив, пока не озадачиваясь разрядностью АЦП,
#   просто выбранные с частотой дискретизации значения с теми уровнями,
#   которые имеет функция в полученных точках.
fd = 2.1 * f
deltaT = 1 / fd
t2 = np.arange(0, 1, deltaT)
y2 = 4 * np.cos(2 * np.pi * f * t2  +  (np.pi / 12))

# 5) Выполнить прямое дискретное преобразование Фурье для массива
#    временных отсчетов сигнала и оценить ширину данного спектра. А также
#    объем памяти, требуемый для хранения данного массива (тип переменных в
#    массиве – на ваше усмотрение, float, int и пр.).
myfftArr = myRfft(y2)
freqsArr = np.fft.rfftfreq(len(y2), d=deltaT)
print(f"y2 type: {y2.dtype} size: {y2.nbytes}")
aplitude = np.abs(myfftArr)
threshold = 0.1 * np.max(aplitude)
active_indices = np.where(aplitude > threshold)[0]
indMax = active_indices.max()
fMax = freqsArr[indMax] 
print("Ширина спектра fftArr: ", fMax)

print("\n\n")

#7)Увеличьте частоту дискретизации в 4 раза и проделайте задания из п.4-6.
fd4 = fd * 4
deltaT4 = 1 / fd4
t24 = np.arange(0, 1, deltaT4)
y24 = 4 * np.cos(2 * np.pi * f * t24  +  (np.pi / 12))

fftArr4 = myRfft(y24)
freqsArr4 = np.fft.rfftfreq(len(y24), d=deltaT4)
print(f"y24 type: {y24.dtype} size: {y24.nbytes}")

aplitude4 = np.abs(fftArr4)
threshold4 = 0.1 * np.max(aplitude4)
active_indices4 = np.where(aplitude4 > threshold4)[0]
indMax4 = active_indices4.max()
fMax4 = freqsArr4[indMax4] 
print("Ширина спектра fftArr4: ", fMax4)

print("\n\n")

plt.plot(t, y, label="Оригинальный сигнал")
plt.plot(t2, y2, label=(f"Востановленный сигнал {fd}"))
plt.plot(t24, y24, label=(f"Востановленный сигнал {fd4}"))
plt.legend()
plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(freqsArr, np.abs(myfftArr))
ax1.set_title(f"fftArr fd = {fd}")
ax2.plot(freqsArr4, np.abs(fftArr4))
ax2.set_title(f"fftArr fd = {fd4}")
ax1.grid()
ax2.grid()
plt.show()

# 13
for raz in [3, 4, 5, 6]:
    y3 = adc(y24, raz)
    mean_err = np.mean(np.abs(y24 - y3))
    print(f"{raz}: cредняя ошибка квантования = {mean_err:.4f}")

    fft_q = myRfft(y3)

    plt.plot(freqsArr4, np.abs(fftArr4), label="Спектр до квантования", color="black", alpha=0.5)
    plt.plot(freqsArr4, np.abs(fft_q), label=f"Спектр после АЦП ({raz} бит)", color="red", linestyle="--")
    plt.title(f"Разрядности АЦП {raz} бит Ошибка: {mean_err:.4f}")
    plt.xlabel("Частота, Гц")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid(True)
    plt.show()
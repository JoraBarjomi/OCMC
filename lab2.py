import matplotlib.pyplot as plt
import numpy as np

# ДАНО:
Power_bs = 46        # дБм
Sectors_bs = 3       # Число секторов на БС
Power_ue = 24        # дБм
Gain_bs = 21         # дБи
PenetrationM = 15    # дБ
IM = 1               # дБ
f_ghz = 1.8          # ГГц
f_mhz = f_ghz * 1000 # МГц
Band_ul = 10         # МГц
Band_dl = 20         # МГц
Noise_bs = 2.4       # дБ
Noise_ue = 6         # дБ
Required_sinr_dl = 2 # дБ
Required_sinr_ul = 4 # дБ
Feeder_loss = 2.9    # дБ 
MIMO_gain = 3        # дБ

Area_total = 100     # кв. км
Area_micro = 4       # кв. км (ТЦ и бизнес-центры: микро/фемтосоты)
Area_macro = Area_total - Area_micro  # кв. км (остальная территория: макросоты)
h_bs = 30            # Высота подвеса антенны БС, м
h_ms = 1.5           # Высота антенны мобильного абонента, м

# Параметры городской застройки для Walfish-Ikegami
h_roof = 20          # h — средняя высота зданий, м
w_street = 20        # w — средняя ширина улиц, м
b_build = 40         # b — среднее расстояние между зданиями, м
phi = 90             # φ — угол между направлением сигнала и улицей, град


# 1) Бюджет восходящего канала UL
# ThermalNoise = -174 + 10*log10(BW[Гц])
ThermalNoise_ul = -174 + 10 * np.log10(Band_ul * 1e6)
# RxSens = NoiseFigure + ThermalNoise + RequiredSINR 
RxSensBs = Noise_bs + ThermalNoise_ul + Required_sinr_ul
# MAPL_UL = TxPowerUE - FeederLoss + AntGainBS + MIMOGain - IM - PenetrationM - RxSensBS
Mapl_UL = Power_ue - Feeder_loss + Gain_bs + MIMO_gain - IM - PenetrationM - RxSensBs

# 2) Бюджет нисходящего канала DL
ThermalNoise_dl = -174 + 10 * np.log10(Band_dl * 1e6)
RxSensUe = Noise_ue + ThermalNoise_dl + Required_sinr_dl
Mapl_DL = Power_bs - Feeder_loss + Gain_bs + MIMO_gain - IM - PenetrationM - RxSensUe

print(f"RxSens BS: {RxSensBs:.2f} дБм | RxSens UE: {RxSensUe:.2f} дБм")
print(f"MAPL_UL: {Mapl_UL:.2f} дБ")
print(f"MAPL_DL: {Mapl_DL:.2f} дБ\n")


# 3) Модели распространения.
d_m = np.linspace(1, 20000, 200000)


#UMiNLOS
def path_loss_uminlos(d):
    return 26 * np.log10(f_ghz) + 22.7 + 36.7 * np.log10(d)


#COST 231 Hata
def path_loss_cost231hata(d, A=46.3, B=33.9, L_clutter=3.0):
    d_km = d / 1000
    a = 3.2 * np.log10(11.75 * h_ms) ** 2 - 4.97
    s = np.where(
        d_km >= 1,
        44.9 - 6.55 * np.log10(f_mhz),
        (47.88 + 13.9 * np.log10(f_mhz) - 13.9 * np.log10(h_bs)) / np.log10(50),
    )
    return (A + B * np.log10(f_mhz) - 13.82 * np.log10(h_bs) - a
            + s * np.log10(d_km) + L_clutter)


#Walfish-Ikegami
def path_loss_walfish_los(d):
    d_km = d / 1000
    return 42.6 + 20 * np.log10(f_mhz) + 26 * np.log10(d_km)        


def path_loss_walfish_nlos(d):
    d_km = d / 1000
    L0 = 32.44 + 20 * np.log10(f_mhz) + 20 * np.log10(d_km)

    if phi < 35:
        L_ori = -10 + 0.354 * phi
    elif phi < 55:
        L_ori = 2.5 + 0.075 * (phi - 35)
    else:
        L_ori = 4.0 - 0.114 * (phi - 55)

    L2 = (-16.9 - 10 * np.log10(w_street) + 10 * np.log10(f_mhz)
          + 20 * np.log10(h_roof - h_ms) + L_ori)

    if h_bs > h_roof:
        L11 = -18 * np.log10(1 + h_bs - h_roof)                       
        ka = 54.0                                                     
        kd = 18.0                                                       
    else:
        L11 = 0.0
        ka = np.where(d_km > 0.5,
                      54 - 0.8 * (h_bs - h_roof),
                      54 - 0.8 * (h_bs - h_roof) * d_km / 0.5)
        kd = 18 - 15 * (h_bs - h_roof) / h_roof
    kf = -4 + 0.7 * (f_mhz / 925 - 1)                                  
    L1 = (L11 + ka + kd * np.log10(d_km) + kf * np.log10(f_mhz)
          - 9 * np.log10(b_build))                                     

    return np.where(L1 + L2 > 0, L0 + L1 + L2, L0)


pl_umi = path_loss_uminlos(d_m)
pl_hata = path_loss_cost231hata(d_m)
pl_wi_nlos = path_loss_walfish_nlos(d_m)
pl_wi_los = path_loss_walfish_los(d_m)


# 4) Радиус соты, площадь сайта, количество БС
def find_radius(pl, mapl):
    assert pl[0] <= mapl <= pl[-1], "MAPL вне диапазона кривой PL(d)"
    return float(np.interp(mapl, pl, d_m))


K_SECTOR = {2: 1.73, 3: 1.95, 6: 2.6}


def cells_needed(R_m, area_km2):
    S_site = K_SECTOR[Sectors_bs] * (R_m / 1000) ** 2
    return S_site, int(np.ceil(area_km2 / S_site))


results = {}
for zone, pl, area in [("Микро (UMiNLOS)", pl_umi, Area_micro),
                       ("Макро (COST 231 Hata)", pl_hata, Area_macro)]:
    d_ul = find_radius(pl, Mapl_UL)
    d_dl = find_radius(pl, Mapl_DL)
    R = min(d_ul, d_dl)  
    S_site, n_sites = cells_needed(R, area)
    results[zone] = (d_ul, d_dl, R, S_site, n_sites, area)
    print(f"{zone}, площадь зоны {area} кв. км")
    print(f"Радиус UL: {d_ul:.1f} м | Радиус DL: {d_dl:.1f} м")
    print(f"Ограничивает {'UL' if d_ul <= d_dl else 'DL'}, R = {R:.1f} м")
    print(f"Площадь одной {Sectors_bs}-секторной БС: {S_site:.4f} кв. км")
    print(f"Требуется БС: {n_sites}\n")


total_sites = sum(r[4] for r in results.values())
print(f"ИТОГО БС на {Area_total} кв. км: {total_sites}")


# 5) Графики
fig1 = plt.figure(figsize=(12, 7))
plt.plot(d_m, pl_umi, label='UMiNLOS (микро/фемто)', color='red', linewidth=2)
plt.plot(d_m, pl_hata, label='COST 231 Hata (макро)', color='purple', linewidth=2)
plt.plot(d_m, pl_wi_nlos, label='Walfish-Ikegami NLOS', color='orange', linewidth=2, linestyle='-.')
plt.plot(d_m, pl_wi_los, label='Walfish-Ikegami LOS', color='gray', linewidth=1, linestyle=':')
plt.axhline(y=Mapl_UL, color='blue', linestyle='--', label=f'MAPL_UL = {Mapl_UL:.1f} дБ')
plt.axhline(y=Mapl_DL, color='green', linestyle='--', label=f'MAPL_DL = {Mapl_DL:.1f} дБ')
plt.title('Сравнение моделей распространения радиосигнала и бюджет каналов LTE')
plt.xlabel('Расстояние между приемником и передатчиком, м')
plt.ylabel('Потери сигнала (Path Loss), дБ')
plt.grid(True, which='both', linestyle=':')
plt.legend(loc='lower right')
plt.xlim(0, 3000)
plt.ylim(40, 165)
fig1.savefig('fig1_models.png', dpi=200, bbox_inches='tight')
 
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pl, name, color in [(axes[0], pl_umi, 'UMiNLOS (микро/фемто)', 'red'),
                            (axes[1], pl_hata, 'COST 231 Hata (макро)', 'purple')]:
    d_ul = find_radius(pl, Mapl_UL)
    d_dl = find_radius(pl, Mapl_DL)
    ax.plot(d_m, pl, color=color, linewidth=2, label=name)
    ax.axhline(Mapl_UL, color='blue', linestyle='--', label=f'MAPL_UL = {Mapl_UL:.1f} дБ')
    ax.axhline(Mapl_DL, color='green', linestyle='--', label=f'MAPL_DL = {Mapl_DL:.1f} дБ')
    ax.vlines(d_ul, 50, Mapl_UL, color='blue', linewidth=2)
    ax.vlines(d_dl, 50, Mapl_DL, color='green', linewidth=2)
    ax.text(d_ul, 53, f'd_UL = {d_ul:.0f} м ', color='blue', ha='right', rotation=90)
    ax.text(d_dl, 53, f'd_DL = {d_dl:.0f} м ', color='green', ha='right', rotation=90)
    ax.set_xlim(0, 1.25 * max(d_ul, d_dl))
    ax.set_ylim(50, 165)
    ax.set_xlabel('Расстояние между приемником и передатчиком, м')
    ax.set_ylabel('Потери сигнала, дБ')
    ax.grid(True, linestyle=':')
    ax.legend(loc='upper left')
fig2.suptitle('Определение радиуса соты в UL и DL')
fig2.savefig('fig2_radius.png', dpi=200, bbox_inches='tight')
 
plt.show()
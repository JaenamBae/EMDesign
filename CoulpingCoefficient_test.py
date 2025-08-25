import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import numpy as np

file_path = "CouplingCoefficients.csv"
#file_path = "FluxLinkage_SPM.csv"

# 데이터프레임으로 읽기
df = pd.read_csv(file_path)

# 유일한 값 추출
harmonic = sorted(df["Harmonic_order"].unique())
tau_so = sorted(df["tau_so"].unique())
ratio = sorted(df["SlotPitchRatio"].unique())

# 빈 배열 생성 (3차원)
flux_array = np.empty((len(harmonic), len(tau_so), len(ratio)))

# 값 채우기
for i, h in enumerate(harmonic):
    for j, w in enumerate(tau_so):
        for k, r in enumerate(ratio):
            value = df[(df["Harmonic_order"] == h) &
                       (df["tau_so"] == w) &
                       (df["SlotPitchRatio"] == r)]["FluxLinkage(Winding)"].values
            flux_array[i, j, k] = value[0] if len(value) > 0 else np.nan

k_wc = np.zeros_like(flux_array)

# for문으로 계산
for i, h in enumerate(harmonic):
    for j in range(len(tau_so)):
        for k, tau_s in enumerate(ratio):
            base_value = flux_array[0, j, k]
            current_value = flux_array[i, j, k]
            k_wp1 = np.sin(np.pi * tau_s / 2)
            k_wpn = np.sin(np.pi * tau_s * h / 2)
            if base_value != 0:
                coupling_coef = (current_value / base_value) * h * k_wp1
                # if coupling_coef > 1: coupling_coef = 1
                # if coupling_coef < - 1: coupling_coef = -1
                k_wc[i, j, k] = coupling_coef
            else:
                k_wc[i, j, k] = 0  # 또는 np.nan, 적절히 처리

# 보간기 생성
interpolator = RegularGridInterpolator(
    (harmonic, tau_so, ratio),
    k_wc,
    method='linear',
    bounds_error=False,
    fill_value=None
)

harmonic = 3
tau_so = 0.1
slot_pitch_ratio = 14/12

k_wc = interpolator([harmonic, tau_so, slot_pitch_ratio])
print(k_wc)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정 (윈도우용)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 읽기 (인코딩 맞게 조정)
df = pd.read_csv("Bg.csv", encoding='cp949')
x = df.iloc[:, 0].values
f_x = df.iloc[:, 1].values

N = len(f_x)
dx = x[1] - x[0]
L = x[-1] - x[0] + (x[1] - x[0])  # 주기 길이 (샘플 간격 고려)

# FFT 결과
fft_vals = np.fft.fft(f_x)
half_N = N // 2

# bn은 허수부에서 추출 (부호 포함)
b_n = -np.imag(fft_vals[:half_N]) * 2 / N  # 정규화 포함 (2/N은 scipy convention)
harmonics = np.arange(1, 21)

# 재합성: f_reconstructed(x) = sum b_n * sin(n * w * x)
w = 2 * np.pi / L
f_recon = np.zeros_like(x)

for n in harmonics:
    f_recon += b_n[n] * np.sin(n * w * x)  # bn[n]는 n번째 고조파

# 결과 비교
plt.figure(figsize=(10, 5))
plt.plot(x, f_x, label="원래 데이터", linewidth=2)
plt.plot(x, f_recon, '--', label="재합성된 파형 (1~20고조파)", linewidth=2)
plt.title("원점대칭 함수의 bn 기반 재합성 결과")
plt.xlabel("x (거리 또는 시간)")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 첫 번째는 DC 성분이므로 제외하고 1~20 고조파까지 출력
plt.stem(harmonics, b_n[1:21], basefmt=" ")
plt.xlabel("고조파 번호")
plt.ylabel("bn 계수 (부호 포함)")
plt.title("원점대칭 파형의 푸리에 bn 성분")
plt.grid(True)
plt.show()

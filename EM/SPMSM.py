import numpy as np
import matplotlib.pyplot as plt

mu_0   = 4*np.pi*1e-7

class SPMSM:
    def __init__(self, info: dict) -> None:
        self._info = info

    def calculate_coefficients(self, n: int) -> np.ndarray:
        R_i = self._info["R_i"]
        R_r = self._info["R_r"]
        R_m = self._info["R_m"]
        R_s = self._info["R_s"]
        R_so = self._info["R_so"]
        p = self._info["p"]
        alpha_p = self._info["alpha_p"]
        B_r = self._info["B_r"]
        mu_r1 = self._info["mu_r1"]
        mu_r2 = self._info["mu_r2"]
        mu_r3 = self._info["mu_r3"]
        mu_r4 = self._info["mu_r4"]

        # 수치계산 오버플로우 방지를 위해 반경을 모두 R_r 기준으로 표준화시킴
        R_i = R_i / R_r
        R_m = R_m / R_r
        R_s = R_s / R_r
        R_so = R_so / R_r
        R_r = R_r / R_r

        # -------------------------------
        # parallel 착자 가정
        k1n = (n * p + 1) * alpha_p * np.pi / (2 * p)
        k2n = (n * p - 1) * alpha_p * np.pi / (2 * p)
        M_1n = np.sin(k1n) / k1n
        M_2n = np.sin(k2n) / k2n

        H_crn = B_r / (mu_0 * mu_r2) * alpha_p * (M_1n + M_2n)
        H_ctn = B_r / (mu_0 * mu_r2) * alpha_p * (M_1n - M_2n)

        # Radial 착자 가정
        kk = n * np.pi * alpha_p / 2
        #H_crn = 2 * B_r / (mu_0 * mu_r2) * alpha_p * np.sin(kk) / kk
        #H_ctn = 0

        # -------------------------------
        alpha = n * p - 1
        beta = n * p + 1

        # -------------------------------
        C_11 = np.power(R_i, alpha + beta)
        C_12 = 1

        C_21 = np.power(R_r, alpha)
        C_22 = np.power(R_r, -beta)
        C_23 = -np.power(R_r, alpha)
        C_24 = -np.power(R_r, -beta)

        C_31 = mu_r2 * np.power(R_r, alpha)
        C_32 = -mu_r2 * np.power(R_r, -beta)
        C_33 = -mu_r1 * np.power(R_r, alpha)
        C_34 = mu_r1 * np.power(R_r, -beta)

        C_43 = np.power(R_m, alpha)
        C_44 = np.power(R_m, -beta)
        C_45 = -np.power(R_m, alpha)
        C_46 = -np.power(R_m, -beta)

        C_53 = mu_r3 * np.power(R_m, alpha)
        C_54 = -mu_r3 * np.power(R_m, -beta)
        C_55 = -mu_r2 * np.power(R_m, alpha)
        C_56 = mu_r2 * np.power(R_m, -beta)

        C_65 = np.power(R_s, alpha)
        C_66 = np.power(R_s, -beta)
        C_67 = -np.power(R_s, alpha)
        C_68 = -np.power(R_s, -beta)

        C_75 = mu_r4 * np.power(R_s, alpha)
        C_76 = -mu_r4 * np.power(R_s, -beta)
        C_77 = -mu_r3 * np.power(R_s, alpha)
        C_78 = mu_r3 * np.power(R_s, -beta)

        C_87 = np.power(R_so, alpha + beta)
        C_88 = 1

        F_2 = -mu_0 * mu_r2 * (H_ctn + n * p * H_crn) / (1 - np.power(n * p, 2))
        F_3 = -mu_0 * mu_r1 * mu_r2 * (H_crn + n * p * H_ctn) / (1 - np.power(n * p, 2))
        F_4 = mu_0 * mu_r2 * (H_ctn + n * p * H_crn) / (1 - np.power(n * p, 2))
        F_5 = mu_0 * mu_r2 * mu_r3 * (H_crn + n * p * H_ctn) / (1 - np.power(n * p, 2))

        CC = np.array([[C_11, C_12, 0, 0, 0, 0, 0, 0],
                       [C_21, C_22, C_23, C_24, 0, 0, 0, 0],
                       [C_31, C_32, C_33, C_34, 0, 0, 0, 0],
                       [0, 0, C_43, C_44, C_45, C_46, 0, 0],
                       [0, 0, C_53, C_54, C_55, C_56, 0, 0],
                       [0, 0, 0, 0, C_65, C_66, C_67, C_68],
                       [0, 0, 0, 0, C_75, C_76, C_77, C_78],
                       [0, 0, 0, 0, 0, 0, C_87, C_88]])
        FF = np.array([0, F_2, F_3, F_4, F_5, 0, 0, 0])

        det = np.linalg.det(CC)
        if np.isclose(det, 0):
            return np.zeros_like(FF)

        iCC = np.linalg.inv(CC)
        XX = np.matmul(iCC, FF)
        return XX

    def calculate_harmonic(self, r, n):
        if n % 2 == 0:
            return 0, 0

        p = self._info["p"]
        R_r = self._info["R_r"]
        norm_r = r / R_r

        XX = self.calculate_coefficients(n)
        D3 = XX[4]
        E3 = XX[5]

        # -------------------------------
        # 특정 반경에서 자속밀도 데이터 뽑기: Maxwell 과 데이터 비교 위함
        alpha = n * p - 1
        beta = n * p + 1
        B_rn = n * p * (D3 * np.power(norm_r, alpha) + E3 * np.power(norm_r, -beta))
        B_tn = -n * p * (D3 * np.power(norm_r, alpha) - E3 * np.power(norm_r, -beta))

        return B_rn, B_tn

    def calculate_harmonic_field(self, r, theta, n):
        B_rn, B_tn = self.calculate_harmonic(r, n)

        B_r = B_rn * np.cos(np.radians(n * theta))
        B_t = B_tn * np.sin(np.radians(n * theta))

        return B_r, B_t

    def plotBg(self, n_harmonics=20, with_waveform=True, with_fft=True):
        R_m = self._info["R_m"]
        R_s = self._info["R_s"]
        R_r = self._info["R_r"]

        g = R_s - R_m
        r = R_m + g / 2

        if with_waveform:
            n_sample = 3600  # 샘플링 데이터 수
            theta = 360 * np.arange(n_sample) / n_sample
            B_r = np.zeros_like(theta)

            for n in range(1, int(n_harmonics/2), 2):
                B_rn, _ = self.calculate_harmonic_field(r, theta, n)
                B_r = B_r + B_rn

            plt.figure(figsize=(12, 6))
            plt.plot(theta, B_r, color="blue", linewidth=3)
            plt.title("Airgap Flux Density")
            plt.xlabel("Electric Angle (degrees)")
            plt.ylabel("Airgap Flux Density [T]")
            plt.xlim([0, 360])
            plt.xticks(np.arange(0, 360, 60))
            plt.grid(True)
            plt.show()

        if with_fft:
            Bg_FFT = np.zeros(n_harmonics)
            for n in range(n_harmonics):
                B_rn, _ = self.calculate_harmonic(r, n)
                Bg_FFT[n] = B_rn

            # 하모닉 차수 생성 (0부터 시작)
            harmonic_order = np.arange(len(Bg_FFT))

            # 크기 계산 (복소수인 경우 절댓값 사용)
            magnitudes = Bg_FFT / Bg_FFT[1]
            #print(magnitudes)

            # 그래프 그리기
            odd_mask = (harmonic_order % 2 == 1) & (harmonic_order <= 22)
            odd_orders = harmonic_order[odd_mask]
            odd_magnitudes = magnitudes[odd_mask]

            plt.figure(figsize=(8, 5))
            plt.stem(odd_orders, odd_magnitudes, basefmt="k-")

            # 막대 위에 퍼센트 값 표시 (소수점 둘째 자리까지)
            for x, y in zip(odd_orders, odd_magnitudes):
                percent = y * 100
                plt.text(x, y + 0.02, f"{percent:.2f}%", ha='center', va='bottom', fontsize=9)

            plt.title("Airgap Flux Density Harmonics")
            plt.xlabel("Harmonic Order")
            plt.ylabel("Magnitude [pu]")
            plt.grid(True)
            plt.xticks(odd_orders)  # ex) [1, 3, 5, 7, ...]

            plt.show()
import numpy as np
from numpy.ma.core import arange

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

        # 수치계산 오버플로우 방지를 위해 반경을 모두 R_i 기준으로 표준화시킴
        R_i = R_i / R_r
        R_m = R_m / R_r
        R_s = R_s / R_r
        R_so = R_so / R_r
        R_r = R_r / R_r

        # -------------------------------
        # 자속축이 x축에 정렬된 상태를 가정
        k1n = (n * p + 1) * alpha_p * np.pi / (2 * p)
        k2n = (n * p - 1) * alpha_p * np.pi / (2 * p)
        M_1n = np.sin(k1n) / k1n
        M_2n = np.sin(k2n) / k2n

        H_crn = B_r / (mu_0 * mu_r2) * alpha_p * (M_1n + M_2n)
        H_ctn = B_r / (mu_0 * mu_r2) * alpha_p * (M_1n - M_2n)

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

        iCC = np.linalg.inv(CC)
        XX = np.matmul(iCC, FF)

        return XX

    def calculate_harmonic_field(self, r, theta, D3, E3, n):
        p = self._info["p"]

        # -------------------------------
        # 특정 반경에서 자속밀도 데이터 뽑기: Maxwell 과 데이터 비교 위함
        alpha = n * p - 1
        beta = n * p + 1
        B_r = n * p * (D3 * np.power(r, alpha) + E3 * np.power(r, -beta)) * np.cos(n * p * theta)
        B_t = -n * p * (D3 * np.power(r, alpha) - E3 * np.power(r, -beta)) * np.sin(n * p * theta)

        return B_r, B_t

    def airgap_fluxdensity(self, n_harmonics, n_samples):
        # 공극 중앙에서의 자속밀도(법선, 접선) 계산
        R_m = self._info["R_m"]
        R_s = self._info["R_s"]
        g = R_s - R_m
        r = R_m + g*0.5

        theta = 2 * np.arange(n_samples) * np.pi / n_samples
        B_r = np.zeros(n_samples)
        B_t = np.zeros(n_samples)

        # 홀수 고려파만 고려되므로최대 차수는 2*n_harmonics + 1 이다
        B_rm = np.zeros(n_harmonics)
        B_tm = np.zeros(n_harmonics)

        for i in range(n_harmonics):
            n = i * 2 + 1
            XX = self.calculate_coefficients(n)
            D3 = XX[4]
            E3 = XX[5]
            B_rn, B_tn = self.calculate_harmonic_field(r, theta, D3, E3, n)
            B_rm[i] = max(B_rn)
            B_tm[i] = max(B_tn)

            B_r = B_r + B_rn
            B_t = B_t + B_tn

        return theta, B_r, B_t
